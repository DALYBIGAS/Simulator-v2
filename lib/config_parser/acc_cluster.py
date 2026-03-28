from .acc_cluster import AccCluster
from .accelerator import Accelerator
from .dma import DMA, StreamDMA
from .variable import Variable, PortedConnection
from .op import Operand, Operation, FusableOps

class AccCluster:
    def __init__(
        self,
        name: str,
        dmas,
        accs,
        base_address: int,
        working_dir: str,
        config_path: str,
        hw_config_path: str = None
    ):
        self.name = name
        self.dmas = dmas
        self.accs = accs
        self.base_address = base_address
        self.top_address = base_address
        self.config_path = config_path
        # Do this to point the hardware configuration to the
        # sys config YAML file when HWPath isn't defined
        self.hw_config_path = hw_config_path
        self.fusable_ops = FusableOps()
        self.process_config(working_dir=working_dir)

    def _get_dimensions(self, dType):
        """Helper function to extract the number of dimensions from a dType string."""
        if not dType.startswith("<") or not dType.endswith(">"):
            raise ValueError(f"Invalid dType format: {dType}")
        # Remove the angle brackets and split by 'x'
        shape_str = dType[1:-1].split("x")
        # The last element is the data type (e.g., "f32"), so ignore it
        return len(shape_str) - 1

    def process_config(self, working_dir):
        dma_class = []
        acc_class = []
        top_address = self.base_address

        # Parse DMAs
        for dma in self.dmas:
            for device_dict in dma['DMA']:
                # Decide whether the DMA is NonCoherent or Stream
                if 'NonCoherent' in device_dict['Type']:
                    pio_size = 21
                    pio_masters = []
                    if 'PIOMaster' in device_dict:
                        pio_masters.extend(
                            (device_dict['PIOMaster'].split(',')))
                    if 'InterruptNum' in device_dict:
                        dma_class.append(
                            DMA(
                                name=device_dict['Name'],
                                pio=pio_size,
                                pio_masters=pio_masters,
                                address=top_address,
                                dmaType=device_dict['Type'],
                                int_num=device_dict['InterruptNum'],
                                size=device_dict['BufferSize'],
                                maxReq=device_dict['MaxReqSize']
                            )
                        )
                    else:
                        dma_class.append(
                            DMA(
                                name=device_dict['Name'],
                                pio=pio_size,
                                pio_masters=pio_masters,
                                address=top_address,
                                dmaType=device_dict['Type'],
                                int_num=device_dict['BufferSize'],
                                size=device_dict['MaxReqSize']
                            )
                        )
                    aligned_inc = int(pio_size) + (64 - (int(pio_size) % 64))
                    top_address = top_address + aligned_inc
                elif 'Stream' in device_dict['Type']:
                    pio_size = 32
                    statusSize = 4
                    pio_masters = []

                    alignedStatusInc = int(statusSize) + \
                        (64 - (int(statusSize) % 64))
                    aligned_inc = int(pio_size) + (64 - (int(pio_size) % 64))

                    statusAddress = top_address + aligned_inc

                    if 'PIOMaster' in device_dict:
                        pio_masters.extend(
                            (device_dict['PIOMaster'].split(',')))
                    # Can come back and get rid of this if/else tree
                    if 'ReadInt' in device_dict:
                        if 'WriteInt' in device_dict:
                            dma_class.append(
                                StreamDMA(
                                    name=device_dict['Name'],
                                    pio=pio_size,
                                    pio_masters=pio_masters,
                                    address=top_address,
                                    statusAddress=statusAddress,
                                    dmaType=device_dict['Type'],
                                    rd_int=device_dict['ReadInt'],
                                    wr_int=device_dict['WriteInt'],
                                    size=device_dict['BufferSize']
                                )
                            )
                        else:
                            dma_class.append(
                                StreamDMA(
                                    name=device_dict['Name'],
                                    pio=pio_size,
                                    pio_masters=pio_masters,
                                    address=top_address,
                                    statusAddress=statusAddress,
                                    dmaType=device_dict['Type'],
                                    rd_int=device_dict['ReadInt'],
                                    wr_int=None,
                                    size=device_dict['BufferSize']
                                )
                            )
                    elif 'WriteInt' in device_dict:
                        dma_class.append(
                            StreamDMA(
                                name=device_dict['Name'],
                                pio=pio_size,
                                pio_masters=pio_masters,
                                address=top_address,
                                statusAddress=statusAddress,
                                dmaType=device_dict['Type'],
                                rd_int=None,
                                wr_int=device_dict['WriteInt'],
                                size=device_dict['BufferSize']
                            )
                        )
                    else:
                        dma_class.append(
                            StreamDMA(
                                name=device_dict['Name'],
                                pio=pio_size,
                                pio_masters=pio_masters,
                                address=top_address,
                                statusAddress=statusAddress,
                                dmaType=device_dict['Type'],
                                rd_int=None,
                                wr_int=None,
                                size=device_dict['BufferSize']
                            )
                        )

                    # Increment Top Address
                    top_address = top_address + aligned_inc + alignedStatusInc
        # Parse Accelerators
        for acc in self.accs:
            # Save top_address before processing accelerator config for copies
            start_addr = self.top_address
            copies = 1
            pio_size_val = None

            name = None
            pio_masters = []
            stream_in = []
            stream_out = []
            local_connections = []
            # variables will now be set per accelerator copy later
            pio_address = None
            pio_size = None
            int_num = None
            ir_path = None
            hw_config_path = self.hw_config_path
            debug = False
            type = None
            operation = None

            for device_dict in acc['Accelerator']:
                if 'Name' in device_dict:
                    name = device_dict['Name']
                if 'Type' in device_dict:
                    type = device_dict['Type']
                    if type == "Mover":
                        self.fusable_ops.add(device_dict['Source'], device_dict['Destination'])
                if 'PIOSize' in device_dict:
                    # Compute PIO size and save for copies
                    pio_size = device_dict['PIOSize'] + (64 - (device_dict['PIOSize'] % 64))
                    pio_size_val = pio_size
                    # For non-copied accelerator, assign and update address
                    self.top_address = self.top_address + pio_size
                    if ((self.top_address + pio_size) % 64) != 0:
                        print("Acc Error: " + hex(self.top_address - pio_size))
                # ...existing code for other parameters...
                if 'PIOMaster' in device_dict:
                    pio_masters.extend((device_dict['PIOMaster'].split(',')))
                if 'StreamIn' in device_dict:
                    stream_in.extend((device_dict['StreamIn'].split(',')))
                if 'StreamOut' in device_dict:
                    stream_out.extend((device_dict['StreamOut'].split(',')))
                if 'LocalSlaves' in device_dict:
                    local_connections.extend((device_dict['LocalSlaves'].split(',')))
                if 'InterruptNum' in device_dict:
                    int_num = device_dict['InterruptNum']
                if 'IrPath' in device_dict:
                    ir_path = device_dict['IrPath']
                if 'HWPath' in device_dict:
                    hw_config_path = device_dict['HWPath']
                if 'Debug' in device_dict:
                    debug = device_dict['Debug']
                if 'Operation' in device_dict:
                    for op in device_dict['Operation']:
                        operands = []
                        results = []
                        for operand in op['Operands']:
                            if operand['InOut'] == 'In':
                                operands.append(Operand(operand['Name'], 'In', operand['Dtype'], operand['VarName']))
                            else:
                                results.append(Operand(operand['Name'], 'Out', operand['Dtype'], operand['VarName']))
                        operation = Operation(op['Name'], operands, results, op['Tile'])
                if 'Copies' in device_dict:
                    copies = device_dict['Copies']

            # Revert top_address if accelerator is to be copied so Var allocation is per copy
            if copies > 1:
                self.top_address = start_addr

            # Instantiate accelerator copies or a single instance with Var parsing
            if copies > 1:
                for i in range(copies):
                    copy_vars = []
                    # Process Var block for this accelerator copy if exists
                    if 'Var' in acc:
                        for var_group in acc['Var']:
                            for var in var_group:
                                varParams = dict(var)
                                varParams['Address'] = self.top_address
                                varParams['AccName'] = name
                                if varParams['Type'] == "Stream":
                                    aligned_inc = int(var['StreamSize'] + 4) + (64 - (int(var['StreamSize'] + 4) % 64))
                                    varParams['StatusAddress'] = self.top_address + aligned_inc
                                copy_vars.append(Variable(**varParams))
                                if varParams['Type'] == "SPM":
                                    aligned_inc = int(var['Size']) + (64 - (int(var['Size']) % 64))
                                    self.top_address += aligned_inc
                                elif varParams['Type'] == "Stream":
                                    statusSize = 4
                                    aligned_inc = int(var['StreamSize'] + 4) + (64 - (int(var['StreamSize'] + 4) % 64))
                                    status_inc = int(statusSize) + (64 - (int(statusSize) % 64))
                                    self.top_address += aligned_inc + status_inc
                                elif varParams['Type'] == "RegisterBank":
                                    aligned_inc = int(var['Size']) + (64 - (int(var['Size']) % 64))
                                    self.top_address += aligned_inc

                    copy_pio_address = None
                    if pio_size_val is not None:
                        copy_pio_address = self.top_address
                        self.top_address += pio_size_val
                    copy_name = f"{name}_{i}"
                    acc_class.append(
                        Accelerator(
                            name=copy_name,
                            pio_masters=pio_masters,
                            local_connections=local_connections,
                            address=copy_pio_address,
                            size=pio_size,
                            stream_in=stream_in,
                            stream_out=stream_out,
                            int_num=int_num,
                            working_dir=working_dir,
                            ir_path=ir_path,
                            config_path=self.config_path,
                            hw_config_path=hw_config_path,
                            variables=copy_vars,
                            debug=debug,
                            type=type,
                            operation=operation
                        )
                    )
            else:
                single_vars = []
                if 'Var' in acc:
                    for var_group in acc['Var']:
                        for var in var_group:
                            varParams = dict(var)
                            varParams['Address'] = self.top_address
                            varParams['AccName'] = name
                            if varParams['Type'] == "Stream":
                                aligned_inc = int(var['StreamSize'] + 4) + (64 - (int(var['StreamSize'] + 4) % 64))
                                varParams['StatusAddress'] = self.top_address + aligned_inc
                            single_vars.append(Variable(**varParams))
                            if varParams['Type'] == "SPM":
                                aligned_inc = int(var['Size']) + (64 - (int(var['Size']) % 64))
                                self.top_address += aligned_inc
                            elif varParams['Type'] == "Stream":
                                statusSize = 4
                                aligned_inc = int(var['StreamSize'] + 4) + (64 - (int(var['StreamSize'] + 4) % 64))
                                status_inc = int(statusSize) + (64 - (int(statusSize) % 64))
                                self.top_address += aligned_inc + status_inc
                            elif varParams['Type'] == "RegisterBank":
                                aligned_inc = int(var['Size']) + (64 - (int(var['Size']) % 64))
                                self.top_address += aligned_inc
                acc_class.append(
                    Accelerator(
                        name=name,
                        pio_masters=pio_masters,
                        local_connections=local_connections,
                        address=pio_address,
                        size=pio_size,
                        stream_in=stream_in,
                        stream_out=stream_out,
                        int_num=int_num,
                        working_dir=working_dir,
                        ir_path=ir_path,
                        config_path=self.config_path,
                        hw_config_path=hw_config_path,
                        variables=single_vars,
                        debug=debug,
                        type=type,
                        operation=operation
                    )
                )
        self.accs = acc_class
        self.dmas = dma_class
        self.top_address = top_address

    def genConfig(self):
        lines = []
        # Need to add some customization here. Consider this a placeholder
        # Also need to edit AccCluster.py's addresses to match the gem5 supported ones
        lines.append("def build" + self.name +
                     "(options, system, clstr):" + "\n")
        lines.append("	local_low = " + hex(self.base_address))
        lines.append("	local_high = " + hex(self.top_address))
        lines.append("	local_range = AddrRange(local_low, local_high)")
        lines.append(
            "	external_range = [AddrRange(0x00000000, local_low-1), AddrRange(local_high+1, 0xFFFFFFFF)]")
        lines.append(
            "	system.iobus.mem_side_ports = clstr.local_bus.cpu_side_ports")
        # Need to define l2coherency in the YAML file?
        lines.append(
            "	clstr._connect_caches(system, options, l2coherent=False)")
        lines.append("	gic = system.realview.gic")
        lines.append("")

        return lines

    def genDriver(self, output_file):
        with open(output_file, 'w') as f:
            # Write header includes and macros
            f.write("#include <stdint.h>\n")
            f.write(f'#include "../{self.name}_hw_defines.h"\n\n')

            # Generate a call function for each accelerator
            for acc in self.accs:
                acc_name = acc.name  # Get accelerator name
                # Generate function arguments
                args = []
                for operand in acc.operation.operands:
                    opName = operand.name
                    num_dims = self._get_dimensions(operand.dType)
                    args.extend([
                        f"float* {opName}_allocated", f"float* {opName}_aligned",
                        f"int64_t {opName}_offset"
                    ])
                    for dim in range(num_dims):
                        args.append(f"int64_t {opName}_size{dim}")
                    for dim in range(num_dims):
                        args.append(f"int64_t {opName}_stride{dim}")

                for result in acc.operation.results:
                    opName = result.varName
                    num_dims = self._get_dimensions(result.dType)
                    args.extend([
                        f"float* {opName}_allocated", f"float* {opName}_aligned",
                        f"int64_t {opName}_offset"
                    ])
                    for dim in range(num_dims):
                        args.append(f"int64_t {opName}_size{dim}")
                    for dim in range(num_dims):
                        args.append(f"int64_t {opName}_stride{dim}")

                # Write function signature
                f.write(f"void {acc_name}Call({', '.join(args)}) {{\n")
                # Transfer operands (inputs) to SPM
                for operand in acc.operation.operands:
                    opName = operand.name
                    varName = operand.varName
                    f.write(f"    // Transfer operand {opName} to SPM\n")
                    f.write(f"    dma_transfer_tensor_to_spm(DMA_Flags, {varName}, 0, {varName}_shape, {varName}_stride, {varName}_len, (uint64_t){varName}_ptr);\n\n")

                # Call the accelerator
                f.write("    // Call the accelerator\n")
                f.write(f"    accelerator_call({acc_name.upper()});\n\n")

                # Transfer results (outputs) to memory
                for result in acc.operation.results:
                    opName = result.name
                    varName = result.varName
                    f.write(f"    // Transfer result {opName} to memory\n")
                    f.write(f"    dma_transfer_tensor_to_mem(DMA_FLAGS, MEM_ADDR, 0, {varName}_shape, {varName}_stride, {varName}_len, (uint64_t){varName}_ptr);\n\n")

                f.write("}\n\n")

            # End of file
            f.write("// End of generated driver code\n")