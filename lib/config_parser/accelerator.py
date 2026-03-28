class Accelerator:

    def __init__(
        self,
        name: str,
        pio_masters: str,
        local_connections: str,
        address: int,
        size: int,
        stream_in: str,
        stream_out: str,
        int_num: int,
        working_dir: str,
        ir_path: str,
        config_path: str,
        hw_config_path: str,
        variables=None,
        debug: bool = False,
        type: str = None,
        operation = None
    ):

        self.name = name.lower()
        self.pio_masters = pio_masters
        self.local_connections = local_connections
        self.address = address
        self.size = size
        self.stream_in = stream_in
        self.stream_out = stream_out
        self.int_num = int_num

        self.working_dir = working_dir
        self.ir_path = ir_path
        self.config_path = config_path
        self.hw_config_path = hw_config_path
        self.variables = variables
        self.debug = debug
        self.type = type
        self.operation = operation

    def genDefinition(self):
        lines = []
        lines.append("# " + self.name + " Definition")
        lines.append("acc = " + "\"" + self.name + "\"")
        lines.append("ir = " + "\"" + self.working_dir +
                     "/" + self.ir_path + "\"")
        lines.append("hw_config = ""\"" + self.hw_config_path + "\"")

        # Add interrupt number if it exists
        if self.int_num is not None:
            lines.append("clstr." + self.name + " = CommInterface(devicename=acc, gic=gic, pio_addr="
                         + str(hex(self.address)) + ", pio_size=" + str(self.size) + ", int_num=" + str(self.int_num) + ")")
        else:
            lines.append("clstr." + self.name + " = CommInterface(devicename=acc, gic=gic, pio_addr="
                         + str(hex(self.address)) + ", pio_size=" + str(self.size) + ")")

        lines.append("AccConfig(clstr." + self.name + ", ir, hw_config)")
        lines.append("")

        return lines

    def genConfig(self):
        lines = []

        lines.append("# " + self.name + " Config")

        for connection in self.local_connections:
            if "LocalBus" in connection:
                lines.append("clstr." + self.name +
                             ".local = clstr.local_bus.cpu_side_ports")
            else:
                lines.append("clstr." + self.name +
                             ".local = clstr." + connection.lower() + ".pio")

        # Assign PIO Masters
        for master in self.pio_masters:
            if "LocalBus" in master:
                lines.append("clstr." + self.name +
                             ".pio = clstr.local_bus.mem_side_ports")
            else:
                assert False, "Shouldn't be here?"
                # lines.append("clstr." + self.name + ".pio " +
                #              "=" " clstr." + i + ".local")
        # Add StreamIn
        for inCon in self.stream_in:
            lines.append("clstr." + self.name +
                         ".stream = clstr." + inCon.lower() + ".stream_in")
        # Add StreamOut
        for outCon in self.stream_out:
            lines.append("clstr." + self.name +
                         ".stream = clstr." + outCon.lower() + ".stream_out")

        lines.append("clstr." + self.name +
                     ".enable_debug_msgs = " + str(self.debug))
        lines.append("")

        # Add scratchpad variables
        for var in self.variables:
            # Have the variable create its config
            lines = var.genConfig(lines)
            lines.append("")
        # Return finished config portion
        return lines

