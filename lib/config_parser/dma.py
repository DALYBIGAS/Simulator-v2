class StreamDMA:
    def __init__(
        self,
        name: str,
        pio: int,
        pio_masters: str,
        address: int,
        statusAddress: int,
        dmaType: str,
        rd_int: int = None,
        wr_int: int = None,
        size: int = 64
    ):
        self.name = name.lower()
        self.pio = pio
        self.pio_masters = pio_masters
        self.size = size
        self.address = address
        self.statusAddress = statusAddress
        self.dmaType = dmaType
        self.rd_int = rd_int
        self.wr_int = wr_int

        for master in self.pio_masters:
            count = 0
            if "localbus" in master.lower():
                pio_masters[count] = "local_bus"
                count += 1
    # Probably could apply the style used here in other genConfigs

    def genConfig(self):
        lines = []
        dmaPath = "clstr." + self.name + "."
        # Need to fix max_pending?
        lines.append("# Stream DMA")
        lines.append("clstr." + self.name + " = StreamDma(pio_addr=" + hex(self.address) +
                     ", status_addr=" + hex(self.statusAddress) + ", pio_size = " + str(self.pio) + ", gic=gic, max_pending = " + str(self.pio) + ")")
        lines.append(dmaPath + "stream_addr = " +
                     hex(self.address) + " + " + str(self.pio))
        lines.append(dmaPath + "stream_size = " + str(self.size))
        lines.append(dmaPath + "pio_delay = '1ns'")
        if self.rd_int != None:
            lines.append(dmaPath + "rd_int = " + str(self.rd_int))
        if self.wr_int != None:
            lines.append(dmaPath + "wr_int = " + str(self.wr_int))
        lines.append("clstr." + self.name +
                     ".dma = clstr.coherency_bus.cpu_side_ports")
        if self.pio_masters is not None:
            for master in self.pio_masters:
                lines.append("clstr." + master.lower() +
                             ".mem_side_ports = clstr." + self.name + ".pio")
        lines.append("")

        return lines


class DMA:
    def __init__(
        self,
        name: str,
        pio: int,
        pio_masters: str,
        address: int,
        dmaType: str,
        int_num=None,
        size: int = 64,
        maxReq: int = 4
    ):
        self.name = name.lower()
        self.pio = pio
        self.pio_masters = pio_masters
        self.size = size
        self.address = address
        self.dmaType = dmaType
        self.int_num = int_num
        self.maxReq = maxReq

        for master in self.pio_masters:
            count = 0
            if "localbus" in master.lower():
                pio_masters[count] = "local_bus"
                count += 1
    # Probably could apply the style used here in other genConfigs

    def genConfig(self):
        lines = []
        dmaPath = "clstr." + self.name + "."
        systemPath = "clstr."
        lines.append("# Noncoherent DMA")
        lines.append("clstr." + self.name + " = NoncoherentDma(pio_addr="
                     + hex(self.address) + ", pio_size = " + str(self.pio)
                     + ", gic=gic, int_num=" + str(self.int_num) + ")")
        lines.append(dmaPath + "cluster_dma = " +
                     systemPath + "local_bus.cpu_side_ports")
        lines.append(dmaPath + "max_req_size = " + str(self.maxReq))
        lines.append(dmaPath + "buffer_size = " + str(self.size))
        lines.append("clstr." + self.name +
                     ".dma = clstr.coherency_bus.cpu_side_ports")
        if self.pio_masters is not None:
            for master in self.pio_masters:
                lines.append("clstr." + master.lower() +
                             ".mem_side_ports = clstr." + self.name + ".pio")
        lines.append("")

        return lines

