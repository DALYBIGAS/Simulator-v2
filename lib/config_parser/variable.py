
class PortedConnection:
    def __init__(self, conName: str, numPorts: int):
        self.conName = conName
        self.numPorts = numPorts


class Variable:
    def __init__(self, **kwargs):
        # Read the type first
        self.type = kwargs.get('Type')
        if self.type == 'SPM':
            self.connections = []
            # Read in SPM args
            self.name = kwargs.get('Name')
            self.accName = kwargs.get('AccName')
            self.size = kwargs.get('Size')
            self.ports = kwargs.get('Ports', 1)
            self.address = kwargs.get('Address')
            self.readyMode = kwargs.get('ReadyMode', False)
            self.resetOnRead = kwargs.get('ResetOnRead', True)
            self.readOnInvalid = kwargs.get('ReadOnInvalid', False)
            self.writeOnValid = kwargs.get('WriteOnValid', True)
            # Append the default connection here... probably need to be more elegant
            self.connections.append(PortedConnection(self.accName, self.ports))
            # Append other connections to the connections list
            if 'Connections' in kwargs:
                for conDef in kwargs.get('Connections').split(','):
                    con, numPorts = conDef.split(':')
                    self.connections.append(PortedConnection(con, numPorts))
        elif self.type == 'Stream':
            # Read in Stream args
            self.name = kwargs.get('Name')
            self.accName = kwargs.get('AccName')
            self.inCon = kwargs.get('InCon')
            self.outCon = kwargs.get('OutCon')
            self.streamSize = kwargs.get('StreamSize')
            self.bufferSize = kwargs.get('BufferSize')
            self.address = kwargs.get('Address')
            self.statusAddress = kwargs.get('StatusAddress')
            # Convert connection definitions to lowercase
            self.inCon = self.inCon.lower()
            self.outCon = self.outCon.lower()
        elif self.type == 'RegisterBank':
            self.connections = []
            # Read in SPM args
            self.name = kwargs.get('Name')
            self.accName = kwargs.get('AccName')
            self.size = kwargs.get('Size')
            self.address = kwargs.get('Address')
            # Append the default connection here... probably need to be more elegant
            self.connections.append(PortedConnection(self.accName, 1))
            # Append other connections to the connections list
            if 'Connections' in kwargs:
                for conDef in kwargs.get('Connections').split(','):
                    con, numPorts = conDef.split(':')
                    self.connections.append(PortedConnection(con, numPorts))
        elif self.type == 'Cache':
            self.name = kwargs.get('Name')
            self.accName = kwargs.get('AccName')
            self.size = kwargs.get('Size')
        else:
            # Throw an exception if we don't know the type
            exceptionString = ("The variable: " + kwargs.get('Name')
                               + " has an invalid type named: " + self.type)
            raise Exception(exceptionString)

    def genConfig(self, lines):
        # Add new variable configs here
        # Stream Buffer Variable
        if self.type == 'Stream':
            lines.append("# " + self.name + " (Stream Variable)")
            lines.append("addr = " + hex(self.address))
            lines.append("clstr." + self.name.lower() + " = StreamBuffer(stream_address = addr, status_address= " + hex(self.statusAddress)
                         + ", stream_size = " + str(self.streamSize) + ", buffer_size = " + str(self.bufferSize) + ")")
            lines.append("clstr." + self.inCon + ".stream = " +
                         "clstr." + self.name.lower() + ".stream_in")
            lines.append("clstr." + self.outCon + ".stream = " +
                         "clstr." + self.name.lower() + ".stream_out")
            lines.append("")
        # Scratchpad Memory
        elif self.type == 'SPM':
            lines.append("# " + self.name + " (Variable)")
            lines.append("addr = " + hex(self.address))
            lines.append(
                "spmRange = AddrRange(addr, addr + " + hex(self.size) + ")")
            # When appending convert all connections to lowercase for standardization
            lines.append("clstr." + self.name.lower() +
                         " = ScratchpadMemory(range = spmRange)")
            # Probably need to add table and read mode to the YAML File
            lines.append("clstr." + self.name.lower() +
                         "." + "conf_table_reported = False")
            lines.append("clstr." + self.name.lower() + "." +
                         "ready_mode = " + str(self.readyMode))
            lines.append("clstr." + self.name.lower() + "." +
                         "reset_on_scratchpad_read = " + str(self.resetOnRead))
            lines.append("clstr." + self.name.lower() + "." +
                         "read_on_invalid = " + str(self.readOnInvalid))
            lines.append("clstr." + self.name.lower() + "." +
                         "write_on_valid = " + str(self.writeOnValid))
            lines.append("clstr." + self.name.lower() + "." +
                         "port" + " = " + "clstr.local_bus.mem_side_ports")
            for con in self.connections:
                lines.append("")
                lines.append("# Connecting " + self.name +
                             " to " + con.conName)
                lines.append("for i in range(" + str(con.numPorts) + "):")
                lines.append("	clstr." + con.conName.lower() + ".spm = " +
                             "clstr." + self.name.lower() + ".spm_ports")
        # RegisterBank
        elif self.type == 'RegisterBank':
            lines.append("# " + self.name + " (Variable)")
            lines.append("addr = " + hex(self.address))
            lines.append(
                "regRange = AddrRange(addr, addr + " + hex(self.size) + ")")
            # When appending convert all connections to lowercase for standardization
            lines.append("clstr." + self.name.lower() +
                         " = RegisterBank(range = regRange)")
            lines.append("clstr." + self.name.lower() + "." +
                         "load_port" + " = " + "clstr.local_bus.mem_side_ports")
            for con in self.connections:
                lines.append("")
                lines.append("# Connecting " + self.name +
                             " to " + con.conName)
                lines.append("clstr." + con.conName.lower() + ".reg = " +
                             "clstr." + self.name.lower() + ".reg_port")
        # L1 Cache, need to add L2 still...
        elif self.type == 'Cache':
            lines.append("# " + self.name + " (Cache)")
            lines.append("clstr." + self.name +
                         " = L1Cache(size = '" + str(self.size) + "B')")
            lines.append("clstr." + self.name +
                         ".mem_side = clstr.coherency_bus.cpu_side_ports")
            lines.append("clstr." + self.name +
                         ".cpu_side = clstr." + self.accName + ".local")
        else:
            # Should never get here... but just in case throw an exception
            exceptionString = ("The variable: " + self.name
                               + " has an invalid type named: " + self.type)
            raise Exception(exceptionString)
        return lines
