#!/usr/bin/python

import sys
import yaml
import math

class Declare:
    def __init__(self, funcName, numOfTensors, tensorDims):
        self.funcName = funcName
        self.numOfTensors = numOfTensors
        self.tensorDims = tensorDims

    def toCString(self):
        line = "void *" + self.funcName + "("
        for i in range(self.numOfTensors):
            addrType = "float*"
            paramsPerTensor = addrType + ", " + addrType + ", int64_t, "
            for j in range(self.tensorDims[i]):
                paramsPerTensor += "int64_t, int64_t, "
            line += paramsPerTensor
        line = line[:-2] + ");"
        return line

class Define:
    def __init__(self, funcName, numOfTensors, tensorDims, tensorSizes, tensorBytes):
        self.funcName = funcName
        self.numOfTensors = numOfTensors
        self.tensorDims = tensorDims
        self.tensorSizes = tensorSizes
        self.tensorBytes = tensorBytes


    def assembleTensor(self, index, numOfDims, isInput = True):
        lines = []
        var = "i" if isInput else "o"

        if numOfDims == 0:
            sizes = "\tvector<int64_t> " + var + str(index) + "_sizes = {};"
            lines.append(sizes)
            strides = "\tvector<int64_t> " + var + str(index) + "_strides = {};"
            lines.append(strides)
        else:
          sizes = "\tvector<int64_t> " + var + str(index) + "_sizes = {"
          for i in range(numOfDims):
              sizes += var + str(index) + "_size" + str(i) + ", "
          sizes = sizes[:-2] + "};"
          lines.append(sizes)

          strides = "\tvector<int64_t> " + var + str(index) + "_strides = {"
          for i in range(numOfDims):
              strides += var + str(index) + "_stride" + str(i) + ", "
          strides = strides[:-2] + "};"
          lines.append(strides)

        memref = "\tMemRef memRef" + str(index) + "(" + var + str(index) + "_allocated, "
        memref += var + str(index) + "_aligned, "
        memref += var + str(index) + "_offset, "
        memref += var + str(index) + "_sizes, "
        memref += var + str(index) + "_strides, "
        memref += str(numOfDims) + ");"
        lines.append(memref)

        return lines

    def codeGenMMR(self):

        lines = []

        lines.append("volatile uint8_t  * top   = (uint8_t  *)(TOP + 0x00);")

        mmr_offset = 0
        for i in range(self.numOfTensors - 1):
            lines.append("volatile uint32_t * mmr_i" + str(i) + "_aligned" + " = (uint32_t *)(TOP + " + str(1+mmr_offset) + ");")
            mmr_offset += 8

            for j in range(self.tensorDims[i]):
                lines.append("volatile int64_t * mmr_i" + str(i) + "_size_" + str(j) + " = (int64_t *)(TOP + " + str(1+mmr_offset) + ");")
                mmr_offset += 8

            for j in range(self.tensorDims[i]):
                lines.append("volatile int64_t * mmr_i" + str(i) + "_stride_" + str(j) + " = (int64_t *)(TOP + " + str(1+mmr_offset) + ");")
                mmr_offset += 8

            lines.append("volatile int64_t * mmr_i" + str(i) + "_offset_" + " = (int64_t *)(TOP + " + str(1+mmr_offset) + ");")
            mmr_offset += 8

        i = self.numOfTensors - 1
        lines.append("volatile uint32_t * mmr_o" + str(i) + "_aligned" + " = (uint32_t *)(TOP + " + str(1+mmr_offset) + ");")
        mmr_offset += 8

        for j in range(self.tensorDims[i]):
            lines.append("volatile int64_t * mmr_o" + str(i) + "_size_" + str(j) + " = (int64_t *)(TOP + " + str(1+mmr_offset) + ");")
            mmr_offset += 8

        for j in range(self.tensorDims[i]):
            lines.append("volatile int64_t * mmr_o" + str(i) + "_stride_" + str(j) + " = (int64_t *)(TOP + " + str(1+mmr_offset) + ");")
            mmr_offset += 8

        lines.append("volatile int64_t * mmr_o" + str(i) + "_offset" + " = (int64_t *)(TOP + " + str(1+mmr_offset) + ");")

        outStr = ""
        for line in lines:
            outStr += line + "\n"
        return outStr
    
    def codeGenDeviceMMR(self):

        lines = []
        lines.append("""	//Define Device MMRs
	volatile uint8_t  * """ 
    + str(self.funcName) 
    + """Flags  = (uint8_t *)""" 
    + str(self.funcName) 
    + """;
	volatile uint8_t  * DmaFlags   = (uint8_t  *)(DMA_Flags);
	volatile uint64_t * DmaRdAddr  = (uint64_t *)(DMA_RdAddr);
	volatile uint64_t * DmaWrAddr  = (uint64_t *)(DMA_WrAddr);
	volatile uint32_t * DmaCopyLen = (uint32_t *)(DMA_CopyLen);
    """)
        return lines

    def codeGenInDMA(self, currentTensorDim, currentTensor):

        lines = []

        if currentTensorDim == (self.tensorDims[currentTensor] - 1):

            lines.append("""// Transfering Input Tensors...
	// Transfer i""" + str(currentTensor) + """...
	*DmaRdAddr  = i""" + str(currentTensor) + "_addr_dim_" + str(currentTensorDim ) + """;
	*DmaWrAddr  = i""" + str(currentTensor) + """_accel_addr;
	*DmaCopyLen = i""" + str(currentTensor) + """_size_""" 
    + str(self.tensorDims[currentTensor]-1) + """;
	*DmaFlags   = DEV_INIT;
    // Poll DMA for finish
	while ((*DmaFlags & DEV_INTR) != DEV_INTR);""")
            lines.append("\t"*(currentTensorDim+2) + "i" + str(currentTensor) + "_accel_addr += 4 * i" + str(currentTensor) + "_size_" + str(self.tensorDims[currentTensor]-1) + ";\n")

        else:

            lines.append("")
            lines.append("\t"*(currentTensorDim+1) + "for(int dim_" + str(currentTensorDim) + " = 0; dim_" + str(currentTensorDim) + " < i" + str(currentTensor) + "_size_" + str(currentTensorDim) + "; dim_" + str(currentTensorDim) + "++){")
            lines.append("")
            lines.extend(self.codeGenInDMA(currentTensorDim + 1, currentTensor))
            lines.append("i" + str(currentTensor) + "_addr_dim_" + str(currentTensorDim) + " += 4 * i" + str(currentTensor) + "_stride_" + str(currentTensorDim) + ";")
            lines.append("i" + str(currentTensor) + "_addr_dim_" + str(currentTensorDim + 1) + " = i" + str(currentTensor) + "_addr_dim_" + str(currentTensorDim) + ";\n" + "}")

        return lines

    def codeGenOutDMA(self, currentTensorDim, currentTensor):

        lines = []

        if currentTensorDim == (self.tensorDims[currentTensor] - 1):

            lines.append("""// Transfering output tensors...
	// Transfering o""" + str(currentTensor) + """...
	*DmaRdAddr  = o""" + str(currentTensor) + "_accel_addr" + """;
	*DmaWrAddr  = o""" + str(currentTensor) + """_addr_dim_""" + str(currentTensorDim) + """;
	*DmaCopyLen = o""" + str(currentTensor) + """_size_""" 
    + str(currentTensorDim) + """;
	*DmaFlags   = DEV_INIT;
    // Poll DMA for finish
	while ((*DmaFlags & DEV_INTR) != DEV_INTR);""")
            lines.append("\t"*(currentTensorDim+2) + "o" + str(currentTensor) + "_accel_addr += 4 * o" + str(currentTensor) + "_size_" + str(self.tensorDims[currentTensor]-1) + ";\n")

        else:

            lines.append("")
            lines.append("\t"*(currentTensorDim+1) + "for(int dim_" + str(currentTensorDim) + " = 0; dim_" + str(currentTensorDim) + " < o" + str(currentTensor) + "_size_" + str(currentTensorDim) + "; dim_" + str(currentTensorDim) + "++){")
            lines.append("")
            lines.extend(self.codeGenOutDMA(currentTensorDim + 1, currentTensor))
            lines.append("o" + str(currentTensor) + "_addr_dim_" + str(currentTensorDim) + " += 4 * o" + str(currentTensor) + "_stride_" + str(currentTensorDim) + ";")
            lines.append("o" + str(currentTensor) + "_addr_dim_" + str(currentTensorDim + 1) + " = o" + str(currentTensor) + "_addr_dim_" + str(currentTensorDim) + ";\n" + "}")

        return lines

    def codeGenAccel(self):

        lines = []
        lines.append("// Start the accelerator " + str(self.funcName))
        lines.append("*" + str(self.funcName) + "Flags = DEV_INIT;")
        lines.append("// Polling for finish...")
        lines.append("while ((*"+ str(self.funcName) +"Flags & DEV_INTR) != DEV_INTR);")

        return lines

    def toStringTop(self):
        lines = []
        head = "void top(\n\t\t"

        for i in range(self.numOfTensors - 1):
            head += "uint64_t i" + str(i) + "_aligned,\n\t\tuint64_t i" + str(i) + "_offset,\n\t\t"
            for j in range(self.tensorDims[i]):
                head += "uint64_t i" + str(i) + "_size_" + str(j) + ",\n\t\t"
            for j in range(self.tensorDims[i]):
                head += "uint64_t i" + str(i) + "_stride_" + str(j) + ",\n\t\t"


        i = self.numOfTensors - 1
        head += "uint64_t o" + str(i) + "_aligned,\n\t\tuint64_t o" + str(i) + "_offset,\n\t\t"
        for j in range(self.tensorDims[i]):
            head += "int64_t o" + str(i) + "_size_" + str(j) + ",\n\t\t"
        for j in range(self.tensorDims[i]):
            head += "int64_t o" + str(i) + "_stride_" + str(j) + ",\n\t\t"

        head = head[:-4] + "\n) {"
        lines.append(head)
        lines.append("")
        lines.extend(self.codeGenDeviceMMR())

        # lines.append("\t// Transfering inputs...")
        lines.append("\t// Generating data address and transfering inputs...")
        for i in range(self.numOfTensors - 1):
            
            lines.append("\tuint64_t i" + str(i) + "_accel_addr = " + str(self.funcName).upper() + "_BUF_" + str(i) + ";")

            for j in range(self.tensorDims[i]):

                if j == 0:
                    lines.append("\tuint64_t i" + str(i) + "_addr_dim_" + str(j) + " = i" + str(i) + "_aligned + 4 * i" + str(i) + "_offset;")
                else:
                    lines.append("\tuint64_t i" + str(i) + "_addr_dim_" + str(j) + " = i" + str(i) + "_addr_dim_" + str(j-1) + ";")
            lines.extend(self.codeGenInDMA(0, i))
        
        lines.extend(self.codeGenAccel())
        
        i = self.numOfTensors - 1
        lines.append("\tuint64_t o" + str(i) + "_accel_addr = " + str(self.funcName).upper() + "_BUF_" + str(i) + ";")

        for j in range(self.tensorDims[i]):

            if j == 0:
                lines.append("\tuint64_t o" + str(i) + "_addr_dim_" + str(j) + " = o" + str(i) + "_aligned + 4 * o" + str(i) + "_offset;")
            else:
                lines.append("\tuint64_t o" + str(i) + "_addr_dim_" + str(j) + " = o" + str(i) + "_addr_dim_" + str(j-1) + ";")
        lines.extend(self.codeGenOutDMA(0, i))
        
        outStr = ""
        for line in lines:
            outStr += line + "\n"
        return outStr

    def toStringCfgYAML(self):
        clusterList = []
        clusterNameDict = {'Name' : str(self.funcName) + '_cluster'}
        
        clusterList.append(clusterNameDict)
        
        dmaDict = {'DMA': [
            {
                'Name': 'dma',
                'MaxReqSize': 64,
                'BufferSize': 128,
                'PIOMaster': 'LocalBus',
                'Type': 'NonCoherent',
                'InterruptNum': 95
            }
        ]}

        clusterList.append(dmaDict)

        # PIO size of flags = 1 byte
        topAccelPIOSize = 1

        # PIO size of tensor attributes (pointer, offset, sizes, strides)
        for i in range(self.numOfTensors):
            topAccelPIOSize += 16 + 8 * (2 * self.tensorDims[i])
    
        topAccelDict = {'Accelerator': [
            {
                'Name': 'Top',
                'IrPath': 'hw/top.ll',
                'PIOSize': topAccelPIOSize,
                'InterruptNum': 68,
                'PIOMaster': 'LocalBus',
                'LocalSlaves': 'LocalBus',
                'Debug': False
            }
        ]}

        clusterList.append(topAccelDict)

        accList = []
        accAttrDict = {
            'Name': str(self.funcName),
            'IrPath': 'hw/'+ str(self.funcName) +'.ll',
            'Debug': False,
            'PIOSize': 1,
            'PIOMaster': 'LocalBus'
        }
        accList.append(accAttrDict)


        varDict = {}
        for i in range(self.numOfTensors):
            varDict = {'Var': [
                {
                    'Name': str(self.funcName).upper() + '_BUF_' + str(i),
                    'Type': 'SPM',
                    'Size': self.tensorBytes[i] * math.prod(self.tensorSizes[i]),
                    'Ports': 2
                }
            ]}
            accList.append(varDict)

        accDict = {'Accelerator' : accList}

        clusterList.append(accDict)

        config = {'acc_cluster' : clusterList}

        # Adding hardware configure (instructions and their functional
        # units) to config.yml

        

        with open('../config.yml', 'w') as file:
            yaml.dump(config, file, sort_keys=False)

    def toStringAlt(self):
        lines = []
        head = "extern \"C\" void __attribute__ ((optimize(\"0\")))" + self.funcName + "("

        for i in range(self.numOfTensors - 1):
            head += "float* i" + str(i) + "_allocated, float* i" + str(i) + "_aligned, int64_t i" + str(i) + "_offset, "
            for j in range(self.tensorDims[i]):
                head += "int64_t i" + str(i) + "_size" + str(j) + ", "
            for j in range(self.tensorDims[i]):
                head += "int64_t i" + str(i) + "_stride" + str(j) + ", "

        i = self.numOfTensors - 1
        head += "float* o" + str(i) + "_allocated, float* o" + str(i) + "_aligned, int64_t o" + str(i) + "_offset, "
        for j in range(self.tensorDims[i]):
            head += "int64_t o" + str(i) + "_size" + str(j) + ", "
        for j in range(self.tensorDims[i]):
            head += "int64_t o" + str(i) + "_stride" + str(j) + ", "

        head = head[:-2] + ") {"
        lines.append(head)
        lines.append("")
        lines.append("""
    m5_reset_stats();

    volatile int count = 0;
    stage = 0;
    """)

        for i in range(self.numOfTensors - 1):
            lines.append("    *mmr_i" + str(i) + "_aligned" + " = (uint32_t)(void *)i" + str(i) + "_aligned;")

            for j in range(self.tensorDims[i]):
                lines.append("    *mmr_i" + str(i) + "_size_" + str(j) + " = (int64_t)i" + str(i) + "_size" + str(j) + ";")

            for j in range(self.tensorDims[i]):
                lines.append("    *mmr_i" + str(i) + "_stride_" + str(j) + " = (int64_t)i" + str(i) + "_stride" + str(j) + ";")

            lines.append("    *mmr_i" + str(i) + "_offset" + " = (int64_t)i" + str(i) + "_offset;")

        i = self.numOfTensors - 1
        lines.append("    *mmr_o" + str(i) + "_aligned" + " = (uint32_t)(void *)o" + str(i) + "_aligned;")

        for j in range(self.tensorDims[i]):
            lines.append("    *mmr_o" + str(i) + "_size_" + str(j) + " = (int64_t)o" + str(i) + "_size" + str(j) + ";")

        for j in range(self.tensorDims[i]):
            lines.append("    *mmr_o" + str(i) + "_stride_" + str(j) + " = (int64_t)o" + str(i) + "_stride" + str(j) + ";")

        lines.append("    *mmr_o" + str(i) + "_offset" + " = (int64_t)o" + str(i) + "_offset;")

        # for i in range(self.numOfTensors-1):
        #   lines.append("    *i" + str(i) + " = (uint32_t)(void *)" + "i" + str(i) + "_aligned;")
        # i = self.numOfTensors - 1
        # lines.append("    *o" + str(i) + " = (uint32_t)(void *)" + "o" + str(i) + "_aligned;") 
        
        lines.append("""    *top = 0x01;
    while (stage < 1) count++;
    m5_dump_stats();
	m5_exit();
}""")
        outStr = ""
        for line in lines:
            outStr += line + "\n"
        return outStr

    def toCString(self):
        lines = []
        head = "extern \"C\" void cgra_" + self.funcName + "("

        for i in range(self.numOfTensors - 1):
            head += "float* i" + str(i) + "_allocated, float* i" + str(i) + "_aligned, int64_t i" + str(i) + "_offset, "
            for j in range(self.tensorDims[i]):
                head += "int64_t i" + str(i) + "_size" + str(j) + ", "
            for j in range(self.tensorDims[i]):
                head += "int64_t i" + str(i) + "_stride" + str(j) + ", "

        i = self.numOfTensors - 1
        head += "float* o" + str(i) + "_allocated, float* o" + str(i) + "_aligned, int64_t o" + str(i) + "_offset, "
        for j in range(self.tensorDims[i]):
            head += "int64_t o" + str(i) + "_size" + str(j) + ", "
        for j in range(self.tensorDims[i]):
            head += "int64_t o" + str(i) + "_stride" + str(j) + ", "

        head = head[:-2] + ") {"
        lines.append(head)
        lines.append("")

        for i in range(self.numOfTensors-1):
            lines.extend(self.assembleTensor(i, self.tensorDims[i], True))
            lines.append("")

        lines.append("\tDataReq input;")
        for i in range(self.numOfTensors-1):
            lines.append("\tinput.assembleReq(memRef" + str(i) + ");")
        lines.append("\t")


        i = self.numOfTensors - 1
        lines.extend(self.assembleTensor(i, self.tensorDims[i], False))
        lines.append("")

        lines.append("\tDataReq output;")
        lines.append("\toutput.assembleReq(memRef" + str(i) + ");")


        lines.append("\tcgra->issueRD(input);")
        accum = "int accum = 1"
        for j in range(self.tensorDims[0]):
            accum += " * i0_size" + str(j)
        accum += ";"
        lines.append("\t" + accum)
        lines.append("\tcgra->issueEX(\"" + self.funcName + "\", accum);")
        lines.append("\tcgra->issueWR(output, false);")

        call = self.funcName + "("
        for i in range(self.numOfTensors - 1):
            call += "i" + str(i) + "_allocated, i" + str(i) + "_aligned, i" + str(i) + "_offset, "
            for j in range(self.tensorDims[i]):
                call += "i" + str(i) + "_size" + str(j) + ", "
            for j in range(self.tensorDims[i]):
                call += "i" + str(i) + "_stride" + str(j) + ", "

        i = self.numOfTensors - 1
        call += "o" + str(i) + "_allocated, o" + str(i) + "_aligned, o" + str(i) + "_offset, "
        for j in range(self.tensorDims[i]):
            call += "o" + str(i) + "_size" + str(j) + ", "
        for j in range(self.tensorDims[i]):
            call += "o" + str(i) + "_stride" + str(j) + ", "
        call = call[:-2] + ");"

        lines.append("\t" + call)
        lines.append("}")

        outStr = ""
        for line in lines:
            outStr += line + "\n"
        return outStr

def codeGenHeader(anchorOp):
    print("""
/*
 * ======================================================================
 * top.cpp
 * ======================================================================
 * This file includes the dma and accelerator scheduling performed by a 
 * virtual top controller.
 *
*/

//Standard Libraries
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <stdint.h>
#include "../defines.h"
#include "../gemm_clstr_hw_defines.h"

#define DEV_INIT	0x01
#define DEV_INTR	0x04

""")

def codeGenGenericDeclares(declares):
    print("extern \"C\" {")
    for dec in declares:
        print("\t" + dec.toCString())
    print("}")
    print()


def codeGenGenericDefines(defines):
    for define in defines:
        # print(define.codeGenDeviceMMR())
        # print(define.toStringTop())
        define.toStringCfgYAML()
        print()

if len(sys.argv) != 3:
    print('provide the source llvm-mlir file name, the outlined anchor op name and the tiling size')

else:
    sourceFileName = sys.argv[1]
    anchorOp = sys.argv[2]
    
    with open(sourceFileName) as sourceFile:


        # codeGenHeader(anchorOp)

        declares = []
        defines = []

        lines = sourceFile.readlines()
        for line in lines:

            if "func.func " in line:
                funcName = anchorOp + "_kernel"
                # funcName = line.split("@")[1].split("(")[0]
                numOfTensors = line.count("memref")
                phrases = line.split("memref")
                tensorDims = []
                tensorSizes = []
                tensorBytes = []
                for phrase in phrases[1:]:

                    dim = phrase.split("<")[1].split(">")[0].count("x")
                    tensorDims.append(dim)

                for phrase in phrases[1:]:

                    parts = phrase.split("<")[1].split(",")[0].split('x')
                    sizes = [int(part) for part in parts[:-1]]
                    data_type = parts[-1]
                    bits = int(data_type[1:])
                    bytes_per_element = bits // 8
                    tensorBytes.append(bytes_per_element)
                    tensorSizes.append(sizes)

                dec = Declare(funcName, numOfTensors, tensorDims)
                define = Define(funcName, numOfTensors, tensorDims, tensorSizes, tensorBytes)
                declares.append(dec)
                defines.append(define)

        # codeGenGenericDeclares(declares)
        codeGenGenericDefines(defines)
  
