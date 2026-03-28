# Copyright (c) 2015, University of Kaiserslautern
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
# OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Authors: 	Matthias Jung
#			Frederik Lauer

CROSS_COMPILE_DIR = /path/to/cross/compile/dir
LLVM_TOOLCHAIN = /path/to/llvm_arm_toolchain

ARCH          ?= arm
CROSS_COMPILE ?= arm-none-eabi-
#CROSS_COMPILE_DIR = /home/samerogers/aarch64-toolchain/bin

#ARCH          ?= aarch64
#CROSS_COMPILE ?= aarch64-elf-

AS           = $(CROSS_COMPILE_DIR)/$(CROSS_COMPILE)as
# Replaced with LLVM
# AS           = $(LLVM_TOOLCHAIN)/bin/clang
LD           = $(CROSS_COMPILE_DIR)/$(CROSS_COMPILE)ld
CC           = $(CROSS_COMPILE_DIR)/$(CROSS_COMPILE)gcc
# Replaced with LLVM
LLVM_CC      = $(LLVM_TOOLCHAIN)/bin/clang
CXX          = $(CROSS_COMPILE_DIR)/$(CROSS_COMPILE)g++
# Replaced with LLVM
LLVM_CXX     = $(LLVM_TOOLCHAIN)/bin/clang++
CPP          = $(CROSS_COMPILE_DIR)/$(CROSS_COMPILE)cpp
# Replaced with LLVM
# CPP          = cpp
AR           = $(CROSS_COMPILE_DIR)/$(CROSS_COMPILE)ar
NM           = $(CROSS_COMPILE_DIR)/$(CROSS_COMPILE)nm
STRIP        = $(CROSS_COMPILE_DIR)/$(CROSS_COMPILE)strip
OBJCOPY      = $(CROSS_COMPILE_DIR)/$(CROSS_COMPILE)objcopy
OBJDUMP      = $(CROSS_COMPILE_DIR)/$(CROSS_COMPILE)objdump

# CFLAGS       = "-g -O3 -lunistd -I . -target arm-none-eabi \
# 				--sysroot=/usr/lib/arm-none-eabi -L/usr/lib/arm-none-eabi/lib \
# 				--rtlib=compiler-rt --stdlib=libc++ \
# 				-I$(LLVM_TOOLCHAIN)/lib/clang/16/include \
# 				--gcc-toolchain=$(CROSS_COMPILE_DIR) \
# 				-L$(LLVM_TOOLCHAIN)/lib"

# Flags used in LLVM toolchain				
# CFLAGS		= --target=arm-none-eabi \
# --sysroot=$(LLVM_TOOLCHAIN)/lib/clang-runtimes/arm-none-eabi/armv4t \
# -I$(LLVM_TOOLCHAIN)/lib/clang-runtimes/arm-none-eabi/armv4t/include/c++/v1 \
# -I$(LLVM_TOOLCHAIN)/lib/clang-runtimes/arm-none-eabi/armv4t/include \
# -L$(LLVM_TOOLCHAIN)/lib/clang-runtimes/arm-none-eabi/armv4t/lib \
# -mfpu=none -g -O3 -I . \
# -fno-exceptions \
# -fno-rtti \
# -lcrt0-semihost \
# -lsemihost \
# -fuse-ld=$(CROSS_COMPILE_DIR)/$(CROSS_COMPILE)ld \
# -lc -lm 

# CXXFLAGS     = "-g -O3 -lunistd -I . -target arm-none-eabi \
# 				--sysroot=/usr/lib/arm-none-eabi -L/usr/lib/arm-none-eabi/lib \
# 				--rtlib=compiler-rt --stdlib=libc++ \
# 				-I$(LLVM_TOOLCHAIN)/lib/clang/16/include \
# 				--gcc-toolchain=$(CROSS_COMPILE_DIR) \
# 				-L$(LLVM_TOOLCHAIN)/lib"

# Flags used in LLVM toolchain				
LLVM_CXXFLAGS     = --target=armv4t \
--sysroot=$(LLVM_TOOLCHAIN)/lib/clang-runtimes/arm-none-eabi/armv4t \
-I$(LLVM_TOOLCHAIN)/lib/clang-runtimes/arm-none-eabi/armv4t/include/c++/v1 \
-I$(LLVM_TOOLCHAIN)/lib/clang-runtimes/arm-none-eabi/armv4t/include \
-L$(LLVM_TOOLCHAIN)/lib/clang-runtimes/arm-none-eabi/armv4t/lib \
-mfpu=none -g -O3 -I . \
-fno-exceptions \
-fno-rtti \
-lcrt0-semihost \
-lsemihost \
-fuse-ld=$(CROSS_COMPILE_DIR)/$(CROSS_COMPILE)ld \
-lc -lm 

# Origin flags
CFLAGS       =  -g -O3 -I . -fstack-usage
CXXFLAGS     =  -g -O3 -I . -fstack-usage -fno-exceptions --sysroot=/usr/lib/arm-none-eabi -L/usr/lib/arm-none-eabi/lib

ASFLAGS      = -EL
LNK_OPT      = -nostartfiles -lc -lm -Wl,-Map=main_picolibc.map
# LNK_OPT      = -nostartfiles -lc -lm -Wl,-Map=main_boot.map

# sample.ld used when using LLVM Toolchain
LNK_SCRIPT   = picolibc_revised.ld
# LNK_SCRIPT   = boot_modified.ld
# LNK_SCRIPT   = boot.ld

#LNK_SCRIPT   = start64.ld
LNK_FILE_OPT = -nostartfiles -Xlinker -T$(LNK_SCRIPT)
