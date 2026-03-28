# 需要编译的源文件
KERN = wavelet.c top.c

# 编译目标，生成 LLVM IR 文件
build: $(KERN)
	clang -O2 -S -fno-slp-vectorize -fno-vectorize -fno-unroll-loops -fno-inline -fno-builtin -emit-llvm $(KERN) -target armv7-pc-none-eabi -mcpu=cortex-a9 -mfpu=neon -g

# 清理生成的中间文件
clean:
	rm -f *.ll *.o *.bc
