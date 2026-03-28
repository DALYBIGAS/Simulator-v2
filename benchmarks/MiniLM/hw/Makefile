KERN=batch_matmul_kernel.cpp top_32bit_mmr.cpp

# KERN

build: $(FILES)
	clang -O1 -S -m32 -fno-slp-vectorize -fno-vectorize -fno-unroll-loops -fno-inline -fno-builtin -emit-llvm $(KERN) -target armv7-pc-none-eabi

clean:
	rm -f *.ll
