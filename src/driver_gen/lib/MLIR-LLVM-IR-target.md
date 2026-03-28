Address Computation ¶
Accesses to a memref element are transformed into an access to an element of the buffer pointed to by the descriptor. The position of the element in the buffer is calculated by linearizing memref indices in row-major order (lexically first index is the slowest varying, similar to C, but accounting for strides). The computation of the linear address is emitted as arithmetic operation in the LLVM IR dialect. Strides are extracted from the memref descriptor.

Examples:

An access to a memref with indices:

%0 = memref.load %m[%1,%2,%3,%4] : memref<?x?x4x8xf32, offset: ?>
is transformed into the equivalent of the following code:

// Compute the linearized index from strides.
// When strides or, in absence of explicit strides, the corresponding sizes are
// dynamic, extract the stride value from the descriptor.
%stride1 = llvm.extractvalue[4, 0] : !llvm.struct<(ptr, ptr, i64,
                                                   array<4xi64>, array<4xi64>)>
%addr1 = arith.muli %stride1, %1 : i64

// When the stride or, in absence of explicit strides, the trailing sizes are
// known statically, this value is used as a constant. The natural value of
// strides is the product of all sizes following the current dimension.
%stride2 = llvm.mlir.constant(32 : index) : i64
%addr2 = arith.muli %stride2, %2 : i64
%addr3 = arith.addi %addr1, %addr2 : i64

%stride3 = llvm.mlir.constant(8 : index) : i64
%addr4 = arith.muli %stride3, %3 : i64
%addr5 = arith.addi %addr3, %addr4 : i64

// Multiplication with the known unit stride can be omitted.
%addr6 = arith.addi %addr5, %4 : i64

// If the linear offset is known to be zero, it can also be omitted. If it is
// dynamic, it is extracted from the descriptor.
%offset = llvm.extractvalue[2] : !llvm.struct<(ptr, ptr, i64,
                                               array<4xi64>, array<4xi64>)>
%addr7 = arith.addi %addr6, %offset : i64

// All accesses are based on the aligned pointer.
%aligned = llvm.extractvalue[1] : !llvm.struct<(ptr, ptr, i64,
                                                array<4xi64>, array<4xi64>)>

// Get the address of the data pointer.
%ptr = llvm.getelementptr %aligned[%addr7]
     : !llvm.struct<(ptr, ptr, i64, array<4xi64>, array<4xi64>)> -> !llvm.ptr

// Perform the actual load.
%0 = llvm.load %ptr : !llvm.ptr -> f32
For stores, the address computation code is identical and only the actual store operation is different.

Note: the conversion does not perform any sort of common subexpression elimination when emitting memref accesses.