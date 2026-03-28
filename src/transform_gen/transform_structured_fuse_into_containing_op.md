transform.structured.fuse_into_containing_op (transform::FuseIntoContainingOp) ¶
Fuse a producer into a containing operation.

Syntax:

operation ::= `transform.structured.fuse_into_containing_op` $producer_op `into` $containing_op attr-dict  `:` functional-type(operands, results)
Fuses the producer_op into the containing_op. Returns a handle to the fused ops and the new_containing_op.

The producer is typically a slice of a tileable op (i.e., implements TilingInterface). In that case, this transform computes the accessed producer slice inside of the containing op (“tile and fuse”) and if required, creates a new containing op with outputs from the fused producer. Otherwise, the entire producer is cloned inside the containing op (“clone and fuse”).

The containing op handle must be associated with exactly one payload op. The producer op handle may be associated with multiple payload ops. This transform fuses producers one-by-one, always picking an unspecified producer that has at least one use inside the containing op among the producers. A producer can be listed multiple times in the handle.

Note: If a producer has multiple uses inside the containing op, it is currently tiled and/or cloned multiple times into the containing op. TODO: Reuse already fused OpResults instead of tiling/cloning a second time when possible. Fuse producers according to a topological sorting to achieve the largest amount of reuse.

Return modes ¶
If at least one producer could not be fused, this operation produces a silenceable failure. This is the case when tiling fails or when no producer op could be found among the remaining producers that has at least one use within the containing op. I.e., “producers” that are not consumed within the containing op are rejected by this operation.

This operation consumes the producer handle. This operation only reads the containing op handle.

Traits: ReportTrackingListenerFailuresOpTrait

Interfaces: MemoryEffectOpInterface, TransformOpInterface

Operands: ¶
Operand	Description
producer_op	TransformHandleTypeInterface instance
containing_op	TransformHandleTypeInterface instance
Results: ¶
Result	Description
fused_op	TransformHandleTypeInterface instance
new_containing_op	TransformHandleTypeInterface instance

Chaining Transformations with Handles ¶
Going back to the transformation sequence, we have tiled the matrix multiplication, but we also want to tile and fuse the elementwise operations. The typical way of doing in the structured operations paradigm is to tile the last operation in some acyclic dataflow graph, and then progressively fuse the operations that produce its operands. This removes the need to explicitly tile all operations as fusion can adapt their sizes and inject recomputation if desired. So instead of tiling the matmul operation, we are going to tile the last operation in the chain, and then fuse the preceding operations into the loops produced by tiling.

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
       %arg0: !transform.any_op,
       %arg1: !transform.op<"linalg.matmul">,
       %arg2: !transform.op<"linalg.elemwise_binary">) {
    // Since the %arg2 handle is associated with both elementwise operations,
    // we need to split it into two handles so we can target only the second
    // elementwise operation.
    %add, %max = transform.split_handle %arg2
        : (!transform.op<"linalg.elemwise_binary">)
        -> (!transform.any_op, !transform.any_op)

    // The actual tiling transformation takes tile sizes as attributes. It
    // produces a handle to the loop generated during tiling.
    %tiled_max, %loop =
        transform.structured.tile_using_forall %max tile_sizes [8, 32]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // We can now fuse the other operations into the loop. Here, we fuse
    // operations one by one. This requires the operation that is being fused to
    // define the value used within the loop, so the order of such fusions is
    // important. We could also use "transform.merge_handles" to obtain a single
    // handle to all operations and give it to `fuse_into_containing_op` that
    // would take care of the ordering in this case.
    %add_fused, %loop_0 =
        transform.structured.fuse_into_containing_op %add into %loop
          : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)
    %matmul_fused, %loop_1 =
        transform.structured.fuse_into_containing_op %arg1 into %loop_0
          : (!transform.op<"linalg.matmul">, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    transform.yield
  }
}
This achieves the desired tiling and fusion.