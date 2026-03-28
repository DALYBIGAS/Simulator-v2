module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %chain = transform.structured.match ops ["linalg.batch_matmul", "linalg.generic", "linalg.generic"] in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled, %loops:2 = transform.structured.tile_using_for %chain tile_sizes [128, 128, 64]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %tiled {llm.kernel_name = "llama_prefill"} : !transform.any_op
    transform.yield
  }
}
