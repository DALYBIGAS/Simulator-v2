module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %chain = transform.structured.match ops ["linalg.matmul", "linalg.softmax", "linalg.matmul"] in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled, %loops:2 = transform.structured.tile_using_for %chain tile_sizes [64, 128, 32]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %tiled {llm.kernel_name = "decode_attention_stage3"} : !transform.any_op
    transform.yield
  }
}
