module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %target = transform.structured.match ops ["linalg.matmul"] in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.annotate %target {llm.outline_target = "decode_attention_kernel"} : !transform.any_op
    transform.yield
  }
}
