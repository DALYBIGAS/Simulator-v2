module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %target = transform.structured.match ops ["linalg.batch_matmul"] in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.annotate %target {llm.outline_target = "llama_prefill_kernel"} : !transform.any_op
    transform.yield
  }
}
