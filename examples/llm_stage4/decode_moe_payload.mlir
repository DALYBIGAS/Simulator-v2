module {
  func.func @main(%query: tensor<1x1x4096xbf16>, %key_cache: tensor<32x4096x128xbf16>, %value_cache: tensor<32x4096x128xbf16>) -> tensor<1x1x4096xbf16> {
    %scores = llm.kv_matmul %query, %key_cache {kv_cache = true} : tensor<1x1x4096xbf16>, tensor<32x4096x128xbf16> -> tensor<1x32x1x4096xbf16>
    %probs = llm.softmax %scores : tensor<1x32x1x4096xbf16> -> tensor<1x32x1x4096xbf16>
    %context = llm.kv_matmul %probs, %value_cache {kv_cache = true} : tensor<1x32x1x4096xbf16>, tensor<32x4096x128xbf16> -> tensor<1x1x4096xbf16>
    %gate = llm.moe_gate %context {experts = 16, top_k = 2} : tensor<1x1x4096xbf16> -> tensor<1x1x16xf32>
    %routed = llm.moe_dispatch %context, %gate : tensor<1x1x4096xbf16>, tensor<1x1x16xf32> -> tensor<2x1x4096xbf16>
    %expert_out = llm.expert_matmul %routed : tensor<2x1x4096xbf16> -> tensor<2x1x4096xbf16>
    %out = llm.moe_combine %expert_out, %gate : tensor<2x1x4096xbf16>, tensor<1x1x16xf32> -> tensor<1x1x4096xbf16>
    return %out : tensor<1x1x4096xbf16>
  }
}
