module {
  func.func @main(%query: tensor<1x1x4096xbf16>, %key_cache: tensor<32x4096x128xbf16>, %value_cache: tensor<32x4096x128xbf16>) -> tensor<1x1x4096xbf16> {
    %scores = llm.kv_matmul %query, %key_cache {kv_cache = true} : tensor<1x1x4096xbf16>, tensor<32x4096x128xbf16> -> tensor<1x32x1x4096xbf16>
    %probs = llm.softmax %scores : tensor<1x32x1x4096xbf16> -> tensor<1x32x1x4096xbf16>
    %context = llm.kv_matmul %probs, %value_cache {kv_cache = true} : tensor<1x32x1x4096xbf16>, tensor<32x4096x128xbf16> -> tensor<1x1x4096xbf16>
    return %context : tensor<1x1x4096xbf16>
  }
}
