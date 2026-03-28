module {
  func.func @main(%arg0: tensor<1x1x4096xbf16>) -> tensor<1x1x4096xbf16> {
    %0 = tensor.empty() : tensor<1x1x4096xbf16>
    return %0 : tensor<1x1x4096xbf16>
  }
}
