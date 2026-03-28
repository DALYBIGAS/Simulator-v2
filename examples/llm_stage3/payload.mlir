module {
  func.func @main(%arg0: tensor<1x1x2048xbf16>, %arg1: tensor<16x2048x128xbf16>, %arg2: tensor<16x2048x128xbf16>) -> tensor<1x1x2048xbf16> {
    %0 = tensor.empty() : tensor<1x1x2048xbf16>
    return %0 : tensor<1x1x2048xbf16>
  }
}
