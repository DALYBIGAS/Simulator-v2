// FALLBACK OPTIMIZED MLIR
// model=llama3-8b
// kernel=llama_prefill
// mode=prefill
// stage:graph -> canonicalize, cse
// stage:kernel -> llm-fuse-attention-chain, llm-tile[128,128,64], llm-fuse-epilogue, llm-select-kernel[qkv-projection]
// stage:buffer -> llm-promote-sram, llm-insert-async-dma
// stage:runtime -> llm-emit-events, llm-emit-launch-plan
// stage:backend -> lower-affine, convert-scf-to-cf, finalize-memref-to-llvm

module attributes {llm.optimized = true} {
  // The original payload is embedded below for deterministic replay.
  // module {
  //   func.func @main(%arg0: tensor<1x1x4096xbf16>) -> tensor<1x1x4096xbf16> {
  //     %0 = tensor.empty() : tensor<1x1x4096xbf16>
  //     return %0 : tensor<1x1x4096xbf16>
  //   }
  // }
}
