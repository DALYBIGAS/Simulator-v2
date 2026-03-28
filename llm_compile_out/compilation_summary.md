# Compilation Summary

- model: llama3-8b
- model_family: llama
- kernel: llama_prefill
- mode: prefill
- chip: legend-llm-chip-v4
- array: 128x128
- dtype: bf16
- selected_kernel: qkv-projection
- tile_sizes: [128, 128, 64]
- kv_cache: False
- payload: examples/llm_stage4/payload.mlir
- est_latency_ms: 6144000.0000
- est_tokens_per_sec: 0.33
