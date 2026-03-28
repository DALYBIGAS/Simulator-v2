# LLM Compiler Refactor Plan

## Stage-3 target tree

```text
Unnamed-Simulator/
├── compile.py
├── tools/
│   └── profile_gem5.py
├── examples/
│   ├── llm_stage2/
│   └── llm_stage3/
├── lib/
│   ├── config_parser/
│   ├── transform_gen/
│   ├── driver_gen/
│   └── llm/
│       ├── kernels/
│       ├── passes/
│       ├── runtime/
│       └── profiler/
└── test/
```

## Framework goals

- make hardware capabilities explicit and compiler-consumable
- represent compile requests with mode-specific LLM metadata
- select a reusable kernel variant per workload
- build a target-aware pass pipeline plan
- materialize a runtime launch plan
- emit reusable driver stubs
- parse gem5-like counters into compact reports

## Recommended next engineering steps

1. connect `pass_pipeline.json` to real MLIR pass execution
2. add kernel lowering for attention, kv-cache, and norm
3. add quantized matmul and grouped GEMM paths
4. connect runtime plans to generated host/device launch code
5. add profiler-to-op attribution and autotune loops
