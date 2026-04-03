# GAMMA-v2
A compiler-oriented AI-chip simulation and evaluation framework for LLM workloads, built on top of the original simulator infrastructure and extended with **explicit hardware capability modeling**, **LLM-aware compile planning**, **runtime launch planning**, and **gem5-style performance reporting**.

---

## Overview

GAMMA-v2 is the second-generation evolution of the original GAMMA-v1 framework.  
It keeps the original idea of combining compiler techniques and system-level simulation, but shifts the workflow from a largely manual, benchmark-specific pipeline to a **structured, compiler-driven, artifact-producing flow** that is better suited to modern LLM and AI-chip evaluation.

At a high level, GAMMA-v2 introduces:

- **Explicit hardware capability descriptions** via YAML
- **LLM-oriented compile specifications** for prefill / decode style workloads
- **Model-family-aware planning** for families such as LLaMA, DeepSeek, OPT, Mixtral, and Qwen
- **Kernel selection and pass pipeline planning** as first-class compiler outputs
- **Runtime launch plan generation** for accelerator execution
- **Estimated metric generation** before backend execution
- **Measured performance report generation** from gem5-like stats
- **Reusable intermediate artifacts** for debugging, replay, and design-space exploration

In other words, GAMMA-v2 is not just “run a benchmark on a simulator”; it is a framework for:

1. describing an AI-chip,
2. describing an LLM workload,
3. compiling a plan for that workload,
4. materializing backend/runtime artifacts,
5. and evaluating performance using generated reports.

---

## What GAMMA-v2 is for

GAMMA-v2 is intended for the following scenarios:

- evaluating whether a proposed AI-chip architecture is suitable for LLM workloads
- comparing multiple hardware configurations under the same workload
- studying how SRAM, DMA bandwidth, array shape, and supported data types affect performance
- generating compiler/runtime artifacts that explain *why* a mapping decision was made
- building a repeatable workflow for AI accelerator exploration instead of relying only on hand-written kernels and manually assembled simulation steps

---

## Key capabilities

### 1. Hardware-aware compile planning
The framework consumes a hardware description that explicitly models:

- SRAM size / banking information
- DMA bandwidth and capabilities
- compute array dimensions
- supported data types
- native MMA tile sizes
- hardware support flags such as KV-cache support, fused epilogues, and prefill/decode split

This makes the simulator more useful for **chip design exploration**, because the compiler output is directly conditioned on the hardware capabilities rather than being hidden in scattered scripts.

### 2. LLM-oriented workload modeling
GAMMA-v2 introduces compile specs that carry workload metadata such as:

- model name / architecture
- kernel name
- mode (`prefill`, `decode`, etc.)
- sequence length
- hidden size
- number of heads
- head dimension
- KV-cache usage
- expert metadata for MoE-style models
- operator chain information

This is a major upgrade over a generic benchmark-driven flow because it lets the simulator reason about **LLM structure**, not just isolated kernels.

### 3. Compiler artifact generation
Instead of only producing a final runnable benchmark, v2 emits a full bundle of compiler artifacts, including:

- tile transform scripts
- fuse transform scripts
- outline transform scripts
- driver stubs
- compile manifest
- model profile metadata
- selected kernel metadata
- pass pipeline plans
- runtime launch plans
- estimated metrics
- backend placeholder IR
- optimized MLIR output

This is valuable for debugging and for methodology sections in papers, because every stage becomes inspectable.

### 4. Measured and estimated evaluation flow
The framework supports two levels of analysis:

- **Estimated metrics** from compile-time modeling
- **Measured metrics** from gem5-like stats files

This helps you answer both:

- “What does the compiler think will happen?”
- “What did the simulator actually measure?”

### 5. Multi-model-family support
The current framework is structured to support multiple LLM families, including:

- LLaMA
- DeepSeek
- OPT
- Mixtral
- Qwen3

This makes v2 more suitable than v1 for modern LLM architecture studies.



