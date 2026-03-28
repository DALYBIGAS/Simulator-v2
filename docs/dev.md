# Develop Notes

## TODO Tasks

### Driver Generator Related

#### Config

 - [ ] YAML
 - [ ] Parser (Process_config)

#### Single accelerator

##### Basic components library

 - [x] DMA (Stream, Noncoherent, Tensor Transfer)
 - [x] Accelerator
 - [ ] A DMA used for tensor-like data

#### Multiple accelerators

 - [ ] Op fusion
 - [ ] Pipelining
 - [ ] Double buffer

### Transform Generator Related

 - [ ] Op fusion
 - [ ] Pipelining
 - [ ] Double buffer

### Design Space Exploration

Add tasks here.

### Docs Related

 - [ ] Update the docs for the new features.
  
### Future Work

 - [ ] How to automate the driver generation process directly through the transform dialect program. How to extract the semantics of the transform and detect any possible memory access and dataflow change to modify the driver program.
 - [ ] How to detect available scheduling and memory access optimization opportunities through template based SoC description file.

# Develop Refenrence

## MLIR Related

- MLIR test codes, which include many examples of how to use Transform dialect (named transform-xxxx.mlir in each dialect's folder). See [github repository](https://github.com/llvm/llvm-project/tree/main/mlir/test).
- Transform dialect's documentation. See [Transform dialect](https://mlir.llvm.org/docs/Dialects/Transform/).

## Recommeded Extensions in VS Code

- MLIR
- CMake
- Todo Tree
- Markdown All in One
- Git Graph
- Git History
- Github Copilot
- Clangd. See [Clangd](https://clangd.llvm.org/installation) for more information.