# Building

## Building from Dockerfile

This is recommended way, but there is still no docker built yet :)

To be filled. For test now.

## Building from Source

Use git to clone this repo.

```bash
git clone https://github.com/lczzz29/Unnamed-Simulator.git
```

Checkout the develop branch. Currently only this branch contains main content of project.

```bash
cd Unnamed-Simulator
git checkout develop
```

Initialize and update the submodules llvm-project, soda-opt and gem5-SALAM.

```bash
git submodule init
git submodule update
```

### Dependency

#### LLVM & MLIR

The environment may resolve the dependencies and meet the requirement by following the instructions in the [LLVM Getting Started](https://llvm.org/docs/GettingStarted.html#requirements) page. In Ubuntu 22.04, you may follow these commands to install the dependencies.

```bash
sudo apt-get install build-essential cmake ninja-build python3-pip

# Using pip as package manager
# You may would like to install the packages in a virtual environment
pip install pybind11 numpy
```

Build LLVM and MLIR from SODA-OPT project. You may refer to the [How to build?](https://github.com/lczzz29/soda-opt?tab=readme-ov-file#how-to-build) section in its README file. The steps using Helper Script are listed below.

```bash
cd soda-opt/build_tools/

# To configure, build, and install
./build_llvm.sh <path/to/llvm/src> <llvm_build_dir> <llvm_install_dir>

# To configure, build, and install
./build_tools/build_soda.sh <source_dir> <install_dir> <build_dir> <path/to/llvm/build/dir> <path/to/llvm/install/dir>
```

After building LLVM, MLIR and SODA-OPT, you may need to add the path to their binaries to the environment variable `PATH`.

```bash
export PATH=$PATH:<path/to/llvm/install/bin>:<path/to/soda-opt/install/bin>
```

#### PyTorch & Torch-MLIR

Download the wheel package of torch and torch-mlir from the [release page](https://github.com/llvm/torch-mlir/releases/download/). So far we tested following specific versions:

```bash
# torch
wget https://github.com/llvm/torch-mlir/releases/download/oneshot-20230101.76/torch-2.0.0.dev20230101+cpu-cp310-cp310-linux_x86_64.whl

# torch-mlir
wget https://github.com/llvm/torch-mlir/releases/download/oneshot-20230101.76/torch_mlir-20230101.76-cp310-cp310-linux_x86_64.whl
```

Install the wheel packages using pip.

```bash
pip install torch-2.0.0.dev20230101+cpu-cp310-cp310-linux_x86_64.whl
pip install torch_mlir-20230101.76-cp310-cp310-linux_x86_64.whl
```

#### gem5-SALAM

Build the gem5-SALAM project. You may refer to the [Building gem5](https://www.gem5.org/documentation/general_docs/building) to install dependencies on different OS.

For Ubuntu 22.04, you may install the dependencies using the following commands.

```bash
sudo apt install build-essential git m4 scons zlib1g zlib1g-dev \
    libprotobuf-dev protobuf-compiler libprotoc-dev libgoogle-perftools-dev \
    python3-dev libboost-all-dev pkg-config python3-tk
```

Then build the gem5-SALAM project.

```bash
cd gem5-SALAM
scons build/ARM/gem5.opt -j`nproc`
```

#### Transformers (Optional)

You may not need this package if you are not going to use the transformer model. You can install other packages like openCV if you want to use related models.

Install the transformers package using pip.

```bash
pip install transformers
```

#### pydot and graphviz (Optional)

pydot and graphviz are used to visualize the system config file in gem5. Install the pydot and graphviz packages using pip.

```bash
pip install pydot graphviz
```



