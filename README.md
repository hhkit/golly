# Golly

Golly (from 'GPU' and '[Polly](https://polly.llvm.org/)') is an LLVM-based static race detector for detecting races in affine CUDA kernels, based on the integer linear programming solver in [isl](https://repo.or.cz/w/isl.git). Race freedom is not guaranteed, but as far as possible, we aim to minimize false positives in error reporting to prevent leading users on wild goose chases.

# Installation

Golly was developed on Ubuntu 22.04 Jammy Jellyfish, which ships with Python 3.10.
The required packages for Ubuntu 22.04 are as follows:
```bash
$ sudo apt install -y cmake llvm clang pkg-config libclang-dev libisl-dev
```

## Building with CMake
With the dependencies installed, Golly can be built with the following commands:
```
$ git clone --recurse-submodules https://github.com/hhkit/golly
$ mkdir build && cd build
$ cmake ..
$ cmake --build .
```

For a quick demo without worrying about dependencies, try opening this repository in a codespace. Note that the container is rather large, so remember to destroy your codespace after use.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/hhkit/golly?quickstart=1)

# Usage

Once built, `golly` can be executed from the base directory on any `.cu` file.
```bash
$ ./golly tests/simple.cu 
```

~~Block and grid dimensions can be specified with `--blockDim` and `--gridDim` flags.~~

Note: Block/grid specification for individual files are currently broken due to development focus on bulk testing.
<details>
<summary>Command, once fixed</summary>
```bash
$ ./golly tests/simple.cu --blockDim 256 --gridDim 1
```
</details>

Two benchmark suites have been provided, PolyBench/GPU and the Gklee test suite. To access these benchmarks, golly must be cloned with submodules. If this has not been done, execute the following command:
```bash
$ git submodule update --init
```

Otherwise, a test suite can be tested with a .yaml configuration. Either of the following commands should work:
```bash
$ ./golly tests/polybenchGpu.yaml
$ ./golly tests/GkleeTests.yaml
```