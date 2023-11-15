# Golly

Golly (from 'GPU' and '[Polly](https://polly.llvm.org/)' [1]) is an LLVM-based static race detector for detecting races in affine CUDA kernels, based on the integer linear programming solver in [isl](https://repo.or.cz/w/isl.git) [2]. Race freedom is not guaranteed, but as far as possible, we aim to minimize false positives in error reporting to prevent leading users on wild goose chases.

# Installation

Golly was developed on Ubuntu 22.04 Jammy Jellyfish, which ships with Python 3.10.
The required packages for Ubuntu 22.04 are as follows:
```bash
$ sudo apt install -y cmake llvm clang pkg-config libclang-dev libisl-dev
```

To be completely certain, please refer to the [install script](.devcontainer/install-dev-tools.sh).

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

Two benchmark suites have been provided, PolyBench/GPU [3] and the Gklee [4] test suite. To access these benchmarks, golly must be cloned with submodules. If this has not been done, execute the following command:
```bash
$ git submodule update --init
```

Otherwise, a test suite can be tested with a .yaml configuration. Either of the following commands should work:
```bash
$ ./golly tests/polybenchGpu.yaml
$ ./golly tests/GkleeTests.yaml
```

# References
[1] T. Grosser, A. Groesslinger, C. Lengauer, "Polly - Performing polyhedral optimizations on a low-level intermediate representation" Parallel Processing Letters 2012 22:04, doi: 10.1142/S0129626412500107

[2] Verdoolaege, S. (2010). isl: An Integer Set Library for the Polyhedral Model. In: Fukuda, K., Hoeven, J.v.d., Joswig, M., Takayama, N. (eds) Mathematical Software – ICMS 2010. ICMS 2010. Lecture Notes in Computer Science, vol 6327. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-642-15582-6_49

[3] S. Grauer-Gray, L. Xu, R. Searles, S. Ayalasomayajula and J. Cavazos, "Auto-tuning a high-level language targeted to GPU codes," 2012 Innovative Parallel Computing (InPar), San Jose, CA, USA, 2012, pp. 1-10, doi: 10.1109/InPar.2012.6339595.

[4] Guodong Li, Peng Li, Geof Sawaya, Ganesh Gopalakrishnan, Indradeep Ghosh, and Sreeranga P. Rajan. 2012. GKLEE: concolic verification and test generation for GPUs. SIGPLAN Not. 47, 8 (August 2012), 215–224. https://doi.org/10.1145/2370036.2145844

  