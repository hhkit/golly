#!/bin/sh
apt update 
apt install -y \
    build-essential \
    cmake \
    llvm \
    clang \
    pkg-config \
    libclang-dev \
    libisl-dev \
    git \
    ninja-build \
    && apt autoremove \
    && apt clean -y \
    && rm -rf /var/lib/apt/lists/*