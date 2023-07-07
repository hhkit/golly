#!/bin/sh

cd $(dirname $0) && find . -type f -name "*.cu" -print0 | xargs -0 -I {} clang -S -emit-llvm -Xclang -disable-O0-optnone {}