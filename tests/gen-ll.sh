#!/bin/sh

cd $(dirname $0)
rm -r inter && mkdir inter
rm -rf canon && mkdir canon
find . -type f -name "*.cu" -print0 | xargs -0 -i sh -c 'cd inter && clang -g -fno-discard-value-names -DPOLYBENCH_USE_SCALAR_LB -DNI=32 -DNJ=32 -S -emit-llvm -Xclang -disable-O0-optnone ../{}'
cd inter && find . -type f -name "*.ll" -print0 | xargs -0 -i sh -c 'opt --polly-canonicalize {} | llvm-dis > ../canon/{}'