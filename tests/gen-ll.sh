#!/bin/sh

cd $(dirname $0)
rm -r inter && mkdir inter
rm -rf canon && mkdir canon
find . -type f -name "*.cu" -print0 | xargs -0 -i sh -c 'cd inter && clang -g -S -emit-llvm -Xclang -disable-O0-optnone ../{} 2> /dev/null'
cd inter && find . -type f -name "*.ll" -print0 | xargs -0 -i sh -c 'opt --polly-canonicalize {} | llvm-dis > ../canon/{}'