#!/bin/sh
# removes all llvm files

find . -name "*.ll" -type f -delete
find . -name "*.pp.cu" -type f -delete
find . -name "*.ext.cu" -type f -delete