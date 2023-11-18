#!/bin/sh

BASEDIR=$(dirname $0)
cd $BASEDIR
# cp ../../build/tools/kernel_extractor/kernel_extractor .
cp -r ../../tools/kernel_extractor/* kernel_extractor.src
docker build --file ./gpuverify.Dockerfile --tag gpuverify .
docker build --file ./gklee.Dockerfile --tag gklee .