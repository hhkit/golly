#!/bin/sh

BASEDIR=$(dirname $0)
cd $BASEDIR
cp ../../build/tools/kernel_extractor/kernel_extractor .
docker build --file ./gpuverify.Dockerfile --tag gpuver .