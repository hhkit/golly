#!/usr/bin/python3
import pathlib as path
import subprocess as sp
import os
import yaml
import argparse
import docker

parser = argparse.ArgumentParser(
    prog="GPUVerify",
    )
parser.add_argument("filename", type=path.Path)
args = parser.parse_args()

def strip(s):
  return str(s).replace(' ', '')


if args.filename.suffix == ".yaml":
  with open(args.filename, 'r') as file:
    dir = os.path.splitext(args.filename)[0]

    for file in yaml.safe_load(file):
      kernels = file['kernels']
      # take the first kernel
      cfg = kernels[0]

      filename=path.Path(str(dir) + "/" + str(file['file']))

      b = strip(str(cfg['block']))
      g = strip(str(cfg['grid']))

      client = docker.from_env()
      cont = client.containers.run(image="gpuver", volumes=[
        f"{os.getcwd()}/tests:/mnt/tests"
      ], command="bash", detach=True, tty=True)


      (_,out) = cont.exec_run(f"bash -c 'clang -E /mnt/{filename} > /mnt/{filename}.pp.cu'")
      (_,out) = cont.exec_run(f"/kernel_extractor /mnt/{filename}.pp.cu", stderr=False)
      (_,out) = cont.exec_run([
        "gpuverify",
        f"--blockDim={b}",
        f"--gridDim={g}",
        f"/mnt/{filename}.pp.cu.ext.cu"
      ])
      print(filename)
      print(out.decode('UTF-8'))

      cont.stop()
      cont.remove()