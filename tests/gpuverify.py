#!/usr/bin/python3
import pathlib as path
import subprocess as sp
import os
import yaml
import argparse
import docker
import re
import time
from statistics import mean

parser = argparse.ArgumentParser(
    prog="GPUVerify profiler",
)
parser.add_argument("filename", type=path.Path)
parser.add_argument("--profile", action="store_true")
parser.add_argument("--profile-out", dest="profileOut", type=path.Path)
parser.add_argument("--iters", type=int, default=1)
args = parser.parse_args()


def strip(s):
    return str(s).replace(" ", "")


race_re = re.compile("error: possible (read|write)-write race")
bd_re = re.compile("error: barrier may be reached by non-uniform control flow")


def extract_error(ec, log):
    if ec != 0:
        if bd_re.search(log) != None:
            return "bd"
        if race_re.search(log) != None:
            return "race"

    return "-"


profile = bool(args.profile)
iters = args.iters

timings = {}

if args.filename.suffix == ".yaml":
    with open(args.filename, "r") as file:
        dir = os.path.splitext(args.filename)[0]

        for file in yaml.safe_load(file):
            kernels = file["kernels"]
            # take the first kernel
            cfg = kernels[0]

            filename = path.Path(str(dir) + "/" + str(file["file"]))

            b = strip(str(cfg["block"]))
            g = strip(str(cfg["grid"]))

            client = docker.from_env()
            cont = client.containers.run(
                image="gpuverify",
                volumes=[f"{os.getcwd()}/tests:/mnt/tests"],
                command="bash",
                detach=True,
                tty=True,
            )

            # (_,out) = cont.exec_run(f"bash -c 'echo | clang -xc++ -E -v -'")
            # (_,out) = cont.exec_run(f"bash -c 'clang -E /mnt/{filename} > /mnt/{filename}.pp.cu'")
            (_, out) = cont.exec_run(f"bash -c 'kernel_extractor /mnt/{filename}'")
            (_, out) = cont.exec_run(f"bash -c 'cat /mnt/{filename}.ext.cu'")
            (_, out) = cont.exec_run(
                f"bash -c 'sed \"/<iostream>/d\" /mnt/{filename}.ext.cu > /mnt/{filename}.fin.cu'"
            )
            (_, out) = cont.exec_run(f"bash -c 'cat /mnt/{filename}.fin.cu'")

            def exec():
                return cont.exec_run(
                    [
                        "timeout",
                        "10m",
                        "gpuverify",
                        f"--blockDim={b}",
                        f"--gridDim={g}",
                        f"--clang-opt=--std=c++17",
                        f"--clang-opt=-nocudainc",
                        f"--clang-opt=-I/usr/include/c++/11",
                        f"--clang-opt=-I/usr/include/x86_64-linux-gnu/c++/11",
                        f"--clang-opt=-I/usr/include/c++/11/backward",
                        f"--clang-opt=-I/usr/lib/llvm-14/lib/clang/14.0.0/include",
                        f"--clang-opt=-I/usr/local/include",
                        f"--clang-opt=-I/usr/x86_64-linux-gnu/include",
                        f"--clang-opt=-I/usr/include/x86_64-linux-gnu",
                        f"--clang-opt=-I/include",
                        f"--clang-opt=-I/usr/include",
                        f"--clang-opt=-D_GNU_SOURCE",  # needed for polybench
                        f"/mnt/{filename}.fin.cu",
                    ]
                )

            if profile:
                for i in range(iters):
                    start = time.perf_counter()
                    (ec, out) = exec()
                    end = time.perf_counter()
                    dur = end - start
                    err = extract_error(ec, out.decode("UTF-8"))

                    if str(filename) not in timings:
                        timings[str(filename)] = {
                            "times": [dur],
                            "error": err,
                            "log": out.decode("UTF-8"),
                        }
                        print(f"{filename} - {err}")
                        # print(out)
                    else:
                        timings[str(filename)]["times"].append(dur)
            else:
                (_, out) = exec()
            # print(filename)
            # print(out.decode("UTF-8"))

        cont.stop()
        cont.remove()

    for k, v in timings.items():
        times = v["times"]
        timings[k]["min"] = min(times)
        timings[k]["avg"] = mean(times)
        timings[k]["max"] = max(times)

    if args.profileOut is not None:
        with open(args.profileOut, "w") as outFile:
            outFile.write(yaml.dump(timings))
    else:
        print(str(timings))
