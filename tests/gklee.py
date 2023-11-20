#!/usr/bin/python3
import pathlib as path
import subprocess as sp
import os
import yaml
import argparse
import re
import docker
import time
from statistics import mean

parser = argparse.ArgumentParser(
    prog="GKLEE driver",
)
parser.add_argument("filename", type=path.Path)
parser.add_argument("--profile", action="store_true")
parser.add_argument("--profile-out", dest="profileOut", type=path.Path)
parser.add_argument("--iters", type=int, default=1)
args = parser.parse_args()


def strip(s):
    return str(s).replace(" ", "")


race_re = re.compile("incur the \(Actual\) (read|write)-write race")
bd_re = re.compile("Found a deadlock")


def extract_error(ec, log):
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
                image="gklee",
                volumes=[f"{os.getcwd()}/tests:/mnt/tests"],
                command="bash",
                detach=True,
                tty=True,
            )

            base_name = os.path.basename(os.path.splitext(filename)[0])
            (ec, out) = cont.exec_run(
                f"bash -c 'gklee-nvcc /mnt/{filename} -o {base_name}.bc -I/mnt/{os.path.dirname(filename)} -I/usr/include '"
            )

            def exec():
                (ec, out) = cont.exec_run(
                    f"bash -c 'timeout 1m gklee --verbose 0 --symbolic-config --race-prune {base_name}.bc'"
                )

                return (ec, out.decode("UTF-8"))

            if profile:
                for i in range(iters):
                    start = time.perf_counter()
                    ec, out = exec()
                    end = time.perf_counter()
                    dur = end - start

                    if str(filename) not in timings:
                        err = extract_error(ec, out)
                        timings[str(filename)] = {
                            "times": [dur],
                            "error": err,
                            "log": out,
                        }
                        print(f"{filename}: {err}")
                    else:
                        timings[str(filename)]["times"].append(dur)
            else:
                exec()

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
