#!/usr/bin/env python3
import pandas as pd
import pathlib as path
import yaml
import itertools as it
import os
import collections.abc


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def open_yaml(filepath: path.Path) -> dict:
    print(f"reading {filepath}")
    with open(filepath) as f:
        return yaml.load(f, Loader=yaml.Loader)


def escape(line: str) -> str:
    return line.replace("_", "\_")


detectors = ["golly", "gklee", "gpuverify"]

benchmarks = ["Gklee", "Polybench"]


cwd = os.curdir

profiles = {}

# parse benchmarks
for path in [
    cwd + "/../tests/polybenchGpu",
    cwd + "/../tests/GkleeTests",
]:
    suite = path.split("/")[-1]
    for case in open_yaml(path + ".yaml"):
        testcase = case["file"].split("/")[-2]
        err = case.get("error", "-")
        profiles = update(profiles, {suite: {testcase: {"err": err}}})

# parse profiles
for d, b in it.product(detectors, benchmarks):
    p = open_yaml(cwd + "/../" + "profiles/" + d + b + ".yaml")

    benchmark: str
    profile: dict
    for benchmark, profile in p.items():
        tokens = benchmark.split("/")
        suite = tokens[1]
        testcase = tokens[-2]
        profiles = update(
            profiles,
            {
                suite: {
                    testcase: {
                        "results": {
                            d: {
                                "time": profile["avg"],
                                "error": profile.get("error", "-"),
                            }
                        }
                    }
                }
            },
        )
# print(profiles)

# benchmark | expected result | golly runtime | golly correct | gpuverify runtime | gv correct | gklee runtime | gklee correct |
df = pd.DataFrame(
    columns=[
        "Filename",
        "Error_{expected}",
        "GoTime",
        "GoRes",
        "GvTime",
        "GvRes",
        "GkTime",
        "GkRes",
    ]
)

for suite, suitedata in profiles.items():
    for casename, data in suitedata.items():
        res = data["results"]
        df.loc[len(df.index)] = [
            escape(casename),
            data["err"],
            res["golly"]["time"],
            res["golly"]["error"],
            res["gpuverify"]["time"],
            res["gpuverify"]["error"],
            res["gklee"]["time"],
            res["gklee"]["error"],
        ]
print(df.to_latex(index=False))
