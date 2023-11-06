#!/usr/bin/python3
import pathlib as path
import subprocess as sp
import io
import os
import uuid
import argparse
import tempfile
import yaml

golly_path = "./build/lib/golly.so"
golly_repair_path = "./build/barrier-repair/golly-repair"

class FuncDetection:
    def __init__(self, name, blockDim, gridDim):
        self.name = name
        self.blockDim = blockDim
        self.gridDim = gridDim

def compile(
    filename: path.Path, workdir: path.Path, showWarnings: bool, clangArgs: [] = None
):
    workdir.mkdir(parents=True, exist_ok=True)
    sp.run(
        [
            "clang",
            "-g",
            "-fno-discard-value-names",
            "-S",
            "-emit-llvm",
            "--cuda-gpu-arch=sm_60",
            "-Xclang",
            "-disable-O0-optnone",
            "-DPOLYBENCH_USE_SCALAR_LB",
            filename.resolve(),
        ]
        + ["" if showWarnings else "-w"]
        + (clangArgs if clangArgs is not None else []),
        cwd=workdir.resolve(),
    )


def canonicalize(outfile: io.TextIOWrapper, workdir: path.Path):
    # find the ptx ll
    ll = next(
        path for path in workdir.iterdir() if path.stem.find("nvptx64-nvidia") != -1
    )
    (out,) = (
        sp.Popen(
            ["opt", "-early-cse", "--polly-canonicalize", ll.resolve()],
            cwd=workdir.resolve(),
            stdout=sp.PIPE,
        ),
    )

    asm = sp.Popen("llvm-dis", shell=True, stdin=out.stdout, stdout=outfile)
    asm.wait()


def analyze(file: path.Path, patchFile: path.Path, config: path.Path, verbose: bool):
    cmd = [
            "opt",
            "-load",
            f"{golly_path}",
            f"--load-pass-plugin={golly_path}",
            "--passes=golly" + (f"<config={config}>" if config is not None else ""),
            "--disable-output",
            file.resolve(),
        ] + (["--golly-verbose"] if verbose else []) + ([f"--golly-out={patchFile}" if patchFile is not None else []])
    print("command: " + " ".join(map(str, cmd)))
    sp.run(cmd)


def analysisPass(filename, workdir, patchFile, showWarnings, clangArgs, verbose,  config: path.Path = None, **kwargs):
    file = filename
    if file.suffix == ".cu":
        # compile and canonicalize
        ll_file = file.with_suffix(".ll")

        compile(file, workdir, args.showWarnings, args.clangArgs)
        with open(ll_file, "w") as out_ll:
            canonicalize(out_ll, workdir)

        file = ll_file

    assert file.exists()

    analyze(file, patchFile=patchFile, config=config, verbose=verbose)

def raceDetected(patchFile: path.Path) -> bool:
    return patchFile.exists() and patchFile.stat().st_size > 2

def repair(file: path.Path, workdir:path.Path, clangArgs, patchFile: path.Path, blockDim: str, gridDim: str):
    scratch = path.Path(f"{workdir}/{uuid.uuid4()}")
    tmp_dir = path.Path(f"{workdir}/patches")

    tmp_dir.mkdir(parents=True, exist_ok=True)

    sp.run([
        golly_repair_path,
        file,
        f"--locsFile={patchFile}",
        f"--golly-tmp={tmp_dir}"
    ])

    for f in tmp_dir.glob("*"):
        print(f"attempt repair with {f}")
        workdir2 = path.Path(f"{workdir}/{hash(f)}")
        analysisPass(f, workdir=workdir2, patchFile=scratch, showWarnings=False, clangArgs=clangArgs, verbose=False)
        if raceDetected(scratch):
            print(f"repair failed with {f}")
            scratch.unlink()
        else:
            print(f"repair success with {f}")
            return

    print('could not repair race')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="golly",
        description="Polyhedral CUDA Analyzer",
    )

    parser.add_argument("filename", type=path.Path)
    parser.add_argument(
        "--blockDim", "-B", help="Block dimensions, specify as [x,y,z] | [x,y] | x"
    )
    parser.add_argument(
        "--gridDim", "-G", help="Grid dimensions, specify [x,y,z] | [x,y] | x"
    )
    parser.add_argument(
        "--showWarnings",
        "-W",
        help="Shows warnings from clang (suppressed otherwise)",
        action="store_true",
    )
    
    parser.add_argument(
        "--repair",
        "-r",
        help="Attempt barrier repair",
        action="store_true",
    )
    parser.add_argument("clangArgs", help="argument to pass to clang", nargs="*")
    parser.add_argument("--verbose", "-v", action="store_true", default=False)

    args = parser.parse_args()


    if args.filename.suffix == ".yaml":
        with open(args.filename, 'r') as file:
            dir = os.path.splitext(args.filename)[0]
            for item in yaml.safe_load(file):
                f = item['file']
                kernels = item.get('kernels')
                assert(type(kernels) == type([])) # kernels should be an array 

                workdir = path.Path(f"{tempfile.gettempdir()}/golly/{uuid.uuid4()}")
                workdir.mkdir(parents=True, exist_ok=True)
                configFile=f"{workdir}/cfg"
                with open(configFile, 'w') as cfg:
                    yaml.safe_dump(kernels, cfg)
                patchFile=f"{workdir}/pairs.out"
                analysisPass(workdir=workdir, patchFile=patchFile, config=configFile, **dict(vars(args), filename=path.Path(f"{dir}/{f}")))
                print("\n")

    else:
        workdir = path.Path(f"{tempfile.gettempdir()}/golly/{uuid.uuid4()}")
        workdir.mkdir(parents=True, exist_ok=True)
        patchFile=f"{workdir}/pairs.out"
        analysisPass(workdir=workdir, patchFile=patchFile, **vars(args))

        if args.repair:
            repair(args.filename, workdir, clangArgs=args.clangArgs, patchFile=patchFile, blockDim=args.blockDim, gridDim=args.gridDim )
