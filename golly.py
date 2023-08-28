import pathlib as path
import subprocess as sp
import io
import uuid
import argparse

golly_path = "./build/lib/golly.so"


def compile(filename: path.Path, workdir: path.Path):
    workdir.mkdir(parents=True, exist_ok=True)
    sp.run(
        [
            "clang",
            "-g",
            "-fno-discard-value-names",
            "-S",
            "-emit-llvm",
            "-Xclang",
            "-disable-O0-optnone",
            filename.resolve(),
        ],
        cwd=workdir.resolve(),
    )


def canonicalize(outfile: io.TextIOWrapper, workdir: path.Path):
    # find the ptx ll
    ll = next(
        path for path in workdir.iterdir() if path.stem.find("nvptx64-nvidia") != -1
    )
    (out,) = (
        sp.Popen(
            ["opt", "--polly-canonicalize", ll.resolve()],
            cwd=workdir.resolve(),
            stdout=sp.PIPE,
        ),
    )

    asm = sp.Popen("llvm-dis", shell=True, stdin=out.stdout, stdout=outfile)
    asm.wait()


def analyze(file: path.Path, blockDim: str, gridDim: str):
    sp.run(
        [
            "opt",
            "-load",
            f"{golly_path}",
            f"--load-pass-plugin={golly_path}",
            "--passes=golly",
            "--disable-output",
            file.resolve(),
        ]
        + (["--golly-block-dims", blockDim] if blockDim is not None else [])
        + (["--golly-grid-dims", gridDim] if gridDim is not None else [])
    )


parser = argparse.ArgumentParser(
    prog="Golly",
    description="Polyhedral CUDA Analyzer",
    epilog="Please direct all issues to Ivan Ho, National University of Singapore [hhkit@nus.edu.sg]",
)

parser.add_argument("filename")
parser.add_argument(
    "--blockDim", help="Block dimensions, specify as [x,y,z] | [x,y] | x"
)
parser.add_argument("--gridDim", help="Grid dimensions, specify [x,y,z] | [x,y] | x")

args = parser.parse_args()

workdir = path.Path(f"/tmp/golly/{uuid.uuid4()}")
cu_file = path.Path(args.filename)
ll_file = cu_file.with_suffix(".ll")

assert cu_file.exists()

compile(cu_file, workdir)
with open(ll_file, "w") as out_ll:
    canonicalize(out_ll, workdir)
analyze(ll_file, blockDim=args.blockDim, gridDim=args.gridDim)
