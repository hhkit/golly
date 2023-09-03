import pathlib as path
import subprocess as sp
import io
import uuid
import argparse
import tempfile

golly_path = "./build/lib/golly.so"


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
            ["opt", "--polly-canonicalize", ll.resolve()],
            cwd=workdir.resolve(),
            stdout=sp.PIPE,
        ),
    )

    asm = sp.Popen("llvm-dis", shell=True, stdin=out.stdout, stdout=outfile)
    asm.wait()


def analyze(file: path.Path, blockDim: str, gridDim: str, verbose: bool):
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
        + (["--golly-verbose"] if verbose else [])
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Golly",
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
    parser.add_argument("clangArgs", help="argument to pass to clang", nargs="*")
    parser.add_argument("--verbose", "-v", action="store_true", default=False)

    args = parser.parse_args()

    workdir = path.Path(f"{tempfile.gettempdir()}/golly/{uuid.uuid4()}")

    file = args.filename
    assert file.exists()

    if file.suffix == ".cu":
        # compile and canonicalize
        ll_file = file.with_suffix(".ll")

        compile(file, workdir, args.showWarnings, args.clangArgs)
        with open(ll_file, "w") as out_ll:
            canonicalize(out_ll, workdir)

        file = ll_file

    assert file.exists()
    analyze(file, blockDim=args.blockDim, gridDim=args.gridDim, verbose=args.verbose)
