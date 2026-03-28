#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

try:
    from .common import build_tile_script
except ImportError:
    from common import build_tile_script


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a tiling Transform dialect script for an MLIR module.")
    parser.add_argument("input_file", help="Path to the payload MLIR file.")
    parser.add_argument("--match-op", default="linalg.matmul")
    parser.add_argument("--tile-sizes", nargs="+", type=int, default=[4, 4, 4])
    parser.add_argument("--output-transform", help="Where to write the generated transform MLIR.")
    parser.add_argument("--mlir-opt", default="mlir-opt")
    parser.add_argument("--emit-only", action="store_true", help="Only emit the transform script and skip mlir-opt.")
    args = parser.parse_args()

    transform_path = Path(args.output_transform or f"{args.input_file}.tile_transform.mlir")
    build_tile_script(args.match_op, args.tile_sizes).write(transform_path)
    print(transform_path)

    if args.emit_only:
        return

    command = [args.mlir_opt, args.input_file, f"--transform-file={transform_path}", "--transform-interpreter"]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)
    print(result.stdout)


if __name__ == "__main__":
    main()
