#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

try:
    from .common import build_fuse_script
except ImportError:
    from common import build_fuse_script


def _parse_ops(raw: str):
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_tiles(raw: str):
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a fusion-oriented Transform dialect script.")
    parser.add_argument("input_file")
    parser.add_argument("--op-chain", required=True, help="Comma separated list of ops to match.")
    parser.add_argument("--kernel-name", required=True)
    parser.add_argument("--last-tile-sizes", required=True, help="Comma separated tile sizes for the last stage.")
    parser.add_argument("--output-transform")
    parser.add_argument("--mlir-opt", default="mlir-opt")
    parser.add_argument("--emit-only", action="store_true")
    args = parser.parse_args()

    ops = _parse_ops(args.op_chain)
    tiles = _parse_tiles(args.last_tile_sizes)
    transform_path = Path(args.output_transform or f"{args.input_file}.fuse_transform.mlir")
    build_fuse_script(ops, args.kernel_name, tiles).write(transform_path)
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
