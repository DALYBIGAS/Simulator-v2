#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

try:
    from .common import build_outline_script
except ImportError:
    from common import build_outline_script


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an outline-oriented Transform dialect script.")
    parser.add_argument("input_file")
    parser.add_argument("--op-name", required=True)
    parser.add_argument("--function-name", required=True)
    parser.add_argument("--output-transform")
    parser.add_argument("--mlir-opt", default="mlir-opt")
    parser.add_argument("--emit-only", action="store_true")
    args = parser.parse_args()

    transform_path = Path(args.output_transform or f"{args.input_file}.outline_transform.mlir")
    build_outline_script(args.op_name, args.function_name).write(transform_path)
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
