import subprocess
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Call the fuse transform program.")
    parser.add_argument("input_file", help="Path to the input MLIR file.")
    parser.add_argument("--op-chain", required=True,
                        help="Comma separated list of operations to fuse (e.g., \"[opA, opB, opC]\").")
    parser.add_argument("--kernel-name", required=True,
                        help="Final hardware kernel name after fusion (e.g., \"opA_opB_opC_fused_kernel\").")
    parser.add_argument("--last-tile-sizes", required=True,
                        help="Tile sizes for the last op tiling as a comma-separated string (e.g., \"4,4,4\").")
    parser.add_argument("--transform-program", default="./fuse_transform",
                        help="Path to the compiled fuse_transform binary.")
    args = parser.parse_args()

    # Prepare the command for the fuse_transform binary.
    command = [
        args.transform_program,
        args.input_file,
        f"--op-chain={args.op_chain}",
        f"--kernel-name={args.kernel_name}",
        f"--last-tile-sizes={args.last_tile_sizes}"
    ]

    print("Running command:", " ".join(command))
    
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print("Fuse transform output:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error running fuse transform:", e.stderr, file=sys.stderr)
        sys.exit(1)

    # Run mlir-opt to apply the transform.
    mlir_opt_command = ["mlir-opt", "--transform-interpreter", args.input_file]
    print("Running command:", " ".join(mlir_opt_command))
    
    try:
        result = subprocess.run(mlir_opt_command, check=True, text=True, capture_output=True)
        print("mlir-opt output:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error running mlir-opt:", e.stderr, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
