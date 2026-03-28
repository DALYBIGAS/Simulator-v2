import subprocess
import argparse
import sys

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Call the MLIR transform program.")
    parser.add_argument("input_file", help="Path to the input MLIR file.")
    parser.add_argument("--match-op", default="linalg.matmul", help="Operation name to match (e.g., linalg.matmul).")
    parser.add_argument("--tile-sizes", nargs="+", type=int, default=[4, 4, 4], help="Tile sizes for tiling (e.g., 4 4 4).")
    parser.add_argument("--transform-program", default="./transform_program", help="Path to the compiled transform program.")
    args = parser.parse_args()

    # Prepare the command to call the transform program
    command = [
        args.transform_program,
        args.input_file,
        f"--match-op={args.match_op}",
        f"--tile-sizes={','.join(map(str, args.tile_sizes))}",
    ]

    # Print the command for debugging
    print("Running command:", " ".join(command))

    # Call the transform program
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print("Transform program output:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error running transform program:", e.stderr, file=sys.stderr)
        sys.exit(1)

    # Run the mlir-opt command to apply the transform
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