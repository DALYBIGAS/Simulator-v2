import subprocess
import sys
class Operand:
    def __init__(self, name: str, inout: str, dtype: str, varname: str = None):
        self.name = name
        self.inOut = inout
        self.dType = dtype
        self.varName = varname

class Operation:
    def __init__(self, name: str, operands: list, results: list, tile: str):
        self.name = name
        self.operands = operands
        self.results = results
        self.tile = tile  # Tile sizes as a comma-separated string (e.g., "4,4,4")

    def apply_transform(self, input_file: str, transform_program: str = "call_transform_program.py"):
        """
        Calls the `tile_call.py` script to apply tiling to the operation.

        Args:
            input_file (str): Path to the input MLIR file.
            transform_program (str): Path to the `tile_call.py` script.
        """
        # Prepare the command to call the transform program
        command = [
            "python3",
            transform_program,
            input_file,
            f"--match-op={self.name}",
            f"--tile-sizes={self.tile}",
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

class FusableOps:
    def __init__(self):
        # Initialize an empty list to store fusable operation pairs
        self.fusable_pairs = []

    def add(self, op1: str, op2: str):
        """
        Add a pair of fusable operations to the list.
        Args:
            op1 (str): The first operation in the pair.
            op2 (str): The second operation in the pair.
        """
        self.fusable_pairs.append((op1, op2))

    def generate_full_list(self):
        """
        Generate all possible fusable operation chains based on the recorded pairs.
        Returns:
            list: A list of lists, where each inner list represents a fusable operation chain.
        """
        # Create a graph to represent fusable operations
        graph = {}
        for op1, op2 in self.fusable_pairs:
            if op1 not in graph:
                graph[op1] = []
            if op2 not in graph:
                graph[op2] = []
            graph[op1].append(op2)
            graph[op2].append(op1)

        # Helper function to perform DFS and generate chains
        def dfs(node, visited, path, all_chains):
            visited.add(node)
            path.append(node)
            all_chains.append(path.copy())
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, visited, path, all_chains)
            path.pop()
            visited.remove(node)

        # Generate all possible chains
        all_chains = []
        for node in graph:
            dfs(node, set(), [], all_chains)

        # Remove duplicate chains (e.g., [A, B] and [B, A] are considered the same)
        unique_chains = []
        for chain in all_chains:
            sorted_chain = tuple(sorted(chain))
            if sorted_chain not in unique_chains:
                unique_chains.append(sorted_chain)

        # Convert tuples back to lists
        return [list(chain) for chain in unique_chains]
