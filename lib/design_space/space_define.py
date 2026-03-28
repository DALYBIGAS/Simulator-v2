from hyperopt import hp
from hyperopt.pyll.base import scope

# List of MLIR ops to accelerate (with priority order)
mlir_ops = ["op1", "op2", "op3", "op4", "op5"]

# Maximum number of accelerators per op
max_accelerators = 8

# Define the search space
search_space = {
    # Choose a subset of MLIR ops to accelerate (with priority order)
    "ops_to_accelerate": hp.choice(
        "ops_to_accelerate",
        [
            # Example: Accelerate 2 ops
            ["op1", "op2"],
            # Example: Accelerate 3 ops
            ["op1", "op2", "op3"],
            # Example: Accelerate all ops
            mlir_ops,
        ],
    ),
    # Number of accelerators per op (uniformly distributed up to max_accelerators)
    "num_accelerators": {
        op: scope.int(hp.quniform(f"num_accelerators_{op}", 1, max_accelerators, 1))
        for op in mlir_ops
    },
    # Double buffer option for each op (True/False)
    "double_buffer": {
        op: hp.choice(f"double_buffer_{op}", [True, False]) for op in mlir_ops
    },
    # Mover option for each op (True/False)
    "mover": {op: hp.choice(f"mover_{op}", [True, False]) for op in mlir_ops},
    # Cache option for each accelerator cluster (True/False)
    "cache": hp.choice("cache", [True, False]),
}

# Example of how to use the search space in a Hyperopt objective function
def objective(params):
    # Extract parameters
    ops_to_accelerate = params["ops_to_accelerate"]
    num_accelerators = params["num_accelerators"]
    double_buffer = params["double_buffer"]
    mover = params["mover"]
    cache = params["cache"]

    # Simulate a cost/performance metric (replace with actual logic)
    cost = 0
    for op in ops_to_accelerate:
        cost += num_accelerators[op]  # Example: cost increases with more accelerators
        if double_buffer[op]:
            cost += 1  # Example: double buffering adds to cost
        if mover[op]:
            cost += 1  # Example: mover adds to cost
    if cache:
        cost += 2  # Example: caching adds to cost

    # Return the cost to minimize
    return cost

# Example of running Hyperopt optimization
from hyperopt import fmin, tpe, Trials

trials = Trials()
best = fmin(
    objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=100,  # Number of evaluations
    trials=trials,
)

print("Best parameters:", best)