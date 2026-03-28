"""Public entrypoints for config parsing.

The legacy simulator parser files remain in-tree, but some of them are not safe
for eager import because of historical circular dependencies. Stage-2 users can
import the structured YAML loaders below without pulling the full legacy stack.
"""

try:  # Legacy compatibility, best effort only.
    from .acc_cluster import AccCluster  # type: ignore
    from .accelerator import Accelerator  # type: ignore
    from .dma import DMA, StreamDMA  # type: ignore
    from .variable import Variable, PortedConnection  # type: ignore
    from .op import Operand, Operation, FusableOps  # type: ignore
except Exception:  # pragma: no cover
    AccCluster = None
    Accelerator = None
    DMA = None
    StreamDMA = None
    Variable = None
    PortedConnection = None
    Operand = None
    Operation = None
    FusableOps = None

from .hw_caps import HardwareCaps, SRAMConfig, DMAConfig, ComputeArrayConfig
from .compiler_options import CompilerOptions, get_compiler_options
from .schema import TensorSpec, CompileSpec
from .parser import load_hardware_caps, load_compile_spec, load_compilation_context, SpecError

__all__ = [
    "AccCluster",
    "Accelerator",
    "DMA",
    "StreamDMA",
    "Variable",
    "PortedConnection",
    "Operand",
    "Operation",
    "FusableOps",
    "HardwareCaps",
    "SRAMConfig",
    "DMAConfig",
    "ComputeArrayConfig",
    "CompilerOptions",
    "get_compiler_options",
    "TensorSpec",
    "CompileSpec",
    "load_hardware_caps",
    "load_compile_spec",
    "load_compilation_context",
    "SpecError",
]
