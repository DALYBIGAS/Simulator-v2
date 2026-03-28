"""Helpers for generating MLIR Transform dialect programs.

The existing C++ prototypes are kept in-tree, but these helpers are now the
stable entrypoint for script generation and orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class TransformScript:
    body: str

    def render(self) -> str:
        return self.body.strip() + "\n"

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render())
        return path


def _tile_attr(tile_sizes: Iterable[int]) -> str:
    return "[" + ", ".join(str(x) for x in tile_sizes) + "]"


def build_tile_script(match_op: str, tile_sizes: List[int]) -> TransformScript:
    return TransformScript(
        f'''
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
    %matched = transform.structured.match ops ["{match_op}"] in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled, %loops:2 = transform.structured.tile_using_for %matched tile_sizes {_tile_attr(tile_sizes)}
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }}
}}
'''
    )


def build_fuse_script(op_chain: List[str], kernel_name: str, last_tile_sizes: List[int]) -> TransformScript:
    op_match_list = ", ".join(f'"{op}"' for op in op_chain)
    return TransformScript(
        f'''
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
    %chain = transform.structured.match ops [{op_match_list}] in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled, %loops:2 = transform.structured.tile_using_for %chain tile_sizes {_tile_attr(last_tile_sizes)}
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %tiled {{llm.kernel_name = "{kernel_name}"}} : !transform.any_op
    transform.yield
  }}
}}
'''
    )


def build_outline_script(op_name: str, function_name: str) -> TransformScript:
    return TransformScript(
        f'''
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
    %target = transform.structured.match ops ["{op_name}"] in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.annotate %target {{llm.outline_target = "{function_name}"}} : !transform.any_op
    transform.yield
  }}
}}
'''
    )
