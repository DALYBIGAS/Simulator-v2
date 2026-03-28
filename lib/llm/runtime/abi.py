"""Shared ABI helpers for the LLM runtime plan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class TensorABI:
    name: str
    dtype: str
    rank: int
    shape: List[int]
    stride: List[int]

    @classmethod
    def contiguous(cls, name: str, dtype: str, shape: List[int]) -> "TensorABI":
        stride = []
        running = 1
        for extent in reversed(shape):
            stride.append(running)
            running *= max(extent, 1)
        return cls(name=name, dtype=dtype, rank=len(shape), shape=shape, stride=list(reversed(stride)))

    @property
    def numel(self) -> int:
        total = 1
        for extent in self.shape:
            total *= max(extent, 1)
        return total
