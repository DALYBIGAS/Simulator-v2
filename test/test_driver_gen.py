import tempfile
import unittest
from pathlib import Path

from lib.driver_gen.driver_gen import build_stub


class DriverGenTests(unittest.TestCase):
    def test_build_stub_contains_kernel_and_tensor_abi(self):
        stub = build_stub("attention_kernel", "prefill", "bf16", ["query", "key"], ["output"])
        self.assertIn("void attention_kernel_launch", stub)
        self.assertIn("typedef struct", stub)
        self.assertIn("ai_tensor_ref query", stub)
        self.assertIn("ai_tensor_ref output", stub)


if __name__ == "__main__":
    unittest.main()
