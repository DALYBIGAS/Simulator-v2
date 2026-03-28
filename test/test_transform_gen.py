import unittest

from lib.transform_gen.common import build_fuse_script, build_outline_script, build_tile_script


class TransformGenTests(unittest.TestCase):
    def test_tile_script_contains_match_and_tile(self):
        script = build_tile_script("linalg.matmul", [128, 128, 64]).render()
        self.assertIn('transform.structured.match ops ["linalg.matmul"]', script)
        self.assertIn('tile_sizes [128, 128, 64]', script)

    def test_fuse_script_contains_kernel_annotation(self):
        script = build_fuse_script(["linalg.matmul", "linalg.generic"], "qkv_fused", [64, 64, 32]).render()
        self.assertIn('llm.kernel_name = "qkv_fused"', script)

    def test_outline_script_contains_function_name(self):
        script = build_outline_script("linalg.matmul", "matmul_kernel").render()
        self.assertIn('llm.outline_target = "matmul_kernel"', script)


if __name__ == "__main__":
    unittest.main()
