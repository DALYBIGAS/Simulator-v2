import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from lib.config_parser import load_compilation_context
from lib.llm.passes.pipeline import build_llm_pipeline


class Stage2CompileTests(unittest.TestCase):
    def test_load_compilation_context(self):
        hw, options, spec = load_compilation_context(
            "examples/llm_stage2/hardware.yaml",
            "examples/llm_stage2/prefill_compile.yaml",
        )
        self.assertEqual(hw.name, "legend-llm-chip")
        self.assertEqual(options.mode, "prefill")
        self.assertEqual(spec.kernel_name, "qkv_prefill")
        self.assertEqual(spec.inputs[0].name, "query")

    def test_build_llm_pipeline(self):
        hw, options, _ = load_compilation_context(
            "examples/llm_stage2/hardware.yaml",
            "examples/llm_stage2/decode_compile.yaml",
        )
        stages = build_llm_pipeline(hw, options)
        self.assertIn("llm-fuse-epilogue", stages["kernel"])
        self.assertIn("llm-insert-async-dma", stages["buffer"])

    def test_compile_entrypoint_emits_manifest_and_driver(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = Path(tmpdir) / "payload.mlir"
            payload.write_text("module {}\n", encoding="utf-8")
            out_dir = Path(tmpdir) / "out"
            subprocess.run(
                [
                    sys.executable,
                    "compile.py",
                    "--hardware",
                    "examples/llm_stage2/hardware.yaml",
                    "--compile-spec",
                    "examples/llm_stage2/decode_compile.yaml",
                    "--payload-mlir",
                    str(payload),
                    "--out-dir",
                    str(out_dir),
                ],
                check=True,
            )
            manifest = json.loads((out_dir / "compile_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["mode"], "decode")
            self.assertTrue((out_dir / "attention_decode_driver.c").exists())
            self.assertTrue((out_dir / "attention_decode.tile.mlir").exists())
            self.assertTrue((out_dir / "attention_decode.outline.mlir").exists())


if __name__ == "__main__":
    unittest.main()
