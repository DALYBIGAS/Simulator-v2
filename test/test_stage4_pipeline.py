import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from lib.config_parser import load_compilation_context
from lib.llm.models.profiles import resolve_model_profile
from lib.llm.profiler.estimator import estimate_metrics
from lib.llm.runtime.codegen import generate_runtime_launch_code
from lib.llm.runtime.plan import build_runtime_plan


class Stage4PipelineTests(unittest.TestCase):
    def test_model_profile_resolution(self):
        profile = resolve_model_profile("mixtral-8x7b", "mixtral")
        self.assertEqual(profile.family, "mixtral")

    def test_estimated_metrics(self):
        hw, _, spec = load_compilation_context(
            "examples/llm_stage4/hardware.yaml",
            "examples/llm_stage4/qwen3_prefill.yaml",
        )
        metrics = estimate_metrics(hw, spec, "qwen3")
        self.assertGreater(metrics.estimated_flops, 0.0)
        self.assertGreater(metrics.estimated_tokens_per_sec, 0.0)

    def test_runtime_codegen(self):
        hw, options, spec = load_compilation_context(
            "examples/llm_stage4/hardware.yaml",
            "examples/llm_stage4/deepseek_decode.yaml",
        )
        plan = build_runtime_plan(hw, options, spec)
        code = generate_runtime_launch_code(plan)
        self.assertIn("ai_launch_kernel", code)
        self.assertIn("experts-done", code)

    def test_compile_and_apply(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        with tempfile.TemporaryDirectory() as tmpdir:
            compile_out = Path(tmpdir) / "compile"
            backend_out = Path(tmpdir) / "backend"
            subprocess.run(
                [
                    sys.executable,
                    "compile.py",
                    "--hardware",
                    "examples/llm_stage4/hardware.yaml",
                    "--compile-spec",
                    "examples/llm_stage4/llama_prefill.yaml",
                    "--payload-mlir",
                    "examples/llm_stage4/payload.mlir",
                    "--out-dir",
                    str(compile_out),
                ],
                cwd=".",
                env=env,
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    "apply_pipeline.py",
                    "--manifest",
                    str(compile_out / "compile_manifest.json"),
                    "--out-dir",
                    str(backend_out),
                ],
                cwd=".",
                env=env,
                check=True,
            )
            self.assertTrue((backend_out / "optimized.mlir").exists())
            self.assertTrue((backend_out / "runtime_launch.c").exists())
            manifest = json.loads((compile_out / "compile_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["model_profile"]["family"], "llama")

if __name__ == "__main__":
    unittest.main()
