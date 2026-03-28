import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from lib.config_parser import load_compilation_context
from lib.llm.kernels.registry import choose_kernel_variant
from lib.llm.passes.pipeline import build_llm_pipeline
from lib.llm.profiler.report import build_perf_report
from lib.llm.runtime.plan import build_runtime_plan


class Stage3FrameworkTests(unittest.TestCase):
    def test_kernel_selection_decode(self):
        _, _, spec = load_compilation_context(
            "examples/llm_stage3/hardware.yaml",
            "examples/llm_stage3/decode_attention.yaml",
        )
        kernel = choose_kernel_variant(spec.match_op, spec.mode, spec.dtype, spec.kv_cache, spec.tags)
        self.assertEqual(kernel.name, "decode-attention")

    def test_runtime_plan_allocates_kv_cache(self):
        hw, options, spec = load_compilation_context(
            "examples/llm_stage3/hardware.yaml",
            "examples/llm_stage3/decode_attention.yaml",
        )
        plan = build_runtime_plan(hw, options, spec)
        spaces = [item.memory_space for item in plan.buffers]
        self.assertIn("kv-cache", spaces)
        self.assertEqual(plan.launches[0].stream_id, 1)

    def test_pipeline_contains_kv_cache_stage(self):
        hw, options, spec = load_compilation_context(
            "examples/llm_stage3/hardware.yaml",
            "examples/llm_stage3/decode_attention.yaml",
        )
        kernel = choose_kernel_variant(spec.match_op, spec.mode, spec.dtype, spec.kv_cache, spec.tags)
        pipeline = build_llm_pipeline(hw, options, spec, kernel)
        self.assertIn("llm-materialize-kv-cache", pipeline["kernel"])
        self.assertIn("llm-allocate-kv-cache", pipeline["runtime"])

    def test_profile_report(self):
        report = build_perf_report("examples/llm_stage3/sample_gem5.stats", clock_ghz=1.0, tokens=2)
        self.assertGreater(report.achieved_bandwidth_gbps, 0.0)
        self.assertGreater(report.token_latency_us, 0.0)

    def test_compile_entrypoint_generates_full_bundle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "out"
            subprocess.run(
                [
                    sys.executable,
                    "compile.py",
                    "--hardware",
                    "examples/llm_stage3/hardware.yaml",
                    "--compile-spec",
                    "examples/llm_stage3/decode_attention.yaml",
                    "--payload-mlir",
                    "examples/llm_stage3/payload.mlir",
                    "--out-dir",
                    str(out_dir),
                ],
                check=True,
            )
            manifest = json.loads((out_dir / "compile_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["selected_kernel"]["name"], "decode-attention")
            self.assertTrue((out_dir / "pass_pipeline.json").exists())
            self.assertTrue((out_dir / "runtime_plan.json").exists())
            self.assertTrue((out_dir / "compilation_summary.md").exists())

    def test_profile_cli(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "perf"
            subprocess.run(
                [
                    sys.executable,
                    "tools/profile_gem5.py",
                    "examples/llm_stage3/sample_gem5.stats",
                    "--clock-ghz",
                    "1.2",
                    "--tokens",
                    "4",
                    "--out-dir",
                    str(out_dir),
                ],
                check=True,
            )
            self.assertTrue((out_dir / "perf_report.json").exists())
            self.assertTrue((out_dir / "perf_report.md").exists())


if __name__ == "__main__":
    unittest.main()
