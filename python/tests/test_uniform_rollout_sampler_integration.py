"""End-to-end smoke test for UniformRolloutSampler on a tiny manifest.

Skipped if the cluster paths aren't available locally (env var SKIP_NAMO_INTEGRATION=1).
"""

import os
import pickle
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "scripts" / "manifests" / "car_envs_100k.txt"
RUN_INTEGRATION = os.environ.get("SKIP_NAMO_INTEGRATION") != "1" and MANIFEST.exists()


@pytest.mark.skipif(not RUN_INTEGRATION, reason="requires car-env manifest")
def test_sampler_smoke_runs_5_envs_and_produces_attempt_results():
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            "python", str(REPO_ROOT / "python/namo/data_collection/modular_parallel_collection.py"),
            "--algorithm", "uniform_rollout_sampler",
            "--manifest", str(MANIFEST),
            "--start-idx", "0",
            "--end-idx", "5",
            "--workers", "1",
            "--output-dir", tmp,
            "--config-file", "config/namo_config_car.yaml",
            "--primitive-prefix", "car_",
            "--sampler-max-chain-depth", "1",
            "--sampler-region-goal-samples", "5",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
        assert result.returncode == 0, f"collection failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

        # At least one pkl was written.
        pkls = list(Path(tmp).rglob("*.pkl"))
        assert pkls, "no pkl files written"

        # First pkl contains episode_results, each with algorithm_stats and a primitive_trial_log.
        with open(pkls[0], "rb") as f:
            data = pickle.load(f)
        episodes = data.get("episode_results", [])
        assert episodes, "no episodes in first pkl"

        ep = episodes[0]
        stats = ep.get("algorithm_stats") or {}
        assert "primitive_trial_log" in stats, "primitive_trial_log missing — batch_collection_classifier needs it"
        assert isinstance(stats["primitive_trial_log"], list)
        # Validate one trial entry's shape matches existing F-char schema.
        if stats["primitive_trial_log"]:
            entry = stats["primitive_trial_log"][0]
            for key in ("edge_idx", "depth", "success", "wall_collision",
                        "movable_collisions", "stuck", "collision", "reachable_after"):
                assert key in entry, f"trial_log entry missing {key}"
