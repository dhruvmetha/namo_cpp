"""Regression test: new sampler reproduces existing F-char per-primitive labels.

Runs the new sampler on one env that was also collected by the old pipeline,
verifies the depth-0 (edge_idx, depth, success) tuples match.

Skipped if the reference F-char pkls aren't accessible. This test is gated
behind explicit invocation because the subprocess can take minutes.
"""

import os
import pickle
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = Path("/common/users/dm1487/namo_data/f_characterization/"
                     "1_push_exhaustive_full/modular_data_rlab7")
RUN_REGRESSION = (
    REFERENCE_DIR.exists()
    and os.environ.get("RUN_NAMO_FCHAR_REGRESSION") == "1"
)


def _extract_grid(trial_log):
    """{(edge, depth) -> success} from a trial_log."""
    return {(entry["edge_idx"], entry["depth"]): bool(entry["success"])
            for entry in trial_log}


@pytest.mark.skipif(
    not RUN_REGRESSION,
    reason="set RUN_NAMO_FCHAR_REGRESSION=1 and ensure reference dir is available"
)
def test_sampler_reproduces_existing_fchar_labels_on_one_env():
    ref_pkls = sorted(REFERENCE_DIR.glob("*.pkl"))
    assert ref_pkls, "no reference pkls found"

    # Find the first reference episode that has a primitive_trial_log.
    ref_ep = None
    for pkl in ref_pkls[:5]:                           # check first few pkls
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        for ep in data.get("episode_results", []):
            stats = ep.get("algorithm_stats") or {}
            if stats.get("primitive_trial_log"):
                ref_ep = ep
                break
        if ref_ep:
            break
    assert ref_ep, "no reference episode with primitive_trial_log found"

    xml_file = ref_ep["xml_file"]
    ref_object = ref_ep["algorithm_stats"]["chosen_object_id"]
    ref_neighbor = ref_ep["algorithm_stats"]["neighbour_region_label"]
    ref_grid = _extract_grid(ref_ep["algorithm_stats"]["primitive_trial_log"])

    # Build a one-line manifest for that env.
    with tempfile.TemporaryDirectory() as tmp:
        manifest = Path(tmp) / "single_env.txt"
        manifest.write_text(xml_file + "\n")

        cmd = [
            "python", str(REPO_ROOT / "python/namo/data_collection/modular_parallel_collection.py"),
            "--algorithm", "uniform_rollout_sampler",
            "--manifest", str(manifest),
            "--start-idx", "0",
            "--end-idx", "1",
            "--workers", "1",
            "--output-dir", tmp,
            "--config-file", "config/namo_config_complete_skill15_car_1x.yaml",
            "--primitive-prefix", "1x_car_d5_",
            "--sampler-max-chain-depth", "1",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
        assert result.returncode == 0, f"sampler failed:\n{result.stdout}\n{result.stderr}"

        new_pkls = list(Path(tmp).rglob("*.pkl"))
        assert new_pkls
        with open(new_pkls[0], "rb") as f:
            new_data = pickle.load(f)

        # Find the matching (object, neighbor) attempt in the new pkl.
        match = None
        for ep in new_data["episode_results"]:
            stats = ep.get("algorithm_stats") or {}
            if (stats.get("chosen_object_id") == ref_object
                    and stats.get("neighbour_region_label") == ref_neighbor):
                match = ep
                break
        assert match, (
            f"no matching attempt for (object={ref_object}, neighbor={ref_neighbor}) "
            f"in new pkl"
        )

        new_grid = _extract_grid(match["algorithm_stats"]["primitive_trial_log"])

        # Compare overlap.
        common = set(ref_grid) & set(new_grid)
        assert common, "no overlapping (edge, depth) pairs"
        mismatches = [(e, d) for (e, d) in common if ref_grid[(e, d)] != new_grid[(e, d)]]
        mismatch_rate = len(mismatches) / len(common)
        # Allow some tolerance — small numerical sim differences are acceptable.
        assert mismatch_rate < 0.05, (
            f"{len(mismatches)}/{len(common)} ({mismatch_rate*100:.1f}%) primitives "
            f"disagree between new sampler and reference F-char. Sample mismatches: "
            f"{mismatches[:5]}"
        )
