import argparse

from namo.boosted_data_collection.run import _build_boosted_config


def test_build_boosted_config_alias_and_legacy_preservation():
    ns = argparse.Namespace(
        output_dir=None,
        start_idx=None,
        end_idx=None,
        workers=None,
        episodes_per_env=None,
        xml_dir=None,
        config_file=None,
        manifest=None,
        seed=None,
        verbose=None,
        run_name=None,
        unique_run_dir=None,
        output_compression=None,
        boosted_output_compression=None,
        boosted_max_horizon=None,
        region_max_chain_depth=3,
        boosted_ignore_xml_goal=None,
        boosted_cell_filter=None,
        boosted_same_object_only=None,
        boosted_use_cpp_grid_fastpath=None,
    )

    yaml_cfg = {
        "output_dir": "/tmp/x",
        "start_idx": 0,
        "end_idx": 1,
        "xml_dir": "/tmp/xml",
        "config_file": "config.yaml",
        "region_allow_collisions": True,
    }

    cfg = _build_boosted_config(ns, yaml_cfg, ["--unknown-flag", "123"])

    assert cfg["boosted_max_horizon"] == 3
    assert cfg["boosted_output_compression"] == "gzip"
    assert cfg["boosted_ignore_xml_goal"] is True
    assert cfg["boosted_cell_filter"] == "newly_reachable"
    assert cfg["legacy_unknown_yaml_keys"]["region_allow_collisions"] is True
    assert cfg["unknown_flag"] == "123"
    assert cfg["unknown_cli_tokens"] == ["--unknown-flag", "123"]
