import json
import os
import numpy as np

from src.sim_system import CLASS_FILES, generate_dataset
from src.data_loader import IQDataset, build_scenario_level_folds


def test_generate_dataset_writes_manifest(tmp_path):
    data_dir = tmp_path / "sim_dataset"
    generate_dataset(
        output_dir=str(data_dir),
        duration_sec=0.02,
        seed=123,
        scenarios_per_class=3,
    )

    manifest_path = data_dir / "dataset_manifest.json"
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text())
    assert manifest["seed"] == 123
    assert manifest["scenarios_per_class"] == 3

    for _, filename in CLASS_FILES.items():
        assert (data_dir / filename).exists()
        class_meta = manifest["classes"][filename]
        assert class_meta["num_scenarios"] == 3
        assert len(class_meta["scenarios"]) == 3


def test_iqdataset_exposes_scenario_ids_and_splits(tmp_path):
    data_dir = tmp_path / "sim_dataset"
    generate_dataset(
        output_dir=str(data_dir),
        duration_sec=0.02,
        seed=321,
        scenarios_per_class=3,
    )

    ds = IQDataset(str(data_dir), precompute_stft=False)
    assert ds.scenario_ids is not None
    assert len(ds.scenario_ids) == len(ds)
    assert len(np.unique(ds.scenario_ids)) == 12

    folds = build_scenario_level_folds(ds.labels, ds.scenario_ids, n_splits=3, seed=42)
    for train_idx, val_idx in folds:
        train_scenarios = set(ds.scenario_ids[train_idx])
        val_scenarios = set(ds.scenario_ids[val_idx])
        assert train_scenarios.isdisjoint(val_scenarios)
