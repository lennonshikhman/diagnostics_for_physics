from __future__ import annotations

from pathlib import Path

from diagnostic_suite.data import PairDataset, TrajectoryCollection, WindowDataset
from diagnostic_suite.evaluate import evaluate_reference
from diagnostic_suite.tasks import build_task_specs
from diagnostic_suite.toy import make_toy_data
import torch
from torch.utils.data import DataLoader


def test_toy_data_roundtrip(tmp_path: Path) -> None:
    make_toy_data(tmp_path, n_train=4, n_val=2, n_test=2, steps=5, grid=12, seed=1)
    tasks = build_task_specs(tmp_path, 64, 32)
    task = tasks["toy2d"]
    coll = TrajectoryCollection(tmp_path, task, "train", spatial_limit=16)
    assert len(coll.readers) == 1
    pair_ds = PairDataset(coll, max_pairs=8, max_delta=2, seed=0)
    item = pair_ds[0]
    assert item["x"].shape[0] == 1
    assert item["x"].shape == item["y"].shape


def test_reference_evaluation_on_toy(tmp_path: Path) -> None:
    make_toy_data(tmp_path, n_train=4, n_val=2, n_test=2, steps=5, grid=12, seed=2)
    task = build_task_specs(tmp_path, 64, 32)["toy2d"]
    coll = TrajectoryCollection(tmp_path, task, "test", spatial_limit=16)
    windows = WindowDataset(coll, max_windows=4, horizon=2, seed=0)
    loader = DataLoader(windows, batch_size=2)
    metrics = evaluate_reference(loader, task, max_delta=2, device=torch.device("cpu"))
    assert metrics["rel_l2"] == 0.0
    assert metrics["semigroup"] == 0.0
    assert metrics["generator_n"] > 0
