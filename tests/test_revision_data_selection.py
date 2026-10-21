import json

import pytest
import torch

from diagnostic_suite.data import CachedDataset
from diagnostic_suite.runner import _model_seed
from diagnostic_suite.tasks import apply_file_selection, build_task_specs


def test_explicit_file_selection_preserves_physics_and_rejects_paths_outside_root(tmp_path):
    tasks = build_task_specs(tmp_path, 64, 32)
    path = tmp_path / "files.json"
    selection = {"active2d": {key: [f"active/{key}/fixed.hdf5"] for key in ("train", "val", "test")}}
    path.write_text(json.dumps(selection))
    selected = apply_file_selection(tasks, path)
    assert selected["active2d"].globs == selection["active2d"]
    assert selected["active2d"].energy_law == tasks["active2d"].energy_law
    assert tasks["active2d"].globs != selected["active2d"].globs
    selection["active2d"]["test"] = ["../different-study.hdf5"]
    path.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="relative"):
        apply_file_selection(tasks, path)


def test_cache_preserves_dataset_values_and_budget_variants_share_initialization():
    original = [{"x": torch.arange(4) + index, "tau": torch.tensor(index / 4)} for index in range(3)]
    cached = CachedDataset(original)
    for index in range(len(original)):
        assert torch.equal(cached[index]["x"], original[index]["x"])
        assert torch.equal(cached[index]["tau"], original[index]["tau"])
    assert _model_seed(3, "fno", "well") == _model_seed(3, "fno", "underfit")
