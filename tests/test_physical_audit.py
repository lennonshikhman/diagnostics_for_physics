import pytest
import torch

from diagnostic_suite.physical_audit import cns_energy_components, physical_samples
from diagnostic_suite.tasks import build_task_specs


def test_cns_energy_fractions_match_known_density_pressure_and_velocity(tmp_path):
    task = build_task_specs(tmp_path, 32, 16)["cns3d"]
    state = torch.ones(2, 5, 8, 8, 8, dtype=torch.float64)
    state[:, 0] = 2
    state[:, 1] = 2
    parts = cns_energy_components(state, task)
    assert torch.allclose(parts["kinetic"], torch.full((2,), 3.0, dtype=torch.float64))
    assert torch.allclose(parts["internal"], torch.full((2,), 5.0, dtype=torch.float64))
    assert torch.allclose(
        parts["internal"] / (parts["internal"] + parts["kinetic"]), torch.full((2,), 5 / 8, dtype=torch.float64)
    )
    with pytest.raises(ValueError, match="CNS"):
        cns_energy_components(state, build_task_specs(tmp_path, 32, 16)["mhd3d"])


def test_octant_balance_retains_reference_baseline_and_model_mass_defect(tmp_path):
    task = build_task_specs(tmp_path, 32, 16)["cns3d"]
    task.grid_spacing = (1 / 8,) * 3
    initial = torch.ones(2, 5, 8, 8, 8, dtype=torch.float64)
    initial[:, 0] = 2
    prediction = initial.clone()
    prediction[:, 0] = 3
    values = physical_samples(initial, initial, prediction, task, delta_t=0.1, control_volume=(slice(0, 4),) * 3)
    assert torch.max(values["reference_subvolume_balance"]) == 0
    assert values["subvolume_balance"].mean().item() == pytest.approx(0.2, abs=1e-8)
    assert values["full_balance"].mean().item() == pytest.approx(0.2, abs=1e-8)
    assert values["functional"].mean() > 0
