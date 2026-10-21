from __future__ import annotations

import pytest
import torch

from diagnostic_suite.models import OversmoothWrapper, avg_pool_nd, build_model


@pytest.mark.parametrize("model_name", ["fno", "deeponet", "unet", "resnet"])
def test_every_public_baseline_model_has_a_valid_2d_forward_pass(model_name: str) -> None:
    model = build_model(
        model_name,
        dim=2,
        channels=2,
        spatial_shape=(16, 16),
        width=8,
        modes=4,
    )
    state = torch.randn(2, 2, 16, 16)
    horizon = torch.tensor([0.25, 0.75])
    output = model(state, horizon)
    assert output.shape == state.shape
    assert torch.isfinite(output).all()


def test_3d_model_forward_pass_preserves_shape() -> None:
    model = build_model(
        "resnet",
        dim=3,
        channels=3,
        spatial_shape=(8, 8, 8),
        width=4,
        modes=2,
    )
    state = torch.randn(1, 3, 8, 8, 8)
    output = model(state, torch.tensor([0.5]))
    assert output.shape == state.shape


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_periodic_smoothing_preserves_mass_and_matches_fourier_gain(dim: int) -> None:
    grid, mode = 16, 3
    coordinate = torch.arange(grid, dtype=torch.float64) / grid
    wave = torch.cos(2 * torch.pi * mode * coordinate)
    shape = (1, 1, grid, *([1] * (dim - 1)))
    state = (2 + wave.reshape(shape)).expand(2, 3, *([grid] * dim)).contiguous()
    filtered = avg_pool_nd(avg_pool_nd(state, dim, padding_mode="circular"), dim, padding_mode="circular")
    gain = ((1 + 2 * torch.cos(torch.tensor(2 * torch.pi * mode / grid))) / 3).double().square()
    assert filtered.mean().item() == pytest.approx(state.mean().item(), abs=1e-13)
    assert torch.max(torch.abs(filtered - (2 + gain * (state - 2)))) < 1e-7


def test_smoothing_boundary_policy_is_explicit_and_rejects_invalid_parameters() -> None:
    class Identity(torch.nn.Module):
        def forward(self, state, horizon):
            return state

    state = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    periodic = OversmoothWrapper(Identity(), 1, passes=1, periodic=True)(state, torch.tensor([1.0]))
    replicated = OversmoothWrapper(Identity(), 1, passes=1, periodic=False)(state, torch.tensor([1.0]))
    assert periodic[0, 0, -1].item() == pytest.approx(1 / 3)
    assert replicated[0, 0, -1].item() == 0
    with pytest.raises(ValueError, match="nonnegative"):
        OversmoothWrapper(Identity(), 1, passes=-1, periodic=True)
    with pytest.raises(TypeError, match="periodic"):
        OversmoothWrapper(Identity(), 1)
    with pytest.raises(ValueError, match="explicit boolean"):
        OversmoothWrapper(Identity(), 1, periodic=None)


def test_cns_like_periodic_smoothing_changes_predictions_and_preserves_channel_sums():
    class Identity(torch.nn.Module):
        def forward(self, state, horizon):
            return state

    coordinate = torch.arange(8, dtype=torch.float64) / 8
    wave = torch.cos(2 * torch.pi * 3 * coordinate).reshape(1, 1, 8, 1, 1)
    state = (2 + wave).expand(2, 5, 8, 8, 8).contiguous()
    result = OversmoothWrapper(Identity(), 3, periodic=True)(state, torch.ones(2))
    assert not torch.allclose(result, state)
    assert torch.max(torch.abs(result.sum((2, 3, 4)) - state.sum((2, 3, 4)))) < 1e-11
