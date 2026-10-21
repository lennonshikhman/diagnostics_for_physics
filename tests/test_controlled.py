import pytest
import torch

from diagnostic_suite.controlled import PROBLEMS, ControlledFNO


@pytest.mark.parametrize("name", list(PROBLEMS))
def test_independent_exact_families_satisfy_pde_and_resolution_consistency(name):
    problem = PROBLEMS[name]
    coefficients = problem.coefficients(3, 13)
    h, dt = 0.08, 1e-6
    state = problem.reference(coefficients, h, 256)
    derivative = (problem.reference(coefficients, h + dt, 256) - problem.reference(coefficients, h - dt, 256)) / (
        2 * dt
    )
    residual = torch.linalg.vector_norm(derivative - problem.rhs(state)) / torch.linalg.vector_norm(derivative)
    assert residual.item() < 1e-7
    coarse = problem.reference(coefficients, h, 64)
    assert torch.max(torch.abs(state[..., ::4] - coarse)).item() < 1e-13
    assert torch.max(torch.abs(state.mean(-1) - problem.mean)).item() < 1e-12


def test_controlled_fno_is_identity_at_zero_and_accepts_finer_physics_grid():
    model = ControlledFNO(width=4, modes=3, layers=2)
    for grid in (16, 32):
        state = torch.randn(2, 1, grid)
        assert torch.equal(model(state, torch.zeros(2)), state)
        result = model(state, torch.tensor([0.2, 0.8]))
        assert result.shape == state.shape
        assert torch.isfinite(result).all()
