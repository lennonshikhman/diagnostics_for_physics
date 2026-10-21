"""Audit an analytic callable propagator at multiple horizons."""

import torch

from diagnostic_suite import CallablePropagator, DiagnosticConfig, DiagnosticPanel

torch.manual_seed(0)
initial = torch.randn(4, 1, 16, 16)
direction = torch.full_like(initial, 0.05)
references = {
    1.0: initial + direction,
    2.0: initial + 2.0 * direction,
}

propagator = CallablePropagator(
    lambda state, horizon: state + horizon * direction,
    name="analytic_translation",
)
panel = DiagnosticPanel(
    DiagnosticConfig(
        dim=2,
        grid_spacing=(1.0, 1.0),
        metadata={"example": "analytic_translation"},
    )
)
print(panel.evaluate_propagator(initial, references, propagator).as_report())
