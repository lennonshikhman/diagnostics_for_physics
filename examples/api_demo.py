"""Minimal public-API demo using synthetic arrays."""

import numpy as np

from diagnostic_suite import DiagnosticConfig, DiagnosticPanel
from diagnostic_suite.toy import make_periodic_advection

# Synthetic trajectory: (N,T,H,W,C). Convert one trajectory to channels-first.
traj = make_periodic_advection(n_traj=1, n_time=3, size=32, channels=1, seed=4)[0]
u0 = np.moveaxis(traj[0], -1, 0)
uref = np.moveaxis(traj[2], -1, 0)

# A deliberately smoothed-ish/biased prediction, just for demonstration.
pred = 0.95 * uref

panel = DiagnosticPanel(DiagnosticConfig(dim=2, energy_type="generic_l2", balance_channel=0))
print(panel.evaluate_pair(u0, uref, pred).as_dict())
print(panel.generator_discrepancy(u0, np.moveaxis(traj[1], -1, 0), pred).as_dict())
