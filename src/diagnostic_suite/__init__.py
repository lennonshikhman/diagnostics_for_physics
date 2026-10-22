"""PDECheck diagnostics for learned PDE time propagators."""

from .constants import EPS
from .tasks import BalanceSpec, ConstraintSpec, TaskSpec, build_all_task_specs, build_task_specs

__version__ = "0.3.3"

_LAZY_EXPORTS = {
    "AutoregressivePropagator": ("diagnostic_suite.propagators", "AutoregressivePropagator"),
    "CallablePropagator": ("diagnostic_suite.propagators", "CallablePropagator"),
    "DiagnosticConfig": ("diagnostic_suite.api", "DiagnosticConfig"),
    "DiagnosticPanel": ("diagnostic_suite.api", "DiagnosticPanel"),
    "DiagnosticResult": ("diagnostic_suite.api", "DiagnosticResult"),
    "SavedPredictionAdapter": ("diagnostic_suite.propagators", "SavedPredictionAdapter"),
    "Normalizer": ("diagnostic_suite.normalization", "Normalizer"),
    "estimate_normalizer": ("diagnostic_suite.normalization", "estimate_normalizer"),
    "METRIC_NAMES": ("diagnostic_suite.metrics", "METRIC_NAMES"),
    "rel_l2": ("diagnostic_suite.metrics", "rel_l2"),
    "semigroup_error": ("diagnostic_suite.metrics", "semigroup_error"),
    "generator_discrepancy": ("diagnostic_suite.metrics", "generator_discrepancy"),
    "energy": ("diagnostic_suite.metrics", "energy"),
    "energy_law_error": ("diagnostic_suite.metrics", "energy_law_error"),
    "energy_reference_error": ("diagnostic_suite.metrics", "energy_reference_error"),
    "constraint_error": ("diagnostic_suite.metrics", "constraint_error"),
    "constraint_errors": ("diagnostic_suite.metrics", "constraint_errors"),
    "integral_balance_error": ("diagnostic_suite.metrics", "integral_balance_error"),
    "global_conserved_quantity_drift": (
        "diagnostic_suite.metrics",
        "global_conserved_quantity_drift",
    ),
    "perturbation_response_error": ("diagnostic_suite.metrics", "perturbation_response_error"),
    "scaling_law_error": ("diagnostic_suite.metrics", "scaling_law_error"),
    "summarize_metric_rows": ("diagnostic_suite.metrics", "summarize_metric_rows"),
    "add_reference_relative_columns": ("diagnostic_suite.metrics", "add_reference_relative_columns"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _LAZY_EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = [
    "AutoregressivePropagator",
    "BalanceSpec",
    "CallablePropagator",
    "ConstraintSpec",
    "DiagnosticConfig",
    "DiagnosticPanel",
    "DiagnosticResult",
    "EPS",
    "SavedPredictionAdapter",
    "TaskSpec",
    "add_reference_relative_columns",
    "build_all_task_specs",
    "build_task_specs",
    "Normalizer",
    "estimate_normalizer",
    "METRIC_NAMES",
    "rel_l2",
    "semigroup_error",
    "generator_discrepancy",
    "energy",
    "energy_law_error",
    "energy_reference_error",
    "constraint_error",
    "constraint_errors",
    "integral_balance_error",
    "global_conserved_quantity_drift",
    "perturbation_response_error",
    "scaling_law_error",
    "summarize_metric_rows",
]
