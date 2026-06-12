"""Diagnostic Suite for learned PDE time propagators."""

from .constants import EPS
from .tasks import TaskSpec, build_all_task_specs, build_task_specs

__version__ = "0.1.0"

_LAZY_EXPORTS = {
    "DiagnosticResult": ("diagnostic_suite.api", "DiagnosticResult"),
    "DiagnosticPanel": ("diagnostic_suite.api", "DiagnosticPanel"),
    "DiagnosticConfig": ("diagnostic_suite.api", "DiagnosticConfig"),
    "Normalizer": ("diagnostic_suite.normalization", "Normalizer"),
    "estimate_normalizer": ("diagnostic_suite.normalization", "estimate_normalizer"),
    "METRIC_NAMES": ("diagnostic_suite.metrics", "METRIC_NAMES"),
    "rel_l2": ("diagnostic_suite.metrics", "rel_l2"),
    "energy": ("diagnostic_suite.metrics", "energy"),
    "constraint_error": ("diagnostic_suite.metrics", "constraint_error"),
    "integral_balance_error": ("diagnostic_suite.metrics", "integral_balance_error"),
    "summarize_metric_rows": ("diagnostic_suite.metrics", "summarize_metric_rows"),
    "add_reference_relative_columns": ("diagnostic_suite.metrics", "add_reference_relative_columns"),
    "DiagnosticConfig": ("diagnostic_suite.api", "DiagnosticConfig"),
    "DiagnosticPanel": ("diagnostic_suite.api", "DiagnosticPanel"),
    "DiagnosticResult": ("diagnostic_suite.api", "DiagnosticResult"),
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
    "DiagnosticResult",
    "DiagnosticPanel",
    "DiagnosticConfig",
    "EPS",
    "TaskSpec",
    "build_task_specs",
    "build_all_task_specs",
    "Normalizer",
    "estimate_normalizer",
    "METRIC_NAMES",
    "rel_l2",
    "energy",
    "constraint_error",
    "integral_balance_error",
    "summarize_metric_rows",
    "add_reference_relative_columns",
    "DiagnosticConfig",
    "DiagnosticPanel",
    "DiagnosticResult",
]
