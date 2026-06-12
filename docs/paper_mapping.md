# Mapping from manuscript diagnostics to package components

| Manuscript quantity | Package function / routine |
|---|---|
| Relative L2 error | `diagnostic_suite.metrics.rel_l2` |
| Semigroup consistency | `DiagnosticPanel.semigroup_error`; `evaluate_diagnostics` |
| Finite-difference generator discrepancy | `DiagnosticPanel.generator_discrepancy`; `evaluate_diagnostics` |
| Energy-law violation | `DiagnosticPanel.evaluate_pair`; `evaluate_diagnostics` |
| Energy-decay error | `DiagnosticPanel.evaluate_pair`; `evaluate_diagnostics` |
| Integral balance | `diagnostic_suite.metrics.integral_balance_error` |
| Constraint/admissibility | `diagnostic_suite.metrics.constraint_error` |
| Reference-normalized generator ratio | `add_reference_relative_columns` |
| Benchmark experiment loop | `diagnostic-suite` CLI |

The public API is intended for quick post-hoc auditing of user-provided arrays.
The CLI reproduces the manuscript-style benchmark workflow.
