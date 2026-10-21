# Examples

Generate and run the toy dataset:

```bash
diagnostic-suite --make-toy --data-root data --toy-grid 24 --toy-steps 6
bash scripts/run_toy_smoke.sh
```

Run the deterministic known-positive/known-negative checks:

```bash
diagnostic-suite --self-check
```

See `api_demo.py` for direct tensor diagnostics and
`callable_propagator_demo.py` for a multi-horizon callable audit.

Run `diagnostic-suite-referee --outdir outputs_referee` for the reproducible
analytic PDE example used in the revised manuscript. The output contains all
horizons, checks, provenance, and a generated LaTeX table.
