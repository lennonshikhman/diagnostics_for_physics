#!/usr/bin/env bash
set -euo pipefail
export DATA_ROOT=${1:-${DATA_ROOT:?Pass the data root as the first argument or set DATA_ROOT.}}
export OUTDIR=${2:-${OUTDIR:-outputs_diagnostic_suite}}
exec bash "$(dirname -- "$0")/reproduce_paper_tables.sh"
