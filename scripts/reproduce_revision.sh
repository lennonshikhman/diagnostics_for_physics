#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
revision_out_root=${REVISION_OUT_ROOT:-outputs_revision_reproduced}
revision_device=${REVISION_DEVICE:-cuda}
case "${1:-analytic}" in
  analytic)
    diagnostic-suite --self-check
    diagnostic-suite-referee --outdir "$revision_out_root/referee"
    diagnostic-suite-sensitivity --outdir "$revision_out_root/sensitivity"
    python scripts/verify_balance_and_constraints.py --outdir "$revision_out_root/balance_constraints"
    ;;
  controlled)
    env -u LD_LIBRARY_PATH diagnostic-suite-controlled \
      --outdir "$revision_out_root/controlled" --device "$revision_device" \
      --problems heat1d advection1d burgers1d --seeds 0 1 2 3 4 \
      --dataset-seed 20261021 --grid 64 --n-train 512 --n-validation 128 --n-test 256 \
      --width 32 --modes 12 --layers 4 --batch-size 64 \
      --warmup-steps 1500 --continuation-steps 1000 --validation-every 50 \
      --composition-weight 10 --physics-weight 0.05 --resume
    python scripts/analyze_controlled_revision.py --root "$revision_out_root/controlled"
    python scripts/verify_matched_endpoint_clocks.py --root "$revision_out_root/controlled"
    ;;
  external)
    revision_raw_root=${REVISION_RAW_ROOT:?Set REVISION_RAW_ROOT to separately acquired data/raw}
    env -u LD_LIBRARY_PATH diagnostic-suite \
      --data-root "$revision_raw_root" --file-selection configs/revision_file_selection.json \
      --outdir "$revision_out_root/external" --device "$revision_device" \
      --tasks adv2d shallow2d active2d cns3d mhd3d --models fno deeponet unet resnet --seeds 0 1 2 \
      --spatial-limit-2d 32 --spatial-limit-3d 16 --width 24 --modes 8 \
      --max-train-pairs 1024 --max-val-pairs 256 --max-test-windows 128 \
      --batch-size 16 --eval-batch-size 16 --epochs-well 30 --epochs-underfit 2 \
      --underfit-width-mult 1 --eval-horizons 1 2 4 --cache-data --hash-data --resume
    python scripts/analyze_external_revision.py --root "$revision_out_root/external"
    ;;
  supplementary)
    revision_saved_controlled=${REVISION_SAVED_CONTROLLED:?Set REVISION_SAVED_CONTROLLED to the existing controlled checkpoint directory}
    revision_saved_external=${REVISION_SAVED_EXTERNAL:?Set REVISION_SAVED_EXTERNAL to the existing external checkpoint directory}
    revision_raw_root=${REVISION_RAW_ROOT:?Set REVISION_RAW_ROOT to separately acquired data/raw}
    python scripts/analyze_rollout_revision.py --root "$revision_saved_controlled" --device "$revision_device"
    python scripts/audit_external_physics_revision.py --root "$revision_saved_external" \
      --data-root "$revision_raw_root" --device "$revision_device"
    ;;
  overhead)
    if [[ "$revision_device" == cpu ]]; then
      python scripts/measure_revision_overhead.py --outdir "$revision_out_root/overhead" --devices cpu
    else
      env -u LD_LIBRARY_PATH python scripts/measure_revision_overhead.py \
        --outdir "$revision_out_root/overhead" --devices cpu "$revision_device"
    fi
    ;;
  *)
    echo "Usage: bash scripts/reproduce_revision.sh {analytic|controlled|external|supplementary|overhead}" >&2
    exit 2
    ;;
esac
