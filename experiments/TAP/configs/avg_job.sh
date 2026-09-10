#!/usr/bin/env bash
# Sample or estimate entropy from a run's AVERAGED weights (the checkpoint's
# `model_avg` shadow -- see `averaging` in the run's config) under a detached tmux
# session, so it survives disconnect. Uses `run.py`'s `--avg` flag, which swaps
# `run.model` for `run.model_avg` before the job runs, and fails loudly if the run
# has no averaged weights (averaging was off, or the checkpoint predates avg_start).
#
#   ./experiments/TAP/configs/avg_job.sh sample  runs/tap_long
#   ./experiments/TAP/configs/avg_job.sh entropy runs/tap_long
#   ./experiments/TAP/configs/avg_job.sh sample  runs/tap_long --n 20000 --integrator heun
#   ./experiments/TAP/configs/avg_job.sh entropy runs/tap_long --checkpoint step_50000.pt
#
#   tmux attach -t $SESSION                watch it (C-b d to detach)
#   tmux send-keys -t $SESSION C-c         stop it
#   tail -f <run>/log.txt                  follow it without attaching

set -euo pipefail

JOB="${1:?usage: avg_job.sh <sample|entropy> <run_dir> [extra run.py flags...]}"
RUN="${2:?usage: avg_job.sh <sample|entropy> <run_dir> [extra run.py flags...]}"
shift 2

case "$JOB" in
  sample|entropy) ;;
  *) echo "job must be 'sample' or 'entropy', got '$JOB'" >&2; exit 1 ;;
esac

SESSION="tap-avg-${JOB}-$(basename "$RUN")"

CMD="python -m eesi.systems.tap.run $JOB --run $RUN --avg $*"

tmux new-session -d -s "$SESSION" "$CMD"
echo "started '$SESSION': $CMD"
