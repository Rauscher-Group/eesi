#!/usr/bin/env bash
# Launch a training/analysis job detached under tmux so it survives disconnect.
# Edit SESSION and CMD below, then run this script.
#
#   ./scripts/tmux_job.sh                  start the job
#   tmux attach -t $SESSION                watch it (C-b d to detach)
#   tmux send-keys -t $SESSION C-c         stop it: finishes the step, checkpoints, exits
#   tail -f runs/.../log.txt               follow it without attaching

SESSION="tap-train"

CMD="python -m eesi.systems.tap.run train \
  --config experiments/TAP/configs/tap_N20_Pe0.yaml \
  --run runs/tap_Pe0_train"

tmux new-session -d -s "$SESSION" "$CMD"
echo "started '$SESSION': $CMD"
