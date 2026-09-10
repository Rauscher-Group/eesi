#!/usr/bin/env bash

SESSION="tap-train2"

CMD="python -m eesi.systems.tap.run train \
  --config experiments/TAP/configs/tap_Pe0_train2.yaml \
  --run runs/tap_Pe0_train2"

tmux new-session -d -s "$SESSION" "$CMD"
echo "started '$SESSION': $CMD"
