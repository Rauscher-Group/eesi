#!/usr/bin/env bash

SESSION="tap-sample-pe0"

CMD="python -m eesi.systems.tap.run sample --run runs/tap_Pe0_train2 \
	--avg --n 10000 --n-steps 50 --chunk 1000 --integrator heun --seed 12"

tmux new-session -d -s "$SESSION" "$CMD"
echo "started '$SESSION': $CMD"
