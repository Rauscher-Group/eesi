#!/usr/bin/env bash

SESSION="tap-entropy-pe0"

CMD="python -m eesi.systems.tap.run entropy --run runs/tap_Pe0_train2 --avg \
	--batch 256 --batches 1000 --methods dot,zdot,div --flow-n 5000 --flow-steps 50"

tmux new-session -d -s "$SESSION" "$CMD"
echo "started '$SESSION': $CMD"
