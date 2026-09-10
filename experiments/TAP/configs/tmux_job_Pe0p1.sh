#!/usr/bin/env bash

SESSION="tap-train-pe0p1"

CMD="python -m eesi.systems.tap.run train \
  		--config experiments/TAP/configs/tap_N20_Pe0p1.yaml \
  		--run runs/tap_Pe0p1 \
    	&& \
     python -m eesi.systems.tap.run sample --run runs/tap_Pe0p1 \
        	--avg --n 10000 --n-steps 50 --chunk 1000 --integrator heun --seed 68951 \
    	&& \
    python -m eesi.systems.tap.run entropy --run runs/tap_Pe0p1 --avg --seed 598701 \
        --batch 256 --batches 1000 --methods dot,zdot,div --flow-n 5000 --flow-steps 50 --flow-seed 19701"

tmux new-session -d -s "$SESSION" "$CMD"
echo "started '$SESSION': $CMD"




