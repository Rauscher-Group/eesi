#!/usr/bin/env bash

SESSION="tap-train-pe3"

CMD="python -m eesi.systems.tap.run train \
  		--config experiments/TAP/configs/tap_N20_Pe3.yaml \
  		--run runs/tap_Pe3 \
    	&& \
     python -m eesi.systems.tap.run sample --run runs/tap_Pe3 \
        	--avg --n 10000 --n-steps 50 --chunk 1000 --integrator heun --seed 6853 \
    	&& \
    python -m eesi.systems.tap.run entropy --run runs/tap_Pe3 --avg --seed 7893 \
        --batch 256 --batches 1000 --methods dot,zdot --no-flow"

tmux new-session -d -s "$SESSION" "$CMD"
echo "started '$SESSION': $CMD"




