#!/usr/bin/env bash

SESSION="tap-pe1"

CMD="python -m eesi.systems.tap.run train \
  		--config experiments/TAP/configs/tap_Pe1.yaml \
  		--run runs/tap_Pe1 \
    	&& \
     python -m eesi.systems.tap.run sample --run runs/tap_Pe1 \
        	--avg --n 10000 --n-steps 50 --chunk 1000 --integrator heun --seed 6851 \
    	&& \
    python -m eesi.systems.tap.run entropy --run runs/tap_Pe1 --avg --seed 789 \
        --batch 256 --batches 1000 --methods dot,zdot,div --flow-n 5000 --flow-steps 50 --flow-seed 5481"

tmux new-session -d -s "$SESSION" "$CMD"
echo "started '$SESSION': $CMD"




