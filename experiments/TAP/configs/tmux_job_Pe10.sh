#!/usr/bin/env bash

SESSION="tap-train-pe10"

CMD="python -m eesi.systems.tap.run train \
  		--config experiments/TAP/configs/tap_N20_Pe10.yaml \
  		--run runs/tap_Pe10 \
    	&& \
     python -m eesi.systems.tap.run sample --run runs/tap_Pe10 \
        	--avg --n 10000 --n-steps 50 --chunk 1000 --integrator heun --seed 68510 \
    	&& \
    python -m eesi.systems.tap.run entropy --run runs/tap_Pe10 --avg --seed 14710 \
        --batch 256 --batches 1000 --methods dot,zdot,div --flow-n 5000 --flow-steps 50 --flow-seed 47910 "

tmux new-session -d -s "$SESSION" "$CMD"
echo "started '$SESSION': $CMD"




