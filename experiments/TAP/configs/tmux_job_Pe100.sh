#!/usr/bin/env bash

SESSION="tap-train-pe100"

CMD="python -m eesi.systems.tap.run train \
  		--config experiments/TAP/configs/tap_N20_Pe100.yaml \
  		--run runs/tap_Pe100 \
    	&& \
     python -m eesi.systems.tap.run sample --run runs/tap_Pe100 \
        	--avg --n 10000 --n-steps 50 --chunk 1000 --integrator heun --seed 68500 \
    	&& \
    python -m eesi.systems.tap.run entropy --run runs/tap_Pe100 --avg --seed 123800\
        --batch 256 --batches 1000 --methods dot,zdot,div --flow-n 5000 --flow-steps 50 --flow-seed 91400"

tmux new-session -d -s "$SESSION" "$CMD"
echo "started '$SESSION': $CMD"




