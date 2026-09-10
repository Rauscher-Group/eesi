#!/usr/bin/env bash

SESSION="tap-train-pe0p03"

CMD="python -m eesi.systems.tap.run train \
  		--config experiments/TAP/configs/tap_N20_Pe0p03.yaml \
  		--run runs/tap_Pe0p03 \
    	&& \
     python -m eesi.systems.tap.run sample --run runs/tap_Pe0p03 \
        	--avg --n 10000 --n-steps 50 --chunk 1000 --integrator heun --seed 68003 \
    	&& \
    python -m eesi.systems.tap.run entropy --run runs/tap_Pe0p03 --avg --seed 59003 \
        --batch 256 --batches 1000 --methods dot,zdot --no-flow"

tmux new-session -d -s "$SESSION" "$CMD"
echo "started '$SESSION': $CMD"




