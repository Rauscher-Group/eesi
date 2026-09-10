#!/usr/bin/env bash

SESSION="tap-train-pe30"

#CMD="python -m eesi.systems.tap.run train \
#  		--config experiments/TAP/configs/tap_N20_Pe30.yaml \
#  		--run runs/tap_Pe30 \
#    	&& \
#     python -m eesi.systems.tap.run sample --run runs/tap_Pe30 \
#        	--avg --n 10000 --n-steps 50 --chunk 1000 --integrator heun --seed 68530 \
#    	&& \
#    python -m eesi.systems.tap.run entropy --run runs/tap_Pe30 --avg --seed 14730 \
#        --batch 256 --batches 1000 --methods dot,zdot,div --flow-n 5000 --flow-steps 50 --flow-seed 47930 "

CMD="python -m eesi.systems.tap.run entropy --run runs/tap_Pe30 --avg --seed 14730 \
        --batch 256 --batches 1000 --methods dot,zdot --no-flow"

tmux new-session -d -s "$SESSION" "$CMD"
echo "started '$SESSION': $CMD"




