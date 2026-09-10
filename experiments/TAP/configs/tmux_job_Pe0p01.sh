#!/usr/bin/env bash
# Launch the Pe=0.01 warm-started training run detached under tmux so it survives
# disconnect. Fill in `init.checkpoint` (or `init.net_b`/`init.net_s`) in
# tap_N20_Pe0p01.yaml before running this.
#
#   ./experiments/TAP/configs/tmux_job_Pe0p01.sh   start the job
#   tmux attach -t $SESSION                        watch it (C-b d to detach)
#   tmux send-keys -t $SESSION C-c                 stop it: finishes the step, checkpoints, exits
#   tail -f runs/.../log.txt                       follow it without attaching

SESSION="tap-train-pe0p01"

#CMD="python -m eesi.systems.tap.run train \
#  		--config experiments/TAP/configs/tap_N20_Pe0p01.yaml \
#  		--run runs/tap_Pe0p01 \
#    	&& \
#"
CMD="python -m eesi.systems.tap.run sample --run runs/tap_Pe0p01 \
        	--avg --n 10000 --n-steps 50 --chunk 1000 --integrator heun --seed 685 \
    	&& \
    python -m eesi.systems.tap.run entropy --run runs/tap_Pe0p01 --avg \
        --batch 256 --batches 1000 --methods dot,zdot,div --flow-n 5000 --flow-steps 50"

tmux new-session -d -s "$SESSION" "$CMD"
echo "started '$SESSION': $CMD"




