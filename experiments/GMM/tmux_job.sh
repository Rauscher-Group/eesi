#!/usr/bin/env bash
# Launch the GMM training detached under tmux so it survives disconnect.
#
#   ./experiments/GMM/tmux_job.sh          start the job
#   tmux attach -t $SESSION                watch it (C-b d to detach)
#   tail -f experiments/GMM/gmm_train.log  follow it without attaching

SESSION="gmm-train"
LOG="experiments/GMM/gmm_train.log"

CMD="python -u experiments/GMM/train_job.py --steps 20000 --batch 1000 --out gmm2 --init gmm --lr 1e-5"

tmux new-session -d -s "$SESSION" "$CMD 2>&1 | tee $LOG"
echo "started '$SESSION': $CMD"
