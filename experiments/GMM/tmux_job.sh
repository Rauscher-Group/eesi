#!/usr/bin/env bash

SESSION="gmm-train"
LOG="experiments/GMM/gmm_train.log"

CMD="python -u experiments/GMM/train_job.py --steps 20000 --batch 1000 --out gmm --lr 1e-5"

tmux new-session -d -s "$SESSION" "$CMD 2>&1 | tee $LOG"
echo "started '$SESSION': $CMD"
