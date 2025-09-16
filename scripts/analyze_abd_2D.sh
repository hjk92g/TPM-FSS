#!/bin/bash

SEED=2021
FOLDS=4 # number of folds (cross-validation splits)
RUNS=3  # number of runs (repetitions)
DATA=/storage/data/CHAOST2

# Run.
for fd in $(seq 0 ${FOLDS})
do
  for r_n in $(seq 1 ${RUNS})
  do
    PRETRAINED=results_abd/train_adnet_alpha_20/fold${fd}_${r_n}/ #ADNet (with T loss)
    python3 main_analyze.py \
    --data_root ${DATA} \
    --pretrained_root ${PRETRAINED} \
    --dataset CHAOST2 \
    --fold $fd \
    --run $r_n \
    --seed ${SEED} \
    --n_sv 5000 \
    --adnet \
    --t_loss_scaler 1.0 \
    --fix_alpha 20.0

    PRETRAINED2=results_abd/train_adnet_alpha_20_no_t_loss/fold${fd}_${r_n}/ #TPM (without T loss)
    python3 main_analyze.py \
    --data_root ${DATA} \
    --pretrained_root ${PRETRAINED2} \
    --dataset CHAOST2 \
    --fold $fd \
    --run $r_n \
    --seed ${SEED} \
    --n_sv 5000 \
    --adnet \
    --t_loss_scaler 0.0 \
    --fix_alpha 20.0
  done
done



