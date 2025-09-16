#!/bin/bash

# Specs.
SEED=2021
FOLDS=4 # number of folds (cross-validation splits)
RUNS=3  # number of runs (repetitions)
DATA=/storage/data/SABS


# Run.
pwd
for fd in $(seq 0 ${FOLDS})
do
  for r_n in $(seq 1 ${RUNS})
  do
    SAVE_FOLDER=results_abd/train_adnet_alpha_20/fold${fd}_${r_n}/ #ADNet (with T loss)
    mkdir -p ${SAVE_FOLDER}
    python3 main_train.py \
    --data_root ${DATA} \
    --save_root ${SAVE_FOLDER} \
    --dataset SABS \
    --n_sv 2000 \
    --fold ${fd} \
    --seed ${SEED} \
    --adnet \
    --t_loss_scaler 1.0 \
    --fix_alpha 20.0

    SAVE_FOLDER2=results_abd/train_adnet_alpha_20_no_t_loss/fold${fd}_${r_n}/ #TPM (ADNet without T loss)
    mkdir -p ${SAVE_FOLDER2}
    python3 main_train.py \
    --data_root ${DATA} \
    --save_root ${SAVE_FOLDER2} \
    --dataset SABS \
    --n_sv 2000 \
    --fold ${fd} \
    --seed ${SEED} \
    --adnet \
    --t_loss_scaler 0.0 \
    --fix_alpha 20.0
  done
done
