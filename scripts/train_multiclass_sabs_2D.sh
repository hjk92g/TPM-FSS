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
    SAVE_FOLDER=results_multiclass_sabs/train_Proto5_learn_T_D_alpha_20_no_t_loss/fold${fd}_${r_n}/
    mkdir -p ${SAVE_FOLDER}
    python3 main_multiclass_train.py \
    --data_root ${DATA} \
    --save_root ${SAVE_FOLDER} \
    --dataset SABS \
    --n_sv 2000 \
    --max_protos 5 \
    --fold ${fd} \
    --seed ${SEED} \
    --learn_T_D \
    --fix_alpha 20.0 \
    --t_loss_scaler 0.0
  done
done
