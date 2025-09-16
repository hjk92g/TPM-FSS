#!/bin/bash

SEED=2021
FOLDS=4 # number of folds (cross-validation splits)
RUNS=3  # number of runs (repetitions)
DATA=/storage/data/SABS

# Run.
for fd in $(seq 0 ${FOLDS})
do
  for r_n in $(seq 1 ${RUNS})
  do
    PRETRAINED=results_sabs/train_adnet_alpha_20/fold${fd}_${r_n}/ #ADNet (with T loss)
    SAVE_FOLDER=results_sabs/test_adnet_alpha_20/fold${fd}_${r_n}
    mkdir -p ${SAVE_FOLDER}
    CUDA_LAUNCH_BLOCKING=1 python3 main_inference.py \
    --data_root ${DATA} \
    --pretrained_root ${PRETRAINED} \
    --save_root ${SAVE_FOLDER} \
    --dataset SABS \
    --n_sv 2000 \
    --train_name train_adnet_alpha_20 \
    --fold ${fd} \
    --run ${r_n} \
    --seed ${SEED} \
    --adnet \
    --t_loss_scaler 1.0 \
    --fix_alpha 20.0

    PRETRAINED2=results_sabs/train_adnet_alpha_20_no_t_loss/fold${fd}_${r_n}/ #TPM (ADNet without T loss)
    SAVE_FOLDER2=results_sabs/test_adnet_alpha_20_no_t_loss/fold${fd}_${r_n}
    mkdir -p ${SAVE_FOLDER2}
    CUDA_LAUNCH_BLOCKING=1 python3 main_inference.py \
    --data_root ${DATA} \
    --pretrained_root ${PRETRAINED2} \
    --save_root ${SAVE_FOLDER2} \
    --dataset SABS \
    --n_sv 2000 \
    --train_name train_adnet_alpha_20_no_t_loss \
    --fold ${fd} \
    --run ${r_n} \
    --seed ${SEED} \
    --adnet \
    --t_loss_scaler 0.0 \
    --fix_alpha 20.0
  done
done

