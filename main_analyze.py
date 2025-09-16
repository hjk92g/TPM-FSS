#!/usr/bin/env python

#Analyze to estimate (train) ICPs using training data

import argparse
import time
import random

import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from models.fewshot_anom import FewShotSeg # Major functions (forward, negSim, getPrototype, ...)
from dataloading.datasets import TrainDataset as TrainDataset
from utils import * # Minor functions (logger, evaluation metrics: Dice, IOU scores, ...)
import numpy as np
import inspect
import wandb, sys

from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import joblib

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--n_sv', type=int, required=True) # supervoxel parameter
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--run', type=int, required=True)
    parser.add_argument('--pretrained_root', type=str, required=True)

    # Training specs.
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--steps', default=50000, type=int) #Iterations
    parser.add_argument('--n_shot', default=1, type=int)
    parser.add_argument('--n_query', default=1, type=int)
    parser.add_argument('--n_way', default=1, type=int)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--non_succesive', action='store_true') #Allow non succesive slices
    parser.add_argument('--max_iterations', default=1000, type=int)
    parser.add_argument('--lr', default=1e-3, type=float) # Learning rate
    parser.add_argument('--lr_gamma', default=0.95, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=0.0005, type=float)

    parser.add_argument('--ADNet', action='store_true') #Exactly the same as ADNet even the parameterization
    parser.add_argument('--adnet', action='store_true') #Make it the same as adnet but different parameterization (learn T_S and T_D with CE loss)
    parser.add_argument('--t_loss_scaler', default=1.0, type=float) # T loss weight. Set it 0 for TPM
    parser.add_argument('--fix_T_D', default=None, type=float)  # Train to fix T_D value
    parser.add_argument('--chg_T_D', action='store_true')  # Train to fit decision boundary to T_D_hat value
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--bg_wt', default=0.1, type=float) # Background class weight to address class imbalance (Foreground class weight=1)

    parser.add_argument('--fix_sig2', default=None, type=float) # Fix sig2 during training
    parser.add_argument('--fix_alpha', default=None, type=float)  # Fix alpha during training
    parser.add_argument('--chg_alpha', action='store_true')  # Change alpha during training using self.alpha_hat
    parser.add_argument('--chg_alpha2', action='store_true') # Change alpha during training using self.alpha_hat2
    parser.add_argument('--learn_alpha', action='store_true')

    parser.add_argument('--fix_p_F', default=None, type=float)  # Fix p_F during training
    parser.add_argument('--chg_p_F', action='store_true')
    parser.add_argument('--learn_p_F', action='store_true')

    parser.add_argument('--EMA_T_D_hat', default=True, action='store_true')
    parser.add_argument('--EMA_alpha_hat', default=True, action='store_true')
    parser.add_argument('--EMA_sig', default=True, action='store_true')
    parser.add_argument('--EMA_p_F', default=True, action='store_true') # EMA (exponential moving average) estimation of foreground class prior

    return parser.parse_args()


def main():
    sys.stdout = open('TPM_analy_logs.txt', 'w')
    args = parse_arguments()

    wandb.login(key='**********') #Your Weights & Biases key
    wandb.init(
        # Set the project where this run will be logged
        project="TPM",
        # We pass a run name (otherwise it’ll be randomly assigned)
        name='TPM_analy_%dfd_%dsd' % (args.fold, args.seed),
        # Track hyperparameters and run metadata
        config={
            "architecture": 'TPM',
            "dataset": args.dataset,
            "n_shot": args.n_shot,
            "n_query": args.n_query,
            "n_way": args.n_way,
            "fold": args.fold,
            "run": args.run,
            "seed": args.seed,
            "non_succesive": args.non_succesive,
            "bg_wt": args.bg_wt,
            "fix_sig2": args.fix_sig2,
            "fix_alpha": args.fix_alpha,
            "chg_alpha": args.chg_alpha,
            "chg_alpha2": args.chg_alpha2,
            "learn_alpha": args.learn_alpha,
            "ADNet": args.ADNet,
            "adnet": args.adnet,
            "t_loss_scaler": args.t_loss_scaler,
            "chg_T_D": args.chg_T_D,
            "fix_T_D": args.fix_T_D,
            "fix_p_F": args.fix_p_F,
            "chg_p_F": args.chg_p_F,
            "learn_p_F": args.learn_p_F,
            "EMA_T_D_hat": args.EMA_T_D_hat,
            "EMA_alpha_hat": args.EMA_alpha_hat,
            "EMA_sig": args.EMA_sig,
            "EMA_p_F": args.EMA_p_F,
        })

    # Deterministic setting for reproducability, but there can be randomness
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # Init model.
    model_ = FewShotSeg(False,
                        ADNet=args.ADNet,
                        adnet = args.adnet,t_loss_scaler=args.t_loss_scaler,
                        fix_alpha=args.fix_alpha,chg_alpha=args.chg_alpha,chg_alpha2=args.chg_alpha2,learn_alpha=args.learn_alpha,
                        fix_sig2=args.fix_sig2,
                        fix_T_D=args.fix_T_D, chg_T_D=args.chg_T_D,
                        fix_p_F=args.fix_p_F, chg_p_F=args.chg_p_F, learn_p_F=args.learn_p_F,
                        EMA_T_D_hat=args.EMA_T_D_hat, EMA_alpha_hat=args.EMA_alpha_hat,
                        EMA_sig=args.EMA_sig, EMA_p_F=args.EMA_p_F, lr=args.lr)

    model = nn.DataParallel(model_.cuda())
    checkpoint = torch.load(args.pretrained_root+'model.pth', map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
    model_.load_params(train=False, verbose=True)

    # Init optimizer.
    model.parameters = [param for nm, param in model.named_parameters()]
    model.sig1 = model_.sig1
    model.sig2 = model_.sig2
    model.T_S = model_.T_S
    model.T_D = model_.T_D
    model.T_D_hat = model_.T_D_hat
    model.p_F_hat = model_.p_F_hat
    model.p_F = model_.p_F
    model.dim_ = model_.dim_
    if args.ADNet:
        model.T_S = model_.T_S
    if args.adnet:
        model.pre_T_S = model_.pre_T_S
    model.alpha = model_.alpha

    # Enable cuDNN benchmark mode to select the fastest convolution algorithm.
    cudnn.enabled = True
    cudnn.benchmark = True

    # Define data set and loader.
    train_dataset = TrainDataset(args)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True, drop_last=True)

    for epoch in range(1):

        # Analyze.
        batch_time, data_time, results_dict = analyze(train_loader, model, args)

        # Log
        sys.stdout.flush()
        time.sleep(0.01)

    wandb.finish()

    # Restore stdout and close the file
    sys.stdout.close()
    sys.stdout = sys.__stdout__


def analyze(train_loader, model, args):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')

    model.eval()

    print('model.p_F_hat', model.p_F_hat.item())

    results_dict={'dists': [], 'loc': [], 'size': [], 'p_F_hat': [], 'p_F_MP_hat': []}

    end = time.time()
    for i, sample in enumerate(train_loader):

        # Extract episode data.
        support_images = [[shot.float().cuda() for shot in way]
                          for way in sample['support_images']]
        support_fg_mask = [[shot.float().cuda() for shot in way]
                           for way in sample['support_fg_labels']]
        support_idx = sample['support_idx']
        support_len = sample['support_len']

        query_images = [query_image.float().cuda() for query_image in sample['query_images']]
        query_labels = torch.cat([query_label.long().cuda() for query_label in sample['query_labels']], dim=0)

        # Log loading time.
        data_time.update(time.time() - end)

        if i%100==(100-1):
            print('i:',i)

        # Compute outputs.
        tmp_dists_sort, [supp_loc, supp_size, tmp_p_F_hat, tmp_p_F_MP_hat] = model(support_images, support_fg_mask, query_images, supp_img_all=support_len*[0],
                                                                        analyze=True, i=i, qry_mask=query_labels, supp_idx=support_idx)  # FewShotSeg()
        results_dict['loc'].append(supp_loc.item())
        results_dict['size'].append(supp_size)
        results_dict['p_F_hat'].append(tmp_p_F_hat.item())
        results_dict['p_F_MP_hat'].append(tmp_p_F_MP_hat.item())

        # Log elapsed time.
        batch_time.update(time.time() - end)
        end = time.time()

    results_dict['loc'] = np.array(results_dict['loc'])
    results_dict['size'] = np.array(results_dict['size'])
    results_dict['p_F_hat'] = np.array(results_dict['p_F_hat'])
    results_dict['p_F_MP_hat'] = np.array(results_dict['p_F_MP_hat'])

    print('loc', results_dict['loc'].shape)
    print('size', results_dict['size'].shape)
    print('p_F_hat', results_dict['p_F_hat'].shape)
    print('p_F_MP_hat', results_dict['p_F_MP_hat'].shape)

    locs = np.array(results_dict['loc'])
    locs2 = (locs - 0.5) ** 2
    locs3 = np.abs(locs - 0.5)
    sizes = np.array(results_dict['size'])
    sizes2 = sizes ** (3 / 2)
    p_F_hats = np.array(results_dict['p_F_hat'])
    p_F_MP_hats = np.array(results_dict['p_F_MP_hat'])
    p_F_hats = np.clip(p_F_hats,a_min=1e-3,a_max=1-1e-3)
    p_F_hats2 = np.log(p_F_hats/(1-p_F_hats)) #Logit function
    p_F_MP_hats = np.clip(p_F_MP_hats, a_min=1e-3, a_max=1 - 1e-3)
    p_F_MP_hats2 = np.log(p_F_MP_hats / (1 - p_F_MP_hats))  # Logit function

    val_inds = np.logical_not(np.isnan(p_F_hats2))
    val_inds2 = np.logical_not(np.isnan(p_F_MP_hats2))

    X = np.concatenate([locs2.reshape([-1, 1]), sizes.reshape([-1, 1])], axis=-1)
    y = p_F_hats2
    y2 = p_F_MP_hats2

    print('Pearsonr with locs:', pearsonr(locs[val_inds], p_F_hats[val_inds]))
    print('Pearsonr with locs2:', pearsonr(locs2[val_inds], p_F_hats[val_inds]))
    print('    Pearsonr with locs2_:', pearsonr(locs2[val_inds], p_F_hats2[val_inds]))
    print('Pearsonr with locs3:', pearsonr(locs3[val_inds], p_F_hats[val_inds]))
    print('Pearsonr with sizes:', pearsonr(sizes[val_inds], p_F_hats[val_inds]))
    print('    Pearsonr with sizes_:', pearsonr(sizes[val_inds], p_F_hats2[val_inds]))
    print('Pearsonr with sizes2:', pearsonr(sizes2[val_inds], p_F_hats[val_inds]), '\n')

    print('Pearsonr with locs:', pearsonr(locs[val_inds2], p_F_MP_hats[val_inds2]))
    print('Pearsonr with locs2:', pearsonr(locs2[val_inds2], p_F_MP_hats[val_inds2]))
    print('    Pearsonr with locs2_:', pearsonr(locs2[val_inds2], p_F_MP_hats2[val_inds2]))
    print('Pearsonr with locs3:', pearsonr(locs3[val_inds2], p_F_MP_hats[val_inds2]))
    print('Pearsonr with sizes:', pearsonr(sizes[val_inds2], p_F_MP_hats[val_inds2]))
    print('    Pearsonr with sizes_:', pearsonr(sizes[val_inds2], p_F_MP_hats2[val_inds2]))
    print('Pearsonr with sizes2:', pearsonr(sizes2[val_inds2], p_F_MP_hats[val_inds2]), '\n')

    ### Use linear regression to estimate ICPs from support foreground size and query location ###
    linr = LinearRegression().fit(X[val_inds], y[val_inds])
    linr2 = LinearRegression().fit(X[val_inds2], y2[val_inds2])

    x_lin = np.linspace(np.min(X[:, 1]), 1.05 * np.max(X[:, 1]), 500).reshape([-1, 1])
    X_lin = np.concatenate([0.0 * np.ones_like(x_lin), x_lin], axis=-1)
    y_pred = linr.predict(X_lin)
    y_pred2 = linr2.predict(X_lin)

    p_F_dict = {}

    plt.figure(1)
    plt.plot(x_lin, 1/(1+np.exp(-y_pred)), 'r', linewidth=1)
    plt.scatter(sizes, p_F_hats, s=1, c=locs, cmap='Spectral')
    p_F_hats_mn = np.nanmean(p_F_hats)
    p_F_dict['p_F_hats_mn'] = p_F_hats_mn
    print('p_F_hats_mn', p_F_hats_mn)
    plt.plot([0, 1.05 * np.max(sizes)], [p_F_hats_mn, p_F_hats_mn], 'k--', linewidth=1)
    plt.xlabel('area')
    plt.ylabel('p_F_hats')
    plt.xlim([0, 1.05 * np.max(sizes)])
    plt.ylim([0, 1.05 * np.nanmax(p_F_hats)])
    plt.savefig(args.pretrained_root+"p_F_hats", dpi=200)
    plt.close()

    plt.figure(2)
    plt.plot(x_lin, 1 / (1 + np.exp(-y_pred2)), 'r', linewidth=1)
    plt.scatter(sizes, p_F_MP_hats, s=1, c=locs, cmap='Spectral')
    p_F_MP_hats_mn = np.nanmean(p_F_MP_hats)
    p_F_dict['p_F_MP_hats_mn'] = p_F_MP_hats_mn
    print('p_F_MP_hats_mn', p_F_MP_hats_mn)
    plt.plot([0, 1.05 * np.max(sizes)], [p_F_MP_hats_mn, p_F_MP_hats_mn], 'k--', linewidth=1)
    plt.xlabel('area')
    plt.ylabel('p_F_MP_hats')
    plt.xlim([0, 1.05 * np.max(sizes)])
    plt.ylim([0, 1.05 * np.nanmax(p_F_MP_hats)])
    plt.savefig(args.pretrained_root + "p_F_MP_hats", dpi=200)
    plt.close()

    # Save linear models for LinEst
    print('linr', linr.coef_, linr.intercept_)
    print('linr2', linr2.coef_, linr2.intercept_)

    joblib.dump(linr, args.pretrained_root+"p_F_hats_model.pkl")
    joblib.dump(linr2, args.pretrained_root + "p_F_MP_hats_model.pkl")

    np.savez(args.pretrained_root + "p_F_dict_one_F.npz", **p_F_dict) # Save average ICPs for AvgEst

    return batch_time.avg, data_time.avg, results_dict

if __name__ == '__main__':
    main()

