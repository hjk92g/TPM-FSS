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

from models.fewshot_multiclass_anom import FewShotSeg # Major functions (forward, negSim, getPrototype, ...)
from dataloading.datasets_multiclass import TrainDataset as TrainDataset
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
    parser.add_argument('--one_F_train', action='store_true')

    # Training specs.
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--steps', default=50000, type=int) #Iterations
    parser.add_argument('--n_shot', default=1, type=int)
    parser.add_argument('--n_query', default=1, type=int)
    parser.add_argument('--n_way', default=1, type=int)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--non_succesive', action='store_true') #Allow non succesive slices
    parser.add_argument('--max_iterations', default=1000, type=int)
    parser.add_argument('--max_protos', default=5, type=int) # Maximum number of prototypes for (multiclass) training
    parser.add_argument('--lr', default=1e-3, type=float) # Learning rate
    parser.add_argument('--lr_gamma', default=0.95, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=0.0005, type=float)
    parser.add_argument('--learn_T_D', action='store_true') #learn T_D with CE loss
    parser.add_argument('--t_loss_scaler', default=0.0, type=float) # T loss weight. Set it 0 for TPM
    parser.add_argument('--fix_T_D', default=None, type=float)  # Train to fix T_D value
    parser.add_argument('--chg_T_D', action='store_true')  # Train to fit decision boundary to T_D_hat value
    parser.add_argument('--chg_T_D_multi', action='store_true')
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--bg_wt', default=0.1, type=float) # Background classs weight to address class imbalance (Foreground classs weight=1)

    parser.add_argument('--fix_sig2', default=None, type=float) # Fix sig2 during training
    parser.add_argument('--fix_alpha', default=None, type=float)  # Fix alpha during training
    parser.add_argument('--chg_alpha', action='store_true')  # Change alpha during training using self.alpha_hat
    parser.add_argument('--chg_alpha2', action='store_true') # Change alpha during training using self.alpha_hat2
    parser.add_argument('--chg_alpha_multi', action='store_true')
    parser.add_argument('--learn_alpha', action='store_true')

    parser.add_argument('--fix_p_F', default=None, type=float)  # Fix p_F during training
    parser.add_argument('--chg_p_F', action='store_true')
    parser.add_argument('--chg_p_F_multi', action='store_true')
    parser.add_argument('--learn_p_F', action='store_true')

    parser.add_argument('--EMA_T_D_hat', default=True, action='store_true')
    parser.add_argument('--EMA_alpha_hat', default=True, action='store_true')
    parser.add_argument('--EMA_sig', default=True, action='store_true')
    parser.add_argument('--EMA_p_F', default=True, action='store_true') # EMA (exponential moving average) estimation of foreground class prior

    return parser.parse_args()


def main():
    sys.stdout = open('TPM_multi_analy_logs.txt', 'w')
    args = parse_arguments()

    wandb.login(key='**********') #Your Weights & Biases key
    wandb.init(
        # Set the project where this run will be logged
        project="TPM",
        # We pass a run name (otherwise it’ll be randomly assigned)
        name='TPM_multi_analy_%dfd_%dsd' % (args.fold, args.seed),
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
            "one_F_train": args.one_F_train,
            "non_succesive": args.non_succesive,
            "bg_wt": args.bg_wt,
            "fix_sig2": args.fix_sig2,
            "fix_alpha": args.fix_alpha,
            "chg_alpha": args.chg_alpha,
            "chg_alpha2": args.chg_alpha2,
            "chg_alpha_multi": args.chg_alpha_multi,
            "learn_alpha": args.learn_alpha,
            "max_protos": args.max_protos,
            "learn_T_D": args.learn_T_D,
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

    # Deterministic setting for reproducability.
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # Init model.
    model_ = FewShotSeg(False, t_loss_scaler=args.t_loss_scaler,
                        fix_alpha=args.fix_alpha,chg_alpha=args.chg_alpha,chg_alpha2=args.chg_alpha2,learn_alpha=args.learn_alpha,
                        chg_alpha_multi=args.chg_alpha_multi,
                        fix_sig2=args.fix_sig2,
                        max_protos=args.max_protos,
                        learn_T_D=args.learn_T_D, fix_T_D=args.fix_T_D, chg_T_D=args.chg_T_D,
                        fix_p_F=args.fix_p_F, chg_p_F=args.chg_p_F, learn_p_F=args.learn_p_F, one_F_train=args.one_F_train,
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
    model.p_F_multi_hat = model_.p_F_multi_hat
    model.p_F = model_.p_F
    model.dim_ = model_.dim_
    if args.learn_T_D:
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
    device = torch.device('cuda')

    print('model.p_F_hat', model.p_F_hat.item())
    print('model.p_F_multi_hat', model.p_F_multi_hat.item())

    results_dict={'loc': [], 'size': [], 'p_F_hat': []} #Note that 'p_F_hat' here actually represent "p_F_multi_hat", i.e., obtained by considering multi. fore. classes classification

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

        if len(query_labels)==args.max_protos:
            [supp_loc, supp_size, tmp_p_F_hat] = model(support_images, support_fg_mask, query_images, supp_img_all=support_len*[0],
                                                                        analyze=True, i=i, qry_mask=query_labels, supp_idx=support_idx)  # FewShotSeg()
            len_supp_size = len(supp_size)
        else:
            len_supp_size = (args.max_protos-1)

        if len_supp_size==args.max_protos:
            results_dict['loc'].append(supp_loc.item())
            results_dict['size'].append(supp_size)
            results_dict['p_F_hat'].append(tmp_p_F_hat)

        # Log elapsed time.
        batch_time.update(time.time() - end)
        end = time.time()

    results_dict['loc'] = np.array(results_dict['loc'])
    results_dict['size'] = np.array(results_dict['size'])
    results_dict['p_F_hat'] = np.array(results_dict['p_F_hat'])

    print('loc', results_dict['loc'].shape)
    print('size', results_dict['size'].shape)
    print('p_F_hat', results_dict['p_F_hat'].shape)

    locs = np.array(results_dict['loc'])
    locs2 = (locs - 0.5) ** 2
    sizes = np.array(results_dict['size'])
    p_F_hats = np.array(results_dict['p_F_hat'])
    p_F_hats = np.clip(p_F_hats, a_min=1e-3, a_max=1 - 1e-3)

    val_inds = np.logical_not(np.isnan(p_F_hats))

    n_protos = sizes.shape[-1]

    Xs = [np.concatenate([locs2.reshape([-1, 1]), sizes[:,i].reshape([-1, 1])], axis=-1) for i in range(n_protos)]#n_protos x [n,2]
    y = np.concatenate([p_F_hats.reshape([-1, 1]),1-p_F_hats.reshape([-1, 1])], axis=-1)
    y_2 = np.concatenate(n_protos*[p_F_hats.reshape([-1, 1])]+[1-p_F_hats.reshape([-1, 1])], axis=-1)
    y_2 = y_2/np.sum(y_2,axis=-1,keepdims=True)
    print(Xs[0].shape, y.shape, y_2.shape)

    ### Use linear regression to estimate ICPs from support foreground size and query location ###
    class LinReg(nn.Module):
        def __init__(self, input_size, output_size):
            super(LinReg, self).__init__()
            self.fc = nn.Linear(input_size, output_size, device=device)

        def forward(self, x):
            lin_x = self.fc(x)
            return lin_x

    # Instantiate the model
    linr = LinReg(input_size=2, output_size=2)

    linr.fc.bias.data[1] = 0.0

    def second_weight_hook(grad):
        grad[1,:] = 0  # Zero out the gradient for the second output weights
        return grad

    def second_bias_hook(grad):
        grad[1] = 0  # Zero out the gradient for the second bias term
        return grad

    linr.fc.weight.register_hook(second_weight_hook)
    linr.fc.bias.register_hook(second_bias_hook)

    def FB2pred(FB, multiclass=True):
        # FB: [n,2] when multiclass=False
        # FB: [n,n_protos,2] when multiclass=True
        if multiclass:
            F = FB[:, :,0] #[n,n_protos]
            B = FB[:, :,1] #[n,n_protos]
            sum_B = torch.sum(B,dim=-1,keepdim=True)#[n,1]
            pred_prob = torch.cat((F,sum_B),dim=-1)
            pred_prob = pred_prob/torch.sum(pred_prob,dim=-1,keepdim=True)#[n,n_protos+1]
        else:
            F = FB[:, 0]
            B = FB[:, 1]
            pred_prob = torch.stack((F/(F+B),B/(F+B)),dim=-1)#[n,2]
        return pred_prob

    def FB_loss(FB, prob, multiclass=True):
        #FB: [n,2] when multiclass=False
        #FB: [n,n_protos,2] when multiclass=True
        pred_prob=FB2pred(FB, multiclass=multiclass)#[n,n_protos+1]
        l_pred_prob = torch.log(pred_prob)
        return nn.functional.kl_div(l_pred_prob,prob)

    # Optimizer
    optimizer = torch.optim.Adam(linr.parameters(), lr=0.1)

    X_tchs = [torch.tensor(Xs[i][val_inds]).to(device).float() for i in range(sizes.shape[-1])]
    y_tch = torch.tensor(y[val_inds]).to(device).float()#[n, 2]
    y_2_tch = torch.tensor(y_2[val_inds]).to(device).float()#[n, n_protos+1]

    # Training loop
    num_epochs = 1000
    for epoch in range(num_epochs):
        # Forward pass
        FBs = [torch.exp(linr(X_tchs[i])) for i in range(n_protos)]  # n_protos x [n x 2]
        Fs = [FBs[i][:, :1] for i in range(n_protos)]  # n_protos x [n x 1]
        Fs = torch.cat(Fs, dim=-1)  # [n x n_protos]
        Bs = [FBs[i][:, 1:] for i in range(n_protos)]  # n_protos x [n x 1]
        Bs = torch.cat(Bs, dim=-1)  # [n x n_protos]
        FB_2s = torch.stack((Fs, Bs), dim=-1)  # [n x n_protos x 2]

        pred_prob_2 = FB2pred(FB_2s, multiclass=True)#[n, n_protos+1]
        pred_F_pri = torch.mean(pred_prob_2[:,:-1],dim=-1)#[n]
        pred_F = pred_F_pri/(1-(n_protos-1)*pred_F_pri)#[n]
        pred_prob = torch.stack((pred_F, 1 - pred_F), dim=-1)
        l_pred_prob = torch.log(pred_prob)
        loss1 = nn.functional.kl_div(l_pred_prob,y_tch)
        loss1_2 = FB_loss(FB_2s, y_2_tch, multiclass=True)

        # Backward pass
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()

        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss1: {loss1.item():.4f}, Loss1_2: {loss1_2.item():.4f}")
            pred_prob = pred_prob.detach().cpu().numpy()
            pred_prob_2 = FB2pred(FB_2s, multiclass=True).detach().cpu().numpy()
            y_np = y_tch.detach().cpu().numpy()
            y_2_np = y_2_tch.detach().cpu().numpy()
            print('    MAE:',np.mean(np.abs(pred_prob[:,0]-y_np[:,0])))
            print('    Pearsonr with pred_prob:',pearsonr(pred_prob[:,0],y_np[:,0]))
            print('    Pearsonr with pred_prob_2:', pearsonr(pred_prob_2[:, -1], y_2_np[:, -1]))
    print()

    print('Finished training')
    print('params:', [(name, param) for name, param in linr.named_parameters()])

    tmp_X = np.eye(2,2)
    tmp_y = np.eye(2,2)
    LinR = LinearRegression().fit(tmp_X, tmp_y)

    LinR.coef_ = linr.fc.weight.data.detach().cpu().numpy()
    LinR.intercept_ = linr.fc.bias.data.detach().cpu().numpy()
    print('LinR', LinR.coef_, LinR.intercept_,'\n')

    print('x_lin',np.min(sizes), 1.05 * np.max(sizes))
    x_lin = np.linspace(np.min(sizes), 1.05 * np.max(sizes), 500).reshape([-1, 1])
    X_lin = np.concatenate([0.0 * np.ones_like(x_lin), x_lin], axis=-1)#[500,2]
    X_lin_tch = torch.tensor(X_lin).to(device).float()
    FB_2s = torch.exp(linr(X_lin_tch)).unsqueeze(1).expand(-1, n_protos, -1) # [n x n_protos x 2]
    pred_prob_2 = FB2pred(FB_2s, multiclass=True)
    pred_F_pri = torch.mean(pred_prob_2[:, :-1], dim=-1)  # [n]
    pred_F = pred_F_pri / (1 - (n_protos - 1) * pred_F_pri)  # [n]
    pred_prob_2 = pred_prob_2.detach().cpu().numpy()
    pred_F = pred_F.detach().cpu().numpy()
    print('pred_F',np.min(pred_F),np.max(pred_F))
    print('pred_prob_2', np.min(pred_prob_2[:, 0]), np.max(pred_prob_2[:, 0]),'\n')

    p_F_dict={}

    plt.figure(0)
    plt.plot(x_lin, pred_F, 'r', linewidth=1)
    for i in range(n_protos):
        plt.scatter(sizes[:,i], p_F_hats, s=1, c=locs, cmap='Spectral')
    p_F_hats_mn= np.nanmean(p_F_hats)
    p_F_dict['p_F_hats_mn']=p_F_hats_mn
    print('p_F_hats_mn',p_F_hats_mn)
    plt.plot([0, 1.05 * np.max(sizes)], [p_F_hats_mn, p_F_hats_mn], 'k--', linewidth=1)
    plt.xlabel('area')
    plt.ylabel('p_F_hats')
    plt.xlim([0, 1.05 * np.max(sizes)])
    plt.ylim([0, 1.05 * np.nanmax(p_F_hats)])
    if args.one_F_train:
        plt.savefig(args.pretrained_root + "p_F_hats2", dpi=200)
    else:
        plt.savefig(args.pretrained_root+"p_F_hats", dpi=200)
    plt.close()

    plt.figure(1)
    plt.plot(x_lin, pred_prob_2[:, 0], 'r', linewidth=1)
    p_F_hats_2 = p_F_hats/(1+(n_protos-1)*p_F_hats)
    for i in range(n_protos):
        plt.scatter(sizes[:, i], p_F_hats_2, s=1, c=locs, cmap='Spectral')
    p_F_hats_mn_2 = p_F_hats_mn/(1+(n_protos-1)*p_F_hats_mn)
    p_F_dict['p_F_hats_mn_2'] = p_F_hats_mn_2
    print('p_F_hats_mn_2',p_F_hats_mn_2)
    plt.plot([0, 1.05 * np.max(sizes)], [p_F_hats_mn_2, p_F_hats_mn_2], 'k--', linewidth=1)
    plt.xlabel('area')
    plt.ylabel('p_F_hats_2')
    plt.xlim([0, 1.05 * np.max(sizes)])
    plt.ylim([0, 1.05 * np.nanmax(p_F_hats_2)])
    plt.savefig(args.pretrained_root + "p_F_hats_2", dpi=200)
    plt.close()

    # Save linear models for LinEst
    if args.one_F_train:
        joblib.dump(LinR, args.pretrained_root + "p_F_hats_model2.pkl")
    else:
        joblib.dump(LinR, args.pretrained_root +"p_F_hats_model.pkl")

    np.savez(args.pretrained_root + "p_F_dict.npz", **p_F_dict) # Save average ICPs for AvgEst

    return batch_time.avg, data_time.avg, results_dict

if __name__ == '__main__':
    main()

