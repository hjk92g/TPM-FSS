#!/usr/bin/env python

import argparse
import random
import numpy as np

from utils import *


import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader

from models.fewshot_anom import FewShotSeg
from dataloading.datasets import TestDataset
from dataloading.dataset_specifics import *
from torch.nn.parameter import Parameter
import pickle, os, sys


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--save_root', type=str, required=True)
    parser.add_argument('--pretrained_root', type=str, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--run', type=int, required=True)
    parser.add_argument('--train_name', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--n_shot', default=1, type=int)
    parser.add_argument('--all_slices', default=False, type=bool)
    parser.add_argument('--EP1', default=False, type=bool)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--non_succesive', action='store_true')  # Allow non succesive slices

    parser.add_argument('--ADNet', action='store_true')  # Make it the same as adnet
    parser.add_argument('--adnet', action='store_true')  # Learn T_D
    parser.add_argument('--t_loss_scaler', default=1.0, type=float) # T loss weight. Set it 0 for TPM
    parser.add_argument('--fix_T_D', default=None, type=float)  # Train to fix T_D value
    parser.add_argument('--chg_T_D', action='store_true')  # Train to fit decision boundary to T_D_hat value

    parser.add_argument('--fix_sig2', default=None, type=float)  # Fix sig2 during training

    parser.add_argument('--fix_alpha', default=None, type=float)  # Fix alpha during training
    parser.add_argument('--chg_alpha', action='store_true')  # Change alpha during training using self.alpha_T_D
    parser.add_argument('--chg_alpha2', action='store_true') # Change alpha during training using self.alpha_T_D2
    parser.add_argument('--learn_alpha', action='store_true')

    parser.add_argument('--fix_p_F', default=None, type=float)  # Fix p_F during training
    parser.add_argument('--chg_p_F', action='store_true')
    parser.add_argument('--learn_p_F', action='store_true')

    parser.add_argument('--EMA_T_D_hat', default=True, action='store_true')
    parser.add_argument('--EMA_alpha_hat', default=True, action='store_true')
    parser.add_argument('--EMA_sig', default=True, action='store_true')
    parser.add_argument('--EMA_p_F', default=True,action='store_true')  # EMA (exponential moving average) estimation of foreground class prior

    parser.add_argument("--n_sv", nargs="+", default=["5000"]) # supervoxel parameter

    return parser.parse_args()

def main():
    sys.stdout = open('TPM_test_logs.txt', 'w')
    args = parse_arguments()

    if args.EP1: # EP2 is used in the paper
        EP_num = 1
    else:
        EP_num = 2

    wandb.login(key='**********') #Your Weights & Biases key
    wandb.init(
        # Set the project where this run will be logged
        project="TPM",
        # We pass a run name (otherwise it’ll be randomly assigned)

        name='TPM_test_EP%d_%ds_%dfd_%dsd' % (EP_num, args.n_shot, args.fold, args.seed),
        # Track hyperparameters and run metadata
        config={
            "architecture": 'TPM',
            "phase": "test",
            "dataset": args.dataset,
            "n_shot": args.n_shot,
            "all_slices": args.all_slices,
            "EP1": args.EP1,
            "fold": args.fold,
            "run": args.run,
            "train_name": args.train_name,
            "seed": args.seed,
            "non_succesive": args.non_succesive,
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
            "n_sv": args.n_sv,
        })

    # Deterministic setting for reproducability.
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # Set up logging.
    logger = set_logger(args.save_root, 'test.log')
    logger.info(args)

    # Setup the path to save.
    args.save = os.path.join(args.save_root)

    # Init model and load state_dict.
    model_ = FewShotSeg(use_coco_init=False,
                        ADNet = args.ADNet, adnet = args.adnet,t_loss_scaler=args.t_loss_scaler,
                        fix_alpha=args.fix_alpha, chg_alpha=args.chg_alpha, chg_alpha2=args.chg_alpha2, learn_alpha=args.learn_alpha,
                        fix_sig2=args.fix_sig2,
                        fix_T_D=args.fix_T_D, chg_T_D=args.chg_T_D,
                        fix_p_F=args.fix_p_F, chg_p_F=args.chg_p_F, learn_p_F=args.learn_p_F,
                        data_root = args.data_root, pretrained_root=args.pretrained_root,
                        EMA_T_D_hat=args.EMA_T_D_hat, EMA_alpha_hat=args.EMA_alpha_hat,
                        EMA_sig=args.EMA_sig, EMA_p_F=args.EMA_p_F)
    print(model_.state_dict().keys())

    model_.logger = logger
    model = nn.DataParallel(model_.cuda())
    model.get_shapes = model_.get_shapes
    model.methods = model_.methods
    model.metrics = model_.metrics

    checkpoint = torch.load(args.pretrained_root+'model.pth', map_location="cpu")

    model.load_state_dict(checkpoint,strict=False)

    # Data loader.
    test_dataset = TestDataset(args)
    query_loader = DataLoader(test_dataset,
                              batch_size=1,
                              shuffle=False,
                              num_workers=args.workers,
                              pin_memory=True,
                              drop_last=True)

    # Inference.
    logger.info('  Start inference ... Note: EP1 is ' + str(args.EP1))
    logger.info('  Support: ' + str(test_dataset.support_dir[len(args.data_root):]))
    logger.info('  Query: ' +
                str([elem[len(args.data_root):] for elem in test_dataset.image_dirs]))

    model_.load_params(train=False,verbose=True)

    ### Show parameters###
    alpha = (model_.alpha.detach().cpu().numpy())[0]
    logger.info('  alpha: {:.5f}'.format(alpha))
    T_S = (model_.T_S.detach().cpu().numpy())[0]
    logger.info('  T_S: {:.5f}'.format(T_S))  # Anomaly score threshold (T_S)
    T_D = (model_.T_D.detach().cpu().numpy())[0]
    logger.info('  T_D: {:.5f}'.format(T_D)) # Distance threshold (T_D)

    sig1 = model_.sig1.detach().cpu().numpy()[0]
    sig2 = model_.sig2.detach().cpu().numpy()[0]
    logger.info('  Sigma 1 : {:.5f}, '.format(sig1)) # Foreground class std. (sigma F)
    logger.info('  Sigma 2 : {:.5f}, '.format(sig2)) # Background class std. (sigma B)

    p_F = (model_.p_F.detach().cpu().numpy())[0]
    logger.info('  p_F: {:.5f}'.format(p_F)) # Foreground class prior (p_F)

    T_D_hat = model_.T_D_hat.detach().cpu().numpy()[0]
    #alpha_hat = model_.alpha_hat.detach().cpu().numpy()[0]
    #alpha_hat2 = model_.alpha_hat2.detach().cpu().numpy()[0]
    p_F_hat = model_.p_F_hat.detach().cpu().numpy()[0]
    logger.info('  T_D_hat : {:.5f}, '.format(T_D_hat))
    logger.info('  p_F_hat : {:.5f}, '.format(p_F_hat)) # AvgEst using exponential moving average (EMA)

    # Get unique labels (classes).
    labels = get_label_names(args.dataset)

    print('labels',labels) #labels {0: 'BG', 1: 'LIVER', 2: 'RK', 3: 'LK', 4: 'SPLEEN'}

    # Loop over classes.
    class_dice = {}
    class_iou = {}
    class_mse = {}

    results_sv = {}

    datasets = ['CHAOST2', 'SABS']
    trains = ['train_adnet_alpha_20', 'train_adnet_alpha_20_no_t_loss']
    folds = ['fold0', 'fold1', 'fold2', 'fold3', 'fold4']
    runs = ['run1', 'run2', 'run3']
    methods = ['CE-T', 'AvgEst', 'LinEst', 'OCP', 'CE-T_MP', 'AvgEst_MP','LinEst_MP', 'OCP_MP']
    metrics = ['iou', 'dice', 'MSE']
    classes = ['LIVER', 'LK', 'RK', 'SPLEEN']

    if args.EP1 is True:
        results_sv_path = "results/results_sv_EP1.pkl"
    else:
        results_sv_path = "results/results_sv_EP2.pkl"

    ### Initialize or load 'results_sv' ###
    os.makedirs("results", exist_ok=True)
    if os.path.exists(results_sv_path):
        with open(results_sv_path, "rb") as f:
            results_sv = pickle.load(f)

        for dataset in datasets:
            results_sv.setdefault(dataset,{})

            for train_ in trains:
                results_sv[dataset].setdefault(train_, {})
                for fold in folds:
                    results_sv[dataset][train_].setdefault(fold, {})
                    for run in runs:
                        results_sv[dataset][train_][fold].setdefault(run, {})
                        for method in methods:
                            results_sv[dataset][train_][fold][run].setdefault(method, {})
                            for metric in metrics:
                                results_sv[dataset][train_][fold][run][method].setdefault(metric, {})
                                for cls in classes:
                                    results_sv[dataset][train_][fold][run][method][metric].setdefault(cls, np.nan)
    else:
        for dataset in datasets:
            results_sv[dataset] = {}

            for train in trains:
                results_sv[dataset][train] = {}
                for fold in folds:
                    results_sv[dataset][train][fold] = {}
                    for run in runs:
                        results_sv[dataset][train][fold][run] = {}
                        for method in methods:
                            results_sv[dataset][train][fold][run][method] = {}
                            for metric in metrics:
                                results_sv[dataset][train][fold][run][method][metric] = {cls: np.nan for cls in classes}
    methods_class_iou = {method: {} for method in methods}
    methods_class_dice = {method: {} for method in methods}
    methods_class_MSE = {method: {} for method in methods}
    for label_val, label_name in labels.items():

        # Skip BG class.
        if label_name is 'BG':
            continue

        logger.info('  *------------------Class: {}--------------------*'.format(label_name))
        logger.info('  *--------------------------------------------------*')

        # Get support sample + mask for current class.
        support_sample = test_dataset.getSupport(label=label_val, all_slices=args.all_slices, N=args.n_shot)
        test_dataset.label = label_val

        # Infer.
        with torch.no_grad():
            scores, q_results_dict = infer(model, query_loader, support_sample, args, logger, label_name)

        # Log class-wise results
        class_dice[label_name] = torch.tensor(scores.patient_dice).mean().item()
        class_iou[label_name] = torch.tensor(scores.patient_iou).mean().item()

        logger.info('      Mean class IoU: {}'.format(class_iou[label_name]))
        logger.info('      Mean class Dice: {}'.format(class_dice[label_name]))
        logger.info('      TP, TN, FP, FN: {}'.format([scores.TP.item(),scores.TN.item(),scores.FP.item(),scores.FN.item()]))

        logger.info('  ---Accuracy:')
        for method in methods:
            methods_class_iou[method][label_name] = torch.tensor(q_results_dict[method]['iou']).mean().item()
            methods_class_dice[method][label_name] = torch.tensor(q_results_dict[method]['dice']).mean().item()
            methods_class_MSE[method][label_name] = torch.tensor(q_results_dict[method]['MSE']).mean().item()
            logger.info('      method: {}'.format(method))
            logger.info('      Mean class IoU ({}): {}'.format(method, methods_class_iou[method][label_name]))
            logger.info('      Mean class Dice ({}): {}'.format(method, methods_class_dice[method][label_name]))
        logger.info('  *-----*')
        for method in methods:
            logger.info('      Mean class MSE ({}): {}'.format(method, methods_class_MSE[method][label_name]))
        logger.info('  *--------------------------------------------------*')


    # Log final results.
    logger.info('  *-----------------Final results--------------------*')
    logger.info('  *--------------------------------------------------*')
    logger.info('  Mean IoU: {}'.format(class_iou))
    logger.info('  Mean Dice: {}'.format(class_dice))

    logger.info('  ---Accuracy:')
    for method in methods:
        logger.info('      Mean IoU ({}): {}'.format(method, methods_class_iou[method]))
        logger.info('      Mean Dice ({}): {}'.format(method, methods_class_dice[method]))
    logger.info('  *-----*')
    for method in methods:
        logger.info('      Mean MSE ({}): {}'.format(method, methods_class_MSE[method]))
    logger.info('  *--------------------------------------------------*\n\n\n')

    for method in methods:
        results_sv[args.dataset][args.train_name]['fold'+str(args.fold)]['run'+str(args.run)][method]['iou']=\
            methods_class_iou[method]
        results_sv[args.dataset][args.train_name]['fold' + str(args.fold)]['run' + str(args.run)][method]['dice'] = \
            methods_class_dice[method]
        results_sv[args.dataset][args.train_name]['fold' + str(args.fold)]['run' + str(args.run)][method]['MSE'] = \
            methods_class_MSE[method]

    with open(results_sv_path, "wb") as f:
        pickle.dump(results_sv, f)




def infer(model, query_loader, support_sample, args, logger, label_name):

    # Test mode.
    model.eval()

    # Unpack support data.
    support_image = [support_sample['image'][[i]].float().cuda() for i in range(support_sample['image'].shape[0])]  # n_shot x [3 x H x W]
    support_fg_mask = [support_sample['label'][[i]].float().cuda() for i in range(support_sample['image'].shape[0])]  # n_shot x [H x W]
    support_image_all = support_sample['image_all'].float().cuda() #[slide_number x 3 x H x W]
    if len(support_image)==1:
        support_idx = support_sample['idx']
    else:
        support_idxs = support_sample['idxs']

    # Loop through query volumes.
    scores = Scores()

    q_results_dict={method: {metric: [] for metric in model.metrics} for method in model.methods}

    for i, sample in enumerate(query_loader):

        # Unpack query data.
        query_image = [sample['image'][i].float().cuda() for i in range(sample['image'].shape[0])]  # [C x 3 x H x W]
        query_label = sample['label'].long()  # [C x H x W]
        query_id = sample['id'][0].split('image_')[1][:-len('.nii.gz')]
        query_fg_mask = [sample['label'][i].float().cuda() for i in range(sample['image'].shape[0])]

        # Compute output.
        if args.EP1 is True:
            # Match support slice and query sub-chunk.
            query_pred = torch.zeros(query_label.shape[-3:])
            C_q = sample['image'].shape[1]
            idx_ = np.linspace(0, C_q, args.n_shot+1).astype('int')
            query_pred_dict = {method: torch.zeros(query_label.shape[-3:]).cuda() for method in model.methods}
            query_pred_dict = {method: torch.stack([query_pred_dict[method],query_pred_dict[method]], dim=1)
                               for method in model.methods}
            for sub_chunck in range(args.n_shot):
                support_image_s = [support_image[sub_chunck]]  # 1 x 3 x H x W
                support_fg_mask_s = [support_fg_mask[sub_chunck]]  # 1 x H x W
                query_image_s = query_image[0][idx_[sub_chunck]:idx_[sub_chunck+1]]  # C' x 3 x H x W
                query_fg_mask_s = query_fg_mask[0][idx_[sub_chunck]:idx_[sub_chunck + 1]]
                (query_pred_s, _, output_dict_s) = model([support_image_s], [support_fg_mask_s], [query_image_s],
                                                    train=False, i=i, qry_mask=query_fg_mask_s[None,:],
                                                    supp_img_all=support_image_all,supp_idx=support_idxs[sub_chunck])# C x 2 x H x W

                query_pred_s = query_pred_s.argmax(dim=1).cpu()  # C x H x W
                query_pred[idx_[sub_chunck]:idx_[sub_chunck+1]] = query_pred_s


                for method in model.methods:
                    query_pred_dict[method][idx_[sub_chunck]:idx_[sub_chunck+1]] = output_dict_s[method]
        else:  # EP 2
            (query_pred, _, output_dict) = model([support_image], [support_fg_mask], query_image,
                                                    train=False, i=i, qry_mask=query_label,
                                                    supp_img_all=support_image_all, supp_idx=support_idx)  # C x 2 x H x W

            query_pred = query_pred.argmax(dim=1).cpu()  # C x H x W

            query_pred_dict = {method: output_dict[method] for method in model.methods}

        # Record scores.
        scores.record(query_pred, query_label)
        preval = torch.mean(query_label[0].float().cuda(),dim=[1,2])
        for method in model.methods:
            tmp_query_pred_ = query_pred_dict[method]
            tmp_query_pred = tmp_query_pred_.argmax(dim=1)

            tp = torch.sum((query_label == 1).cuda() * (tmp_query_pred == 1)).cpu()
            fp = torch.sum((query_label == 0).cuda() * (tmp_query_pred == 1)).cpu()
            fn = torch.sum((query_label == 1).cuda() * (tmp_query_pred == 0)).cpu()
            tmp_iou = tp / (tp + fp + fn)
            q_results_dict[method]['iou'].append(tmp_iou)
            tmp_dice =2 * tp / (2 * tp + fp + fn)
            q_results_dict[method]['dice'].append(tmp_dice)

            pred_pos = torch.mean(tmp_query_pred.float(),dim=[1,2])
            tmp_MSE = torch.mean((preval-pred_pos)**2).cpu()
            q_results_dict[method]['MSE'].append(tmp_MSE)

        # Log.
        logger.info('    Tested query volume: ' + sample['id'][0][len(args.data_root):]
                    + '. Dice score:  ' + str(scores.patient_dice[-1].item()))

        # Save predictions.
        file_name = 'image_' + query_id + '_' + label_name + '.pt'
        torch.save(query_pred, os.path.join(args.save, file_name))

    return scores, q_results_dict

if __name__ == '__main__':
    main()
