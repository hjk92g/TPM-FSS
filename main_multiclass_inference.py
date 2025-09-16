#!/usr/bin/env python

import argparse
import random
import numpy as np
import itertools

from utils import *


import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader

from models.fewshot_multiclass_anom import FewShotSeg
from dataloading.datasets_multiclass import TestDataset
from dataloading.dataset_specifics import *
from torch.nn.parameter import Parameter
import pickle, os, sys


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--save_root', type=str, required=True)
    parser.add_argument('--pretrained_root', type=str, required=True)
    parser.add_argument('--one_F_train', action='store_true')
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--run', type=int, required=True)
    parser.add_argument('--train_name', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--n_shot', default=1, type=int)
    parser.add_argument('--all_slices', default=False, type=bool)
    parser.add_argument('--supp_slice', type=str, required=True, help='bal/rand') #Most balanced slice or Random slice from slices contain all classes

    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--non_succesive', action='store_true')  #Allow non succesive slices

    parser.add_argument('--learn_T_D', action='store_true')  #Learn T_D
    parser.add_argument('--t_loss_scaler', default=1.0, type=float) #. Set it 0 for TPM
    parser.add_argument('--fix_T_D', default=None, type=float)  # Train to fix T_D value
    parser.add_argument('--chg_T_D', action='store_true')  # Train to fit decision boundary to T_D_hat value
    parser.add_argument('--chg_T_D_multi', action='store_true')
    parser.add_argument('--max_protos', default=5, type=int) # Maximum number of prototypes for (multiclass) training

    parser.add_argument('--F_importance', default=1.0, type=float)
    parser.add_argument('--sig2_scale', default=1.0, type=float)
    parser.add_argument('--fix_sig2', default=None, type=float)  # Fix sig2 during training
    parser.add_argument('--fix_alpha', default=None, type=float)  # Fix alpha during training
    parser.add_argument('--chg_alpha', action='store_true')  # Change alpha during training using self.alpha_t_C
    parser.add_argument('--chg_alpha2', action='store_true')  # Change alpha during training using self.alpha_hat2
    parser.add_argument('--chg_alpha_multi', action='store_true')
    parser.add_argument('--learn_alpha', action='store_true')

    parser.add_argument('--fix_p_F', default=None, type=float)  # Fix p_F during training
    parser.add_argument('--chg_p_F', action='store_true')
    parser.add_argument('--chg_p_F_multi', action='store_true')
    parser.add_argument('--learn_p_F', action='store_true')

    parser.add_argument('--EMA_T_D_hat', default=True, action='store_true')
    parser.add_argument('--EMA_alpha_hat', default=True, action='store_true')
    parser.add_argument('--EMA_sig', default=True, action='store_true')
    parser.add_argument('--EMA_p_F', default=True, action='store_true')  # EMA (exponential moving average) estimation of foreground class prior

    parser.add_argument("--n_sv", nargs="+", default=["5000"])
    return parser.parse_args()

def main():
    sys.stdout = open('TPM_multi_test_logs.txt', 'w')
    args = parse_arguments()

    wandb.login(key='**********') #Your Weights & Biases key
    wandb.init(
        # Set the project where this run will be logged
        project="TPM",
        # We pass a run name (otherwise it’ll be randomly assigned)

        name='TPM_multi_test_%ds_%dfd_%dsd' % (args.n_shot, args.fold, args.seed),
        # Track hyperparameters and run metadata
        config={
            "architecture": 'TPM',
            "phase": "test",
            "dataset": args.dataset,
            "pretrained_root": args.pretrained_root,
            "one_F_train": args.one_F_train,
            "n_shot": args.n_shot,
            "all_slices": args.all_slices,
            "supp_slice": args.supp_slice,
            "fold": args.fold,
            "train_name": args.train_name,
            "seed": args.seed,
            "non_succesive": args.non_succesive,
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
                        pretrained_root =args.pretrained_root,
                        t_loss_scaler=args.t_loss_scaler,
                        fix_alpha=args.fix_alpha, chg_alpha=args.chg_alpha, chg_alpha2=args.chg_alpha2,
                        chg_alpha_multi=args.chg_alpha_multi, learn_alpha=args.learn_alpha,
                        fix_sig2=args.fix_sig2,
                        max_protos=args.max_protos,
                        learn_T_D=args.learn_T_D, fix_T_D=args.fix_T_D, chg_T_D=args.chg_T_D,
                        fix_p_F=args.fix_p_F, chg_p_F=args.chg_p_F, learn_p_F=args.learn_p_F, one_F_train=args.one_F_train,
                        EMA_T_D_hat=args.EMA_T_D_hat, EMA_alpha_hat=args.EMA_alpha_hat,
                        EMA_sig=args.EMA_sig, EMA_p_F=args.EMA_p_F)

    print(model_.state_dict().keys())

    model_.logger = logger
    model = nn.DataParallel(model_.cuda())
    model.get_shape = model_.get_shapes
    model.methods = model_.methods
    model.metrics = model_.metrics

    checkpoint = torch.load(args.pretrained_root+"model.pth", map_location="cpu")


    model.load_state_dict(checkpoint, strict=False)

    # Data loader.
    test_dataset = TestDataset(args, binarize=False)
    query_loader = DataLoader(test_dataset,
                              batch_size=1,
                              shuffle=False,
                              num_workers=args.workers,
                              pin_memory=True,
                              drop_last=True)#All query slices (some may not contain multiple classes)
    test_dataset_multi = TestDataset(args, binarize=False,only_multi=True)
    query_loader_multi = DataLoader(test_dataset_multi,
                              batch_size=1,
                              shuffle=False,
                              num_workers=args.workers,
                              pin_memory=True,
                              drop_last=True) #Only query slices with multiple classes

    # Inference.
    logger.info('  Start inference ...')
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
    T_D_multi_hat = model_.T_D_multi_hat.detach().cpu().numpy()[0]
    #alpha_hat = model_.alpha_hat.detach().cpu().numpy()[0]
    #alpha_hat2 = model_.alpha_hat2.detach().cpu().numpy()[0]
    #alpha_multi_hat = model_.alpha_multi_hat.detach().cpu().numpy()[0]
    p_F_hat = model_.p_F_hat.detach().cpu().numpy()[0]
    p_F_multi_hat = model_.p_F_multi_hat.detach().cpu().numpy()[0]
    logger.info('  T_D_hat : {:.5f}, '.format(T_D_hat))
    logger.info('  T_D_multi_hat : {:.5f}, '.format(T_D_multi_hat))
    logger.info('  p_F_hat : {:.5f}, '.format(p_F_hat))
    logger.info('  p_F_multi_hat : {:.5f}, '.format(p_F_multi_hat)) # AvgEst using exponential moving average (EMA)

    # Get unique labels (classes).
    labels = get_label_names(args.dataset)

    print('labels',labels) #labels {0: 'BG', 1: 'LIVER', 2: 'RK', 3: 'LK', 4: 'SPLEEN'}

    # Store dice scores patient-wise.
    patient_dict = {}
    for label_val, label_name in labels.items():
        if label_name is 'BG':
            continue
        patient_dict[label_name] = {}

    class_dice = {}
    class_iou = {}
    class_multi_dice = {}
    class_multi_iou = {}

    results_sv = {}

    datasets = ['CHAOST2', 'SABS']
    trains = ['train_Proto5_learn_T_D_alpha_20_no_t_loss','train_adnet_alpha_20_no_t_loss']
    folds = ['fold0', 'fold1', 'fold2', 'fold3', 'fold4']
    runs = ['run1', 'run2', 'run3']
    methods = ['ADNet++', 'CE-T', 'AvgEst', 'LinEst', 'OCP']
    metrics = ['iou', 'dice', 'MSE']
    classes = ['LIVER', 'LK', 'RK', 'SPLEEN']

    results_sv_path = "results_multiclass/results_sv.pkl"
    os.makedirs("results_multiclass", exist_ok=True)
    if os.path.exists(results_sv_path):
        with open(results_sv_path, "rb") as f:
            results_sv = pickle.load(f)

        for dataset in datasets:

            results_sv.setdefault(dataset, {})

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

    model.classes = classes

    #Using all slices (some may not contain mutiple-foreground classes)
    methods_class_iou = {method: {} for method in methods}
    methods_class_dice = {method: {} for method in methods}
    methods_class_MSE = {method: {} for method in methods}
    #Only using slices with multi-foreground classes
    methods_class_multi_iou = {method: {} for method in methods}
    methods_class_multi_dice = {method: {} for method in methods}
    methods_class_multi_MSE = {method: {} for method in methods}

    for label_val, label_name in labels.items():
        # Skip BG class.
        if label_name is 'BG':
            continue
        else:
            class_dice[label_name] = []
            class_iou[label_name] = []
            class_multi_dice[label_name] = []
            class_multi_iou[label_name] = []

        for method in methods:
            if label_name is 'BG':
                continue
            else:
                methods_class_dice[method][label_name] = []
                methods_class_iou[method][label_name] = []
                methods_class_MSE[method][label_name] = []

                methods_class_multi_dice[method][label_name] = []
                methods_class_multi_iou[method][label_name] = []
                methods_class_multi_MSE[method][label_name] = []

    perms = len(list(itertools.combinations(test_dataset.FOLD[args.fold], args.n_shot)))
    for perm_i in range(perms):#reversed(range(perms)):

        # Get support sample + mask for current class.
        test_dataset.perm = perm_i
        support_sample = test_dataset.getSupport(label=-10000)

        # Log.
        logger.info('  Permutation: ' + str(perm_i + 1) + '/' + str(perms))
        logger.info('  Support: ' + str(
            [elem[len(args.data_root):] for elem in list(test_dataset.support_combinaitons[perm_i])]))
        logger.info('  Query: ' +
                    str([elem[len(args.data_root):] for elem in test_dataset.image_dirs if
                         elem not in list(test_dataset.support_combinaitons[perm_i])]))

        # Infer.
        with torch.no_grad():
            scores, q_results_dict = infer(model, query_loader, support_sample, args, logger, labels)
            print('\n\n\n')
            scores_multi, q_results_multi_dict = infer(model, query_loader_multi, support_sample, args, logger, labels)

        # Log patient-wise results.
        for label_val, label_name in labels.items():
            # Skip BG class.
            if label_name is 'BG':
                continue

            logger.info('  *---Class: {}---*'.format(label_name))

            # Log class-wise results
            class_dice[label_name] += scores[label_name].patient_dice
            class_iou[label_name] += scores[label_name].patient_iou
            class_multi_dice[label_name] += scores_multi[label_name].patient_dice
            class_multi_iou[label_name] += scores_multi[label_name].patient_iou

            for method in methods:
                iou = q_results_dict[method]['iou'][label_name]
                dice = q_results_dict[method]['dice'][label_name]
                MSE = q_results_dict[method]['MSE'][label_name]


                iou_multi = q_results_multi_dict[method]['iou'][label_name]
                dice_multi = q_results_multi_dict[method]['dice'][label_name]
                MSE_multi = q_results_multi_dict[method]['MSE'][label_name]
                methods_class_iou[method][label_name] += [torch.tensor(iou)]
                methods_class_dice[method][label_name] += [torch.tensor(dice)]
                methods_class_MSE[method][label_name] += [torch.tensor(MSE)]
                methods_class_multi_iou[method][label_name] += [torch.tensor(iou_multi)]
                methods_class_multi_dice[method][label_name] += [torch.tensor(dice_multi)]
                methods_class_multi_MSE[method][label_name] += [torch.tensor(MSE_multi)]
                logger.info('      method: {}'.format(method))
                logger.info('      Class IoU ({}): {}'.format(method, methods_class_iou[method][label_name][-1].mean().item()))
                logger.info('      Class Dice ({}): {}'.format(method, methods_class_dice[method][label_name][-1].mean().item()))
                logger.info('      Class IoU (multi-class slices) ({}): {}'.format(method, methods_class_multi_iou[method][label_name][-1].mean().item()))
                logger.info('      Class Dice (multi-class slices) ({}): {}'.format(method, methods_class_multi_dice[method][label_name][-1].mean().item()))

            logger.info('  *-----*')
            for method in methods:
                logger.info('      Class MSE ({}): {}'.format(method, methods_class_MSE[method][label_name][-1].mean().item()))
                logger.info('      Class MSE (multi-class slices) ({}): {}'.format(method, methods_class_multi_MSE[method][label_name][-1].mean().item()))
            logger.info('  *--------------------------------------------------*')


    # Show evaluation results
    for label_val, label_name in labels.items():

        # Skip BG class.
        if label_name is 'BG':
            continue
        logger.info('  # dice scores per class: ' + str(len(class_dice[label_name])))
        logger.info('       ' + label_name + ' before: ' + str(class_dice[label_name]))
        logger.info('  dice per patient:')

        print('class_iou[label_name]',class_iou[label_name])
        print('class_dice[label_name]', class_dice[label_name])

        class_iou[label_name] = torch.tensor(class_iou[label_name]).mean().item()
        class_dice[label_name] = torch.tensor(class_dice[label_name]).mean().item()
        class_multi_iou[label_name] = torch.tensor(class_multi_iou[label_name]).mean().item()
        class_multi_dice[label_name] = torch.tensor(class_multi_dice[label_name]).mean().item()

        logger.info('      Mean class iou: {}'.format(class_iou[label_name]))
        logger.info('      Mean class Dice: {}'.format(class_dice[label_name]))
        logger.info('      Mean class iou (multi-class slices): {}'.format(class_multi_iou[label_name]))
        logger.info('      Mean class Dice (multi-class slices): {}'.format(class_multi_dice[label_name]))

        logger.info('  ---Accuracy:')
        for method in methods:
            methods_class_iou[method][label_name] = torch.tensor(torch.cat(methods_class_iou[method][label_name])).mean().item()
            methods_class_dice[method][label_name] = torch.tensor(torch.cat(methods_class_dice[method][label_name])).mean().item()
            methods_class_MSE[method][label_name] = torch.tensor(torch.cat(methods_class_MSE[method][label_name])).mean().item()
            methods_class_multi_iou[method][label_name] = torch.tensor(
                                        torch.cat(methods_class_multi_iou[method][label_name])).mean().item()
            methods_class_multi_dice[method][label_name] = torch.tensor(
                                        torch.cat(methods_class_multi_dice[method][label_name])).mean().item()
            methods_class_multi_MSE[method][label_name] = torch.tensor(
                                        torch.cat(methods_class_multi_MSE[method][label_name])).mean().item()
            logger.info('        method: {}'.format(method))
            logger.info('        Mean class iou ({}): {}'.format(method, methods_class_iou[method][label_name]))
            logger.info('        Mean class Dice ({}): {}'.format(method,methods_class_dice[method][label_name]))
            logger.info('        Mean class iou (multi-class slices) ({}): {}'.format(method, methods_class_multi_iou[method][label_name]))
            logger.info('        Mean class Dice (multi-class slices) ({}): {}'.format(method, methods_class_multi_dice[method][label_name]))
        logger.info('  *-----*')
        for method in methods:
            logger.info('        Mean class MSE ({}): {}'.format(method, methods_class_MSE[method][label_name]))
            logger.info('        Mean class MSE (multi-class slices) ({}): {}'.format(method, methods_class_multi_MSE[method][label_name]))
        logger.info('  *--------------------------------------------------*')

    logger.info('  ')
    # Log final results.
    logger.info('  *-----------------Final results--------------------*')
    logger.info('  *--------------------------------------------------*')

    logger.info('  ---Accuracy:')
    for method in methods:
        logger.info('      Mean IoU ({}): {}'.format(method, methods_class_iou[method]))
        logger.info('      Mean Dice ({}): {}'.format(method, methods_class_dice[method]))
    logger.info('  *-----*')
    for method in methods:
        logger.info('      Mean MSE ({}): {}'.format(method, methods_class_MSE[method]))
    logger.info('  *--------------------------------------------------*')

    logger.info('  ')
    logger.info('  ---Accuracy:')
    for method in methods:
        logger.info('      Mean IoU (multi-class slices) ({}): {}'.format(method, methods_class_multi_iou[method]))
        logger.info('      Mean Dice (multi-class slices) ({}): {}'.format(method, methods_class_multi_dice[method]))
    logger.info('  *-----*')
    for method in methods:
        logger.info('      Mean MSE (multi-class slices) ({}): {}'.format(method, methods_class_multi_MSE[method]))
    logger.info('  *--------------------------------------------------*\n\n\n')

    for method in methods:
        results_sv[args.dataset][args.train_name]['fold' + str(args.fold)]['run' + str(args.run)][method]['iou'] = \
            methods_class_iou[method]
        results_sv[args.dataset][args.train_name]['fold' + str(args.fold)]['run' + str(args.run)][method]['dice'] = \
            methods_class_dice[method]
        results_sv[args.dataset][args.train_name]['fold' + str(args.fold)]['run' + str(args.run)][method]['MSE'] = \
            methods_class_MSE[method]

    with open(results_sv_path, "wb") as f:
        pickle.dump(results_sv, f)



def infer(model, query_loader, support_sample, args, logger, labels):

    # Test mode.
    model.eval()

    support_image = [support_sample[i]['image'].float().cuda() for i in range(len(support_sample))]  #n_shot x [H x W]
    support_fg_mask = [support_sample[i]['label'].float().cuda() for i in range(len(support_sample))]  #n_shot x [n_class x  H x W]
    #support_image_shape=model.get_shape(support_image)
    #support_fg_mask_shape = model.get_shape(support_fg_mask)

    # Init recording metrics for query volumes.
    scores = {}

    q_results_dict = {method: {metric: {cls: [] for cls in model.classes}
                               for metric in model.metrics} for method in model.methods}

    for label_val, label_name in labels.items():
        # Skip BG class.
        if label_name is 'BG':
            continue
        scores[label_name] = Scores()


    # Unpack support data.
    support_spr = [support_sample[i]['sprvxl'].float().cuda() for i in range(len(support_sample))]

    support_idx= [support_sample[i]['ind'] for i in range(len(support_sample))]
    if len(support_sample)!=1:
        raise

    # Loop through query volumes.
    for i, sample in enumerate(query_loader):

        # Unpack query data.
        query_image = [sample['image'][j].float().cuda() for j in range(sample['image'].shape[0])]  # [C x 3 x H x W]# 1 x [n_slides x 3 x H x W]
        query_label = sample['label'].long()[0]  #[C x H x W]? #[n_slides x n_class x H x W]?


        with torch.no_grad():
            (query_pred, _, query_labels2, output_dict) \
                = model([support_image],[support_fg_mask],query_image,train=False, i=i,qry_mask=query_label,
                    supp_spr=support_spr[0][0],supp_idx=support_idx[0])  # C x 2 x H x W

        for method in model.methods:
            tmp_query_pred_ = output_dict[method]
            nan_mask = torch.isnan(tmp_query_pred_).all()
            tmp_query_pred = tmp_query_pred_.argmax(dim=1)

            preval = torch.mean(query_label.float().cuda(), dim=[2, 3])

            for i_c, cls in enumerate(model.classes):
                tp = torch.sum((query_label[i_c] == 1).cuda() * (tmp_query_pred == i_c)).cpu()
                fp = torch.sum((query_label[i_c] == 0).cuda() * (tmp_query_pred == i_c)).cpu()
                fn = torch.sum((query_label[i_c] == 1).cuda() * (tmp_query_pred != i_c)).cpu()
                if nan_mask:
                    tp =np.nan*tp
                tmp_iou = tp / (tp + fp + fn)
                q_results_dict[method]['iou'][cls].append(tmp_iou)
                tmp_dice = 2 * tp / (2 * tp + fp + fn)
                q_results_dict[method]['dice'][cls].append(tmp_dice)

                pred_pos = torch.mean((tmp_query_pred== i_c).float(), dim=[1, 2])
                tmp_MSE = torch.mean((preval[i_c] - pred_pos) ** 2).cpu()
                if nan_mask:
                    tmp_MSE =np.nan*tmp_MSE
                q_results_dict[method]['MSE'][cls].append(tmp_MSE)

        for label_val, label_name in labels.items():
            # Skip BG class.
            if label_name is 'BG':
                continue

            # Record scores.
            scores[label_name].record(torch.argmax(query_pred,dim=1)[0]== label_val, query_labels2== label_val)

    return scores, q_results_dict

if __name__ == '__main__':
    main()
