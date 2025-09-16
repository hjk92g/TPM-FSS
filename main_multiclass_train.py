#!/usr/bin/env python

import argparse
import time
import random

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.optim.lr_scheduler import MultiStepLR

from models.fewshot_multiclass_anom import FewShotSeg ### Major functions (forward, negSim, getPrototype, ...)
from dataloading.datasets_multiclass import TrainDataset as TrainDataset
from utils import * ### Minor functions (logger, evaluation metrics: Dice, IOU scores, ...)
import numpy as np
import inspect
import wandb, sys

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--save_root', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--n_sv', type=int, required=True) # supervoxel parameter
    parser.add_argument('--fold', type=int, required=True)

    # Training specs.
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--steps', default=50000, type=int) #Iterations
    parser.add_argument('--n_shot', default=1, type=int)
    parser.add_argument('--n_query', default=1, type=int)
    parser.add_argument('--n_way', default=1, type=int)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--non_succesive', action='store_true') #Allow non succesive slices
    parser.add_argument('--max_iterations', default=1000, type=int)
    parser.add_argument('--max_protos', default=5, type=int) # Maximum number of prototypes for multi-class training
    parser.add_argument('--lr', default=1e-3, type=float) # Learning rate
    parser.add_argument('--lr_gamma', default=0.95, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=0.0005, type=float)
    parser.add_argument('--learn_T_D', action='store_true') #Make it the same as adnet but different parameterization (learn T_S and T_D with CE loss)
    parser.add_argument('--t_loss_scaler', default=0.0, type=float) # T loss weight. Set it 0 for TPM
    parser.add_argument('--fix_T_D', default=None, type=float)  # Train to fix T_D value
    parser.add_argument('--chg_T_D', action='store_true')  # Train to fit decision boundary to T_D_hat value
    parser.add_argument('--chg_T_D_multi', action='store_true')
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--bg_wt', default=0.1, type=float) # Background class weight to address class imbalance (Foreground class weight=1)
    parser.add_argument('--fix_sig2', default=None, type=float) # Fix sig2 during training
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
    parser.add_argument('--EMA_p_F', default=True, action='store_true') # EMA (exponential moving average) estimation of foreground class prior

    return parser.parse_args()


def main():
    sys.stdout = open('TPM_multi_train_logs.txt', 'w')
    args = parse_arguments()

    wandb.login(key='**********') #Your Weights & Biases key
    wandb.init(
        # Set the project where this run will be logged
        project="TPM",
        # We pass a run name (otherwise it’ll be randomly assigned)
        name='TPM_multi_train_%dfd_%dsd' % (args.fold, args.seed),
        # Track hyperparameters and run metadata
        config={
            "architecture": 'TPM',
            "dataset": args.dataset,
            "n_shot": args.n_shot,
            "n_query": args.n_query,
            "n_way": args.n_way,
            "fold": args.fold,
            "seed": args.seed,
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
            "EMA_T_D_hat": args.EMA_T_D_hat,
            "EMA_alpha_hat": args.EMA_alpha_hat,
            "EMA_sig": args.EMA_sig,
            "EMA_p_F": args.EMA_p_F,
        })

    # Deterministic setting for reproducability, but it there can be randomness
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # Set up logging.
    logger = set_logger(args.save_root, 'train.log')
    logger.info(args)

    # Setup the path to save.
    args.save_model_path = os.path.join(args.save_root, 'model.pth')

    # Init model.
    model_ = FewShotSeg(False, t_loss_scaler=args.t_loss_scaler,
                        fix_alpha=args.fix_alpha, chg_alpha=args.chg_alpha, chg_alpha2=args.chg_alpha2,
                        chg_alpha_multi=args.chg_alpha_multi,learn_alpha=args.learn_alpha,
                        fix_sig2=args.fix_sig2,
                        max_protos=args.max_protos,
                        learn_T_D=args.learn_T_D, fix_T_D=args.fix_T_D, chg_T_D=args.chg_T_D,
                        fix_p_F=args.fix_p_F, chg_p_F=args.chg_p_F, learn_p_F=args.learn_p_F,
                        EMA_T_D_hat=args.EMA_T_D_hat, EMA_alpha_hat=args.EMA_alpha_hat,
                        EMA_sig=args.EMA_sig, EMA_p_F=args.EMA_p_F, lr=args.lr)
    model = nn.DataParallel(model_.cuda())

    # Init optimizer.
    model.parameters = [param for nm, param in model.named_parameters()]
    model.sig1 = model_.sig1
    model.sig2 = model_.sig2
    model.T_S = model_.T_S
    model.T_D = model_.T_D
    model.p_F = model_.p_F
    model.dim_ = model_.dim_
    if args.learn_T_D:
        model.pre_T_S = model_.pre_T_S
    model.alpha = model_.alpha

    optimizer = torch.optim.SGD(model.parameters, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    milestones = [(ii + 1) * 1000 for ii in range(args.steps // 1000 - 1)]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_gamma) ### pytorch: Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones.

    # Define loss function.
    my_weight = torch.FloatTensor(args.max_protos*[1.0]+ [args.bg_wt]).cuda()
    criterion = nn.NLLLoss(ignore_index=255, weight=my_weight)

    # Enable cuDNN benchmark mode to select the fastest convolution algorithm.
    cudnn.enabled = True
    cudnn.benchmark = True

    # Define data set and loader.
    train_dataset = TrainDataset(args)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True, drop_last=True)

    logger.info('  Training on images not in test fold: ' +
                str([elem[len(args.data_root):] for elem in train_dataset.image_dirs]))

    # Start training.
    sub_epochs = args.steps // args.max_iterations
    logger.info('  Start training ...')

    for epoch in range(sub_epochs):

        print('Epoch:', epoch)

        # Train.
        batch_time, data_time, losses, q_loss, align_loss, t_loss = train(
            train_loader, model, criterion, optimizer, scheduler, args)

        # Log
        logger.info('============== Epoch [{}] =============='.format(epoch))
        logger.info('  Batch time: {:6.3f}'.format(batch_time))
        logger.info('  Loading time: {:6.3f}'.format(data_time))
        logger.info('  Total Loss  : {:.5f}'.format(losses))
        logger.info('  Query Loss  : {:.5f}'.format(q_loss))
        logger.info('  Align Loss  : {:.5f}'.format(align_loss))

        if args.learn_T_D:
            logger.info('  t Loss  : {:.5f}'.format(t_loss))
            wandb.log({'t_loss': t_loss})

        wandb.log({'epoch': epoch})
        wandb.log({'losses': losses, 'q_loss': q_loss, 'align_loss': align_loss})

        alpha = (model_.alpha.detach().cpu().numpy())[0]
        logger.info('  alpha: {:.5f}'.format(alpha))
        T_S = (model_.T_S.detach().cpu().numpy())[0]
        logger.info('  T_S: {:.5f}'.format(T_S))  # Anomaly score threshold (T_S)
        T_D  = (model_.T_D.detach().cpu().numpy())[0]
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
        logger.info('  p_F_hat : {:.5f}, '.format(p_F_hat))

        sys.stdout.flush()
        time.sleep(0.01)

    # Save trained model.
    logger.info('  Saving model ...')
    torch.save(model.state_dict(), args.save_model_path)
    wandb.finish()

    # Restore stdout and close the file
    sys.stdout.close()
    sys.stdout = sys.__stdout__


def train(train_loader, model, criterion, optimizer, scheduler, args):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    q_loss = AverageMeter('Query loss', ':.4f')
    a_loss = AverageMeter('Align loss', ':.4f')
    t_loss = AverageMeter('Threshold loss', ':.4f')

    # Train mode.
    model.train()

    end = time.time()
    for i, sample in enumerate(train_loader):

        # Extract episode data.
        support_images = [[shot.float().cuda() for shot in way]
                          for way in sample['support_images']]
        support_fg_mask = [[shot.float().cuda() for shot in way]
                           for way in sample['support_fg_labels']]

        query_images = [query_image.float().cuda() for query_image in sample['query_images']]
        query_labels = torch.cat([query_label.long().cuda() for query_label in sample['query_labels']], dim=0)

        # Log loading time.
        data_time.update(time.time() - end)

        if i%100==(100-1):
            print('i:',i)

        # Compute outputs and losses.
        (query_pred, align_loss, thresh_loss, query_labels2) = model(support_images, support_fg_mask, query_images, train=True, i=i,qry_mask=query_labels)  # FewShotSeg()

        query_log_prob = torch.log(torch.clamp(query_pred, torch.finfo(torch.float32).eps,1 - torch.finfo(torch.float32).eps))

        if (len(query_labels)<args.max_protos): #Temporary change criterion
            my_weight = torch.FloatTensor(len(query_labels) * [1.0] + [args.bg_wt]).cuda()
            criterion = nn.NLLLoss(ignore_index=255, weight=my_weight)

        query_loss = criterion(query_log_prob, query_labels2)

        if (len(query_labels)<args.max_protos): #Recover original criterion
            my_weight = torch.FloatTensor(args.max_protos * [1.0] + [args.bg_wt]).cuda()
            criterion = nn.NLLLoss(ignore_index=255, weight=my_weight)

        # query_loss: By comparing query_pred and query_labels (supp_prototypes: based on support set)

        if args.learn_T_D:
            loss = query_loss + align_loss + thresh_loss
        else:
            loss = query_loss + align_loss

        # compute gradient and do SGD step
        for param in model.parameters:
            param.grad = None

        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters, 1.0)
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(torch.min(param.grad))|torch.isnan(torch.max(param.grad)):
                    print('Parameter: ', name, param.grad.shape, torch.min(param.grad), torch.max(param.grad))
                    param.grad = torch.zeros(param.grad)
        optimizer.step()
        scheduler.step()

        # Log loss.
        losses.update(loss.item(), query_pred.size(0))
        q_loss.update(query_loss.item(), query_pred.size(0))
        a_loss.update(align_loss.item(), query_pred.size(0))
        if args.learn_T_D:
            t_loss.update(thresh_loss.item(), query_pred.size(0))

        # Log elapsed time.
        batch_time.update(time.time() - end)
        end = time.time()

    return batch_time.avg, data_time.avg, losses.avg, q_loss.avg, a_loss.avg, t_loss.avg

if __name__ == '__main__':
    main()

