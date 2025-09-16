import torch
import random


def get_label_names(dataset):
    label_names = {}
    if dataset == 'CHAOST2':
        label_names[0] = 'BG'
        label_names[1] = 'LIVER'
        label_names[2] = 'RK'
        label_names[3] = 'LK'
        label_names[4] = 'SPLEEN'

    elif dataset == 'SABS':
        label_names[0] = 'BG'
        label_names[6] = 'LIVER'
        label_names[2] = 'RK'
        label_names[3] = 'LK'
        label_names[1] = 'SPLEEN'

    return label_names


def get_folds(dataset):
    FOLD = {}
    if dataset == 'CHAOST2':
        FOLD[0] = set(range(0, 5))
        FOLD[1] = set(range(4, 9))
        FOLD[2] = set(range(8, 13))
        FOLD[3] = set(range(12, 17))
        FOLD[4] = set(range(16, 20))
        FOLD[4].update([0])
        return FOLD

    elif dataset == 'SABS':
        FOLD[0] = set(range(0, 6))
        FOLD[1] = set(range(6, 12))
        FOLD[2] = set(range(12, 18))
        FOLD[3] = set(range(18, 24))
        FOLD[4] = set(range(24, 30))
        return FOLD

    else:
        raise ValueError(f'Dataset: {dataset} not found')


def get_label_values(dataset):
    label_values = {}
    if dataset == 'CHAOST2':
        label_values['BG'] = 0
        label_values['LIVER'] = 1
        label_values['RK'] = 2
        label_values['LK'] = 3
        label_values['SPLEEN'] = 4

    elif dataset == 'SABS':
        label_values['BG'] = 0
        label_values['LIVER'] = 6
        label_values['RK'] = 2
        label_values['LK'] = 3
        label_values['SPLEEN'] = 1

    return label_values


def sample_xy(spr, k=0, b=215):

    _, h, v = torch.where(spr)

    if len(h) == 0 or len(v) == 0:
        horizontal = 0
        vertical = 0
    else:

        h_min = min(h)
        h_max = max(h)
        if b > (h_max - h_min):
            kk = min(k, int((h_max - h_min) / 2))
            horizontal = random.randint(max(h_max - b - kk, 0), min(h_min + kk, 256 - b - 1))
        else:
            kk = min(k, int(b / 2))
            horizontal = random.randint(max(h_min - kk, 0), min(h_max - b + kk, 256 - b - 1))

        v_min = min(v)
        v_max = max(v)
        if b > (v_max - v_min):
            kk = min(k, int((v_max - v_min) / 2))
            vertical = random.randint(max(v_max - b - kk, 0), min(v_min + kk, 256 - b - 1))
        else:
            kk = min(k, int(b / 2))
            vertical = random.randint(max(v_min - kk, 0), min(v_max - b + kk, 256 - b - 1))

    return horizontal, vertical
