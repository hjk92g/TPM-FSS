import torch
from torch.utils.data import Dataset
import torchvision.transforms as deftfx
import glob
import os
import SimpleITK as sitk
import random
import numpy as np
from . import image_transforms as myit
from .dataset_specifics import *
import itertools
import sys, time


class TestDataset(Dataset):

    def __init__(self, args, standardize=True, binarize=True, only_multi=False):

        # reading the paths
        self.binarize = binarize
        self.data_root = args.data_root
        self.dataset = args.dataset
        self.only_multi = only_multi
        if args.dataset == 'CHAOST2':
            self.image_dirs = glob.glob(os.path.join(args.data_root, 'chaos_MR_T2_normalized/image*'))
        elif args.dataset == 'SABS':
            self.image_dirs = glob.glob(os.path.join(args.data_root, 'sabs_CT_normalized/image*'))
        self.image_dirs = sorted(self.image_dirs, key=lambda x: int(x.split('_')[-1].split('.nii.gz')[0]))
        self.supp_slice=args.supp_slice
        print('self.supp_slice',self.supp_slice)

        # remove test fold!
        self.FOLD = get_folds(args.dataset)
        self.image_dirs = [elem for idx, elem in enumerate(self.image_dirs) if idx in self.FOLD[args.fold]]

        # split into support/query
        self.support_dir = self.image_dirs[-1]
        self.support_combinaitons = list(itertools.combinations(self.image_dirs, args.n_shot))
        self.image_dirs = self.image_dirs[:-1]  # remove support
        self.label = None
        self.perm = None
        self.n_shot = args.n_shot

        if args.n_sv.__class__ is list:
            self.n_sv = args.n_sv
        else:
            self.n_sv = [str(args.n_sv)]

        # standardize before/after loading
        self.standardize = standardize
        self.mu = None
        self.sigma = None

    def __len__(self):
        return len(self.image_dirs) - self.n_shot

    def __getitem__(self, idx):

        img_path = self.image_dirs[idx]
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        img = (img - img.mean()) / img.std()
        img = np.stack(3 * [img], axis=1)


        sprs = []
        for n_sv in self.n_sv:
            spr_path = os.path.join(self.data_root, 'supervoxels_' + n_sv,
                                    'superpix-MIDDLE_' + img_path.split('image_')[1])
            sprs.append(torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(spr_path))))
        spr = torch.stack(sprs, dim=0)

        lbl = sitk.GetArrayFromImage(
            sitk.ReadImage(img_path.split('image_')[0] + 'label_' + img_path.split('image_')[-1]))

        if self.dataset == 'SABS':
            lbl_ = np.zeros(lbl.shape)
            lbl_[lbl == 1] = 1
            lbl_[lbl == 2] = 2
            lbl_[lbl == 3] = 3
            lbl_[lbl == 6] = 4
            lbl = lbl_


        if self.binarize:
            lbl = 1 * (lbl == self.label)

        # sample class(es) (label)
        unique = list(np.unique(lbl))
        unique.remove(0)

        if self.only_multi:
            unq_len_list = []
            for i in range(len(lbl)):
                tmp_unique = np.unique(lbl[i])
                unq_len_list.append(len(tmp_unique))
            unq_len_np = np.array(unq_len_list)
            multi_inds=np.where(unq_len_np == np.max(unq_len_np))[0]
            print('   multi_inds', multi_inds)

        lbl_l = [lbl == unq for unq in unique]  # n_protos x [n_slides x H x W]
        lbl = np.stack(lbl_l, axis=0)

        if self.only_multi:
            sample = {'id': img_path, 'image': torch.from_numpy(img[multi_inds]), 'label': torch.from_numpy(lbl[:,multi_inds]), 'sprvxl': spr[:,multi_inds]}
        else:
            sample = {'id': img_path, 'image': torch.from_numpy(img), 'label': torch.from_numpy(lbl), 'sprvxl': spr}

        return sample

    def get_support_index(self, n_shot, C):
        """
        Selecting intervals according to ALPNet (Ouyang et al.)
        """
        if n_shot == 1:
            pcts = [0.5]
        else:
            half_part = 1 / (n_shot * 2)
            part_interval = (1.0 - 1.0 / n_shot) / (n_shot - 1)
            pcts = [half_part + part_interval * ii for ii in range(n_shot)]

        return (np.array(pcts) * C).astype('int')

    def getSupport(self, label=None):
        if label is None:
            raise ValueError('Need to specify label class!')

        samples = []
        for img_path in list(self.support_combinaitons[self.perm]):
            img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
            if self.standardize:
                if self.dataset == 'SABS':
                    img = (img - img.mean()) / img.std()
                else:
                    img = (img - img.mean()) / img.std()
            img = np.stack(3 * [img], axis=0)

            lbl = sitk.GetArrayFromImage(
                sitk.ReadImage(img_path.split('image_')[0] + 'label_' + img_path.split('image_')[-1]))

            if self.dataset == 'SABS':
                lbl_ = np.zeros(lbl.shape)
                lbl_[lbl == 1] = 1
                lbl_[lbl == 2] = 2
                lbl_[lbl == 3] = 3
                lbl_[lbl == 6] = 4
                lbl = lbl_

            if self.binarize:
                lbl = 1 * (lbl == label)

            # sample class(es) (label)
            unique = list(np.unique(lbl))
            unique.remove(0)

            unq_len_list = []
            for i in range(len(lbl)):
                tmp_unique = np.unique(lbl[i])
                unq_len_list.append(len(tmp_unique))
            unq_len_np=np.array(unq_len_list)
            max_inds=np.where(unq_len_np==np.max(unq_len_np))[0]

            if self.supp_slice=='bal':
                CE_list = []
                for i in max_inds:
                    tmp_unique = np.unique(lbl[i], return_counts=True)
                    tmp_prior = tmp_unique[1] / np.sum(tmp_unique[1])
                    tmp_CE = -np.sum(np.log(tmp_prior)) / len(tmp_prior)
                    CE_list.append(tmp_CE)
                ind = max_inds[np.nanargmin(CE_list)]
            elif self.supp_slice=='rand':
                ind = random.choice(max_inds)
            else:
                print('Unknown supp_slice!!!')
                raise

            img=img[:,ind] #[3xHxW]
            lbl=lbl[ind]#[HxW]


            sprs = []
            for n_sv in self.n_sv:
                spr_path = os.path.join(self.data_root, 'supervoxels_' + n_sv,
                                        'superpix-MIDDLE_' + img_path.split('image_')[1])
                sprs.append(torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(spr_path))))
            spr = torch.stack(sprs, dim=0)

            lbl_l = [lbl==unq for unq in unique]#n_protos x [H x W]
            lbl = np.stack(lbl_l,axis=0)

            sample = {'image': torch.from_numpy(img)[None],
                      'label': torch.from_numpy(lbl),
                      'sprvxl': spr,
                      'ind': ind}

            samples.append(sample)

        return samples

class TrainDataset(Dataset):

    def __init__(self, args):
        self.n_shot = args.n_shot
        self.n_way = args.n_way
        self.n_query = args.n_query
        self.n_sv = args.n_sv
        self.max_iter = args.max_iterations
        self.max_protos = args.max_protos
        self.read = True  # read images before get_item
        self.train_sampling = 'neighbors'
        self.min_size = 200

        # reading the paths (leaving the reading of images into memory to __getitem__)
        if args.dataset == 'CHAOST2':
            self.image_dirs = glob.glob(os.path.join(args.data_root, 'chaos_MR_T2_normalized/image*'))
        elif args.dataset == 'SABS':
            self.image_dirs = glob.glob(os.path.join(args.data_root, 'sabs_CT_normalized/image*'))
        self.image_dirs = sorted(self.image_dirs, key=lambda x: int(x.split('_')[-1].split('.nii.gz')[0]))
        self.sprvxl_dirs = glob.glob(os.path.join(args.data_root, 'supervoxels_' + str(args.n_sv), 'super*'))
        self.sprvxl_dirs = sorted(self.sprvxl_dirs, key=lambda x: int(x.split('_')[-1].split('.nii.gz')[0]))

        # remove test fold!
        self.FOLD = get_folds(args.dataset)
        self.image_dirs = [elem for idx, elem in enumerate(self.image_dirs) if idx not in self.FOLD[args.fold]]
        self.sprvxl_dirs = [elem for idx, elem in enumerate(self.sprvxl_dirs) if idx not in self.FOLD[args.fold]]

        # read images
        if self.read:
            self.images = {}
            self.sprvxls = {}
            for image_dir, sprvxl_dir in zip(self.image_dirs, self.sprvxl_dirs):
                self.images[image_dir] = sitk.GetArrayFromImage(sitk.ReadImage(image_dir))
                self.sprvxls[sprvxl_dir] = sitk.GetArrayFromImage(sitk.ReadImage(sprvxl_dir))

    def __len__(self):
        return self.max_iter

    def gamma_tansform(self, img):
        gamma_range = (0.5, 1.5)
        gamma = np.random.rand() * (gamma_range[1] - gamma_range[0]) + gamma_range[0]
        cmin = img.min()
        irange = (img.max() - cmin + 1e-5)

        img = img - cmin + 1e-5
        img = irange * np.power(img * 1.0 / irange, gamma)
        img = img + cmin

        return img

    def geom_transform(self, img, mask):

        affine = {'rotate': 5, 'shift': (5, 5), 'shear': 5, 'scale': (0.9, 1.2)}
        alpha = 10
        sigma = 5
        order = 3

        tfx = []
        tfx.append(myit.RandomAffine(affine.get('rotate'),
                                     affine.get('shift'),
                                     affine.get('shear'),
                                     affine.get('scale'),
                                     affine.get('scale_iso', True),
                                     order=order))
        tfx.append(myit.ElasticTransform(alpha, sigma))
        transform = deftfx.Compose(tfx)

        img_ = img.copy()
        mask_ = mask.copy()



        if len(img.shape) > 4:
            n_shot = img_.shape[1]
            for shot in range(n_shot):
                cat = np.concatenate((img[0, shot], mask[:, shot])).transpose(1, 2, 0)
                cat = transform(cat).transpose(2, 0, 1)
                img_[0, shot] = cat[:3, :, :]
                mask_[:, shot] = np.rint(cat[3:, :, :])
                if (np.sum(mask_) == 0):
                    print('No foreground!(datasets, geom_t)')
                    print('Pre geom_t:', np.sum(mask))
                    print('Post geom_t:', np.sum(mask_))

        else:
            for q in range(img.shape[0]):
                cat = np.concatenate((img[q], mask)).transpose(1, 2, 0)
                cat = transform(cat).transpose(2, 0, 1)
                img_[q] = cat[:3, :, :]
                mask_ = np.rint(cat[3:, :, :].squeeze())
                if (np.sum(mask_) == 0):
                    print('No foreground!(datasets, geom_t, q)')
                    print('Pre geom_t:', np.sum(mask))
                    print('Post geom_t:', np.sum(mask_))
        return img_, mask_

    def __getitem__(self, idx):

        # sample patient idx
        pat_idx = random.choice(range(len(self.image_dirs)))

        if self.read:
            # get image/supervoxel volume from dictionary
            img = self.images[self.image_dirs[pat_idx]]
            sprvxl = self.sprvxls[self.sprvxl_dirs[pat_idx]]
        else:
            # read image/supervoxel volume into memory
            img = sitk.GetArrayFromImage(sitk.ReadImage(self.image_dirs[pat_idx]))
            sprvxl = sitk.GetArrayFromImage(sitk.ReadImage(self.sprvxl_dirs[pat_idx]))

        # normalize
        img = (img - img.mean()) / img.std()

        # sample class(es) (supervoxel)
        unique = list(np.unique(sprvxl))
        unique.remove(0)
        if len(unique)<15:
            print('len(unique)', len(unique))
            print('unique', unique,'\n')

        if self.n_shot!=1:
            print('self.n_shot!=1')
            print(self.n_shot)

        if self.n_way!=1:
            print('self.n_way!=1')
            print(self.n_way)

        if self.n_query!=1:
            print('self.n_query!=1')
            print(self.n_query)

        if ((self.n_shot * self.n_way) + self.n_query)!=2:
            print('((self.n_shot * self.n_way) + self.n_query)!=2')
            print(((self.n_shot * self.n_way) + self.n_query))

        '''Select image slice with at least one foreground''' #(The same process as one foreground class code)
        size = 0
        while size < self.min_size:
            n_slices = (self.n_shot * self.n_way) + self.n_query - 1
            while n_slices < ((self.n_shot * self.n_way) + self.n_query):
                cls_idx = random.choice(unique)

                # extract slices containing the sampled class
                sli_idx = np.sum(sprvxl == cls_idx, axis=(1, 2)) > 20
                n_slices = np.sum(sli_idx)

            img_slices = img[sli_idx]
            sprvxl_slices = 1 * (sprvxl[sli_idx] == cls_idx)

            origin_idx = np.where(sli_idx)[0]

            # sample support and query slices
            i = random.choice(
                np.arange(n_slices - ((self.n_shot * self.n_way) + self.n_query) + 1))  # successive slices
            sample = np.arange(i, i + (self.n_shot * self.n_way) + self.n_query)

            size = np.sum(sprvxl_slices[sample[0]])

        # invert order
        if np.random.random(1) > 0.5:
            sample = sample[::-1]  # successive slices (inverted)

        sup_idx = origin_idx[sample[:self.n_shot * self.n_way]]
        qry_idx = origin_idx[sample[self.n_shot * self.n_way:]]

        unique.remove(cls_idx)
        sup_lbl = sprvxl_slices[sample[:self.n_shot * self.n_way]][None,]  # n_way * (n_shot * C) * H * W
        qry_lbl = sprvxl_slices[sample[self.n_shot * self.n_way:]]  # n_qry * C * H * W
        for _ in range(1,self.max_protos):
            cnt = 0
            sup_size=0
            while (sup_size < self.min_size)&(cnt<10):

                if len(unique)>=1:
                    cls_idx = random.choice(unique)
                    sup_size = np.sum(sprvxl[sup_idx][0] == cls_idx, axis=(0, 1))

                try:
                    unique.remove(cls_idx)
                except:
                    pass
                cnt += 1

            if sup_size>=self.min_size:
                tmp_sup_lbl = 1 * (sprvxl[sup_idx] == cls_idx)[None,]  # n_way * (n_shot * C) * H * W
                tmp_qry_lbl = 1 * (sprvxl[qry_idx] == cls_idx)  # n_qry * C * H * W

                sup_lbl = np.concatenate([sup_lbl, tmp_sup_lbl], axis=0)
                qry_lbl = np.concatenate([qry_lbl, tmp_qry_lbl], axis=0)
                ###n_protos: between 1 and max_protos

        pre_sup_lbl_cnt = np.sum(sup_lbl)
        pre_qry_lbl_cnt = np.sum(qry_lbl)

        if (np.sum(sup_lbl)==0)|(np.sum(qry_lbl)==0):
            print('No foreground!(datasets)')
            print(np.sum(sup_lbl),np.sum(qry_lbl))

        sup_img = img_slices[sample[:self.n_shot * self.n_way]][None,]  # n_way * (n_shot * C) * H * W
        sup_img = np.stack((sup_img, sup_img, sup_img), axis=2)
        qry_img = img_slices[sample[self.n_shot * self.n_way:]]  # n_qry * C * H * W
        qry_img = np.stack((qry_img, qry_img, qry_img), axis=1)

        # gamma transform
        if np.random.random(1) > 0.5:
            qry_img = self.gamma_tansform(qry_img)
        else:
            sup_img = self.gamma_tansform(sup_img)

        # geom transform
        cnt = 0
        if (np.random.random(1) > 0.5)&(np.sum(qry_lbl)>0):
            qry_lbl_cnt=0
            while (qry_lbl_cnt == 0)&(cnt<5): #Make sure foreground label pixel count is not zero.
                qry_img_, qry_lbl_ = self.geom_transform(qry_img, qry_lbl)
                qry_lbl_cnt = np.sum(qry_lbl_)
                cnt+=1
            qry_img, qry_lbl = qry_img_+0.0, qry_lbl_+0.0
        else:
            sup_lbl_cnt = 0
            while (sup_lbl_cnt == 0)&(cnt<15): #Make sure foreground label pixel count is not zero.
                sup_img_, sup_lbl_ = self.geom_transform(sup_img, sup_lbl)
                sup_lbl_cnt = np.sum(sup_lbl_)
                cnt += 1
            sup_img, sup_lbl = sup_img_+0.0, sup_lbl_+0.0
        if (np.sum(sup_lbl) == 0) | (np.sum(qry_lbl) == 0):
            print('No foreground!(datasets, geom_transform)')
            print('Pre geom_transform:',pre_sup_lbl_cnt,pre_qry_lbl_cnt)
            print('Post geom_transform:',np.sum(sup_lbl), np.sum(qry_lbl))
        if cnt>1:
            sys.stdout.flush()
            time.sleep(0.01)

        #Remove improper foreground labels
        chk = np.sum(sup_lbl, axis=(2, 3))<self.min_size
        chk = chk[:,0]
        if chk.any():
            print('    sup_lbl (pre-filter)', sup_lbl.shape, np.unique(sup_lbl), np.sum(sup_lbl, axis=(2, 3)).reshape(-1))
            sup_lbl = sup_lbl[np.logical_not(chk)]
            qry_lbl = qry_lbl[np.logical_not(chk)]
            print('    sup_lbl (post-filter)', sup_lbl.shape, np.unique(sup_lbl), np.sum(sup_lbl, axis=(2, 3)).reshape(-1),'\n')

        sup_lbl = sup_lbl.transpose(1,0,2,3)

        sample = {'support_images': sup_img,
                  'support_fg_labels': sup_lbl,
                  'support_idx': sample[0].item(),
                  'support_len': len(img),
                  'query_images': qry_img,
                  'query_labels': qry_lbl}

        return sample
