import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from .backbone.torchvision_backbones import TVDeeplabRes101Encoder
import scipy.optimize as optimize
from matplotlib import pyplot as plt
import wandb, sys, time, joblib
from scipy.stats import pearsonr, spearmanr, pointbiserialr
from scipy.ndimage import distance_transform_edt
from scipy.special import gamma, lambertw


#TPM: Tied Prototype Model
class FewShotSeg(nn.Module):

    def __init__(self, use_coco_init=True,
                 fix_alpha=None, chg_alpha=False, chg_alpha2=False, chg_alpha_multi=False, learn_alpha=False,
                 learn_T_D=False, t_loss_scaler=0.0,
                 fix_T_D=None, chg_T_D=False, chg_T_D_multi=False,
                 fix_p_F=None, chg_p_F=False, chg_p_F_multi=False, learn_p_F=False, one_F_train=False,
                 fix_sig2=None,
                 dim_=1, max_protos=None, pretrained_root=None,
                 EMA_T_D_hat=True, EMA_alpha_hat=True, EMA_sig=True, EMA_p_F=True, lr=1e-3):
        super().__init__()

        self.learn_T_D = learn_T_D #adnet in one fore. class
        # Encoder
        self.encoder = TVDeeplabRes101Encoder(use_coco_init) # 256 feature dimension
        self.device = torch.device('cuda')

        self.fix_sig2=fix_sig2
        self.fix_alpha = fix_alpha
        self.chg_alpha = chg_alpha
        self.chg_alpha2 = chg_alpha2
        self.chg_alpha_multi = chg_alpha_multi
        self.learn_alpha = learn_alpha
        self.max_protos = max_protos
        if (self.fix_alpha is not None)+self.chg_alpha+self.chg_alpha2+self.chg_alpha_multi+self.learn_alpha>1:
            print('Only one option for alpha!!!') #Either fix_alpha is None or chg_alpha is False. They cannot be both: fix_alpha!=None and chg_alpha==True.........
            raise

        self.fix_p_F = fix_p_F
        self.chg_p_F = chg_p_F
        self.chg_p_F_multi = chg_p_F_multi
        self.learn_p_F = learn_p_F
        if (self.fix_p_F is not None) + self.chg_p_F+ self.chg_p_F_multi + self.learn_p_F > 1:
            print('Only one option for p_F!!!')
            raise

        self.fix_T_D = fix_T_D
        self.chg_T_D = chg_T_D
        self.chg_T_D_multi = chg_T_D_multi
        if self.learn_T_D+(self.fix_T_D is not None)+self.chg_T_D+self.chg_T_D_multi>1:
            print('Only one option for T_D!!!')
            raise

        if ((self.fix_p_F is not None)|self.chg_p_F|self.chg_p_F_multi|self.learn_p_F)\
                &(self.learn_T_D|(self.fix_T_D is not None)|self.chg_T_D|self.chg_T_D_multi):
            if self.learn_p_F&self.learn_T_D:
                print('Note that both self.learn_p_F and self.learn_T_D options are used')
            else:
                print('Only use option for p_F or T_D!!!')
                raise

        self.t_loss_scaler=t_loss_scaler

        self.EMA_T_D_hat = EMA_T_D_hat
        self.EMA_alpha_hat=EMA_alpha_hat
        self.EMA_sig=EMA_sig
        self.EMA_p_F=EMA_p_F
        self.lr = lr
        self.softplus = nn.Softplus()
        self.softsign = nn.Softsign()
        self.dim_ = dim_ #d=1 in equations (while actually using 256 dimensional feature space)

        self.T_D_hat = Parameter(torch.Tensor([(2*(1-10/20.0))**0.5]).to(self.device),requires_grad=False)  # Estimate good classification threshold based on distance by considering single fore. class classification
        self.T_D_multi_hat = Parameter(torch.Tensor([(2*(1-10/20.0))**0.5]).to(self.device),requires_grad=False)  #Estimate good classification threshold based on distance by considering multi. fore. classes classification
        # Ignore "alpha_hat", "alpha_hat2", and "alpha_multi_hat". They are not used in the TPM code (may appear and explained in later work)
        self.alpha_hat = Parameter(torch.Tensor([20.0]).to(self.device),requires_grad=False)
        self.alpha_hat2 = Parameter(torch.Tensor([20.0]).to(self.device),requires_grad=False)
        self.alpha_multi_hat = Parameter(torch.Tensor([20.0]).to(self.device),requires_grad=False)
        self.p_F_hat = Parameter(torch.Tensor([0.5]).to(self.device),requires_grad=False) ##Estimate good p_F by considering single fore. class classification
        self.p_F_multi_hat = Parameter(torch.Tensor([0.5]).to(self.device), requires_grad=False)  ##Estimate good p_F by considering multi. fore. classes classification

        if self.fix_alpha is None:
            if self.learn_alpha:
                self.l_alpha = Parameter(torch.Tensor([np.log(20.0)]).to(self.device))
                self.alpha = torch.exp(self.l_alpha)
            else:
                if self.chg_alpha:#model alpha follows self.alpha_hat
                    self.alpha = self.alpha_hat.data.detach()
                elif self.chg_alpha2:
                    self.alpha = self.alpha_hat2.data.detach()
                elif self.chg_alpha_multi:
                    self.alpha = self.alpha_hat_multi.data.detach()
                else:
                    self.alpha = Parameter(torch.Tensor([20.0]).to(self.device), requires_grad=False)
        else:
            self.alpha = Parameter(torch.Tensor([self.fix_alpha]).to(self.device), requires_grad=False)
        self.alpha = self.alpha.to(self.device)

        if self.fix_p_F is None: #p_F options: without changing sigma values
            if self.learn_p_F:
                self.pre_p_F = Parameter(torch.Tensor([0.0]).to(self.device))
                self.p_F = self.softsign2(self.pre_p_F)
            else:
                if self.chg_p_F:#follows self.p_F_hat
                    self.p_F = self.p_F_hat.data.detach()
                elif self.chg_p_F_multi:#follows self.p_F_multi_hat
                    self.p_F = self.p_F_multi_hat.data.detach()
                else:
                    self.p_F = Parameter(torch.Tensor([0.5]).to(self.device), requires_grad=False)
        else:
            self.p_F = Parameter(torch.Tensor([self.fix_p_F]).to(self.device), requires_grad=False)
        self.p_F = self.p_F.to(self.device)
        self.p_B = 1 - self.p_F

        if self.fix_T_D is None: #T_D options: indirectly changes sigma values
            if self.learn_T_D: #adnet: self.learn_T_D
                pre_T_S_ = np.log(np.exp((-10+self.alpha.item()-2*np.log(self.p_F.item()/self.p_B.item()))/self.dim_) - 1) #9.99995459903963
                self.pre_T_S = Parameter(torch.Tensor([pre_T_S_]).to(self.device))  #Use softplus to avoid saturation
                self.T_S = self.dim_*self.softplus(self.pre_T_S)+2*torch.log(self.p_F/self.p_B)-self.alpha
                self.T_D = torch.sqrt(2 * (1 + self.T_S / self.alpha))
            else:
                if self.chg_T_D:
                    self.T_D = self.T_D_hat.data.detach()
                    self.T_S = self.alpha*(self.T_D**2/2-1)
                elif self.chg_T_D_multi:
                    self.T_D = self.T_D_multi_hat.data.detach()
                    self.T_S = self.alpha*(self.T_D**2/2-1)
                else:
                    self.T_S = Parameter(torch.Tensor([-10.0]).to(self.device), requires_grad=False)
                    self.T_D = torch.sqrt(2 * (1 + self.T_S / self.alpha))
        else:
            self.T_D = Parameter(torch.Tensor([self.fix_T_D]).to(self.device), requires_grad=False)
            self.T_S = self.alpha * (self.T_D ** 2 / 2 - 1)
        self.T_S = self.T_S.to(self.device)
        self.T_D = self.T_D.to(self.device)

        self.qry_p_F_thres = None

        sig_part = (self.p_F / self.p_B) ** (-2 / self.dim_)*torch.exp(self.alpha * self.T_D ** 2 / (2 * self.dim_))
        self.sig1_ = (2 / self.alpha * (1 - 1/sig_part)) ** 0.5
        self.sig2_ = (2 / self.alpha * (sig_part - 1)) ** 0.5
        self.sig1 = Parameter(torch.Tensor([self.sig1_]).to(self.device), requires_grad=False)  # For standard inference
        self.sig2 = Parameter(torch.Tensor([self.sig2_]).to(self.device), requires_grad=False)
        if self.fix_sig2 is None:
            pass
        else:
            if (self.fix_alpha is not None) | self.chg_alpha | self.chg_alpha2| self.chg_alpha_multi | self.learn_alpha:
                print('alpha options are prioritized over fixing sig2!!!')
                print('There should be no option selected for sig2!!!')
                raise
            else:
                self.sig2_ = torch.from_numpy(np.array(self.fix_sig2))
                self.sig2.data = self.sig2_.to(self.device)

                beta2 = -self.T_D ** 2 / (self.dim_ * self.sig2 ** 2)
                self.sig1_ = self.sig2 * (beta2 / (lambertw(float(beta2 * torch.exp(beta2)), -1).real)) ** 0.5
                self.sig1.data = self.sig1_
                self.alpha.data = 2 * (1 / self.sig1_ ** 2 - 1 / self.sig2.data ** 2)
                if self.p_F.item()!=0.5:
                    print('This is implemented only for self.p_F.item()==0.5!!!')
                    raise

        self.criterion = nn.NLLLoss()

        self.cnt=0
        self.softmax = nn.Softmax(dim=0)
        self.zero_tensor = torch.tensor([0.0]).to(self.device)
        self.color_list = ['r', 'g', 'b', 'c','m','y','k']
        self.methods = ['ADNet++', 'CE-T', 'AvgEst', 'LinEst', 'OCP']
        self.metrics = ['iou', 'dice', 'MSE']
        self.pretrained_root = pretrained_root
        self.one_F_train=one_F_train
        self.nan_chk=False

    def forward(self, supp_imgs, fore_mask, qry_imgs, train=False, analyze=False, i=None, qry_mask=None, supp_spr=None,
                supp_img_all=None, supp_idx=None):
        """
        Args:
            # B: batch_size
            # N: query size
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
                ->multiclass: way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
                ->multiclass: way x shot x [n_protos x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
                ->multiclass: way x shot x [n_protos x H x W], list of lists of tensors
            qry_imgs: query images
                N_q x [B_q x 3 x H x W], list of tensors
                ->multiclass: N_q x [B_q x 3 x H x W], list of tensors
            qry_mask: query mask
                training: [N_q x H x W] or inference: [N_q x B_q x H x W], tensor
                E.g. training: torch.Size([1, 256, 256]), inference: torch.Size([1, 34, 256, 256])
                ->multiclass: training: [n_protos x H x W] or inference: [n_protos x B_q x H x W], tensor
                E.g. training: torch.Size([5, 256, 256]), inference: torch.Size([5, 34, 256, 256])
        """

        if i<50:
            print('i:',i)
        n_ways = len(supp_imgs)
        self.n_ways = n_ways+0
        self.n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)
        batch_size_q = qry_imgs[0].shape[0]
        self.batch_size_q = batch_size_q+0
        batch_size = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:] # [H, W]

        self.n_protos = len(fore_mask[0][0])

        if self.n_ways!=1:
            print('    self.n_ways!=1')
            print('    self.n_ways',self.n_ways)

        if self.n_shots!=1:
            print('    self.n_shots!=1')
            print('    self.n_shots',self.n_shots)

        if batch_size!=1:
            print('    batch_size!=1')
            print('    batch_size',batch_size)
            print('    This code was written for batch_size==1!!!')
            raise

        if n_queries!=1:
            print('    n_queries!=1')
            print('    n_queries',n_queries)
            print('    This code was written for n_queries==1!!!')
            raise

        if self.batch_size_q!=1:
            print('    self.batch_size_q!=1')
            print('    self.batch_size_q',self.batch_size_q)

        if self.n_protos<self.max_protos:
            print('    self.n_protos<self.max_protos')
            print('    self.n_protos',self.n_protos)

        if self.n_protos!=len(qry_mask):
            print('    self.n_protos!=len(qry_mask)')
            print('    self.n_protos', self.n_protos, 'len(qry_mask)',len(qry_mask))
            print('    This code was written for self.n_protos==len(qry_mask)!!!')
            raise


        if qry_mask is not None:
            if train|analyze:
                qry_mask_ = qry_mask.unsqueeze(1)
            else:
                qry_mask_ = qry_mask+0.0
                #Multiclass: [n_protos x B_q x H x W]
                print('    qry_mask_', qry_mask_.shape)

        # ###### Extract features ######
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)
        img_fts = self.encoder(imgs_concat, low_level=False)

        fts_size = img_fts.shape[-2:] #[H' W']

        if train|analyze:
            img_fts_n = img_fts / torch.norm(img_fts, dim=1, keepdim=True)
            img_fts_n_reshaped = (img_fts_n.permute(0, 2, 3, 1)).reshape(-1, img_fts.shape[1])

            MRL = torch.norm(torch.mean(img_fts_n_reshaped, dim=0, keepdim=True), dim=1)  # Mean Resultant Length
            L_unif = torch.logsumexp(input=1.0 * torch.matmul(img_fts_n_reshaped, img_fts_n_reshaped.T), dim=(0, 1))
            # L_unif is from "Hubs and Hyperspheres: Reducing Hubness and Improving Transductive Few-shot Learning with Hyperspherical Embeddings"

            wandb.log({'Mean_Resultant_Length': MRL.item()})
            wandb.log({'L_unif': L_unif.item()})
        else:
            # Load ICP estimating linear models for LinEst
            if self.one_F_train: #For single foreground trained
                linr = joblib.load(self.pretrained_root + 'p_F_hats_model2.pkl')
            else: #For multi foreground classes trained
                linr = joblib.load(self.pretrained_root+'p_F_hats_model.pkl')
            p_F_dict= np.load(self.pretrained_root + "p_F_dict.npz")
            p_F_dict = dict(p_F_dict)

        supp_fts = img_fts[:self.n_ways * self.n_shots * batch_size].view(
            self.n_ways, self.n_shots, batch_size, -1, *fts_size)  # Wa x Sh x B x C x H' x W'
        # Wa: way, Sh: shot, C: channel (feature) dimension, H', W': height and width after applying CNN encoder
        qry_fts = img_fts[self.n_ways * self.n_shots * batch_size:].view(
            n_queries, batch_size_q, -1, *fts_size)  #Inference: N_q x B_q x C x H' x W'
        #N: N_queries, B: batch_size_q
        chk_pre = torch.sum(fore_mask[0][0], axis=(1, 2))==0
        if chk_pre.any():
            print('    No foreground(pre-stack)')
            print('    chk_pre:',chk_pre)
        fore_mask = torch.stack([torch.stack(way, dim=0) for way in fore_mask], dim=0)

        supp_cnts = torch.cat([torch.sum(fore_mask[:,:,i_p]).reshape(1) for i_p in range(self.n_protos)])

        wandb.log({'p_F_cnt': torch.sum(supp_cnts)/torch.prod(torch.tensor(fore_mask[:,:,0].shape))})

        chk_post = torch.sum(fore_mask[0][0], axis=(1, 2)) == 0
        if chk_post.any():
            print('    No foreground(post-stack)')
            print('    chk_post:', chk_post)
        #back_mask = 1-fore_mask

        if train:
            self.load_params(train=True,verbose=False) #Load self.T_S, self.T_D, self.alpha, ...
        else:
            self.load_params(train=False,verbose=False)

        ###### Compute loss ######
        align_loss = torch.zeros(1).to(self.device)
        outputs = []
        outputs_dict = {method: [] for method in self.methods}
        for epi in range(batch_size):
            ###### Extract prototypes ######
            supp_fts_ = [[[self.getFeatures(supp_fts[i_w, i_s, [epi]],
                                           fore_mask[i_w, i_s, [i_p]])
                           for i_p in range(self.n_protos)] for i_s in range(self.n_shots)] for i_w in range(self.n_ways)] ### getFeatures: get feature vectors using masked area
            fg_prototypes = self.getPrototype(supp_fts_, multiclass=True) ### getPrototype(): just average & not l2 normalized

            if ((i < 50)&train)|False:
                self.analyze_fts(supp_fts, fore_mask, fg_prototypes, epi, verbose= True, train=train, support=True)
                if train|analyze:
                    self.analyze_fts(qry_fts, qry_mask_, fg_prototypes, epi, verbose=True, train=train, support=False)
                else:
                    self.analyze_fts(qry_fts, qry_mask_, fg_prototypes, epi, verbose=True, train=train, support=False)
            else:
                if analyze:
                    self.analyze_fts(supp_fts, fore_mask, fg_prototypes, epi, verbose=False,train=True, support=True)
                    self.analyze_fts(qry_fts, qry_mask_, fg_prototypes, epi,verbose=False, train=True, support=False)
                else:
                    self.target_F_cnt = [[torch.sum(qry_mask_[:,q])
                                          for q in range(qry_mask_.shape[1])] for _ in range(self.n_ways)]
                    tmp_p_cnt=torch.mean(qry_mask_.float(),dim=[2,3])#[n_protos x B_q]
                    self.p_cnt_list = [[torch.cat((tmp_p_cnt[:,q],1-torch.sum(tmp_p_cnt[:,q],dim=-1,keepdim=True)),dim=-1).unsqueeze(0)
                                          for q in range(qry_mask_.shape[1])] for _ in range(self.n_ways)]

            sys.stdout.flush()
            torch.cuda.empty_cache()

            if train:
                self.T_D_hat_MSE = torch.mean((self.T_D_hat - self.qry_p_F_thres) ** 2)

            ###### Compute anom. scores ######
            cos_s = [[self.cosSim(qry_fts[epi], fg_prototypes[i_w][i_p])
                       for i_p in range(self.n_protos)] for i_w in range(self.n_ways)]

            ###### Get threshold (Legacy of ADNet) #######
            if self.learn_T_D:
                self.t_loss = self.T_S / self.alpha

            ###### Get predictions #######
            if train:
                pred_protos_ = self.getPred2(cos_s, multiclass=True, cls_prior='uniform',train=True) # N x Wa x H' x W' # Foreground class probability: 1-sigmoid((S(,)-T)/2)
            else:
                pred_protos_ = self.getPred2(cos_s, multiclass=True,cls_prior='uniform',train=False)  # N x Wa x H' x W' # Foreground class probability: 1-sigmoid((S(,)-T)/2)

            pred_protos = F.interpolate(pred_protos_, size=img_size, mode='bilinear', align_corners=True) # inference: [B_q x (n_protos+1)x H x W]
            pred_protos = pred_protos/torch.sum(pred_protos,dim=1,keepdim=True) #Make sure that class probabilities sum to 1
            #multiclass: [N_q? x (n_protos+1)x H x W]
            if (not train):
                pred_cnts= torch.bincount(torch.argmax(pred_protos,dim=1).flatten(),minlength=self.n_protos+1)
                print('    pred_cnts/torch.sum(pred_cnts):', pred_cnts/torch.sum(pred_cnts),'\n')

            if ((i < 50) & train) | (not train)&(not analyze):
                pred_uniform_protos_ = self.getPred2(cos_s, multiclass=True,cls_prior='uniform',train=False)  # N x Wa x H' x W' ### Foreground class probability: 1-sigmoid((S(,)-T)/2)
                pred_uniform_protos = F.interpolate(pred_uniform_protos_, size=img_size, mode='bilinear',align_corners=True)
                pred_uniform_protos = pred_uniform_protos / torch.sum(pred_uniform_protos,dim=1,keepdim=True)  ###Make sure that class probabilities sum to 1

                pred_p_F_protos_ = self.getPred2(cos_s, multiclass=True,cls_prior='qry_p_F',train=False)  # N x Wa x H' x W' ### Foreground class probability: 1-sigmoid((S(,)-T)/2)
                pred_p_F_protos = F.interpolate(pred_p_F_protos_, size=img_size, mode='bilinear', align_corners=True)
                pred_p_F_protos = pred_p_F_protos / torch.sum(pred_p_F_protos,dim=1,keepdim=True)  ###Make sure that class probabilities sum to 1

                pred_uniform_ADNetPP_ = self.getPred2(cos_s, multiclass=True,cls_prior='uniform',method='ADNet++',train=False)  # N x Wa x H' x W' ### Foreground class probability: 1-sigmoid((S(,)-T)/2)
                pred_uniform_ADNetPP = F.interpolate(pred_uniform_ADNetPP_, size=img_size, mode='bilinear',align_corners=True)
                pred_uniform_ADNetPP = pred_uniform_ADNetPP / torch.sum(pred_uniform_ADNetPP, dim=1,keepdim=True)  ###Make sure that class probabilities sum to 1

                outputs_dict['CE-T'].append(pred_uniform_protos)
                outputs_dict['OCP'].append(pred_p_F_protos) #OCP only considering total foreground count
                outputs_dict['ADNet++'].append(pred_uniform_ADNetPP)
                if not train:
                    p_F_multi_hat = torch.tensor(p_F_dict['p_F_hats_mn']).to(self.device)

                    pred_p_F_multi_hat_protos_ = self.getPred2(cos_s, multiclass=True, cls_prior=p_F_multi_hat,train=False)  # N x Wa x H' x W' ### Foreground class probability: 1-sigmoid((S(,)-T)/2)
                    pred_p_F_multi_hat_protos = F.interpolate(pred_p_F_multi_hat_protos_, size=img_size, mode='bilinear',align_corners=True)
                    pred_p_F_multi_hat_protos = pred_p_F_multi_hat_protos / torch.sum(pred_p_F_multi_hat_protos, dim=1,keepdim=True)  ###Make sure that class probabilities sum to 1
                    outputs_dict['AvgEst'].append(pred_p_F_multi_hat_protos)

                    ### Prediction using LinEst ###
                    supp_loc = supp_idx / (len(supp_spr) - 1)
                    supp_loc2 = (supp_loc - 0.5) ** 2
                    assert (len(fore_mask)==1)&(len(fore_mask[0])==1)

                    sizes = torch.mean(fore_mask[0][0], dim=[1, 2]).detach().cpu().numpy()
                    Xs = [np.concatenate([supp_loc2.reshape([-1, 1]), sizes[i_p].reshape([-1, 1])], axis=-1)
                          for i_p in range(self.n_protos)]  # n_protos x [1, 2]
                    FBs = [np.exp(linr.predict(Xs[i_p])) for i_p in range(self.n_protos)]  # n_protos x [n x 2]
                    Fs = [FBs[i_p][:, :1] for i_p in range(self.n_protos)]  # n_protos x [n x 1]
                    Fs = np.concatenate(Fs, axis=-1)  # [n x n_protos]
                    Bs = [FBs[i_p][:, 1:] for i_p in range(self.n_protos)]  # n_protos x [n x 1]
                    Bs = np.concatenate(Bs, axis=-1)  # [n x n_protos]
                    sum_B = np.sum(Bs, axis=-1, keepdims=True)  # [n,1]
                    pred_prob = np.concatenate((Fs, sum_B), axis=-1)
                    pred_prob_2 = pred_prob / np.sum(pred_prob, axis=-1, keepdims=True)  # [n,n_protos+1]
                    pred_F_pri = np.mean(pred_prob_2[:, :-1], axis=-1)  # [n]
                    pred_F = pred_F_pri / (1 - (self.n_protos - 1) * pred_F_pri)  # [n]

                    estim_p_F_hat = pred_F
                    estim_p_F_hat = np.clip(estim_p_F_hat, a_min=1e-3, a_max=2 - 1e-3)[0]
                    estim_p_F_hat = torch.tensor(estim_p_F_hat).to(self.device)

                    pred_estim_p_F_hat_protos_ = self.getPred2(cos_s, multiclass=True, cls_prior=estim_p_F_hat,train=False)  # N x Wa x H' x W' ### Foreground class probability: 1-sigmoid((S(,)-T)/2)
                    pred_estim_p_F_hat_protos = F.interpolate(pred_estim_p_F_hat_protos_, size=img_size, mode='bilinear',align_corners=True)
                    pred_estim_p_F_hat_protos = pred_estim_p_F_hat_protos / torch.sum(pred_estim_p_F_hat_protos, dim=1,keepdim=True)  ###Make sure that class probabilities sum to 1
                    outputs_dict['LinEst'].append(pred_estim_p_F_hat_protos)

            outputs.append(pred_protos)

            ###### Prototype alignment loss from "PANet" paper######
            if train:
                align_loss_epi = self.alignLoss(qry_fts[:, epi], pred_protos_,
                                                supp_fts[:, :, epi],
                                                fore_mask[:, :])
                align_loss += align_loss_epi

        output = torch.stack(outputs, dim=1)  # [N x B x (n_protos+1) x H x W] or Inference: [B_q x B x (n_protos+1) x H x W]

        output = output.view(-1, *output.shape[2:])
        if (not train)&(not analyze):
            output_dict = {method: torch.stack(outputs_dict[method], dim=1) for method in self.methods}
            output_dict = {method: output_dict[method].view(-1, *output.shape[1:]) for method in self.methods}
            print()

        query_pred_mask = torch.argmax(output, dim=1)  ###[(N * B) x H x W]
        qry_pred_cnts = torch.cat([torch.sum(query_pred_mask == i_p).reshape(1) for i_p in range(self.n_protos + 1)])

        wandb.log({'p_F_pred_cnt': 1-qry_pred_cnts[-1]/torch.sum(qry_pred_cnts)})

        if train:
            self.apply_EMA()

        qry_labels = torch.full_like(qry_mask[0], self.n_protos, device=img_fts.device)

        for i_p in range(self.n_protos):
            qry_labels[qry_mask[i_p] == 1] = i_p  # foreground

        if train:
            if self.learn_T_D:
                return output, (align_loss / batch_size), (self.t_loss_scaler * self.t_loss), qry_labels[None,:]
            else:
                return output, (align_loss / batch_size), np.nan, qry_labels[None,:]
        else:
            if analyze:
                ###### Get ICP values of training data ######
                tmp_sq_dists = [[2 * (1 - cos_s[i_w][i_p]) for i_p in range(self.n_protos)] for i_w in range(self.n_ways)]
                tmp_dists = [[(tmp_sq_dists[i_w][i_p] + torch.finfo(torch.float32).eps) ** 0.5
                                   for i_w in range(self.n_ways)] for i_p in range(self.n_protos)]
                tmp_dists = [F.interpolate(torch.stack(tmp_dists[i_p], dim=0), size=img_size, mode='bilinear',align_corners=True)
                                  for i_p in range(self.n_protos)]
                tmp_dists = [tmp_dists[i_p].flatten() for i_p in range(self.n_protos)]#5 x [65536]
                tmp_dists = torch.stack(tmp_dists,dim=0)#[5 x 65536]
                supp_loc = supp_idx / (len(supp_img_all) - 1)

                supp_size = torch.mean(fore_mask,dim=[0,1,3,4]).detach().cpu().numpy()

                sig1 = self.sig1.detach()
                sig2 = self.sig2.detach()

                h_a = torch.exp(-tmp_dists**2/(2*sig1 ** 2)) / sig1 ** self.dim_#[5 x 65536]
                h_b = torch.exp(-tmp_dists**2/(2*sig2 ** 2)) / sig2 ** self.dim_#[5 x 65536]
                h_a_max = torch.max(h_a, dim=0)[0]  ###[65536]
                H_b = torch.sum(h_b, dim=0)  ###[65536]
                loc_ratio_sort = torch.sort((H_b / (h_a_max + H_b)))[0]
                qry_F_tot = torch.sum(torch.max(qry_mask_,dim=0)[0])
                if int(qry_F_tot) > 0:
                    tmp_p_F_hat = loc_ratio_sort[int(qry_F_tot) - 1]
                else:
                    tmp_p_F_hat = torch.clip(2 * loc_ratio_sort[0] - loc_ratio_sort[1], min=0.0, max=1)
                tmp_p_F_hat = tmp_p_F_hat.detach()

                tmp_p_F_hat = tmp_p_F_hat.item()
                return [supp_loc, supp_size, tmp_p_F_hat]
            else:
                return (output, (align_loss / batch_size), qry_labels[None,:], output_dict)

    def getSqDistance(self, fts, prototype):
        """
        Calculate the (squared) distance between features and prototypes

        ### C: channel dimension
        Args:
            fts: input features
                expect shape: [1 x C x N_fg] or [1 x C x N_bg]
            prototype: prototype of one semantic class
                expect shape: [1 x C]
        """
        ### Normalized setting: ||X-Y||**2 = ||X||**2+||Y||**2-2*X*Y = 2-2*cosine_similarity(X,Y)
        
        sq_dist = 2*(1-F.cosine_similarity(fts, prototype[..., None], dim=1)) # Squared distance

        return sq_dist

    def negSim(self, fts, prototype):
        """
        Calculate the distance between features and prototypes

        ### C: channel dimension
        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """

        if self.learn_alpha: #Allow gradient flow
            sim = - F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.alpha
        else: #Make sure there is no gradient flow
            sim = - F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.alpha.detach()
        return sim

    def cosSim(self, fts, prototype):
        """
        Calculate the cosine similarity between features and prototypes

        ### C: channel dimension
        Args:
            fts: input features
                expect shape: N x C x H x W
                (Inference: B_q x C x H x W)
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        sim = F.cosine_similarity(fts, prototype[..., None, None], dim=1)

        return sim

    def getFeatures(self, fts, mask, MAP=True):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear', align_corners = False)

        if MAP:
            # masked fg features
            masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3)) \
                         / (mask[None, ...].sum(dim=(2, 3)) + 1e-5)  # 1 x C
            return masked_fts
        else:
            fts_reshape = fts.view(*fts.shape[:2],-1) ### 1 x C x (H*W)

            mask2 = mask.clone().view(-1)  ### (H*W)
            mask2 = mask2.bool()
            fg_fts = fts_reshape[:, :, mask2]
            return fg_fts ### 1 x C x N_fg

    def getPrototype(self, fg_fts, multiclass=False):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
                expect shape (multiclass=True): Wa x Sh x n_proto x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]

        Output:
            fg_prototypes
                expect shape: Wa x [1 x C]
                expect shape (multiclass=True): Wa x n_proto x [1 x C]
        """

        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        if multiclass:
            fg_fts_reshaped = [[[fg_fts[i_w][i_s][i_p] for i_s in range(n_shots)]
                    for i_p in range(self.n_protos)] for i_w in range(n_ways)]

            fg_prototypes = [[torch.sum(torch.cat([tr for tr in way[i_p]], dim=0), dim=0, keepdim=True) / n_shots
                             for i_p in range(self.n_protos)] for way in fg_fts_reshaped]
        else:
            fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots
                             for way in fg_fts]  # concat all fg_fts
        ### fg_prototypes: just averaged and not l2 normalized
        return fg_prototypes

    def alignLoss(self, qry_fts, pred, supp_fts, fore_mask):#PAR loss from "PANet" paper
        '''qry_fts[:, epi]: [N x C x H' x W']
        pred: [N_q? x (n_protos+1)x H' x W'] (background class: the last class)
        supp_fts[:, :, epi]: [Wa x Sh x C x H' x W']
        fore_mask[:, :]: [Wa x Sh x n_protos x H x W]'''
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        if n_ways!=1:
            print('The code is written for n_ways==1!')
            raise

        # Mask and get query prototype
        pred_mask_ = pred.argmax(dim=1, keepdim=True)  # N? x 1 x H' x W'
        binary_masks = [pred_mask_ == i_p for i_p in range(self.n_protos+1)] #(n_protos+1) x [N_q? x 1 x H' x W']
        nonskip_ips = [i_p for i_p in range(self.n_protos) if binary_masks[i_p].sum()> 0]  #Nonskipping mask index
        pred_mask = torch.stack(binary_masks, dim=1).float()  # N x (n_protos+1) x 1 x H' x W'
        qry_prototypes = torch.sum(qry_fts.unsqueeze(1) * pred_mask[:,:self.n_protos], dim=(0, 3, 4))# [N x n_protos x C x H' x W']->[n_protos x C]
        qry_prototypes = qry_prototypes / (pred_mask[:,:self.n_protos].sum((0, 3, 4)) + 1e-5)  # [n_protos x C]

        # Compute the support loss
        loss = torch.zeros(1).to(self.device)

        if len(nonskip_ips)>0:
            # Get the query prototypes
            for shot in range(n_shots):
                img_fts = supp_fts[0, [shot]]  #[1 x C x H' x W']

                supp_sim = [[self.cosSim(img_fts, qry_prototypes[i_p][None, :]) for i_p in range(self.n_protos) if i_p in nonskip_ips]]

                pred_protos_ = self.getPred2(supp_sim, multiclass=True,train=True)  # [N_q x (len(nonskip_ips)+1) x H' x W']

                pred_protos = F.interpolate(pred_protos_, size=fore_mask.shape[-2:], mode='bilinear',align_corners=True)  ### ???
                pred_protos = pred_protos / torch.sum(pred_protos,dim=1)  ###Make sure that class probabilities sum to 1
                # [N_q x (len(nonskip_ips)+1) x H x W]

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[0, shot, 0], len(nonskip_ips), device=img_fts.device)

                for i_, i_p in enumerate(nonskip_ips):
                    supp_label[fore_mask[0, shot, i_p] == 1] = i_  # foreground

                # Compute Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(pred_protos, eps, 1 - eps))
                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

        return loss

    def getPred2(self, sim, multiclass=True, cls_prior='param',method='TPM',alpha=None, train=True, p_T=None):
        # We actually use log version for TPM calculation for numerical stability
        '''
        Inputs:
            sim
                (multiclass=False): way x [1, H', W']
                (multiclass=True, train): way x n_protos x [1, H', W']
                (multiclass=True, inference): way x n_protos x [B_q, H', W']
            #thresh: way x [1]
            cls_prior: 'uniform', 'param', 'qry_p_F'
        Output:
            pred
                (multiclass=False): [N_q x Wa x H' x W']
                (multiclass=True, train): [N_q x n_protos x Wa x H' x W']???
                                    [1 x (n_protos+1) x H' x W']
                (multiclass=True, inference): [B_q x (n_protos+1) x H' x W']
        '''
        pred = []
        self.nan_chk=False
        if train:
            if self.learn_alpha|self.learn_T_D|self.learn_p_F:
                sig_part = (self.p_F / self.p_B) ** (-2 / self.dim_)*torch.exp((self.alpha + self.T_S) / self.dim_)
                sig1 = (2 / self.alpha * (1 - 1 / sig_part)) ** 0.5
                sig2 = (2 / self.alpha * (sig_part - 1)) ** 0.5
            else:
                sig1 = self.sig1+0.0
                sig2 = self.sig2+0.0
            L_sig1 = torch.log(sig1)
            L_sig2 = torch.log(sig2)
        else:
            sig1 = self.sig1.detach()
            sig2 = self.sig2.detach()
            L_sig1 = torch.log(sig1)
            L_sig2 = torch.log(sig2)
            if alpha is not None:
                T_S = -2*self.dim_ * torch.log(sig1/sig2) + 2 * torch.log(self.p_F / (1-self.p_F)) - self.alpha
                T_D = (2*(1+T_S/self.alpha))**0.5
                if isinstance(cls_prior, torch.Tensor):
                    if len(cls_prior.shape) <= 1:
                        T_D_hat =(2*(torch.log(cls_prior / (1-cls_prior))-self.dim_ * torch.log(sig1/sig2))/(1/sig1**2-1/sig2**2))**0.5
                        L_sig_part = alpha * T_D_hat ** 2 / (2 * self.dim_) - 2 / self.dim_ * torch.log((cls_prior / (1 - cls_prior)))
                    else:
                        L_sig_part = alpha * T_D ** 2 / (2 * self.dim_) - 2 / self.dim_ * torch.log((self.p_F / (1 - self.p_F)))
                else:
                    L_sig_part = alpha * T_D ** 2 / (2 * self.dim_) - 2 / self.dim_ * torch.log((self.p_F / (1 - self.p_F)))
                if L_sig_part>0:
                    L_sig1 = 0.5*(torch.log(2 / alpha)+torch.log1p(-torch.exp(-L_sig_part)))
                    L_sig2 = 0.5 * (torch.log(2 / alpha) + L_sig_part + torch.log1p(-torch.exp(-L_sig_part)))
                    sig1 = torch.exp(L_sig1)  # (2 / alpha * (1 - 1 / sig_part)) ** 0.5
                    sig2 = torch.exp(L_sig2)  # (2 / alpha * (sig_part - 1)) ** 0.5
                    if L_sig_part > 100:
                        print('    L_sig_part>100!!!')
                        print('    L_sig_part:', L_sig_part)
                        print('    sig1',sig1)
                        print('    sig2', sig2)
                        self.nan_chk = True
                else:
                    sig_part = torch.exp(L_sig_part)
                    sig1 = (2 / alpha * (1 - 1 / sig_part)) ** 0.5
                    sig2 = (2 / alpha * (sig_part - 1)) ** 0.5
                    L_sig1 = torch.log(sig1)
                    L_sig2 = torch.log(sig2)

        loc_n_protos = len(sim[0])#local_n_protos

        if multiclass:
            for s in sim:
                L_pred_F = []  # Foregrounds
                L_pred_B = []  # Backgrounds

                if (cls_prior=='qry_p_F')|(p_T is not None):
                    L_h_a_list = [(s[i_p] - 1) / sig1 ** 2 - self.dim_*L_sig1
                                  for i_p in range(loc_n_protos)]  #4 x [34, 32, 32]
                    L_h_b_list = [(s[i_p] - 1) / sig2 ** 2 - self.dim_*L_sig2
                                  for i_p in range(loc_n_protos)]  #4 x [34, 32, 32]
                    L_h_a_max = torch.max(torch.stack(L_h_a_list, dim=0), dim=0)[0]  #[34, 32, 32]
                    L_H_b = torch.logsumexp(torch.stack(L_h_b_list, dim=0), dim=0)  #[34, 32, 32]
                    L_h_a_max_H_b = torch.stack([L_h_a_max, L_H_b],dim=0) #[2, 34, 32, 32]
                    L_loc_ratio_sort = torch.sort((L_H_b - torch.logsumexp(L_h_a_max_H_b,dim=0)).reshape(len(L_H_b), -1))[0]
                    loc_ratio_sort = torch.exp(L_loc_ratio_sort)
                    if p_T is None:
                        loc_qry_p_F_E = [loc_ratio_sort[sh][int(self.target_F_cnt[0][sh]*32**2/65536)-1] if int(self.target_F_cnt[0][sh]*32**2/65536)>0
                                         else torch.clip(2*loc_ratio_sort[sh][0]-loc_ratio_sort[sh][1],min=0.0,max=1)
                                         for sh in range(len(L_H_b))]
                    else:
                        loc_qry_p_F_E = [loc_ratio_sort[sh][int(p_T[sh] * 32 ** 2 / 65536) - 1] if int(p_T[sh] * 32 ** 2 / 65536) > 0
                                         else torch.clip(2 * loc_ratio_sort[sh][0] - loc_ratio_sort[sh][1], min=0.0,max=1)
                                         for sh in range(len(L_H_b))]
                    loc_qry_p_F_E = torch.stack(loc_qry_p_F_E, dim=0).unsqueeze(-1).unsqueeze(-1).detach()

                #h_a_list=[]
                L_h_a_list = []
                for i_p in range(loc_n_protos):
                    L_h_a = (s[i_p] - 1) / sig1 ** 2 - self.dim_*L_sig1  #h_a=h(a,proto_i)
                    L_h_b = (s[i_p] - 1) / sig2 ** 2 - self.dim_*L_sig2  #h_b=h(b,proto_i)
                    if cls_prior == 'uniform':
                        L_pred_F.append(np.log(0.5 / loc_n_protos) + L_h_a)
                        L_pred_B.append(np.log(0.5 / loc_n_protos) + L_h_b)
                    elif cls_prior == 'param':
                        L_pred_F.append(torch.log(self.p_F) + L_h_a)
                        L_pred_B.append(torch.log(self.p_B) + L_h_b)
                    elif (cls_prior == 'qry_p_F')|(p_T is not None):
                        L_pred_F.append(torch.log(loc_qry_p_F_E / loc_n_protos) + L_h_a)
                        L_pred_B.append(torch.log((1 - loc_qry_p_F_E) / loc_n_protos) + L_h_b)
                    elif isinstance(cls_prior, torch.Tensor):
                        if len(cls_prior.shape)<=1:
                            L_pred_F.append(torch.log(cls_prior / loc_n_protos) +L_h_a)
                            L_pred_B.append(torch.log((1 - cls_prior) / loc_n_protos) + L_h_b)
                        elif len(cls_prior.shape)==2:
                            assert cls_prior.shape[0]==1
                            L_pred_F.append(torch.log(cls_prior[:,i_p] / loc_n_protos) + L_h_a)
                            L_pred_B.append(torch.log(cls_prior[:,-1] / loc_n_protos) + L_h_b)
                        else:
                            raise
                    else:
                        print('Unknown cls_prior setting!!!')
                        raise

                if method == 'TPM':
                    pred_F = [torch.exp(L_pred_F_tensor) for L_pred_F_tensor in L_pred_F]
                    L_pred_B = torch.logsumexp(torch.stack(L_pred_B,dim=0),dim=0)
                    pred_B = [torch.exp(L_pred_B)]
                    pred.append(pred_F+pred_B)
                elif method=='ADNet++':
                    L_pred_F_B = torch.stack([L_pred_F[i_p],L_pred_B[i_p]],dim=0)
                    L_pred_Fs = torch.stack([L_pred_F[i_p] - torch.logsumexp(L_pred_F_B,dim=0) for i_p in range(loc_n_protos)],dim=0)
                    pred_Fs = torch.exp(L_pred_Fs)
                    pred_Fs_max = torch.max(pred_Fs,dim=0,keepdim =True)[0] #[1 x 34 x 32 x 32]
                    pred_B_ = 1-pred_Fs_max
                else:
                    raise

            if method=='TPM':
                stacked_pred_l = [torch.stack(pred_l) for pred_l in pred]
                pred = torch.stack(stacked_pred_l)  #[N_q x (n_protos+1)x 1 x H' x W'] or [N_q x (n_protos+1)x B_q x H' x W']
                pred = pred/torch.sum(pred,dim=1)
                pred_argmax = torch.argmax(pred,dim=1)
                pred_fore_cnt = torch.sum(pred_argmax!=loc_n_protos,dim=[2,3])/32**2
                if not train:
                    print('Method:', method)
                    print('  cls_prior:', cls_prior)
                    print('  pred_fore_cnt:',pred_fore_cnt)
                    print()
            elif method=='ADNet++':
                pred_ = torch.cat([pred_Fs]+[pred_B_],dim=0)
                pred = self.softmax(pred_).unsqueeze(0) #[1 x (4+1) x 34 x 32 x 32]
                pred_argmax = torch.argmax(pred, dim=1)
                pred_fore_cnt = torch.sum(pred_argmax != loc_n_protos, dim=[2, 3]) / 32 ** 2
                if not train:
                    print('Method:', method)
                    print('cls_prior:', cls_prior)
                    print('  pred_fore_cnt:', pred_fore_cnt)
            else:
                print('Unknown method!!!')
                raise

            if len(pred):
                return pred[0].transpose(dim0=0, dim1=1)
            else:
                print('    len(pred)>1!!!')
                raise
        else:
            raise

    def analyze_fts(self, fts_, fore_mask, fg_prototypes_, epi, verbose=True, train=False, support=True):
        '''
        inputs: fts, fore_mask, fg_prototypes
            fts: Wa x Sh x B x C x H' x W'???

            fore_mask (multiclass): way x shot x [n_protos x H x W]

            qry_mask_ (multiclass) training: [n_protos x 1 x H x W] or inference: [n_protos x B_q x H x W], tensor

            fg_prototypes (multiclass): Wa x n_proto x [1 x C]
        '''
        fts = fts_.detach() #No gradient computation
        fg_prototypes = [[fg_prototypes_[i_w][i_p].detach() for i_p in range(self.n_protos)] for i_w in range(self.n_ways)]
        if support:
            supp_tag = 'supp_'
        else:
            supp_tag = 'qry_'
        back_mask=1-fore_mask
        if support:
            union_mask = torch.sum(fore_mask, dim=2, keepdim=True)  #[way x shot x 1 x H x W]
        else:
            union_mask = torch.sum(fore_mask, dim=0, keepdim=True)  #[1 x B_q x H x W]

        if torch.max(union_mask) <= 1:
            back_mask2 = 1 - union_mask #torch.Size([1, 1, 256, 256]) #Based on union
        else:
            raise

        if support:
            fore_fts_ = [[[self.getFeatures(fts[i_w, i_s, [epi]], fore_mask[i_w, i_s, [i_p]], MAP=False) for i_p in range(self.n_protos)]
                          for i_s in range(self.n_shots)] for i_w in range(self.n_ways)]  #Way x shot x n_protos x [1xCxN_fg]

            back_fts_ = [[[self.getFeatures(fts[i_w, i_s, [epi]], back_mask[i_w, i_s, [i_p]], MAP=False) for i_p in range(self.n_protos)]
                          for i_s in range(self.n_shots)] for i_w in range(self.n_ways)]  #Way x shot x n_protos x [1xCxN_bg]

            n_sh = self.n_shots+0
        else:
            n_sh = self.batch_size_q + 0

            fore_fts_ = [[[self.getFeatures(fts[epi, q][None, :], fore_mask[i_p, q][None, :], MAP=False)
                           for i_p in range(self.n_protos)] for q in range(self.batch_size_q)]]  #1 x N? x n_protos x [1xCxN_fg]
            back_fts_ = [[[self.getFeatures(fts[epi, q][None, :], back_mask[i_p, q][None, :], MAP=False)
                           for i_p in range(self.n_protos)] for q in range(self.batch_size_q)]]  #1 x N? x n_protos x [1xCxN_bg]
            Fore_fts_ = [[self.getFeatures(fts[epi, q][None, :], union_mask[0, q][None, :], MAP=False)]
                         for q in range(self.batch_size_q)]  #1 x N? x [1xCxN_fg]
            Back_fts_ = [[self.getFeatures(fts[epi, q][None, :], back_mask2[0, q][None, :], MAP=False)]
                           for q in range(self.batch_size_q)]  #1 x N? x [1xCxN_bg]

            fts_ = [[[torch.cat([fore_fts_[0][q][0], back_fts_[0][q][0]],dim=-1)]
                     for q in range(self.batch_size_q)]]#torch.Size([1, 256, 65536])

            fts_reshaped = [[[fts_[0][q][0].squeeze(0).transpose(0, 1)]
                     for q in range(self.batch_size_q)]]#torch.Size([65536, 256])

            if train:
                fts_n = [[[fts_reshaped[0][q][0] / torch.norm(fts_reshaped[0][q][0], dim=1, keepdim=True)]
                          for q in range(self.batch_size_q)]]

                log_fts_n = [[[self.log_map(fts_n[0][q][0])] for q in range(self.batch_size_q)]]
                log_mn = [[[torch.mean(log_fts_n[0][q][0], dim=0, keepdim=True)] for q in range(self.batch_size_q)]]

                dist_mn = [[[torch.mean(torch.norm(log_fts_n[0][q][0] - log_mn[0][q][0], dim=-1), dim=0)]
                            for q in range(self.batch_size_q)]]

                dist_std = [[[torch.std(torch.norm(log_fts_n[0][q][0] - log_mn[0][q][0], dim=-1), dim=0)]
                             for q in range(self.batch_size_q)]]

                wandb.log({'dist_mn': dist_mn[0][0][0].item(),'dist_std_': dist_std[0][0][0].item()})


        sq_dist_fg = [[[self.getSqDistance(fore_fts_[way][sh][i_p], fg_prototypes[way][i_p]) for i_p in range(self.n_protos)]
                       for sh in range(n_sh)] for way in range(self.n_ways)]  #Way x sh x n_proto x [1 x C x N_fg] #sh: shot or q
        sq_dist_bg = [[[self.getSqDistance(back_fts_[way][sh][i_p], fg_prototypes[way][i_p]) for i_p in range(self.n_protos)]
                       for sh in range(n_sh)] for way in range(self.n_ways)]  #Way x sh x n_proto x [1 x C x N_bg] #sh: shot or q

        ### Analyze prototypes ###
        if (self.n_ways==1)&(support):
            fg_protos=torch.cat(fg_prototypes[0])#[N_protos, C]
            fg_protos = fg_protos/torch.norm(fg_protos,dim=1,keepdim=True)
            if self.n_protos>1:
                fg_protos_diff = fg_protos.unsqueeze(1) - fg_protos.unsqueeze(0)#[N_protos, N_protos, C]
                fg_protos_dists = torch.sqrt((fg_protos_diff ** 2).sum(-1))#[N_protos, N_protos]
                fg_protos_angs = torch.acos((2-fg_protos_dists**2)/2) #[N_protos, N_protos]

                # Compute the mean of the upper triangle of the distance matrix, excluding the diagonal
                self.proto_dist_mn = torch.mean(fg_protos_dists[torch.triu_indices(self.n_protos, self.n_protos, offset=1)])
                proto_dist_std = torch.std(fg_protos_dists[torch.triu_indices(self.n_protos, self.n_protos, offset=1)])
                proto_ang_mn = torch.mean(fg_protos_angs[torch.triu_indices(self.n_protos, self.n_protos, offset=1)])
                proto_ang_std = torch.std(fg_protos_angs[torch.triu_indices(self.n_protos, self.n_protos, offset=1)])
                if verbose:
                    print(f'    proto_dist_mn:', self.proto_dist_mn.item())
                    print(f'    proto_dist_std:', proto_dist_std.item())
                    print(f'    proto_ang_mn:', proto_ang_mn.item())
                    print(f'    proto_ang_std:', proto_ang_std.item())
                wandb.log({'proto_dist_mn': self.proto_dist_mn.item(), 'proto_dist_std': proto_dist_std.item()})
                wandb.log({'proto_ang_mn': proto_ang_mn.item(), 'proto_ang_std': proto_ang_std.item()})
                if np.isnan(self.proto_dist_mn.item()):
                    print('    NaN Value!!!')
                    raise

        if (self.n_shots > 1)&support:
            print('    supp_fore_fts_:', len(fore_fts_), len(fore_fts_[0]), fore_fts_[0][0].shape)
        if (self.batch_size_q > 1)&(not support):
            print('    self.batch_size_q:', self.batch_size_q, fore_fts_[0][0][0].shape)

        ### Estimate T_D: decision threshold of distance ###
        p_F_cnt_list = [[[sq_dist_fg[way][sh][i_p].shape[-1] / (sq_dist_fg[way][sh][i_p].shape[-1] + sq_dist_bg[way][sh][i_p].shape[-1])
                        for i_p in range(self.n_protos)] for sh in range(n_sh)] for way in range(self.n_ways)] #sh: shot or q
        p_F_cnt_list = [[[torch.tensor([p_F_cnt_list[way][sh][i_p]], device=self.device)
                        for i_p in range(self.n_protos)] for sh in range(n_sh)] for way in range(self.n_ways)] #sh: shot or q]]
        p_F_cnt_list = [[torch.cat(p_F_cnt_list[way][sh]) for sh in range(n_sh)] for way in range(self.n_ways)]

        if (train)|verbose:
            fore_mn_list = [[[torch.mean(torch.sqrt(sq_dist_fg[way][sh][i_p] + torch.finfo(torch.float32).eps)).detach()
                             for i_p in range(self.n_protos)] for sh in range(n_sh)] for way in range(self.n_ways)]  #sh: shot or q
            back_mn_list = [[[torch.mean(torch.sqrt(sq_dist_bg[way][sh][i_p] + torch.finfo(torch.float32).eps)).detach()
                             for i_p in range(self.n_protos)] for sh in range(n_sh)] for way in range(self.n_ways)]  #sh: shot or q

            mid_thres_list = [[[(fore_mn_list[way][sh][i_p] + back_mn_list[way][sh][i_p]) / 2
                            for i_p in range(self.n_protos)] for sh in range(n_sh)] for way in range(self.n_ways)]  #sh: shot or q

            dists_list = [[[torch.sort(torch.sqrt(torch.cat([sq_dist_fg[way][sh][i_p], sq_dist_bg[way][sh][i_p]], dim=1) + torch.finfo(torch.float32).eps))[0]
                for i_p in range(self.n_protos)] for sh in range(n_sh)] for way in range(self.n_ways)]  #sh: shot or q
            p_F_thres_list = [[[(dists_list[way][sh][i_p][0][sq_dist_fg[way][sh][i_p].shape[-1]]+dists_list[way][sh][i_p][0][sq_dist_fg[way][sh][i_p].shape[-1]+1])/2
                for i_p in range(self.n_protos)] for sh in range(n_sh)] for way in range(self.n_ways)]  #sh: shot or q

            if not support:
                self.qry_p_F_thres =p_F_thres_list

            sig1 = self.sig1.detach()
            sig2 = self.sig2.detach()

            Num_F_list = [[[torch.exp(-dists_list[way][sh][i_p] ** 2 / (2 * sig1 ** 2)) / sig1
                            for i_p in range(self.n_protos)] for sh in range(n_sh)] for way in range(self.n_ways)]  #Numerators for foregrounds #sh: shot or q
            Num_F_max = [[torch.max(torch.cat(Num_F_list[way][sh], dim=0), dim=0)[0]
                          for sh in range(n_sh)] for way in range(self.n_ways)]  #Maximum values of numerators for foregrounds
            Num_B_list = [[[torch.exp(-dists_list[way][sh][i_p] ** 2 / (2 * sig2 ** 2)) / sig2
                            for i_p in range(self.n_protos)] for sh in range(n_sh)] for way in range(self.n_ways)]  #Numerators for backgrounds #sh: shot or q
            Num_B = [[sum(Num_B_list[way][sh]) for sh in range(n_sh)] for way in range(self.n_ways)]  #Numerators for backgrounds ###sh: shot or q
            #Num_list = [[torch.cat(Num_F_list[way][sh] + [Num_B[way][sh]], dim=0).t()
            #             for sh in range(n_sh)] for way in range(self.n_ways)]
            #Num_list = [torch.stack(Num_list[way],dim=0) for way in range(self.n_ways)]
            target_F_cnt = [[sum(p_F_cnt_list[way][sh]) * (sq_dist_fg[way][sh][0].shape[-1] + sq_dist_bg[way][sh][0].shape[-1])
                        for sh in range(n_sh)] for way in range(self.n_ways)]
            ratio_sort = [[torch.sort(Num_B[way][sh][0] / (Num_F_max[way][sh] + Num_B[way][sh][0]))[0]
                           for sh in range(n_sh)] for way in range(self.n_ways)]  #H(b)/(h_max(a)+H(b))

            p_cnt_list = [[torch.cat([p_F_cnt_list[way][sh], 1 - torch.sum(p_F_cnt_list[way][sh], dim=0, keepdim=True)],dim=0).unsqueeze(0)
                           for sh in range(n_sh)] for way in range(self.n_ways)]
            p_cnt_list = [torch.stack(p_cnt_list[way],dim=0) for way in range(self.n_ways)]
            p_cnt_mn = [p_cnt_list[way].mean(dim=0) for way in range(self.n_ways)]
            if support:
                if (verbose):
                    self.supp_p_F = [[ratio_sort[way][sh][int(target_F_cnt[way][sh])-1] if int(target_F_cnt[way][sh])>0
                                       else torch.clip(2*ratio_sort[way][sh][0]-ratio_sort[way][sh][1],min=0.0,max=1)
                                      for sh in range(n_sh)] for way in range(self.n_ways)]

                self.supp_T_D_list = [[[p_F_thres_list[way][sh][i_p].item()
                                        for i_p in range(self.n_protos)] for sh in range(n_sh)] for way in range(self.n_ways)]  #sh: shot or q

            else:
                self.target_F_cnt = target_F_cnt
                self.p_cnt_list = p_cnt_list
                self.qry_p_F_E = [[ratio_sort[way][sh][int(target_F_cnt[way][sh]) - 1] if int(target_F_cnt[way][sh]) > 0
                                   else torch.clip(2 * ratio_sort[way][sh][0] - ratio_sort[way][sh][1], min=0.0, max=1)
                                   for sh in range(n_sh)] for way in range(self.n_ways)]
                self.qry_p_F_E = [torch.stack(self.qry_p_F_E[way], dim=0).unsqueeze(-1) for way in range(self.n_ways)]

                tmp_mn = [self.qry_p_F_E[way].mean() for way in range(self.n_ways)]

                FP_ind_list = [[[torch.sqrt(sq_dist_bg[way][sh][i_p] + torch.finfo(torch.float32).eps) <= p_F_thres_list[way][sh][i_p].item()
                     for i_p in range(self.n_protos)] for sh in range(n_sh)] for way in range(self.n_ways)]  #sh: shot or q
                FN_ind_list = [[[torch.sqrt(sq_dist_fg[way][sh][i_p] + torch.finfo(torch.float32).eps) > p_F_thres_list[way][sh][i_p].item()
                     for i_p in range(self.n_protos)] for sh in range(n_sh)] for way in range(self.n_ways)]  #sh: shot or q
                self.FP_len_list = [[[torch.sum(FP_ind_list[way][sh][i_p])
                     for i_p in range(self.n_protos)] for sh in range(n_sh)] for way in range(self.n_ways)]
                self.FN_len_list = [[[torch.sum(FN_ind_list[way][sh][i_p])
                     for i_p in range(self.n_protos)] for sh in range(n_sh)] for way in range(self.n_ways)]
                self.F_list = [[[sq_dist_fg[way][sh][i_p].shape[-1]
                     for i_p in range(self.n_protos)] for sh in range(n_sh)] for way in range(self.n_ways)]  #sh: shot or q
                self.B_list = [[[sq_dist_bg[way][sh][i_p].shape[-1]
                     for i_p in range(self.n_protos)] for sh in range(n_sh)] for way in range(self.n_ways)]  #sh: shot or q


                self.FP_med_list = [[[torch.median(torch.sqrt(sq_dist_bg[way][sh][i_p][FP_ind_list[way][sh][i_p]] + torch.finfo(torch.float32).eps))
                                     if self.FP_len_list[way][sh][i_p] > 0 else torch.tensor(float('nan'))
                                     for i_p in range(self.n_protos)] for sh in range(n_sh)] for way in range(self.n_ways)]  #sh: shot or q
                self.FN_med_list = [[[torch.median(torch.sqrt(sq_dist_fg[way][sh][i_p][FN_ind_list[way][sh][i_p]] + torch.finfo(torch.float32).eps))
                                     if self.FN_len_list[way][sh][i_p] > 0 else torch.tensor(float('nan'))
                                     for i_p in range(self.n_protos)] for sh in range(n_sh)] for way in range(self.n_ways)]  #sh: shot or q

                lgm_list= [[[torch.mean(torch.log(dists_list[way][sh][i_p][0]))
                             for i_p in range(self.n_protos)] for sh in range(n_sh)] for way in range(self.n_ways)]

            if (self.n_shots == 1) & (not support):
                lgm_tensor = torch.tensor(lgm_list)
                lgm_list_ = torch.mean(lgm_tensor, dim=-1).tolist()

                self.qry_p_F_thres = [[torch.stack(self.qry_p_F_thres[way][sh]) for sh in range(n_sh)] for way in range(self.n_ways)]
                self.qry_p_F_thres = [torch.stack(self.qry_p_F_thres[way]) for way in range(self.n_ways)]
                self.qry_p_F_thres = torch.stack(self.qry_p_F_thres)

                if train:
                    self.qry_lgm = torch.tensor(lgm_list_[0][0] + 0.0)
                    T_D_hat_h_proto_dist_ratio = torch.mean(self.qry_p_F_thres) / (self.proto_dist_mn/2)
                    wandb.log({'GeoMean': (torch.exp(self.qry_lgm)).item(), 'T_D_hat_h_proto_dist_ratio': T_D_hat_h_proto_dist_ratio})

        if train&(not support)&(self.n_shots == 1)&(self.n_ways == 1):

            tmp_qry_p_F_thres = self.qry_p_F_thres.detach() + 0.0
            T_D_hat_med = torch.median(tmp_qry_p_F_thres)

            self.T_D_hat_ = T_D_hat_med

            # 1/ (1+(self.sig2/self.sig1)**self.dim_*torch.exp(self.T_D**2/2*(1/self.sig1**2-1/self.sig2**2)))

            self.p_F_multi_hat_ = tmp_mn[0]+0.0 #self.qry_p_F_E[way].mean()
            print('  self.p_F_multi_hat_',self.p_F_multi_hat_)
        else:
            if (not train) & (not support):
                pass

        for way in range(self.n_ways):
            for sh in range(n_sh):
                for i_p in range(self.n_protos):
                    if verbose:
                        print('      %sp_F_cnt: %f' % (supp_tag, p_F_cnt_list[way][sh][i_p].item()))
                        if p_F_cnt_list[way][sh][i_p].item() > 0:
                            print('      %sfore_mn: %f back_mn: %f mid_thres: %f' % (
                            supp_tag, fore_mn_list[way][sh][i_p].item(),
                            back_mn_list[way][sh][i_p].item(),
                            mid_thres_list[way][sh][i_p].item()))
                    if train & (verbose ):
                        wandb.log({supp_tag + 'p_F_cnt': p_F_cnt_list[way][sh][i_p].item(),
                                   supp_tag + 'fore_mn': fore_mn_list[way][sh][i_p].item(),
                                   supp_tag + 'back_mn': back_mn_list[way][sh][i_p].item(),
                                   supp_tag + 'mid_thres': mid_thres_list[way][sh][i_p].item()})
                        if (not support):
                            wandb.log({'FP_med': self.FP_med_list[way][sh][i_p].item()})
                            wandb.log({'FN_med': self.FN_med_list[way][sh][i_p].item()})
                            wandb.log({'FP_len': self.FP_len_list[way][sh][i_p].item()})
                            wandb.log({'FN_len': self.FN_len_list[way][sh][i_p].item()})
                            if self.B_list[way][sh][i_p] > 0:
                                wandb.log({'FP_rate': self.FP_len_list[way][sh][i_p].item() / self.B_list[way][sh][i_p]})
                            if self.F_list[way][sh][i_p] > 0:
                                wandb.log({'FN_rate': self.FN_len_list[way][sh][i_p].item() / self.F_list[way][sh][i_p]})

    def log_map(self, w, base=None):
        # Sphere->Tangent
        # Tangent space: <v,base>=0 (v is orthogonal to base)
        # w: [n,dim]
        if base is None:
            mn = torch.mean(w, dim=0, keepdim=True)
            base = mn / (1e-5 + torch.norm(mn, dim=-1, keepdim=True))  # [1,dim]

        cos_sim = torch.sum(base * w, axis=-1, keepdim=True)  # [n, 1]

        # Difference vector in tangent plane
        diff = w - base * cos_sim
        diff_norm = torch.norm(diff, dim=-1, keepdim=True)

        # Calculate the vector on the tangent space
        v = torch.acos(0.999*cos_sim) * diff / (1e-5 + diff_norm) # Apply tricks for numerical stability
        return v

    def load_params(self, train=True, verbose=False):
        # Load parameter values for "analyze" and "inference"
        if verbose:
            print('load_params')
            if self.learn_alpha:
                print('    self.learn_alpha', self.learn_alpha)
            if self.fix_alpha is not None:
                print('    self.fix_alpha', self.fix_alpha)
            if self.chg_alpha:
                print('    self.chg_alpha', self.chg_alpha)
            if self.chg_alpha2:
                print('    self.chg_alpha2', self.chg_alpha2)
            if self.chg_alpha_multi:
                print('    self.chg_alpha_multi', self.chg_alpha_multi)
            print('    self.alpha(pre)', self.alpha)

        if self.fix_alpha is None:
            if self.learn_alpha:
                self.alpha = torch.exp(self.l_alpha)
            else:
                if self.chg_alpha:  # model alpha follows self.alpha_hat
                    self.alpha = self.alpha_hat.data.detach()
                elif self.chg_alpha2:
                    self.alpha = self.alpha_hat2.data.detach()
                elif self.chg_alpha_multi:  # model alpha follows self.alpha_hat_multi
                    self.alpha = self.alpha_multi_hat.data.detach()
                else:
                    pass
        if verbose:
            print('    self.alpha(post)', self.alpha)
            print()

            if self.learn_T_D:
                print('    self.learn_T_D', self.learn_T_D)
            if self.fix_T_D is not None:
                print('    self.fix_T_D', self.fix_T_D)
            if self.chg_T_D:
                print('    self.chg_T_D', self.chg_T_D)
            if self.chg_T_D_multi:
                print('    self.chg_T_D_multi', self.chg_T_D_multi)
            print('    self.T_D(pre)', self.T_D, 'self.T_S', self.T_S)
        if self.fix_T_D is None:
            if self.learn_T_D:
                self.T_S = self.dim_ * self.softplus(self.pre_T_S) + 2 * torch.log(self.p_F / self.p_B) - self.alpha
                self.T_D = torch.sqrt(2 * (1 + self.T_S / self.alpha))
                sig_part = torch.exp(self.alpha * self.T_D ** 2 / (2 * self.dim_))/ (self.p_F / self.p_B) ** (2 / self.dim_)
                self.sig1_ = (2 / self.alpha * (1 - 1/sig_part)) ** 0.5
                self.sig2_ = (2 / self.alpha * (sig_part - 1)) ** 0.5
            else:
                if self.chg_T_D|self.chg_T_D_multi:
                    if self.chg_T_D:
                        self.T_D = self.T_D_hat.data.detach()
                    elif self.chg_T_D_multi:
                        self.T_D = self.T_D_multi_hat.data.detach()
                    else:
                        raise

                    self.T_S = self.alpha * (self.T_D ** 2 / 2 - 1)
                    if self.fix_sig2 is None:  # sigma values are based on self.alpha and self.T_D
                        sig_part = torch.exp(self.alpha * self.T_D ** 2 / (2 * self.dim_))/ (self.p_F / self.p_B) ** (2 / self.dim_)
                        self.sig1_ = (2 / self.alpha * (1 - 1/sig_part)) ** 0.5
                        self.sig2_ = (2 / self.alpha * (sig_part - 1)) ** 0.5
                    else:  # self.sig2 is fixed. self.sig1 value is based on self.sig2 and self.T_D
                        if (self.fix_alpha is not None) | self.chg_alpha | self.chg_alpha2 | self.chg_alpha_multi | self.learn_alpha:
                            print('alpha options are prioritized over fixing sig2!!!')
                            print('There should be no option selected for sig2!!!')
                            raise
                        else:  # When no option is chosen for alpha
                            beta2 = -self.T_D ** 2 / (self.dim_ * self.sig2 ** 2)
                            self.sig1_ = self.sig2 * (beta2 / (lambertw(float(beta2 * torch.exp(beta2)), -1).real)) ** 0.5
                            self.sig2_ = self.sig2.data.detach()
                            self.alpha_ = 2 * (1 / self.sig1_ ** 2 - 1 / self.sig2 ** 2)
                            if self.p_F.item() != 0.5:
                                raise
                else:
                    if self.fix_sig2 is None:  # Underdetermined situation
                        #sigma values are based on self.alpha and default self.T_D
                        if not self.chg_p_F:
                            self.T_S = self.alpha * (self.T_D ** 2 / 2 - 1)
                            sig_part = torch.exp(self.alpha * self.T_D ** 2 / (2 * self.dim_)) / (self.p_F / self.p_B) ** (2 / self.dim_)
                            self.sig1_ = (2 / self.alpha * (1 - 1/sig_part)) ** 0.5
                            self.sig2_ = (2 / self.alpha * (sig_part - 1)) ** 0.5
                    else:  # self.sig2 is fixed. self.sig1 and self.T_D values are based on self.sig2 and self.alpha
                        if (self.fix_alpha is not None) | self.chg_alpha | self.chg_alpha2 | self.chg_alpha_multi| self.learn_alpha:
                            print('alpha options are prioritized over fixing sig2!!!')
                            print('There should be no option selected for sig2!!!')
                            raise
                        else:  # When no option is chosen for alpha. Underdetermined situation
                            print('Not implemented!')  # self.sig1 and self.T_D values are based on self.sig2 and self.alpha
                            raise
        else:
            sig_part = torch.exp(self.alpha * self.T_D ** 2 / (2 * self.dim_)) / (self.p_F / self.p_B) ** (2 / self.dim_)
            self.sig1_ = (2 / self.alpha * (1 - 1/sig_part)) ** 0.5
            self.sig2_ = (2 / self.alpha * (sig_part - 1)) ** 0.5


        if verbose:
            if self.learn_p_F:
                print('    self.learn_p_F', self.learn_p_F)
            if self.fix_p_F is not None:
                print('    self.fix_p_F', self.fix_p_F)
            if self.chg_p_F:
                print('    self.chg_p_F', self.chg_p_F)
            print('    self.p_F(pre)', self.p_F)
            print()

        # Calculate self.p_F value using (given) self.alpha, self.T_D, sigma values
        if self.fix_p_F is None:
            if self.learn_p_F:
                self.p_F = self.softsign2(self.pre_p_F)
                self.p_B = 1- self.p_F
            else:
                if self.chg_p_F:#follows self.p_F_hat
                    self.p_F.data = self.p_F_hat.data.detach()
                    self.p_B = 1 - self.p_F
                    if not (self.chg_T_D|self.chg_T_D_multi|self.learn_T_D|(self.fix_T_D is not None)):
                        sig_part = (self.p_F / self.p_B) ** (-2/self.dim_)*torch.exp(self.alpha * self.T_D_hat **2 / (2*self.dim_))
                        if sig_part<1:
                            print('self.sig1_ and self.sig2_ are undefined (NaN)!!!')
                            raise

                        self.sig1_ = (2 / self.alpha * (1 - 1/sig_part)) ** 0.5
                        self.sig2_ = (2 / self.alpha * (sig_part - 1)) ** 0.5
                        self.T_S.data = -2*self.dim_ * torch.log(self.sig1 / self.sig2) + 2 * torch.log(self.p_F / self.p_B) - self.alpha
                        self.T_D = torch.sqrt(2 * (1 + self.T_S / self.alpha))
                if self.chg_p_F_multi:#follows self.p_F_hat
                    self.p_F.data = self.p_F_multi_hat.data.detach()
                    self.p_B = 1 - self.p_F
                    if not (self.chg_T_D|self.chg_T_D_multi|self.learn_T_D|(self.fix_T_D is not None)):
                        sig_part = (self.p_F / self.p_B) ** (-2/self.dim_)*torch.exp(self.alpha * self.T_D_multi_hat **2 / (2*self.dim_))
                        if sig_part<1:
                            print('self.sig1_ and self.sig2_ are undefined (NaN)!!!')
                            raise

                        self.sig1_ = (2 / self.alpha * (1 - 1/sig_part)) ** 0.5
                        self.sig2_ = (2 / self.alpha * (sig_part - 1)) ** 0.5
                        self.T_S.data = -2*self.dim_ * torch.log(self.sig1 / self.sig2) + 2 * torch.log(self.p_F / self.p_B) - self.alpha
                        self.T_D = torch.sqrt(2 * (1 + self.T_S / self.alpha))
        if verbose:
            print('    self.T_D(post)', self.T_D, 'self.T_S', self.T_S)
            print('    self.p_F(post)', self.p_F)
            print()

        if verbose:
            if train:
                print('    self.sig1_', self.sig1_, 'self.sig2_', self.sig2_)
            else:
                print('    self.sig1', self.sig1, 'self.sig2', self.sig2)
            print()

    def apply_EMA(self):
        # Exponential moving average of certain parameters
        if (not torch.isnan(self.sig1_)) & self.EMA_sig:
            self.sig1.data = (1e-3) * self.sig1_.detach() + (1 - 1e-3) * self.sig1.data
        elif torch.isnan(self.sig1_) & self.EMA_sig:
            pass
        else:
            self.sig1.data = self.sig1_.detach() + 0.0

        if (not torch.isnan(self.sig2_)) & self.EMA_sig:
            self.sig2.data = (1e-3) * self.sig2_.detach() + (1 - 1e-3) * self.sig2.data
        elif torch.isnan(self.sig2_) & self.EMA_sig:
            pass
        else:
            self.sig2.data = self.sig2_.detach() + 0.0

        if (not torch.isnan(self.T_D_hat_) & self.EMA_T_D_hat):
            self.T_D_hat.data = (1e-3) * self.T_D_hat_.detach() + (1 - 1e-3) * self.T_D_hat.data  #Use EMA to get more smooth change of T_C
            self.T_D_hat.data = torch.clip(self.T_D_hat.data, min=1e-3, max=2 ** 0.5 - 1e-3)
            self.T_D_hat_ = torch.tensor(np.nan)

        self.T_D_multi_hat.data = torch.sqrt(2 * (torch.log(self.p_F_multi_hat / (1 - self.p_F_multi_hat)) - self.dim_ * torch.log(self.sig1 / self.sig2)) \
                            / (1 / self.sig1 ** 2 - 1 / self.sig2 ** 2))

        p_F_part = self.dim_ * torch.log((self.sig2 / self.sig1)) \
                   - self.T_D_hat ** 2 / 2 * (1 / self.sig1 ** 2 - 1 / self.sig2 ** 2)
        self.p_F_hat.data = torch.exp(-torch.logsumexp(torch.cat([self.zero_tensor, p_F_part]), dim=0)).unsqueeze(0)

        if (not torch.isnan(self.p_F_multi_hat_)) & self.EMA_p_F:
            self.p_F_multi_hat.data = (1e-3) * self.p_F_multi_hat_.detach() + (1 - 1e-3) * self.p_F_multi_hat.data
            self.p_F_multi_hat_ = torch.tensor(np.nan)

        wandb.log({'self.T_D_hat': self.T_D_hat.data.item(), 'self.T_D_multi_hat': self.T_D_multi_hat.data.item(), 'self.T_D_hat_MSE': self.T_D_hat_MSE.item(),
                   'self.p_F': self.p_F.item(), 'self.p_F_hat': self.p_F_hat.item(),'self.p_F_multi_hat': self.p_F_multi_hat.item()
                   #'self.alpha_hat': self.alpha_hat.item(),
                   #'self.alpha_hat2': self.alpha_hat2.item(),
                   })
        wandb.log({'self.sig1': self.sig1.item(), 'self.sig2': self.sig2.item(),
                   'self.T_D': self.T_D.item(), 'self.T_S': self.T_S.item(), 'self.alpha': self.alpha.item()})

    def list_tensors_on_gpu(self, variables_dict):
        gpu_tensors = {}
        gpu_params = {}
        for name, var in variables_dict.items():
            if torch.is_tensor(var) and var.is_cuda:
                gpu_tensors[name] = [list(var.shape), var.requires_grad] #var.device
            elif hasattr(var, 'parameters'):  # Check if it's a nn.Module (or similar)
                for n, p in var.named_parameters():
                    if p.is_cuda:
                        gpu_params[f"{name}.{n}"] = [list(p.shape), p.requires_grad] #p.device
        return gpu_tensors, gpu_params

    def get_shapes(self, data):
        """
        Recursively get the shapes of tensors and lengths of lists in a nested list/tensor structure.

        Args:
        - data: The input data which could be a tensor, a list of tensors, or a list of lists of tensors.

        Returns:
        - A string representation of the lengths and shapes.  (e.g. '4 x 2 x [3 x 5 x 1]')
        """
        if isinstance(data, torch.Tensor):
            # Return the shape of the tensor enclosed in brackets
            return f"[{' x '.join(map(str, data.shape))}]"
        elif isinstance(data, np.ndarray):
            # Return the shape of the tensor enclosed in brackets
            return f"[{' x '.join(map(str, data.shape))}] (Numpy)"
        elif isinstance(data, (int, float, bool, np.generic)):
            return f"1 (Scalar)"
        elif isinstance(data, list):
            # If it's a list, process each item in the list recursively
            # Collect all shapes or descriptions of the elements in the list
            shapes = [self.get_shapes(elem) for elem in data]
            return str(len(data)) + ' x ' + shapes[0]
        else:
            return "Unsupported type"
