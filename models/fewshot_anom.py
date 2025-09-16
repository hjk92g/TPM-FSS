import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from .backbone.torchvision_backbones import TVDeeplabRes101Encoder
import wandb, sys, joblib
from scipy.special import gamma, lambertw


#TPM: Tied Prototype Model
class FewShotSeg(nn.Module):

    def __init__(self, use_coco_init=True,
                 fix_alpha=None, chg_alpha=False, chg_alpha2=False, learn_alpha=False,
                 ADNet=False,
                 adnet=False, t_loss_scaler=1.0,
                 fix_T_D=None, chg_T_D=False,
                 fix_p_F=None, chg_p_F=False, learn_p_F=False,
                 fix_sig2=None,
                 dim_=1, data_root=None, pretrained_root=None,
                 EMA_T_D_hat=True, EMA_alpha_hat=True, EMA_sig=True, EMA_p_F=True, lr=1e-3):
        super().__init__()

        self.ADNet = ADNet
        self.adnet = adnet #learn T_D

        # Encoder
        self.encoder = TVDeeplabRes101Encoder(use_coco_init) # 256 feature dimension
        self.device = torch.device('cuda')

        self.fix_sig2=fix_sig2
        self.fix_alpha = fix_alpha
        self.chg_alpha = chg_alpha
        self.chg_alpha2 = chg_alpha2
        self.learn_alpha = learn_alpha
        if (self.fix_alpha is not None)+self.chg_alpha+self.chg_alpha2+self.learn_alpha>1:
            print('Only one option for alpha!!!') #Either fix_alpha is None or chg_alpha is False. They cannot be both: fix_alpha!=None and chg_alpha==True.........
            raise

        self.fix_p_F = fix_p_F
        self.chg_p_F = chg_p_F
        self.learn_p_F = learn_p_F
        if (self.fix_p_F is not None) + self.chg_p_F + self.learn_p_F > 1:
            print('Only one option for p_F!!!')
            raise

        self.fix_T_D = fix_T_D
        self.chg_T_D = chg_T_D
        if self.ADNet+self.adnet+(self.fix_T_D is not None)+self.chg_T_D>1:
            print('Only one option for T_D!!!')
            raise

        if ((self.fix_p_F is not None)|self.chg_p_F|self.learn_p_F)&(self.ADNet|self.adnet|(self.fix_T_D is not None)|self.chg_T_D):
            if self.learn_p_F&self.adnet:
                print('Note that both self.learn_p_F and self.adnet options are used')
            elif self.learn_p_F&self.ADNet:
                print('Note that both self.learn_p_F and self.ADNet options are used')
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

        self.T_D_hat = Parameter(torch.Tensor([(2*(1-10/20.0))**0.5]).to(self.device),requires_grad=False)  #Estimate good classification threshold based on distance
        # Ignore "alpha_hat" and "alpha_hat2". They are not used in the TPM code (may appear and explained in later work)
        self.alpha_hat = Parameter(torch.Tensor([20.0]).to(self.device),requires_grad=False)
        self.alpha_hat2 = Parameter(torch.Tensor([20.0]).to(self.device),requires_grad=False)
        self.p_F_hat = Parameter(torch.Tensor([0.5]).to(self.device),requires_grad=False) ##Estimate good p_F (AvgEst using EMA)

        if self.fix_alpha is None:
            if self.learn_alpha:
                self.l_alpha = Parameter(torch.Tensor([np.log(20.0)]).to(self.device))
                self.alpha = torch.exp(self.l_alpha)
            else:
                if self.chg_alpha:#model alpha follows self.alpha_hat
                    self.alpha = self.alpha_hat.data.detach()
                elif self.chg_alpha2:
                    self.alpha = self.alpha_hat2.data.detach()
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
                else:
                    self.p_F = Parameter(torch.Tensor([0.5]).to(self.device), requires_grad=False)
        else:
            self.p_F = Parameter(torch.Tensor([self.fix_p_F]).to(self.device), requires_grad=False)
        self.p_F = self.p_F.to(self.device)
        self.p_B = 1 - self.p_F

        if self.fix_T_D is None: #T_D options: indirectly changes sigma values
            if self.ADNet:
                self.T_S = Parameter(torch.Tensor([-10.0]).to(self.device))
                self.T_D = torch.sqrt(2 * (1 + self.T_S / self.alpha))
            elif self.adnet: #adnet: self.learn_T_D
                pre_T_S_ = np.log(np.exp((-10+self.alpha.item()-2*np.log(self.p_F.item()/self.p_B.item()))/self.dim_) - 1) #9.99995459903963
                self.pre_T_S = Parameter(torch.Tensor([pre_T_S_]).to(self.device))  #Use softplus to avoid saturation
                self.T_S = self.dim_*self.softplus(self.pre_T_S)+2*torch.log(self.p_F/self.p_B)-self.alpha
                self.T_D = torch.sqrt(2 * (1 + self.T_S / self.alpha))
            else:
                if self.chg_T_D:
                    self.T_D = self.T_D_hat.data.detach()
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
            if (self.fix_alpha is not None) | self.chg_alpha | self.chg_alpha2 | self.learn_alpha:
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
        self.methods = ['CE-T', 'AvgEst', 'LinEst', 'OCP', 'CE-T_MP', 'AvgEst_MP','LinEst_MP', 'OCP_MP']
        self.metrics = ['iou', 'dice', 'MSE']
        self.data_root = data_root
        self.pretrained_root = pretrained_root

    def forward(self, supp_imgs, fore_mask, qry_imgs, train=False, analyze=False, i=None, qry_mask=None,
                supp_img_all=None, supp_idx=None):
        """
        Args:
            # B: batch_size
            # N: query size
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                B? x [N? x 3 x H x W], list of tensors
            qry_mask: query mask
                training: [B? x H x W] or inference: [B? x N x H x W], tensor
                E.g. training: torch.Size([1, 256, 256]), inference: torch.Size([1, 34, 256, 256])
        """

        if i<50:
            print('i:',i)
        n_ways = len(supp_imgs)
        self.n_ways = n_ways+0
        self.n_shots = len(supp_imgs[0])
        if self.n_shots>1:
            print('   self.n_shots:',self.n_shots)
        n_queries = len(qry_imgs)
        batch_size_q = qry_imgs[0].shape[0]
        self.batch_size_q = batch_size_q+0
        batch_size = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:] # [H, W]

        if qry_mask is not None:
            if train|analyze:
                qry_mask_ = qry_mask.unsqueeze(0)
            else:
                qry_mask_ = qry_mask+0.0

        ###### Extract features ######
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
            #Load ICP estimating linear models for LinEst
            linr = joblib.load(self.pretrained_root+'p_F_hats_model.pkl')
            linr2 = joblib.load(self.pretrained_root + 'p_F_MP_hats_model.pkl')
            p_F_dict = np.load(self.pretrained_root + "p_F_dict_one_F.npz")
            p_F_dict = dict(p_F_dict)

        supp_fts = img_fts[:n_ways * self.n_shots * batch_size].view(
            n_ways, self.n_shots, batch_size, -1, *fts_size)  # Wa x Sh x B x C x H' x W'
        qry_fts = img_fts[self.n_ways * self.n_shots * batch_size:].view(
            n_queries, batch_size_q, -1, *fts_size)  #Inference: N_q x B_q x C x H' x W'

        if torch.sum(fore_mask[0][0]) == 0:
            print('    No foreground(pre-stack)')
        fore_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in fore_mask], dim=0)

        if torch.sum(fore_mask) == 0:
            print('    No foreground(post-stack)')
        back_mask = 1-fore_mask

        if train:
            self.load_params(train=True,verbose=False)
        else:
            self.load_params(train=False,verbose=False)

        ###### Compute loss ######
        align_loss = torch.zeros(1).to(self.device)
        outputs = []
        outputs_dict={method: [] for method in self.methods}
        for epi in range(batch_size):
            ###### Extract prototypes ######
            supp_fts_ = [[self.getFeatures(supp_fts[way, shot, [epi]],
                                           fore_mask[way, shot, [epi]])
                          for shot in range(self.n_shots)] for way in range(n_ways)] # getFeatures: get feature vectors using masked area
            fg_prototypes = self.getPrototype(supp_fts_) # getPrototype(): just average & not l2 normalized

            if torch.sum(fore_mask)==0:
                print('    No foreground')
            if torch.sum(back_mask)==0:
                print('    No background')

            supp_fore_fts_ = [[self.getFeatures(supp_fts[way, shot, [epi]], fore_mask[way, shot, [epi]], MAP=False)
                          for shot in range(self.n_shots)] for way in range(self.n_ways)]  #Way x shot x [1xCxN_fg]
            if not train:
                fg_multi_prototypes, fg_multi_ws = self.getPrototype(supp_fore_fts_, n_protos=5)

            if ((i < 50) & train) | (not train) & (not analyze):  #verbose
                self.analyze_fts(supp_fts, fore_mask, fg_prototypes, epi, verbose=True, train=train, support=True)
                self.analyze_fts(qry_fts, qry_mask_, fg_prototypes, epi, verbose=True, train=train, support=False)
            else:
                if analyze:
                    self.analyze_fts(supp_fts, fore_mask, fg_prototypes, epi, verbose=False, train=True, support=True)
                    self.analyze_fts(qry_fts, qry_mask_, fg_prototypes, epi, verbose=False, train=True, support=False)
                else:  #
                    self.analyze_fts(supp_fts, fore_mask, fg_prototypes, epi, verbose=False, train=train, support=True)
                    self.analyze_fts(qry_fts, qry_mask_, fg_prototypes, epi, verbose=False, train=train, support=False)


            sys.stdout.flush()
            torch.cuda.empty_cache()

            if train:
                print(self.T_D_hat, self.qry_p_F_thres)
                self.T_D_hat_MSE = torch.mean((self.T_D_hat - self.qry_p_F_thres) ** 2)

            ###### Compute anom. scores ######
            cos_s = [self.cosSim(qry_fts[epi], prototype) for prototype in fg_prototypes]
            if not train:
                cos_s_multi = [[self.cosSim(qry_fts[epi], prototype[i_p,None]) for i_p in range(5)] for prototype in fg_multi_prototypes]

            anom_s = [self.negSim(qry_fts[epi], prototype) for prototype in fg_prototypes]

            ###### Get threshold (Legacy of ADNet) #######
            self.thresh_pred = [self.T_S for _ in range(n_ways)]  # Anomaly score threshold (T_S)
            if self.ADNet|self.adnet:
                self.t_loss = self.T_S / self.alpha

            ###### Get predictions #######
            pred = self.getPred(anom_s, self.thresh_pred) # N x Wa x H' x W' # Foreground class probability: 1-sigmoid((S(,)-T)/2)

            if (not train)&(torch.sum(fore_mask)> 0).item():
                print('    torch.mean((pred>=0.5).float()):', torch.mean((pred>=0.5).float()),'\n')

            pred_ups = F.interpolate(pred, size=img_size, mode='bilinear', align_corners=True)
            pred_ups = torch.cat((1.0 - pred_ups, pred_ups), dim=1) # Softmax version of probability

            outputs.append(pred_ups)

            ###### Prototype alignment loss from "PANet" paper######
            if train:
                align_loss_epi = self.alignLoss(qry_fts[:, epi], torch.cat((1.0 - pred, pred), dim=1),
                                                supp_fts[:, :, epi],
                                                fore_mask[:, :, epi])
                align_loss += align_loss_epi

            ### Get prediction for each inference method###
            if (not train)&(not analyze): #Test (inference) only
                ### Single prototype (SP) predictions ###
                tmp_anom_s = [-cos_*self.alpha for cos_ in cos_s]
                tmp_thresh_pred = [self.T_S for _ in range(n_ways)]
                tmp_pred = self.getPred(tmp_anom_s, tmp_thresh_pred, log=False, foreground=True)
                tmp_pred_ups = F.interpolate(tmp_pred, size=img_size, mode='bilinear', align_corners=True)
                tmp_pred_ups = torch.cat((1-tmp_pred_ups, tmp_pred_ups),dim=1)  # Softmax version of probability
                outputs_dict['CE-T'].append(tmp_pred_ups) #This will correspond to "ADNet" when T loss is turned on

                tmp_anom_s = [-cos_ * self.alpha for cos_ in cos_s]
                tmp_thresh_pred = [self.T_S for _ in range(n_ways)]
                tmp_pred = self.getPred(tmp_anom_s, tmp_thresh_pred, p_F = self.p_F_hat, log=False, foreground=True) #self.p_F_hat: AvgEst using EMA
                tmp_pred_ups = F.interpolate(tmp_pred, size=img_size, mode='bilinear',align_corners=True)
                tmp_pred_ups = torch.cat((1 - tmp_pred_ups, tmp_pred_ups), dim=1)  # Softmax version of probability
                outputs_dict['AvgEst'].append(tmp_pred_ups)

                tmp_sq_dists = [2*(1-cos_) for cos_ in cos_s]
                tmp_dists = [(tmp_sq_dist+torch.finfo(torch.float32).eps)**0.5 for tmp_sq_dist in tmp_sq_dists]
                tmp_dists = F.interpolate(torch.stack(tmp_dists, dim=0), size=img_size, mode='bilinear', align_corners=True)
                tmp_dists_fg = [[tmp_dists[batch,q][qry_mask_[batch,q].to(torch.bool)]
                                 for q in range(self.batch_size_q)] for batch in range(batch_size)]
                tmp_dists_bg = [[tmp_dists[batch, q][(1-qry_mask_[batch, q]).to(torch.bool)]
                                 for q in range(self.batch_size_q)] for batch in range(batch_size)]
                tmp_dists_sort = [[torch.sort(torch.cat([tmp_dists_fg[batch][q], tmp_dists_bg[batch][q]],dim=0))[0]
                    for q in range(self.batch_size_q)] for batch in range(batch_size)]
                tmp_p_F_thres = [[(tmp_dists_sort[batch][q][len(tmp_dists_fg[batch][q])]+tmp_dists_sort[batch][q][len(tmp_dists_fg[batch][q])+1])/2
                    for q in range(self.batch_size_q)] for batch in range(batch_size)]
                tmp_p_F_thres = [torch.stack(tmp_p_F_thres[batch]) for batch in range(batch_size)]
                tmp_p_F_thres = torch.stack(tmp_p_F_thres).unsqueeze(-1).unsqueeze(-1)
                tmp_S = self.alpha*(tmp_dists**2/2-1)
                tmp_T_S = self.alpha*(tmp_p_F_thres**2/2-1)
                tmp_pred_ups = 1.0 - torch.sigmoid(0.5 * (tmp_S - tmp_T_S))
                tmp_pred_ups = torch.cat((1 - tmp_pred_ups, tmp_pred_ups), dim=0).transpose(0,1)  # Softmax version of probability
                outputs_dict['OCP'].append(tmp_pred_ups) #Oracle (ICP) result using test labels

                supp_loc = supp_idx / (len(supp_img_all) - 1)
                X_lin = np.array([[(supp_loc - 0.5) ** 2, torch.mean(fore_mask).item()]])
                tmp_anom_s = [-cos_ * self.alpha for cos_ in cos_s]

                estim_p_F_hat = linr.predict(X_lin)
                estim_p_F_hat = 1/(1+np.exp(-estim_p_F_hat))
                estim_p_F_hat = np.clip(estim_p_F_hat, a_min=1e-3, a_max=1 - 1e-3)
                estim_p_F_hat = torch.tensor(estim_p_F_hat).to(self.device)
                tmp_thresh_pred = [self.T_S for _ in range(n_ways)]
                tmp_pred = self.getPred(tmp_anom_s, tmp_thresh_pred, p_F=estim_p_F_hat, log=False, foreground=True)
                tmp_pred_ups = F.interpolate(tmp_pred, size=img_size, mode='bilinear',align_corners=True)
                tmp_pred_ups = torch.cat((1 - tmp_pred_ups, tmp_pred_ups), dim=1)
                outputs_dict['LinEst'].append(tmp_pred_ups)

                ### Multi prototype (MP) predictions ###
                eta_a = [[fg_multi_ws[i_w][i_p] * torch.exp((cos_s_multi[i_w][i_p] - 1) / self.sig1 ** 2) / self.sig1 ** self.dim_
                                for i_p in range(5)] for i_w in range(self.n_ways)]  # 1 x 5 x [36 x 32 x 32]
                eta_b = [[fg_multi_ws[i_w][i_p] * torch.exp((cos_s_multi[i_w][i_p] - 1) / self.sig2 ** 2) / self.sig2 ** self.dim_
                                for i_p in range(5)] for i_w in range(self.n_ways)]  # 1 x 5 x [36 x 32 x 32]
                h_a = [sum(eta_a[i_w]) for i_w in range(self.n_ways)]  # 1 x [36 x 32 x 32]
                h_b = [sum(eta_b[i_w]) for i_w in range(self.n_ways)]  # 1 x [36 x 32 x 32]

                tmp_ratio = [h_a[i_w] / (h_a[i_w] + h_b[i_w]) for i_w in range(self.n_ways)]  # 1 x [36 x 32 x 32]
                tmp_ratio = torch.stack(tmp_ratio, dim=0)# [1 x 36 x 32 x 32]
                tmp_pred_ups = F.interpolate(tmp_ratio, size=img_size, mode='bilinear', align_corners=True)
                tmp_pred_ups = torch.cat((1 - tmp_pred_ups, tmp_pred_ups), dim=0).transpose(0,1)  # Softmax version of probability
                outputs_dict['CE-T_MP'].append(tmp_pred_ups)#[36, 2, 256, 256]

                p_F_MP_hats_mn = torch.tensor(p_F_dict['p_F_MP_hats_mn']).to(self.device)
                tmp_ratio = [p_F_MP_hats_mn*h_a[i_w] / (p_F_MP_hats_mn*h_a[i_w] + (1-p_F_MP_hats_mn)*h_b[i_w]) for i_w in range(self.n_ways)]  # 1 x [36 x 32 x 32]
                tmp_ratio = torch.stack(tmp_ratio, dim=0)  # [1 x 36 x 32 x 32]
                tmp_pred_ups = F.interpolate(tmp_ratio, size=img_size, mode='bilinear', align_corners=True)  #
                tmp_pred_ups = torch.cat((1 - tmp_pred_ups, tmp_pred_ups), dim=0).transpose(0,1)  # Softmax version of probability
                outputs_dict['AvgEst_MP'].append(tmp_pred_ups)

                cos_s_multi2 = [[cos_s_multi[i_w][i_p] for i_w in range(self.n_ways)] for i_p in range(5)]# 5 x 1 x [36 x 32 x 32]
                cos_s_multi2 = [torch.stack(cos_s_multi2[i_p],dim=0) for i_p in range(5)]# 5 x [1 x 36 x 32 x 32]
                cos_s_multi2 = [F.interpolate(cos_s_multi2[i_p], size=img_size, mode='bilinear', align_corners=True) for i_p in range(5)]
                eta_a2 = [[fg_multi_ws[i_w][i_p] * torch.exp((cos_s_multi2[i_p][i_w] - 1) / self.sig1 ** 2) / self.sig1 ** self.dim_
                          for i_p in range(5)] for i_w in range(self.n_ways)]  # 1 x 5 x [36 x 256 x 256]
                eta_b2 = [[fg_multi_ws[i_w][i_p] * torch.exp((cos_s_multi2[i_p][i_w] - 1) / self.sig2 ** 2) / self.sig2 ** self.dim_
                          for i_p in range(5)] for i_w in range(self.n_ways)]  # 1 x 5 x [36 x 256 x 256]
                h_a2 = [sum(eta_a2[i_w]) for i_w in range(self.n_ways)]#1 x [36 x 256 x 256]
                h_b2 = [sum(eta_b2[i_w]) for i_w in range(self.n_ways)]

                tmp_ratio = [h_b2[i_w] / (h_a2[i_w] + h_b2[i_w]) for i_w in range(self.n_ways)]  # 1 x [36 x 256 x 256]
                tmp_ratio = torch.stack(tmp_ratio, dim=0)  # [1 x 36 x 256 x 256]
                ratio_sort = torch.sort(tmp_ratio.flatten(start_dim=2, end_dim=-1),dim=-1)[0]

                tmp_p_F_thres = [[(ratio_sort[batch,q,int(torch.sum(qry_mask_[batch,q]))] \
                                   + ratio_sort[batch,q,int(torch.sum(qry_mask_[batch,q])) + 1]) / 2
                                  for q in range(self.batch_size_q)] for batch in range(batch_size)]#1 x 36 x [1]
                tmp_p_F_thres = [torch.stack(tmp_p_F_thres[batch]).unsqueeze(-1).unsqueeze(-1) for batch in range(batch_size)]#1 x [36 x 1 x 1]

                tmp_ratio = [tmp_p_F_thres[i_w]*h_a2[i_w] \
                        / (tmp_p_F_thres[i_w]*h_a2[i_w] + (1-tmp_p_F_thres[i_w])*h_b2[i_w]) for i_w in range(self.n_ways)]  # 1 x [36 x 256 x 256]
                tmp_pred_ups = torch.stack(tmp_ratio, dim=0)  # [1 x 36 x 256 x 256]
                tmp_pred_ups = torch.cat((1 - tmp_pred_ups, tmp_pred_ups), dim=0).transpose(0,1)  # Softmax version of probability
                outputs_dict['OCP_MP'].append(tmp_pred_ups)

                estim_p_F_MP_hat = linr2.predict(X_lin)
                estim_p_F_MP_hat = 1 / (1 + np.exp(-estim_p_F_MP_hat))
                estim_p_F_MP_hat = np.clip(estim_p_F_MP_hat, a_min=1e-3, a_max=1 - 1e-3)
                estim_p_F_MP_hat = torch.tensor(estim_p_F_MP_hat).to(self.device)

                tmp_ratio = [estim_p_F_MP_hat * h_a[i_w] / (estim_p_F_MP_hat * h_a[i_w] + (1 - estim_p_F_MP_hat) * h_b[i_w])
                             for i_w in range(self.n_ways)]  # 1 x [36 x 32 x 32]
                tmp_ratio = torch.stack(tmp_ratio, dim=0)  # [1 x 36 x 32 x 32]
                tmp_pred_ups = F.interpolate(tmp_ratio, size=img_size, mode='bilinear', align_corners=True)  #
                tmp_pred_ups = torch.cat((1 - tmp_pred_ups, tmp_pred_ups), dim=0).transpose(0,1)  # Softmax version of probability
                outputs_dict['LinEst_MP'].append(tmp_pred_ups)

        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])
        if ((i < 50)&train)|(not train):
            print()
        if (not train)&(not analyze):
            output_dict = {method: torch.stack(outputs_dict[method], dim=1) for method in self.methods}
            output_dict = {method: output_dict[method].view(-1, *output.shape[1:]) for method in self.methods}

        if train:
            self.apply_EMA()

        if (not train) & (i < 50):
            self.cnt = self.cnt + 1

        if train:
            if self.ADNet|self.adnet:
                return output, (align_loss / batch_size), (self.t_loss_scaler * self.t_loss)
            else:
                return output, (align_loss / batch_size), np.nan
        else:
            if analyze:
                ###### Get ICP values of training data ######
                tmp_sq_dists = [2 * (1 - cos_) for cos_ in cos_s]
                tmp_dists = [(tmp_sq_dist + torch.finfo(torch.float32).eps) ** 0.5 for tmp_sq_dist in tmp_sq_dists]
                tmp_dists = torch.stack(tmp_dists, dim=0)
                tmp_dists = F.interpolate(tmp_dists, size=img_size, mode='bilinear', align_corners=True)
                tmp_dists_sort = torch.sort(tmp_dists.flatten(),dim=0)[0]
                supp_loc = supp_idx / (len(supp_img_all) - 1)
                supp_size = torch.mean(fore_mask).item()

                tmp_T_D_hat = (tmp_dists_sort[int(torch.sum(qry_mask_).item())]+tmp_dists_sort[int(torch.sum(qry_mask_).item())+1])/2

                p_F_part = self.dim_ * torch.log((self.sig2 / self.sig1)) - tmp_T_D_hat ** 2 / 2 * (1 / self.sig1 ** 2 - 1 / self.sig2 ** 2)
                tmp_p_F_hat = torch.exp(-torch.logsumexp(torch.cat([self.zero_tensor, p_F_part]), dim=0)).unsqueeze(0)

                eta_a = [[fg_multi_ws[i_w][i_p]*torch.exp((cos_s_multi[i_w][i_p] - 1) / self.sig1 ** 2) / self.sig1 ** self.dim_
                                                        for i_p in range(5)] for i_w in range(self.n_ways)]#1 x 5 x [1 x 32 x 32]
                eta_b = [[fg_multi_ws[i_w][i_p]*torch.exp((cos_s_multi[i_w][i_p] - 1) / self.sig2 ** 2) / self.sig2 ** self.dim_
                                                        for i_p in range(5)] for i_w in range(self.n_ways)] # 1 x 5 x [1 x 32 x 32]
                h_a  = [sum(eta_a[i_w]) for i_w in range(self.n_ways)] #1 x [1 x 32 x 32]
                h_b = [sum(eta_b[i_w]) for i_w in range(self.n_ways)]  # 1 x [1 x 32 x 32]

                tmp_ratio = [h_b[i_w]/(h_a[i_w]+h_b[i_w]) for i_w in range(self.n_ways)]#1 x [1 x 32 x 32]
                tmp_ratio = torch.stack(tmp_ratio, dim=0)
                tmp_ratio = F.interpolate(tmp_ratio, size=img_size, mode='bilinear', align_corners=True)
                ratio_sort = torch.sort(tmp_ratio.flatten())[0]
                if int(torch.sum(qry_mask_).item())>0:
                    tmp_p_F_MP_hat = ratio_sort[int(torch.sum(qry_mask_).item())-1]#When using multiple prototypes
                else:
                    tmp_p_F_MP_hat = torch.clip(2 * ratio_sort[0] - ratio_sort[1], min=0.0, max=1)

                return tmp_dists_sort, [supp_loc, supp_size, tmp_p_F_hat, tmp_p_F_MP_hat]
            else:
                return (output, (align_loss / batch_size), output_dict)

    def getSqDistance(self, fts, prototype):
        """
        Calculate the (squared) distance between features and prototypes

        # C: channel dimension
        Args:
            fts: input features
                expect shape: [1 x C x N_fg] or [1 x C x N_bg]
            prototype: prototype of one semantic class
                expect shape: [1 x C]
        """
        # Normalized setting (spherical embedding): ||X-Y||**2 = ||X||**2+||Y||**2-2*X*Y = 2-2*cosine_similarity(X,Y)
        sq_dist = 2*(1-F.cosine_similarity(fts, prototype[..., None], dim=1)) # Squared distance
        return sq_dist

    def negSim(self, fts, prototype):
        """
        Calculate the distance between features and prototypes

        # C: channel dimension
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

        # C: channel dimension
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
            # Only get corresponding features from mask
            fts_reshape = fts.view(*fts.shape[:2],-1) # 1 x C x (H*W)

            mask2 = mask.clone().view(-1)  # (H*W)
            mask2 = mask2.bool()
            fg_fts = fts_reshape[:, :, mask2]
            return fg_fts

    def getPrototype(self, fg_fts, n_protos=1, bg_fts=None):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """

        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        if n_protos==1:
            fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots for way in fg_fts]  # concat all fg_fts
            # fg_prototypes: just averaged and not l2 normalized
            return fg_prototypes
        else:
            #fg_fts: Way x shot x [1 x C x N_fg]
            #bg_fts: Way x shot x [1 x C x N_bg]
            assert (n_ways==1)
            assert (n_shots==1)
            fg_fts_np = fg_fts[0][0].detach().cpu().numpy()#[1 x C x N_fg]
            fg_fts_np = fg_fts_np[0].transpose()#[N_fg x C]
            fg_fts_np = fg_fts_np/np.linalg.norm(fg_fts_np, axis=-1, keepdims=True)
            if len(fg_fts_np)<n_protos:
                fg_fts_np = np.tile(fg_fts_np,(n_protos, 1))
            if bg_fts is not None:
                bg_fts_np = bg_fts[0][0].detach().cpu().numpy()  # [1 x C x N_bg]
                bg_fts_np = bg_fts_np[0].transpose()  # [N_bg x C]
                bg_fts_np = bg_fts_np / np.linalg.norm(bg_fts_np, axis=-1, keepdims=True)

            fg_feats = torch.from_numpy(fg_fts_np).float().to(self.device)
            if bg_fts is not None:
                bg_feats = torch.from_numpy(bg_fts_np).float().to(self.device)

            if bg_fts is not None:
                protos, w = self.estim_prototypes(fg_feats, n_protos=n_protos, max_iter=100, bg_feats=bg_feats)
            else:
                protos, w = self.estim_prototypes(fg_feats, n_protos=n_protos, max_iter=100)

            fg_prototypes = [protos for _ in range(n_ways)]
            fg_ws = [w for _ in range(n_ways)]

            return fg_prototypes, fg_ws

    def alignLoss(self, qry_fts, pred, supp_fts, fore_mask): #PAR loss from "PANet" paper
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Mask and get query prototype
        pred_mask = pred.argmax(dim=1, keepdim=True)  # N x 1 x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=1).float()  # N x (1 + Wa) x 1 x H' x W'
        qry_prototypes = torch.sum(qry_fts.unsqueeze(1) * pred_mask, dim=(0, 3, 4))
        qry_prototypes = qry_prototypes / (pred_mask.sum((0, 3, 4)) + 1e-5)  # (1 + Wa) x C

        # Compute the support loss
        loss = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                img_fts = supp_fts[way, [shot]]
                supp_sim = self.negSim(img_fts, qry_prototypes[[way + 1]])

                pred = self.getPred([supp_sim], [self.thresh_pred[way]])  # N x Wa x H' x W'
                pred_ups = F.interpolate(pred, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)
                pred_ups = torch.cat((1.0 - pred_ups, pred_ups), dim=1)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=img_fts.device)
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                # Compute Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(pred_ups, eps, 1 - eps))
                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

        return loss
    def getPred(self, sim, thresh, p_F=None, log=False, foreground=True):
        pred = []
        # We actually use equivalent equations of TPM using "sim" (anomaly score) for numerical stability
        for s, t in zip(sim, thresh):

            if p_F is None:
                if log:
                    if foreground:
                        pred.append(-self.softplus(0.5 * (s - t)))
                    else:
                        pred.append(-self.softplus(-0.5 * (s - t)))
                else:
                    if foreground:
                        pred.append(1.0 - torch.sigmoid(0.5 * (s - t)))
                    else:
                        pred.append(torch.sigmoid(0.5 * (s - t)))
            else:
                if log:
                    if foreground:
                        pred.append(-self.softplus(0.5 * (s - t)+ torch.log((1-p_F) / p_F)))
                    else:
                        pred.append(-self.softplus(-(0.5 * (s - t) + torch.log((1 - p_F) / p_F))))
                else:
                    if foreground:
                        pred.append(1.0 - torch.sigmoid(0.5 * (s - t)+ torch.log((1-p_F) / p_F)))
                    else:
                        pred.append(torch.sigmoid(0.5 * (s - t) + torch.log((1 - p_F) / p_F)))

        return torch.stack(pred, dim=1)  # N x Wa x H' x W'

    def softsign2(self,x):
        #(softsign(x)+1)/2x/(2*(1+|x|))+1/2
        return (self.softsign(x)+1)/2

    def analyze_fts(self, fts, fore_mask, fg_prototypes, epi, verbose=True, train=False, support=True):
        '''
            fore_mask
                ...
            query mask
                training: [1 x B? x H x W] or inference: [B? x N x H x W]
        '''
        if support:
            supp_tag = 'supp_'
        else:
            supp_tag = 'qry_'
        back_mask=1-fore_mask

        if support:
            fore_fts_ = [[self.getFeatures(fts[way, shot, [epi]], fore_mask[way, shot, [epi]], MAP=False)
                          for shot in range(self.n_shots)] for way in range(self.n_ways)]  #Way x shot x [1xCxN_fg]
            back_fts_ = [[self.getFeatures(fts[way, shot, [epi]], back_mask[way, shot, [epi]], MAP=False)
                          for shot in range(self.n_shots)] for way in range(self.n_ways)]  #Way x shot x [1xCxN_bg]
            n_sh = self.n_shots+0
        else:
            n_sh = self.batch_size_q + 0

            fore_fts_ = [[self.getFeatures(fts[epi, q][None, :], fore_mask[epi, q][None, :], MAP=False)
                             for q in range(self.batch_size_q)]]  #[N? x [1xCxN_fg]]
            back_fts_ = [[self.getFeatures(fts[epi, q][None, :], back_mask[epi, q][None, :], MAP=False)
                             for q in range(self.batch_size_q)]]  #[N? x [1xCxN_bg]]

            fts_ = [[torch.cat([fore_fts_[_][q], back_fts_[_][q]],dim=-1)
                     for q in range(self.batch_size_q)] for _ in range(len(fore_fts_))]

            fts_reshaped = [[fts_[_][q].squeeze(0).transpose(0, 1)
                     for q in range(self.batch_size_q)] for _ in range(len(fore_fts_))]

            if train:
                fts_n = [[fts_reshaped[_][q] / torch.norm(fts_reshaped[_][q], dim=1, keepdim=True)
                          for q in range(self.batch_size_q)] for _ in range(len(fore_fts_))]

                log_fts_n = [[self.log_map(fts_n[_][q])
                              for q in range(self.batch_size_q)] for _ in range(len(fore_fts_))]
                log_mn = [[torch.mean(log_fts_n[_][q], dim=0, keepdim=True)
                           for q in range(self.batch_size_q)] for _ in range(len(fore_fts_))]

                dist_mn = [[torch.mean(torch.norm(log_fts_n[_][q] - log_mn[_][q], dim=-1), dim=0)
                             for q in range(self.batch_size_q)] for _ in range(len(fore_fts_))]

                dist_std = [[torch.std(torch.norm(log_fts_n[_][q] - log_mn[_][q], dim=-1), dim=0)
                             for q in range(self.batch_size_q)] for _ in range(len(fore_fts_))]

                wandb.log({'dist_mn': dist_mn[0][0].item(),'dist_std_': dist_std[0][0].item()})

        sq_dist_fg = [[self.getSqDistance(fore_fts_[way][sh], fg_prototypes[way])
                       for sh in range(n_sh)] for way in range(self.n_ways)]  #Way x sh x [1 x C x N_fg]
        sq_dist_bg = [[self.getSqDistance(back_fts_[way][sh], fg_prototypes[way])
                       for sh in range(n_sh)] for way in range(self.n_ways)]  #Way x sh x [1 x C x N_bg]

        if (self.n_shots > 1)&support:
            print('    supp_fore_fts_:', len(fore_fts_), len(fore_fts_[0]), fore_fts_[0][0].shape)
        if (self.batch_size_q > 1)&(not support):
            print('    self.batch_size_q:', self.batch_size_q, fore_fts_[0][0].shape)

        ### Estimate T_D: decision threshold of distance ###
        p_F_cnt_list = [[sq_dist_fg[way][sh].shape[-1] / (sq_dist_fg[way][sh].shape[-1] + sq_dist_bg[way][sh].shape[-1])
                         for sh in range(n_sh)] for way in range(self.n_ways)] #sh: shot or q
        p_F_cnt_list = [[torch.tensor([p_F_cnt_list[way][sh]], device=self.device) for sh in range(n_sh)] for way in range(self.n_ways)] #sh: shot or q]]

        if (train)|verbose:
            fore_mn_list = [[torch.mean(torch.sqrt(sq_dist_fg[way][sh] + torch.finfo(torch.float32).eps)).detach()
                             for sh in range(n_sh)] for way in range(self.n_ways)]  #sh: shot or q
            back_mn_list = [[torch.mean(torch.sqrt(sq_dist_bg[way][sh] + torch.finfo(torch.float32).eps)).detach()
                             for sh in range(n_sh)] for way in range(self.n_ways)]  #sh: shot or q

            mid_thres_list = [[(fore_mn_list[way][sh] + back_mn_list[way][sh]) / 2
                               for sh in range(n_sh)] for way in range(self.n_ways)]  #sh: shot or q

            dists_list = [[torch.sort(torch.sqrt(torch.cat([sq_dist_fg[way][sh], sq_dist_bg[way][sh]], dim=1) + torch.finfo(torch.float32).eps))[0]
                 for sh in range(n_sh)] for way in range(self.n_ways)]  #sh: shot or q
            p_F_thres_list = [[(dists_list[way][sh][0][sq_dist_fg[way][sh].shape[-1]]+dists_list[way][sh][0][sq_dist_fg[way][sh].shape[-1]+1])/2
                 for sh in range(n_sh)] for way in range(self.n_ways)]  #sh: shot or q

            self.T_D_list = [[p_F_thres_list[way][sh].item() for sh in range(n_sh)] for way in range(self.n_ways)]
            if support:
                self.supp_T_D_list = [[p_F_thres_list[way][sh].item() for sh in range(n_sh)] for way in
                                          range(self.n_ways)]  #sh: shot or q
            else:
                self.qry_T_D_list = [[p_F_thres_list[way][sh].item() for sh in range(n_sh)] for way in
                                     range(self.n_ways)]  #sh: shot or q
                FP_ind_list = [[torch.sqrt(sq_dist_bg[way][sh] + torch.finfo(torch.float32).eps) <= self.qry_T_D_list[way][sh]
                     for sh in range(n_sh)] for way in range(self.n_ways)]  #sh: shot or q
                FN_ind_list = [[torch.sqrt(sq_dist_fg[way][sh] + torch.finfo(torch.float32).eps) > self.qry_T_D_list[way][sh]
                     for sh in range(n_sh)] for way in range(self.n_ways)]  #sh: shot or q
                self.FP_len_list = [[torch.sum(FP_ind_list[way][sh]) for sh in range(n_sh)] for way in range(self.n_ways)]
                self.FN_len_list = [[torch.sum(FN_ind_list[way][sh]) for sh in range(n_sh)] for way in range(self.n_ways)]
                self.F_list = [[sq_dist_fg[way][sh].shape[-1]
                     for sh in range(n_sh)] for way in range(self.n_ways)]  #sh: shot or q
                self.B_list = [[sq_dist_bg[way][sh].shape[-1]
                     for sh in range(n_sh)] for way in range(self.n_ways)]  #sh: shot or q

                self.FP_med_list = [[torch.median(torch.sqrt(sq_dist_bg[way][sh][FP_ind_list[way][sh]] + torch.finfo(torch.float32).eps))
                                     if self.FP_len_list[way][sh] > 0 else torch.tensor(float('nan'))
                                     for sh in range(n_sh)] for way in range(self.n_ways)]  #sh: shot or q
                self.FN_med_list = [[torch.median(torch.sqrt(sq_dist_fg[way][sh][FN_ind_list[way][sh]] + torch.finfo(torch.float32).eps))
                                     if self.FN_len_list[way][sh] > 0 else torch.tensor(float('nan'))
                                     for sh in range(n_sh)] for way in range(self.n_ways)]  #sh: shot or q
                lgm_list= [[torch.mean(torch.log(dists_list[way][sh][0])) for sh in range(n_sh)] for way in range(self.n_ways)]

            if (self.n_shots == 1) & (not support):
                self.qry_p_F_thres = p_F_thres_list[0][0].clone()
                self.qry_lgm = lgm_list[0][0].clone()
                if train:
                    wandb.log({'GeoMean': (torch.exp(self.qry_lgm)).item()})

        if train&(not support)&(self.n_shots == 1)&(self.n_ways == 1):
            self.T_D_hat_ = self.qry_p_F_thres.detach() + 0.0

        for way in range(self.n_ways):
            for sh in range(n_sh):
                if verbose:
                    if support:
                        print(f'    way: {way}, shot: {sh}')
                    else:
                        print(f'    way: {way}, q: {sh}')
                    print('      %sp_F_cnt: %f' % (supp_tag, p_F_cnt_list[way][sh].item()))
                    if p_F_cnt_list[way][sh].item()>0:
                        print('      %sfore_mn: %f back_mn: %f mid_thres: %f' % (supp_tag, fore_mn_list[way][sh].item(),
                                                                                 back_mn_list[way][sh].item(),
                                                                                 mid_thres_list[way][sh].item()))
                        print('      %sself.T_D_: %f' % (supp_tag, self.T_D_list[way][sh]))
                        print('      self.T_D-%sself.T_D_: %f' % (supp_tag, self.T_D.item()-self.T_D_list[way][sh]))
                if train&verbose:
                    wandb.log({supp_tag+'p_F_cnt': p_F_cnt_list[way][sh].item(),
                               supp_tag+'fore_mn': fore_mn_list[way][sh].item(), supp_tag+'back_mn': back_mn_list[way][sh].item(),
                               supp_tag+'mid_thres': mid_thres_list[way][sh].item()})
                    if (not support):
                        wandb.log({'FP_med': self.FP_med_list[way][sh].item()})
                        wandb.log({'FN_med': self.FN_med_list[way][sh].item()})
                        wandb.log({'FP_len': self.FP_len_list[way][sh].item()})
                        wandb.log({'FN_len': self.FN_len_list[way][sh].item()})
                        if self.B_list[way][sh]>0:
                            wandb.log({'FP_rate': self.FP_len_list[way][sh].item()/self.B_list[way][sh]})
                        if self.F_list[way][sh] > 0:
                            wandb.log({'FN_rate': self.FN_len_list[way][sh].item() / self.F_list[way][sh]})


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
        #Load parameter values for "analyze" and "inference"
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
            print('    self.alpha(pre)', self.alpha)

        if self.fix_alpha is None:
            if self.learn_alpha:
                self.alpha = torch.exp(self.l_alpha)
            else:
                if self.chg_alpha:  # model alpha follows self.alpha_hat
                    self.alpha = self.alpha_hat.data.detach()
                elif self.chg_alpha2:
                    self.alpha = self.alpha_hat2.data.detach()
                else:
                    pass
        if verbose:
            print('    self.alpha(post)', self.alpha)
            print()
            if self.ADNet:
                print('    self.ADNet', self.ADNet)
            if self.adnet:
                print('    self.adnet', self.adnet)
            if self.fix_T_D is not None:
                print('    self.fix_T_D', self.fix_T_D)
            if self.chg_T_D:
                print('    self.chg_T_D', self.chg_T_D)
            print('    self.T_D(pre)', self.T_D, 'self.T_S', self.T_S)

        #Calculate sigma values using (given) self.alpha, self.T_D, self.p_F
        if self.fix_T_D is None:
            if self.ADNet:
                self.T_D = torch.sqrt(2 * (1 + self.T_S / self.alpha))
                sig_part = torch.exp(self.alpha * self.T_D ** 2 / (2 * self.dim_)) / (self.p_F / self.p_B) ** (2 / self.dim_)
                self.sig1_ = (2 / self.alpha * (1 - 1 / sig_part)) ** 0.5
                self.sig2_ = (2 / self.alpha * (sig_part - 1)) ** 0.5
            elif self.adnet:
                self.T_S = self.dim_ * self.softplus(self.pre_T_S) + 2 * torch.log(self.p_F / self.p_B) - self.alpha
                self.T_D = torch.sqrt(2 * (1 + self.T_S / self.alpha))
                sig_part = torch.exp(self.alpha * self.T_D ** 2 / (2 * self.dim_))/ (self.p_F / self.p_B) ** (2 / self.dim_)
                self.sig1_ = (2 / self.alpha * (1 - 1/sig_part)) ** 0.5
                self.sig2_ = (2 / self.alpha * (sig_part - 1)) ** 0.5
            else:
                if self.chg_T_D:
                    self.T_D = self.T_D_hat.data.detach()
                    self.T_S = self.alpha * (self.T_D ** 2 / 2 - 1)
                    if self.fix_sig2 is None:  # sigma values are based on self.alpha and self.T_D
                        sig_part = torch.exp(self.alpha * self.T_D ** 2 / (2 * self.dim_))/ (self.p_F / self.p_B) ** (2 / self.dim_)
                        self.sig1_ = (2 / self.alpha * (1 - 1/sig_part)) ** 0.5
                        self.sig2_ = (2 / self.alpha * (sig_part - 1)) ** 0.5
                    else:  # self.sig2 is fixed. self.sig1 value is based on self.sig2 and self.T_D
                        if (self.fix_alpha is not None) | self.chg_alpha | self.chg_alpha2 | self.learn_alpha:
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
                            self.T_S.data = self.alpha * (self.T_D ** 2 / 2 - 1)
                            sig_part = torch.exp(self.alpha * self.T_D ** 2 / (2 * self.dim_)) / (self.p_F / self.p_B) ** (2 / self.dim_)
                            self.sig1_ = (2 / self.alpha * (1 - 1/sig_part)) ** 0.5
                            self.sig2_ = (2 / self.alpha * (sig_part - 1)) ** 0.5
                    else:  # self.sig2 is fixed. self.sig1 and self.T_D values are based on self.sig2 and self.alpha
                        if (self.fix_alpha is not None) | self.chg_alpha | self.chg_alpha2 | self.learn_alpha:
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
                    if not (self.chg_T_D|self.ADNet|self.adnet|(self.fix_T_D is not None)):
                        sig_part = (self.p_F / self.p_B) ** (-2/self.dim_)*torch.exp(self.alpha * self.T_D_hat **2 / (2*self.dim_))
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
        #Exponential moving average of certain parameters
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
            self.T_D_hat.data = (1e-3) * self.T_D_hat_.detach() + (1 - 1e-3) * self.T_D_hat.data  #Use EMA to get more smooth change of T_D
            self.T_D_hat.data = torch.clip(self.T_D_hat.data, min=1e-3, max=2 ** 0.5 - 1e-3)
            self.T_D_hat_ = torch.tensor(np.nan)

        p_F_part = self.dim_ * torch.log((self.sig2 / self.sig1)) \
                   - self.T_D_hat ** 2 / 2 * (1 / self.sig1 ** 2 - 1 / self.sig2 ** 2)
        self.p_F_hat.data = torch.exp(-torch.logsumexp(torch.cat([self.zero_tensor, p_F_part]), dim=0)).unsqueeze(0)

        wandb.log({'self.T_D_hat': self.T_D_hat.data.item(), 'self.T_D_hat_MSE': self.T_D_hat_MSE.item(),
                   'self.p_F': self.p_F.item(), 'self.p_F_hat': self.p_F_hat.item()
                   #'self.alpha_hat': self.alpha_hat.item(),
                   #'self.alpha_hat2': self.alpha_hat2.item()
                    })
        wandb.log({'self.sig1': self.sig1.item(), 'self.sig2': self.sig2.item(),
                   'self.T_D': self.T_D.item(), 'self.T_S': self.T_S.item(), 'self.alpha': self.alpha.item()})

    def estim_prototypes(self, fg_feats, n_protos=1, protos_init=None, w_init=None, bg_feats=None, max_iter=100):
        # Use spherical EM algorithm for estimating GMMs (prototypes and weights)
        # feats: [n, dim], only use foreground features

        sig1 = self.sig1.detach()
        fg_feats = fg_feats/torch.norm(fg_feats,dim=-1,keepdim=True)

        if bg_feats is not None:
            print('Not implemented!!!')
            raise

        if protos_init is None:
            protos = torch.mean(fg_feats, dim=0, keepdim=True)  # [1, dim]
            protos = torch.tile(protos, [n_protos, 1])  # [n_protos, dim]
            protos = protos + 0.1 * (torch.rand(size=[n_protos, fg_feats.shape[-1]]).to(self.device) - 0.5)  # [n_protos, dim]
        else:
            protos = protos_init.clone()
        protos = protos/torch.norm(protos,dim=-1,keepdim=True)

        if w_init is None:
            l_w = torch.zeros([n_protos, 1]).to(self.device)  # [n_protos,1]
            l_w = l_w + 0.1 * torch.rand([n_protos, 1]).to(self.device)
        else:
            l_w = torch.log(w_init + 1e-8)
        w = torch.exp(l_w) / torch.sum(torch.exp(l_w), dim=0, keepdim=True)
        w = w.t()  # [1,n_protos]

        protos = protos / torch.norm(protos, dim=-1, keepdim=True)
        protos_dist = torch.cdist(fg_feats, protos)  # [n,n_protos]
        L_F = torch.log(w) - 1 / 2 * protos_dist ** 2 / sig1 ** 2  # [n_fg,n_protos]
        L_F2 = torch.logsumexp(L_F, dim=1, keepdim=True)  # [n_fg,1]
        L_F3 = torch.sum(L_F2, dim=0)  # []
        loss_best = -L_F3.item()
        loss_avg = loss_best+0.0
        chk=False

        for i in range(max_iter):
            protos = protos / torch.norm(protos, dim=-1, keepdim=True)
            protos_dist = torch.cdist(fg_feats, protos)  # [n,n_protos]

            # E-step
            p_z_ = w * torch.exp(-1 / 2 * protos_dist ** 2 / sig1 ** 2) / sig1 ** 1  # [n_fg,n_protos]
            p_z = p_z_ / torch.sum(p_z_, dim=1, keepdim=True)  # [n_fg,n_protos]

            # M-step
            protos = torch.sum(fg_feats.unsqueeze(1) * p_z.unsqueeze(-1), dim=0) / torch.sum(p_z, dim=0).unsqueeze(-1)  # [n_protos, dim]
            w = torch.mean(p_z, dim=0, keepdim=True)  # [1,n_protos]
            w = w / torch.sum(w)

            L_F = torch.log(w) - 1 / 2 * protos_dist ** 2 / sig1 ** 2  # [n_fg,n_protos]
            L_F2 = torch.logsumexp(L_F, dim=1, keepdim=True)  # [n_fg,1]
            L_F3 = torch.sum(L_F2, dim=0)  # []
            loss = -L_F3.item()

            if (loss_avg / (loss+1e-8) - 1) <1e-3:
                chk=True
                if i>=5:
                    break
            else:
                loss_avg = (loss+loss_avg)/2

            if loss<loss_best:
                loss_best = loss

            if chk&(i % 2 == 0):
                print('    i_', i,'  loss', loss)
        protos = protos / torch.norm(protos, dim=-1, keepdim=True)
        w = w.t()  # [n_protos, 1]

        return protos, w

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

