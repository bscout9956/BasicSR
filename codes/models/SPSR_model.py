import os
import logging
from utils.util import OrderedDefaultDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

import models.networks as networks
from .base_model import BaseModel
from models.modules.LPIPS import perceptual_loss as models #import models.modules.LPIPS as models
from models.modules.loss import GANLoss, GradientPenaltyLoss, HFENLoss, TVLoss, CharbonnierLoss, ElasticLoss, RelativeL1, L1CosineSim
from models.modules.losses.ssim2 import SSIM, MS_SSIM #implementation for use with any PyTorch
# from models.modules.losses.ssim3 import SSIM, MS_SSIM #for use of the PyTorch 1.1.1+ optimized implementation
logger = logging.getLogger('base')

import models.lr_schedulerR as lr_schedulerR


class Get_gradient(nn.Module):
    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = [[0, -1, 0], 
                    [0, 0, 0], 
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0], 
                    [-1, 0, 1], 
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=2)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=2)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=2)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=2)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=2)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=2)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x

class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0], 
                    [0, 0, 0], 
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0], 
                    [-1, 0, 1], 
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding = 1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding = 1)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding = 1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding = 1)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding = 1)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding = 1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x


class SPSRModel(BaseModel):
    def __init__(self, opt):
        super(SPSRModel, self).__init__(opt)
        train_opt = opt['train']
        
        # define networks and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)  # G
        if self.is_train:
            self.netG.train()
            if train_opt['gan_weight']:
                self.netD = networks.define_D(opt).to(self.device) # D
                self.netD.train()
            if train_opt['gradient_gan_weight']:
                self.netD_grad = networks.define_D(opt).to(self.device) # D_grad
                self.netD_grad.train()
        self.load()  # load G, D, and D_grad if needed

        # define losses, optimizer and scheduler
        if self.is_train:
            # G pixel loss
            if train_opt['pixel_weight']:
                if train_opt['pixel_criterion']:
                    l_pix_type = train_opt['pixel_criterion']
                else: # default to l1
                    l_pix_type = 'l1'
                
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                elif l_pix_type == 'cb':
                    self.cri_pix = CharbonnierLoss().to(self.device)
                elif l_pix_type == 'elastic':
                    self.cri_pix = ElasticLoss().to(self.device)
                elif l_pix_type == 'relativel1':
                    self.cri_pix = RelativeL1().to(self.device)
                elif l_pix_type == 'l1cosinesim':
                    self.cri_pix = L1CosineSim().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = train_opt['pixel_weight']
            else:
                logger.info('Remove pixel loss.')
                self.cri_pix = None

            # G feature loss
            if train_opt['feature_weight']:
                if train_opt['feature_criterion']:
                    l_fea_type = train_opt['feature_criterion']
                else: #default to l1
                    l_fea_type = 'l1'
                
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                elif l_fea_type == 'cb':
                    self.cri_fea = CharbonnierLoss().to(self.device)
                elif l_fea_type == 'elastic':
                    self.cri_fea = ElasticLoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
                self.l_fea_w = train_opt['feature_weight']
            else:
                logger.info('Remove feature loss.')
                self.cri_fea = None
            if self.cri_fea:  # load VGG perceptual loss
                self.netF = networks.define_F(opt, use_bn=False).to(self.device)

            #HFEN loss
            if train_opt['hfen_weight']:
                l_hfen_type = train_opt['hfen_criterion']
                if train_opt['hfen_presmooth']:
                    pre_smooth = train_opt['hfen_presmooth']
                else:
                    pre_smooth = False #train_opt['hfen_presmooth']
                if l_hfen_type:
                    if l_hfen_type == 'rel_l1' or l_hfen_type == 'rel_l2':
                        relative = True
                    else:
                        relative = False #True #train_opt['hfen_relative']
                if l_hfen_type:
                    self.cri_hfen =  HFENLoss(loss_f=l_hfen_type, device=self.device, pre_smooth=pre_smooth, relative=relative).to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_hfen_type))
                self.l_hfen_w = train_opt['hfen_weight']
            else:
                logger.info('Remove HFEN loss.')
                self.cri_hfen = None

            #TV loss
            if train_opt['tv_weight']:
                self.l_tv_w = train_opt['tv_weight']
                l_tv_type = train_opt['tv_type']
                if train_opt['tv_norm']:
                    tv_norm = train_opt['tv_norm']
                else:
                    tv_norm = 1
                
                if l_tv_type == 'normal':
                    self.cri_tv = TVLoss(self.l_tv_w, p=tv_norm).to(self.device) 
                elif l_tv_type == '4D':
                    self.cri_tv = TVLoss4D(self.l_tv_w).to(self.device) #Total Variation regularization in 4 directions
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_tv_type))
            else:
                logger.info('Remove TV loss.')
                self.cri_tv = None

            #SSIM loss
            if train_opt['ssim_weight']:
                self.l_ssim_w = train_opt['ssim_weight']
                
                if train_opt['ssim_type']:
                    l_ssim_type = train_opt['ssim_type']
                else: #default to ms-ssim
                    l_ssim_type = 'ms-ssim'
                    
                if l_ssim_type == 'ssim':
                    self.cri_ssim = SSIM(win_size=11, win_sigma=1.5, size_average=True, data_range=1., channel=3).to(self.device)
                elif l_ssim_type == 'ms-ssim':
                    self.cri_ssim = MS_SSIM(win_size=11, win_sigma=1.5, size_average=True, data_range=1., channel=3).to(self.device)
            else:
                logger.info('Remove SSIM loss.')
                self.cri_ssim = None

            #LPIPS loss
            """
            lpips_spatial = False
            if train_opt['lpips_spatial']:
                #lpips_spatial = True if train_opt['lpips_spatial'] == True else False
                lpips_spatial = True if train_opt['lpips_spatial'] else False
            lpips_GPU = False
            if train_opt['lpips_GPU']:
                #lpips_GPU = True if train_opt['lpips_GPU'] == True else False
                lpips_GPU = True if train_opt['lpips_GPU'] else False
            #"""
            #"""
            lpips_spatial = True #False # Return a spatial map of perceptual distance. Meeds to use .mean() for the backprop if True, the mean distance is approximately the same as the non-spatial distance
            lpips_GPU = True # Whether to use GPU for LPIPS calculations
            if train_opt['lpips_weight']:
                self.l_lpips_w = train_opt['lpips_weight']
                # Can use original off-the-shelf uncalibrated networks 'net' or Linearly calibrated models (LPIPS) 'net-lin'
                if train_opt['lpips_type']:
                    lpips_type = train_opt['lpips_type']
                else: # Default use linearly calibrated models, better results
                    lpips_type = 'net-lin'
                # Can set net = 'alex', 'squeeze' or 'vgg' or Low-level metrics 'L2' or 'ssim'
                if train_opt['lpips_net']:
                    lpips_net = train_opt['lpips_net']
                else: # Default use VGG for feature extraction
                    lpips_net = 'vgg'
                self.cri_lpips = models.PerceptualLoss(model=lpips_type, net=lpips_net, use_gpu=lpips_GPU, model_path=None, spatial=lpips_spatial) #.to(self.device) 
                # Linearly calibrated models (LPIPS)
                # self.cri_lpips = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=lpips_GPU, model_path=None, spatial=lpips_spatial) #.to(self.device) 
                # self.cri_lpips = models.PerceptualLoss(model='net-lin', net='vgg', use_gpu=lpips_GPU, model_path=None, spatial=lpips_spatial) #.to(self.device) 
                # Off-the-shelf uncalibrated networks
                # Can set net = 'alex', 'squeeze' or 'vgg'
                # self.cri_lpips = models.PerceptualLoss(model='net', net='alex', use_gpu=lpips_GPU, model_path=None, spatial=lpips_spatial)
                # Low-level metrics
                # self.cri_lpips = models.PerceptualLoss(model='L2', colorspace='Lab', use_gpu=lpips_GPU)
                # self.cri_lpips = models.PerceptualLoss(model='ssim', colorspace='RGB', use_gpu=lpips_GPU)
            else:
                logger.info('Remove LPIPS loss.')
                self.cri_lpips = None
            #"""
            
            # GD gan loss
            if train_opt['gan_weight']:
                self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
                self.l_gan_w = train_opt['gan_weight']
                # D_update_ratio and D_init_iters are for WGAN
                self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
                self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0

                if train_opt['gan_type'] == 'wgan-gp':
                    self.random_pt = torch.Tensor(1, 1, 1, 1).to(self.device)
                    # gradient penalty loss
                    self.cri_gp = GradientPenaltyLoss(device=self.device).to(self.device)
                    self.l_gp_w = train_opt['gp_weigth']
            else:
                logger.info('Remove GAN loss.')
                self.cri_gan = None

            # gradient_pixel_loss
            if train_opt['gradient_pixel_weight']:
                self.cri_pix_grad = nn.MSELoss().to(self.device)
                self.l_pix_grad_w = train_opt['gradient_pixel_weight']
            else:
                logger.info('Remove gradient pixel loss.')
                self.cri_pix_grad = None

            # gradient_gan_loss
            if train_opt['gradient_gan_weight']:
                self.cri_grad_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
                self.l_gan_grad_w = train_opt['gradient_gan_weight']
            else:
                logger.info('Remove gradient GAN loss.')
                self.cri_grad_gan = None

            # G_grad pixel loss
            if train_opt['pixel_branch_weight']:
                l_pix_type = train_opt['pixel_branch_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix_branch = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix_branch = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_branch_w = train_opt['pixel_branch_weight']
            else:
                logger.info('Remove G_grad pixel loss.')
                self.cri_pix_branch = None

            # optimizers
            # G
            optim_params = []
            for k, v in self.netG.named_parameters():  # optimize part of the model
                if v.requires_grad:
                    optim_params.append(v)
                elif 'module.get_g_nopadding.weight_' not in k:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = networks.define_optim(train_opt, optim_params, 'G')
            self.optimizers.append(self.optimizer_G)
            
            # D
            if self.cri_gan:
                self.optimizer_D = networks.define_optim(train_opt, self.netD.parameters(), 'D')
                self.optimizers.append(self.optimizer_D)

            # D_grad
            if self.cri_grad_gan:
                self.optimizer_D_grad = networks.define_optim(train_opt, self.netD_grad.parameters(), 'D_grad')
                self.optimizers.append(self.optimizer_D_grad)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, \
                        train_opt['lr_steps'], train_opt['lr_gamma']))
            elif train_opt['lr_scheme'] == 'MultiStepLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_schedulerR.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'StepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.StepLR(optimizer, \
                        train_opt['lr_step_size'], train_opt['lr_gamma']))
            elif train_opt['lr_scheme'] == 'StepLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_schedulerR.StepLR_Restart(optimizer, step_sizes=train_opt['lr_step_sizes'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_schedulerR.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('Learning rate scheme ("lr_scheme") not defined or not recognized.')

            self.get_grad = Get_gradient()
            self.get_grad_nopadding = Get_gradient_nopadding()
            # Branch_init_iters
            self.Branch_pretrain = train_opt['Branch_pretrain'] if train_opt['Branch_pretrain'] else 0
            self.Branch_init_iters = train_opt['Branch_init_iters'] if train_opt['Branch_init_iters'] else 1
            self.log_dict = OrderedDefaultDict()
        # print network
        #self.print_network()

    def feed_data(self, data, need_HR=True):
        # LR
        self.var_L = data['LR'].to(self.device)
        if need_HR:  # train or val
            self.var_H = data['HR'].to(self.device)
            input_ref = data['ref'] if 'ref' in data else data['HR']
            self.var_ref = input_ref.to(self.device)

    def optimize_parameters(self, gen, step):
        self.log_dict.clear()
        self.optimizer_G.zero_grad()
        if self.cri_gan:
            self.optimizer_D.zero_grad()
        if self.cri_grad_gan:
            self.optimizer_D_grad.zero_grad()
        
        bm = self.opt['batch_multiplier']
        for _ in range(bm):
            self.feed_data(next(gen))
            
            self.fake_H_branch, self.fake_H, self.grad_LR = self.netG(self.var_L)
            
            self.fake_H_grad = self.get_grad(self.fake_H)
            self.var_H_grad = self.get_grad(self.var_H)
            self.var_ref_grad = self.get_grad(self.var_ref)
            self.var_H_grad_nopadding = self.get_grad_nopadding(self.var_H)
            
            # G
            l_g_total = 0
            if self.cri_gan:
                for p in self.netD.parameters():
                    p.requires_grad = False
            
            if self.cri_grad_gan:
                for p in self.netD_grad.parameters():
                    p.requires_grad = False
            
            if(self.Branch_pretrain): 
                if(step < self.Branch_init_iters):
                    for k,v in self.netG.named_parameters():
                        if 'f_' not in k :
                            v.requires_grad=False
                else:
                    for k,v in self.netG.named_parameters():
                        if 'f_' not in k :
                            v.requires_grad=True
            
            if not self.cri_gan or (step % self.D_update_ratio == 0 and step > self.D_init_iters):
                if self.cri_pix:  # pixel loss
                    l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
                    l_g_total += l_g_pix
                    self.log_dict['l_g_pix'] += l_g_pix.item()
                if self.cri_ssim: # structural loss (Structural Dissimilarity)
                    l_g_ssim = (1. - (self.l_ssim_w * self.cri_ssim(self.fake_H, self.var_H))) / bm #using ssim2.py
                    if not torch.isnan(l_g_ssim).any(): #at random, l_g_ssim is returning NaN for ms-ssim, which breaks the model. Temporary hack, until I find out what's going on.
                        l_g_total += l_g_ssim
                        self.log_dict['l_g_ssim'] += l_g_ssim.item()
                if self.cri_fea:  # feature loss
                    real_fea = self.netF(self.var_H).detach()
                    fake_fea = self.netF(self.fake_H)
                    l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea) / bm
                    l_g_total += l_g_fea
                    self.log_dict['l_g_fea'] += l_g_fea.item()
                if self.cri_hfen:  # HFEN loss
                    l_g_HFEN = self.l_hfen_w * self.cri_hfen(self.fake_H, self.var_H) / bm
                    l_g_total += l_g_HFEN
                    self.log_dict['l_g_HFEN'] += l_g_HFEN.item()
                if self.cri_tv: #TV loss
                    l_g_tv = self.cri_tv(self.fake_H) / bm #note: the weight is already multiplied inside the function, doesn't need to be here
                    l_g_total += l_g_tv
                    self.log_dict['l_g_tv'] += l_g_tv.item()
                if self.cri_lpips: #LPIPS loss
                    # If "spatial = False" .forward() returns a scalar value, if "spatial = True", returns a map (5 layers for vgg and alex or 7 for squeeze)
                    l_g_lpips = self.cri_lpips.forward(self.fake_H, self.var_H, normalize=True).mean() / bm # -> # If normalize is True, assumes the images are between [0,1] and then scales them between [-1,+1]
                    l_g_total += l_g_lpips
                    self.log_dict['l_g_lpips'] += l_g_lpips.item()
                if self.cri_pix_grad: #gradient pixel loss
                    l_g_pix_grad = self.l_pix_grad_w * self.cri_pix_grad(self.fake_H_grad, self.var_H_grad) / bm
                    l_g_total += l_g_pix_grad
                    self.log_dict['l_g_pix_grad'] += l_g_pix_grad.item()
                if self.cri_pix_branch: #branch pixel loss
                    l_g_pix_grad_branch = self.l_pix_branch_w * self.cri_pix_branch(self.fake_H_branch, self.var_H_grad_nopadding) / bm
                    l_g_total += l_g_pix_grad_branch
                    self.log_dict['l_g_pix_grad_branch'] += l_g_pix_grad_branch.item()
                if self.cri_gan: # G gan + cls loss
                    pred_g_fake = self.netD(self.fake_H)
                    pred_d_real = self.netD(self.var_ref).detach()
                    l_g_gan = self.l_gan_w * (self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                                              self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / (bm * 2)
                    l_g_total += l_g_gan
                    self.log_dict['l_g_gan'] += l_g_gan.item()
                if self.cri_grad_gan: # grad G gan + cls loss
                    pred_g_fake_grad = self.netD_grad(self.fake_H_grad)
                    pred_d_real_grad = self.netD_grad(self.var_ref_grad).detach()
                    l_g_gan_grad = self.l_gan_grad_w * (self.cri_grad_gan(pred_d_real_grad - torch.mean(pred_g_fake_grad), False) + 
                                                        self.cri_grad_gan(pred_g_fake_grad - torch.mean(pred_d_real_grad), True)) / (bm * 2)
                    l_g_total += l_g_gan_grad
                    self.log_dict['l_g_gan_grad'] += l_g_gan_grad.item()
                
                l_g_total.backward()
            
            if self.cri_gan:
                # D
                for p in self.netD.parameters():
                    p.requires_grad = True
                
                l_d_total = 0
                pred_d_real = self.netD(self.var_ref)
                pred_d_fake = self.netD(self.fake_H.detach())  # detach to avoid BP to G
                l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True) / bm
                l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), False) / bm
                self.log_dict['l_d_real'] += l_d_real.item()
                self.log_dict['l_d_fake'] += l_d_fake.item()
                
                l_d_total = (l_d_real + l_d_fake) / 2
                
                if self.opt['train']['gan_type'] == 'wgan-gp':
                    batch_size = self.var_ref.size(0)
                    if self.random_pt.size(0) != batch_size:
                        self.random_pt.resize_(batch_size, 1, 1, 1)
                    self.random_pt.uniform_()  # Draw random interpolation points
                    interp = self.random_pt * self.fake_H.detach() + (1 - self.random_pt) * self.var_ref
                    interp.requires_grad = True
                    interp_crit, _ = self.netD(interp)
                    l_d_gp = self.l_gp_w * self.cri_gp(interp, interp_crit) / bm
                    l_d_total += l_d_gp
                    self.log_dict['l_d_gp'] += l_d_gp.item()
                
                l_d_total.backward()
                
                # D outputs
                self.log_dict['D_real'] += torch.mean(pred_d_real.detach()).item() / bm
                self.log_dict['D_fake'] += torch.mean(pred_d_fake.detach()).item() / bm
            
            if self.cri_grad_gan:
                # D_grad
                for p in self.netD_grad.parameters():
                    p.requires_grad = True
                
                l_d_total_grad = 0
                
                pred_d_real_grad = self.netD_grad(self.var_ref_grad)
                pred_d_fake_grad = self.netD_grad(self.fake_H_grad.detach())  # detach to avoid BP to G
                l_d_real_grad = self.cri_grad_gan(pred_d_real_grad - torch.mean(pred_d_fake_grad), True) / bm
                l_d_fake_grad = self.cri_grad_gan(pred_d_fake_grad - torch.mean(pred_d_real_grad), False) / bm
                self.log_dict['l_d_real_grad'] += l_d_real_grad.item()
                self.log_dict['l_d_fake_grad'] += l_d_fake_grad.item()
                
                l_d_total_grad = (l_d_real_grad + l_d_fake_grad) / 2
                
                l_d_total_grad.backward()
                
                # D_grad outputs
                self.log_dict['D_real_grad'] += torch.mean(pred_d_real_grad.detach()).item() / bm
                self.log_dict['D_fake_grad'] += torch.mean(pred_d_fake_grad.detach()).item() / bm
        
        if not self.cri_gan or (step % self.D_update_ratio == 0 and step > self.D_init_iters):
            self.optimizer_G.step()
        if self.cri_gan:
            self.optimizer_D.step()
        if self.cri_grad_gan:
            self.optimizer_D_grad.step()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H_branch, self.fake_H, self.grad_LR = self.netG(self.var_L)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_HR=True):
        out_dict = {}
        out_dict['LR'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        out_dict['SR_branch'] = self.fake_H_branch.detach()[0].float().cpu()
        out_dict['LR_grad'] = self.grad_LR.detach()[0].float().cpu()
        if need_HR:
            out_dict['HR'] = self.var_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)
        if self.is_train:
            # Discriminator
            if self.cri_gan:
                s, n = self.get_network_description(self.netD)
                if isinstance(self.netD, nn.DataParallel):
                    net_struc_str = '{} - {}'.format(self.netD.__class__.__name__,
                                                    self.netD.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netD.__class__.__name__)

                logger.info('Network D structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
                logger.info(s)

            # Discriminator grad
            if self.cri_grad_gan:
                s, n = self.get_network_description(self.netD_grad)
                if isinstance(self.netD_grad, nn.DataParallel):
                    net_struc_str = '{} - {}'.format(self.netD_grad.__class__.__name__,
                                                    self.netD_grad.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netD_grad.__class__.__name__)

                logger.info('Network D_grad structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
                logger.info(s)

            if self.cri_fea:  # F, Perceptual Network
                s, n = self.get_network_description(self.netF)
                if isinstance(self.netF, nn.DataParallel):
                    net_struc_str = '{} - {}'.format(self.netF.__class__.__name__,
                                                    self.netF.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netF.__class__.__name__)

                logger.info('Network F structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
                logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, strict=False)
        if self.opt['is_train']:
            load_path_D = self.opt['path']['pretrain_model_D']
            if self.opt['train']['gan_weight'] and load_path_D is not None:
                logger.info('Loading pretrained model for D [{:s}] ...'.format(load_path_D))
                self.load_network(load_path_D, self.netD)
            load_path_D_grad = self.opt['path']['pretrain_model_D_grad']
            if self.opt['train']['gradient_gan_weight'] and load_path_D_grad is not None:
                logger.info('Loading pretrained model for D_grad [{:s}] ...'.format(load_path_D_grad))
                self.load_network(load_path_D_grad, self.netD_grad)

    def save(self, iter_step, name=None):
        self.save_network(self.netG, 'G', iter_step, name)
        if self.cri_gan:
            self.save_network(self.netD, 'D', iter_step, name)
        if self.cri_grad_gan:
            self.save_network(self.netD_grad, 'D_grad', iter_step, name)
