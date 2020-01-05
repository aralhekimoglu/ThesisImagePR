import os,sys
import itertools
import numpy as np
import math
import random
import copy
from collections import OrderedDict

## Torch imports
import torch
from torch.autograd import Variable
from torch.nn import functional as F

## Networks
from model.networks.encoder import ImageEncoder
from model.networks.verification import create_verification_net
from model.networks.generator import create_generator_net
from model.networks.discriminator import IDDiscriminator,NLayerDiscriminator
from model.losses import GANLoss

## Utils
from utils.network_utils import remove_module_key,get_step_scheduler,set_bn_fix

class FDGANModel(object):

    def __init__(self, args):
        self.args = args
        
        self._init_models()
        self._init_losses()
        self._init_optimizers()
        
        print('---------- Networks initialized -------------')
    
    def _init_models(self):
        self.net_E = ImageEncoder(self.args)
        self.net_V = create_verification_net(self.args)
        self.net_G = create_generator_net(self.args)
        self.net_Di = IDDiscriminator(self.args)
        self.net_Dp = NLayerDiscriminator(3+18)
        #print("Nets")
        
        if self.args.stage == 3 or self.args.pretrained:
            self._load_state_dict(self.net_E, self.args.netE_pretrain)
            self._load_state_dict(self.net_V, self.args.netV_pretrain)
            self._load_state_dict(self.net_G, self.args.netG_pretrain)
            self._load_state_dict(self.net_Di, self.args.netDi_pretrain)
            self._load_state_dict(self.net_Dp, self.args.netDp_pretrain)

        elif self.args.stage == 2:
            #print("E")
            self._load_state_dict(self.net_E, self.args.netE_pretrain)
            #print("V")
            self._load_state_dict(self.net_V, self.args.netV_pretrain)
            #print("Di")
            self._load_state_dict(self.net_Di.encoder, self.args.netE_pretrain)
            self.net_Di.load_verification()
            
        #print("Load")
        self.net_E = torch.nn.DataParallel(self.net_E).cuda()
        self.net_V = torch.nn.DataParallel(self.net_V).cuda()
        self.net_G = torch.nn.DataParallel(self.net_G).cuda()
        self.net_Di = torch.nn.DataParallel(self.net_Di).cuda()
        self.net_Dp = torch.nn.DataParallel(self.net_Dp).cuda()
        #print("Cuda")
        
    def _load_state_dict(self, net, path):
        #print("loading")
        state_dict = remove_module_key(torch.load(path))
        #print("state_dict")
        net.load_state_dict(state_dict)
        #print("net load")
        
    def _init_losses(self):
        if self.args.smooth_label:
            self.criterionGAN_D = GANLoss(smooth=True).cuda()
            self.rand_list = [True] * 1 + [False] * 10000
        else:
            self.criterionGAN_D = GANLoss(smooth=False).cuda()
            self.rand_list = [False]
        self.criterionGAN_G = GANLoss(smooth=False).cuda()
        
    def _init_optimizers(self):
        if self.args.stage==2:
            self.optimizer_G = torch.optim.Adam(self.net_G.parameters(),
                                                lr=self.args.g_lr, betas=(0.5, 0.999))
            self.optimizer_Di = torch.optim.SGD(self.net_Di.parameters(),
                                                lr=self.args.di_lr, momentum=0.9,
                                                weight_decay=1e-4)
            self.optimizer_Dp = torch.optim.SGD(self.net_Dp.parameters(),
                                                lr=self.args.dp_lr, momentum=0.9,
                                                weight_decay=1e-4)
        else:
            param_groups = [
                {'params': self.net_E.module.base.parameters(), 
                'init_lr': self.args.e_lr,'lr':self.args.e_lr},
                {'params': self.net_V.parameters(), 
                'init_lr': self.args.v_lr,'lr':self.args.v_lr},
                {'params': self.net_G.parameters(), 
                'init_lr': self.args.g_lr,'lr':self.args.g_lr}]
        
            if self.args.num_parts != 0:
                param_groups.append({'params': self.net_E.module.pcb.parameters(),
                'init_lr': self.args.pcb_lr,'lr':self.args.pcb_lr})
            
            self.optimizer_G = torch.optim.Adam(param_groups,
                                                lr=self.args.g_lr, betas=(0.5, 0.999))
            self.optimizer_Dp = torch.optim.Adam(self.net_Dp.parameters(),
                                                lr=self.args.dp_lr, betas=(0.5, 0.999))
            self.optimizer_Di = torch.optim.SGD(self.net_Di.parameters(),
                                                lr=self.args.di_lr, momentum=0.9,
                                                weight_decay=1e-4)

        self.optimizers = []
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_Di)
        self.optimizers.append(self.optimizer_Dp)
        
        self.schedulers = []
        for optimizer in self.optimizers:
            self.schedulers.append(get_step_scheduler(optimizer, self.args))
    
    def reset_model_status(self):
        if self.args.stage==2:
            self.net_E.eval()
            self.net_V.eval()
            self.net_G.train()
            self.net_Di.train()
            self.net_Di.apply(set_bn_fix)
            self.net_Dp.train()
            
        elif self.args.stage==3:
            self.net_E.train()
            self.net_E.apply(set_bn_fix)
            self.net_V.train()
            self.net_G.eval()
            self.net_Di.eval()
            self.net_Dp.eval()
            
    def set_input(self, input):
        input1, input2 = input
        labels = (input1['pid']==input2['pid']).long()
        pids = [Variable(input1['pid']).cuda(), Variable(input2['pid']).cuda()]
        
        self.pid1 = Variable(input1['pid']).cuda()[:,0]
        self.pid2 = Variable(input2['pid']).cuda()[:,0]
        
        noise = torch.randn(labels.size(0), self.args.noise_dim)
        mask = labels.view(-1,1,1,1).expand_as(input1['posemap'])
        input2['posemap'] = input1['posemap']*mask.float() + input2['posemap']*(1-mask.float())
        mask = labels.view(-1,1,1,1).expand_as(input1['target'])
        input2['target'] = input1['target']*mask.float() + input2['target']*(1-mask.float())
      
        origin = torch.cat([input1['origin'], input2['origin']])
        #origin_gt = torch.cat([input1['origin_gt'], input2['origin_gt']])
        target = torch.cat([input1['target'], input2['target']])
        posemap = torch.cat([input1['posemap'], input2['posemap']])
        #origin_posemap = torch.cat([input1['origin_posemap'], input2['origin_posemap']])
        noise = torch.cat((noise, noise))

        self.origin = origin.cuda()
        #self.origin_gt = origin_gt.cuda()
        self.target = target.cuda()
        self.posemap = posemap.cuda()
        #self.origin_posemap = origin_posemap.cuda()
        self.labels = labels.cuda()
        self.noise = noise.cuda()
        self.pids = pids
        
    def forward(self):
        A = Variable(self.origin)
        B_map = Variable(self.posemap)
        z = Variable(self.noise)
        bs = A.size(0)

        A_id1, A_id2 = self.net_E(A[:bs//2]),self.net_E(A[bs//2:])
        self.logits = self.net_V(A_id1,A_id2)
        
        A_id = torch.cat((A_id1,A_id2))
        self.fake = self.net_G(B_map, A_id.view(A_id.size(0), A_id.size(1), 1, 1), z.view(z.size(0), z.size(1), 1, 1))
        """
        B_map = Variable(self.origin_posemap)
        self.fake_self = self.net_G(B_map, A_id.view(A_id.size(0), A_id.size(1), 1, 1), z.view(z.size(0), z.size(1), 1, 1))
        """
    
    def calculate_Di_loss(self,prediction,target=True):
        if self.args.num_parts != 0:
            loss_di = 0
            for p in range(6):
                loss = self.criterionGAN_D(prediction[p],target)
                loss_di += loss
        else:
            loss_di = self.criterionGAN_D(prediction, target)

        return loss_di


    def backward_Di(self):
        pred_real = self.net_Di(Variable(self.origin), Variable(self.target))
        pred_fake = self.net_Di(Variable(self.origin), self.fake.detach())

        ## self generation id loss
        #pred_fake_self = self.net_Di(Variable(self.origin), self.fake_self.detach())

        if random.choice(self.rand_list):
            loss_D_real = self.calculate_Di_loss(pred_fake, True)
            loss_D_fake = self.calculate_Di_loss(pred_real, False)
            #loss_D_self = self.calculate_Di_loss(pred_fake_self, True)
        else:
            loss_D_real = self.calculate_Di_loss(pred_real, True)
            loss_D_fake = self.calculate_Di_loss(pred_fake, False)
            #loss_D_self = self.calculate_Di_loss(pred_fake_self, False)
        #loss_D = (loss_D_real + loss_D_fake + loss_D_self) * 0.33
        loss_D = (loss_D_real + loss_D_fake ) * 0.5
        loss_D.backward()
        self.loss_Di = loss_D.data
    
    def backward_Dp(self):
        real_pose = torch.cat((Variable(self.posemap), Variable(self.target)),dim=1)
        fake_pose = torch.cat((Variable(self.posemap), self.fake.detach()),dim=1)
        pred_real = self.net_Dp(real_pose)
        pred_fake = self.net_Dp(fake_pose)
        
        ## self generation pose disc loss
        """
        real_self_pose = torch.cat((Variable(self.origin_posemap), Variable(self.origin_gt)),dim=1)
        fake_self_pose = torch.cat((Variable(self.origin_posemap), self.fake_self.detach()),dim=1)
        pred_real_self = self.net_Dp(real_self_pose)
        pred_fake_self = self.net_Dp(fake_self_pose)
        """
        
        if random.choice(self.rand_list):
            loss_D_real = self.criterionGAN_D(pred_fake, True)
            loss_D_fake = self.criterionGAN_D(pred_real, False)
            #loss_D_real_self = self.criterionGAN_D(pred_fake_self, True)
            #loss_D_fake_self = self.criterionGAN_D(pred_real_self, False)
        else:
            loss_D_real = self.criterionGAN_D(pred_real, True)
            loss_D_fake = self.criterionGAN_D(pred_fake, False)
            #loss_D_real_self = self.criterionGAN_D(pred_real_self, True)
            #loss_D_fake_self = self.criterionGAN_D(pred_fake_self, False)
        #loss_D = (loss_D_real + loss_D_fake + loss_D_real_self + loss_D_fake_self) * 0.25
        loss_D = (loss_D_real + loss_D_fake ) * 0.5
        loss_D.backward()
        self.loss_Dp = loss_D.data
        

    def backward_G(self):
        loss_v = self.net_V.module.calculate_loss(self.pid1,self.pid2,self.logits)
        loss_r = F.l1_loss(self.fake, Variable(self.target))

        ## self generation loss
        #loss_r += F.l1_loss(self.fake_self, Variable(self.origin_gt))
        
        fake_1 = self.fake[:self.fake.size(0)//2] 
        fake_2 = self.fake[self.fake.size(0)//2:]
        loss_sp = F.l1_loss(fake_1[self.labels.view(self.labels.size(0),1,1,1).expand_as(fake_1)==1], 
                            fake_2[self.labels.view(self.labels.size(0),1,1,1).expand_as(fake_1)==1])

        
        pred_fake_Di = self.net_Di(Variable(self.origin), self.fake)
        loss_G_GAN_Di = self.calculate_Di_loss(pred_fake_Di, True)

        pred_fake_Dp = self.net_Dp(torch.cat((Variable(self.posemap),self.fake),dim=1))
        loss_G_GAN_Dp = self.criterionGAN_G(pred_fake_Dp, True)
        """
        ## Self generation losses
        pred_fake_self_Di = self.net_Di(Variable(self.origin), self.fake_self)
        loss_G_GAN_Di += self.calculate_Di_loss(pred_fake_self_Di, True)
        loss_G_GAN_Di *= 0.5

        pred_fake_self_Dp = self.net_Dp(torch.cat((Variable(self.origin_posemap),self.fake_self),dim=1))
        loss_G_GAN_Dp += self.criterionGAN_G(pred_fake_self_Dp, True)
        loss_G_GAN_Dp *= 0.5
        #### end of self generation losses
        """

        loss_G = loss_G_GAN_Di * self.args.lambda_gan_di + \
                 loss_G_GAN_Dp * self.args.lambda_gan_dp + \
                 loss_r * self.args.lambda_recon + \
                 loss_v * self.args.lambda_veri + \
                 loss_sp * self.args.lambda_sp
        loss_G.backward()

        self.loss_G = loss_G.data
        self.loss_v = loss_v.data
        self.loss_sp = loss_sp.data
        self.loss_r = loss_r.data
        self.loss_G_GAN_Di = loss_G_GAN_Di.data
        self.loss_G_GAN_Dp = loss_G_GAN_Dp.data
        self.fake = self.fake.data
        
    def clear_memory(self):
        del self.fake
        del self.logits
        
    def optimize_parameters(self):
        self.set_train_status()
        self.forward()

        self.optimizer_Di.zero_grad()
        self.backward_Di()
        self.optimizer_Di.step()
        
        self.optimizer_Dp.zero_grad()
        self.backward_Dp()
        self.optimizer_Dp.step()
        
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def set_train_status(self):
        if self.args.stage==2:
            self.net_E.eval()
            self.net_V.eval()
            self.net_G.train()
            self.net_Di.train()
            self.net_Dp.train()

        elif self.args.stage==3:
            self.net_E.train()
            self.net_V.train()
            self.net_G.train()
            self.net_Di.train()
            self.net_Dp.train()
    
    def set_eval_status(self):
        self.net_E.eval()
        self.net_V.eval()
        self.net_G.eval()
        self.net_Di.eval()
        self.net_Dp.eval()
    
    def get_loss(self):
        return self.loss_v

    def get_current_errors(self):
        return OrderedDict([('G_v', self.loss_v),
                            ('G_r', self.loss_r),
                            ('G_sp', self.loss_sp),
                            ('G_gan_Di', self.loss_G_GAN_Di),
                            ('G_gan_Dp', self.loss_G_GAN_Dp),
                            ('D_i', self.loss_Di),
                            ('D_p', self.loss_Dp)
                            ])

    def save_model(self, epoch):
        self.save_network(self.net_E, 'E', epoch)
        self.save_network(self.net_V, 'V', epoch)
        self.save_network(self.net_G, 'G', epoch)
        self.save_network(self.net_Di, 'Di', epoch)
        self.save_network(self.net_Dp, 'Dp', epoch)

    def save_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.args.ckpt_dir, save_filename)
        torch.save(network.state_dict(), save_path)

    def update_lr(self):
        for scheduler in self.schedulers:
            scheduler.step()
