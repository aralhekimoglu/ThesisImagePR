from __future__ import print_function, absolute_import
import os

## PyTorch
import torch
from torch import nn

## Networks
from model.networks.encoder import ImageEncoder
from model.networks.verification import create_verification_net

## Utils
from utils.network_utils import get_lambda_scheduler,get_step_scheduler,load_state_dict

class REIDModel:
    def __init__(self,args):
        self.args = args
        
        self._init_models()
        self._init_optimizers()
        
        print('---------- Networks initialized -------------')
    
    def _init_models(self):
        self.net_E = ImageEncoder(self.args)
        self.net_V = create_verification_net(self.args)

        if False:
            print("why are you here")
            self.load_state_dict(self.net_E, self.args.netE_pretrain)
        if False:
            self.load_state_dict(self.net_V, self.args.netV_pretrain)

        self.net_E = torch.nn.DataParallel(self.net_E).cuda()
        self.net_V = torch.nn.DataParallel(self.net_V).cuda()
        #print(self.net_E)
        #print(self.net_V)
    
    def _init_optimizers(self):
        param_groups = [
            {'params': self.net_E.module.base.parameters(), 'init_lr': self.args.e_lr,'lr':self.args.e_lr},
            {'params': self.net_V.module.parameters(), 'init_lr': self.args.v_lr,'lr':self.args.v_lr}]
        
        if self.args.num_parts != 0:
            param_groups.append({'params': self.net_E.module.pcb.parameters(), 'init_lr': self.args.pcb_lr,'lr':self.args.pcb_lr})
            
        self.optimizer = torch.optim.SGD(param_groups, 1e-2, 
                                         momentum=self.args.momentum,
                                         weight_decay=self.args.weight_decay)
        
        #self.scheduler = get_lambda_scheduler(self.optimizer, self.args)
        self.scheduler = get_step_scheduler(self.optimizer, self.args)
    
    def debug(self,*msg):
        if not self.args.debug: return
        for m in msg:
            print(m)
        
    def set_input(self,input):
        (self.img1, _, self.pid1, _),(self.img2, _, self.pid2, _) = input
        #print("pid1",self.pid1)
        #print("pid2",self.pid2)
        self.img1 = self.img1.cuda()
        self.img2 = self.img2.cuda()
        self.pid1 = self.pid1.cuda()
        self.pid2 = self.pid2.cuda()
        
    def forward(self):
        self.f1 = self.net_E(self.img1)
        self.debug("f1 shape",self.f1.shape)
        self.f2 = self.net_E(self.img2)
        self.debug("f2 shape")
        self.debug(self.f2.shape)
        #print("f1",self.f1.shape)
        #print("f2",self.f2.shape)
    
    def optimize_parameters(self,gpu_logger=None):
        self.debug("optimizer_parameters")
        self.net_E.train()
        self.net_V.train()
        self.forward()
        #print("forwarded")
        #gpu_logger.show()
        self.optimizer.zero_grad()
        self.logits = self.net_V(self.f1,self.f2)
        #print("logits")
        #gpu_logger.show()
        self.loss = self.net_V.module.calculate_loss(self.pid1,self.pid2,self.logits)
        #print("loss")
        #gpu_logger.show()
        self.loss.backward()
        self.optimizer.step()
    
    def get_loss(self):
        return self.loss.data
    
    def clear_memory(self):
        del self.f1
        del self.f2
        del self.logits
    
    def save_model(self,epoch):
        self.save_network(self.net_E, 'E', epoch)
        self.save_network(self.net_V, 'V', epoch)

    def save_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.args.ckpt_dir, save_filename)
        torch.save(network.state_dict(), save_path)

    def update_lr(self):
        self.scheduler.step()
        
