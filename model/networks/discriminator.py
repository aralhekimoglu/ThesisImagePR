from __future__ import absolute_import
import os, sys
import functools

import torch

from model.networks.encoder import ImageEncoder
from model.networks.verification import SiameseNet, PartSiameseNet

from utils.network_utils import remove_module_key

import torch.nn as nn

class IDDiscriminator(nn.Module):
    ## Maybe have a cross entropy loss for two probability distrubitions
    ## For now takes two input images and returns 0-1 probability
    def __init__(self,args):
        super(IDDiscriminator, self).__init__()
        self.args = args
        self.encoder = ImageEncoder(self.args)
        if args.num_parts == 0:
            self.verification = SiameseNet(embed_dim = args.embed_dim, num_class = 1)
        else:
            self.verification = PartSiameseNet(num_parts = args.num_parts,
                                               part_dims = args.part_dims,
                                               num_class=1)
    def load_verification(self):
        """
        path is the path to (part)siamese verification net
        """
        state_dict = remove_module_key(torch.load(self.args.netDi_pretrain))
        if self.args.num_parts != 0:
            for p in range(6):
                w_str = 'embed_model.'+str(p)+'.classifier.weight'
                b_str = 'embed_model.'+str(p)+'.classifier.bias'
                state_dict[w_str] = state_dict[w_str][1].reshape(1,self.args.part_dims)
                state_dict[b_str] = torch.FloatTensor([state_dict[b_str][1]])
        else:
            w_str = 'embed_model.classifier.weight'
            b_str = 'embed_model.classifier.bias'
            state_dict[w_str] = state_dict[w_str][1].reshape(1,self.args.embed_dim)
            state_dict[b_str] = torch.FloatTensor([state_dict[b_str][1]])
        
        self.verification.load_state_dict(state_dict,strict = False)
        
        
    def forward(self,img1,img2):
        f1 = self.encoder(img1)
        f2 = self.encoder(img2)
        logit = self.verification(f1,f2) # prediction for 0/1 
        return logit
    
    
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, norm_layer=nn.BatchNorm2d):
        super(NLayerDiscriminator, self).__init__()
        ndf = 64
        n_layers = 3
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        # sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
