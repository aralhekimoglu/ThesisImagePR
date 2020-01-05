from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision

class PCB(nn.Module):
    def __init__(self,num_parts,in_feat_dim = 512,part_feat_dim = 256):
        super(PCB, self).__init__()
        self.num_parts = num_parts
        self.part_convs = nn.ModuleList()
        for _ in range(num_parts):
            self.part_convs.append(nn.Sequential(
                nn.Conv2d(in_feat_dim, part_feat_dim, 1),
                nn.BatchNorm2d(part_feat_dim),
                nn.ReLU(inplace=True)))

    def forward(self,x):
        part_feats = []
        assert x.size(2) % self.num_parts == 0
        part_h = int(x.size(2) / self.num_parts)
        for p in range(self.num_parts):
            part_feat =  x[:,:,p*part_h:(p+1)*part_h,:]
            part_feat = F.avg_pool2d(part_feat,(part_h,x.size(3)))
            part_feat = self.part_convs[p](part_feat)
            part_feat = part_feat.view(x.size(0), -1)
            part_feats.append(part_feat)
            
        return torch.cat(part_feats,dim=1)

class ImageEncoder(nn.Module):
    __factory = {
        'resnet18': torchvision.models.resnet18,
        'resnet34': torchvision.models.resnet34,
        'resnet50': torchvision.models.resnet50,
        'resnet101': torchvision.models.resnet101,
        'resnet152': torchvision.models.resnet152,
    }

    __feat_dims = {
        'resnet18': 512,
        'resnet34': 512,
        'resnet50': 2048,
    }

    def __init__(self, args):
        super(ImageEncoder, self).__init__()

        self.args = args

        # Construct base (pretrained) resnet
        if args.arch not in ImageEncoder.__factory:
            raise KeyError("Unsupported arch:", args.arch)
        self.base = ImageEncoder.__factory[args.arch](pretrained=True)
       
        if args.no_downsampling:
            downsampling_block = list(self.base.layer4.children())[0]
            downsampling_block.downsample[0].stride = (1,1)
            for l in downsampling_block.children():
                if type(l) == nn.Conv2d and l.stride == (2,2):
                    l.stride = (1,1)
            
            new_list = [downsampling_block]+list(self.base.layer4.children())[1:]
            self.base.layer4 = nn.Sequential(*new_list)
            
        if args.num_parts != 0:
            self.pcb = PCB(args.num_parts,
                           in_feat_dim = ImageEncoder.__feat_dims[args.arch], 
                           part_feat_dim = args.part_dims)
        
    def forward(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        if self.args.num_parts != 0:
            x = self.pcb(x)
        else:
            x = F.avg_pool2d(x, x.size()[2:])
            x = x.view(x.size(0), -1)
            
        return x
