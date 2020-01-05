import math
import copy

import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

class EltwiseSubEmbed(nn.Module):
    def __init__(self, nonlinearity='square', use_batch_norm=False,
                 use_classifier=False, num_features=0, num_classes=0):
        super(EltwiseSubEmbed, self).__init__()
        self.nonlinearity = nonlinearity
        if nonlinearity is not None and nonlinearity not in ['square', 'abs']:
            raise KeyError("Unknown nonlinearity:", nonlinearity)
        self.use_batch_norm = use_batch_norm
        self.use_classifier = use_classifier
        if self.use_batch_norm:
            self.bn = nn.BatchNorm1d(num_features)
            self.bn.weight.data.fill_(1)
            self.bn.bias.data.zero_()
        if self.use_classifier:
            assert num_features > 0 and num_classes > 0
            self.classifier = nn.Linear(num_features, num_classes)
            self.classifier.weight.data.normal_(0, 0.001)
            self.classifier.bias.data.zero_()

    def forward(self, x1, x2):
        x = x1 - x2
        if self.nonlinearity == 'square':
            x = x.pow(2)
        elif self.nonlinearity == 'abs':
            x = x.abs()
        if self.use_batch_norm:
            x = self.bn(x)
        if self.use_classifier:
            x = x.view(x.size(0),-1)
            x = self.classifier(x)
        else:
            x = x.sum(1)

        return x

def create_verification_net(args,num_class = 2):
    if args.num_parts == 0 and args.classifier_type == 'siam':
        return SiameseNet(embed_dim = args.embed_dim, num_class = num_class)
    
    elif args.num_parts == 0 and args.classifier_type == 'class':
        return ClassifierNet(embed_dim = args.embed_dim,num_class = args.num_class)

    elif args.num_parts != 0 and args.classifier_type == 'siam':
        return PartSiameseNet(num_parts = args.num_parts, part_dims=args.part_dims, num_class=num_class)

    elif args.num_parts != 0 and args.classifier_type == 'class':
        return PartClassifierNet(num_parts = args.num_parts, part_dims=args.part_dims, num_class = args.num_class)
    
    elif args.classifier_type == 'combined':
        return PartCombinedNet(num_parts = args.num_parts, part_dims=args.part_dims, num_class = args.num_class)

    else:
        print("Unknown type for model")
        exit()
        
class SiameseNet(nn.Module):
    def __init__(self, embed_dim=2048,num_class = 2):
        print("Using SiameseNet")
        super(SiameseNet, self).__init__()
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.embed_model = EltwiseSubEmbed(use_batch_norm=True, use_classifier=True,
                                           num_features=embed_dim, num_classes=num_class)
    
    def forward(self, f1, f2):
        logits = self.embed_model(f1, f2)
        return logits
        
    def calculate_loss(self,pids1,pids2,logits):
        targets = Variable((pids1 == pids2).long().cuda())
        loss = self.criterion(logits, targets)
        return loss

class ClassifierNet(nn.Module):
    def __init__(self, embed_dim=2048,num_class = 751):
        print("Using ClassifierNet")
        super(ClassifierNet, self).__init__()
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.embed_model = self.make_classifier(in_dim = embed_dim, num_class = num_class)

    def make_classifier(self,in_dim=256,num_class=751):
        block = [nn.BatchNorm1d(in_dim)]
        block[0].weight.data.fill_(1)
        block[0].bias.data.zero_()
        block.append(nn.Linear(in_dim,num_class))
        block[1].weight.data.normal_(0, 0.001)
        block[1].bias.data.zero_()
        return nn.Sequential(*block)

    def forward(self,f1,f2):
        logits1 = self.embed_model(f1)
        logits2 = self.embed_model(f2)
        return (logits1,logits2)
        
    def calculate_loss(self,pids1,pids2,logits):
        logits1,logits2 = logits
        targets1,targets2 = [Variable(pids1).cuda(), Variable(pids2).cuda()]
        loss1 = self.criterion(logits1, targets1)
        loss2 = self.criterion(logits2, targets2)
        loss = loss1 + loss2

        return loss

class PartSiameseNet(nn.Module):
    def __init__(self,num_parts = 6,part_dims = 256,num_class=2):
        print("Using PartSiameseNet")
        super(PartSiameseNet,self).__init__()
        self.part_dims = part_dims
        self.num_parts = num_parts
        self.criterion = nn.CrossEntropyLoss().cuda()

        self.embed_model = nn.ModuleList()
        for p in range(num_parts):
            embed_model = EltwiseSubEmbed(use_batch_norm=True, use_classifier=True,
                                          num_features=part_dims, num_classes=num_class)
            self.embed_model.append(embed_model)

    def forward(self,f1,f2):
        logits = []
        for p in range(self.num_parts):
            part_f1 = f1[:,p*self.part_dims:(p+1)*self.part_dims]
            part_f2 = f2[:,p*self.part_dims:(p+1)*self.part_dims]
            part_logit = self.embed_model[p](part_f1,part_f2)
            logits.append(part_logit)
        return logits
        
    def calculate_loss(self,pids1,pids2,logits):
        targets = Variable((pids1 == pids2).long().cuda())
        loss = 0
        for p in range(self.num_parts):
            loss += self.criterion(logits[p], targets)
            
        return loss

class PartClassifierNet(nn.Module):
    def __init__(self,num_parts = 6, part_dims=256, num_class = 751):
        print("Using PartClassifierNet")
        super(PartClassifierNet,self).__init__()
        self.num_parts = num_parts
        self.part_dims = part_dims
        self.criterion = nn.CrossEntropyLoss().cuda()
        
        self.embed_model = nn.ModuleList()
        for p in range(num_parts):
            classifier = self.make_classifier(in_dim = part_dims, num_class = num_class)
            self.embed_model.append(classifier)

    def make_classifier(self,in_dim=256,num_class=751):
        block = [nn.BatchNorm1d(in_dim)]
        block[0].weight.data.fill_(1)
        block[0].bias.data.zero_()
        block.append(nn.Linear(in_dim,num_class))
        block[1].weight.data.normal_(0, 0.001)
        block[1].bias.data.zero_()
        return nn.Sequential(*block)
        
    def forward(self,f1,f2):
        logits1 = []
        logits2 = []
        for p in range(self.num_parts):
            part_f1 = f1[:,p*self.part_dims:(p+1)*self.part_dims]
            part_f2 = f2[:,p*self.part_dims:(p+1)*self.part_dims]
            part_logit1 = self.embed_model[p](part_f1)
            part_logit2 = self.embed_model[p](part_f2)
            logits1.append(part_logit1)
            logits2.append(part_logit2)
        return (logits1,logits2)
        
    def calculate_loss(self,pids1,pids2,logits):
        targets1,targets2 = [Variable(pids1).cuda(), Variable(pids2).cuda()]
        logits1,logits2 = logits
        loss = 0
        for p in range(self.num_parts):
            loss1 = self.criterion(logits1[p], targets1)
            loss2 = self.criterion(logits2[p], targets2)
            loss += loss1 + loss2

        return loss

class PartCombinedNet(nn.Module):
    def __init__(self,num_parts = 6, part_dims=256, num_class = 751):
        print("Using PartCombinedNet")
        super(PartCombinedNet,self).__init__()
        self.siamese = PartSiameseNet(num_parts = num_parts, part_dims=part_dims, num_class=2)
        self.classifier = PartClassifierNet(num_parts = num_parts, part_dims=part_dims, num_class = num_class)

    def forward(self,f1,f2):
        class_logits = self.classifier(f1,f2)
        siam_logits = self.siamese(f1,f2)
        return (class_logits,siam_logits)
        
    def calculate_loss(self,pids1,pids2,logits):
        class_logits,siam_logits = logits
        class_loss = self.classifier.calculate_loss(pids1,pids2,class_logits)
        siam_loss = self.siamese.calculate_loss(pids1,pids2,siam_logits)
        loss = (class_loss + siam_loss)/2
        return loss