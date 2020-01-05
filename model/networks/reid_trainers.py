from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable
from tqdm import tqdm
from .evaluation_metrics import accuracy

criterion = nn.CrossEntropyLoss().cuda()
        

class BaseTrainer(object):
    def __init__(self, model, criterion, num_classes=0, num_instances=4):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.num_classes = num_classes
        self.num_instances = num_instances

    def train(self, epoch, data_loader,optimizer, print_freq=1,train_embed=False):
        if train_embed:
            self.model.module.embed_model.train()
            self.model.module.base_model.eval()
        else:
            self.model.train()

        losses = AverageMeter()
        losses = 0

        end = time.time()
        for inputs in tqdm(data_loader):
            inputs, targets = self._parse_data(inputs)
            loss, prec1 = self._forward(inputs, targets)
            losses += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return losses

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError

class SiameseTrainer(BaseTrainer):
    def _parse_data(self, inputs):
        (imgs1, _, pids1, _), (imgs2, _, pids2, _) = inputs
        inputs = [Variable(imgs1), Variable(imgs2)]
        targets = Variable((pids1 == pids2).long().cuda())
        return inputs, targets

    def _forward(self, inputs, targets):
        _, _, outputs = self.model(*inputs)
        loss = self.criterion(outputs, targets)
        prec1, = accuracy(outputs.data, targets.data)
        return loss, prec1[0]

class ClassifierTrainer(BaseTrainer):
    def _parse_data(self, inputs):
        (imgs1, _, pids1, _), (imgs2, _, pids2, _) = inputs
        inputs = [Variable(imgs1), Variable(imgs2)]
        targets = [Variable(pids1).cuda(), Variable(pids2).cuda()]
        return inputs, targets

    def _forward(self, inputs, targets):
        _,_,logits = self.model(*inputs)
        logits1,logits2 = logits
        targets1,targets2 = targets

        loss1 = self.criterion(logits1, targets1)
        loss2 = self.criterion(logits2, targets2)
        losses = loss1 + loss2

        return losses, None

class PartClassTrainer(BaseTrainer):
    def _parse_data(self, inputs):
        (imgs1, _, pids1, _), (imgs2, _, pids2, _) = inputs
        inputs = [Variable(imgs1), Variable(imgs2)]
        targets = [Variable(pids1).cuda(), Variable(pids2).cuda()]
        return inputs, targets

    def _forward(self, inputs, targets):
        _,_,logits = self.model(*inputs)
        logits1,logits2 = logits

        targets1,targets2 = targets

        losses = 0
        for p in range(6):
            loss1 = self.criterion(logits1[p], targets1)
            loss2 = self.criterion(logits2[p], targets2)
            losses += loss1 + loss2

        return losses, None

class PartSiameseTrainer(BaseTrainer):
    def _parse_data(self, inputs):
        (imgs1, _, pids1, _), (imgs2, _, pids2, _) = inputs
        inputs = [Variable(imgs1), Variable(imgs2)]
        #print("pids",pids1)
        targets = Variable((pids1 == pids2).long().cuda())
        return inputs, targets

    def _forward(self, inputs, targets):
        _, _, outputs = self.model(*inputs)
        losses = 0
        for p in range(6):
            loss = self.criterion(outputs[p], targets)
            #print("loss",loss)
            losses += loss
        #print("losses",losses)
        prec1 = None
        return losses, prec1
