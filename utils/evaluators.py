from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch
import numpy as np
from utils import to_torch,to_numpy
from torch.autograd import Variable
from sklearn.metrics import average_precision_score

import torch.nn.functional as F
import torch.backends.cudnn as cudnn
cudnn.enabled = True
cudnn.benchmark = True

def extract_cnn_feature_pcb(model, inputs):
    model.eval()
    inputs = to_torch(inputs)
    inputs = Variable(inputs, volatile=True).cuda()
    outputs = model(inputs)
    local_feat_list,logits_list = outputs
    outputs = torch.cat(local_feat_list,dim=1).data.cpu()
    return outputs

def extract_cnn_feature(model, inputs):
    model.eval()
    inputs = to_torch(inputs)
    inputs = Variable(inputs).cuda()
    outputs = model(inputs)
    outputs = outputs.data.cpu()
    return outputs

def extract_features(model, data_loader, print_freq=20):
    model.eval()

    features = OrderedDict()
    
    for i, (imgs, fnames, pids, _) in enumerate(data_loader):
        outputs = extract_cnn_feature(model, imgs)
        #outputs = extract_cnn_feature_pcb(model, imgs)
        
        for fname, output, pid in zip(fnames, outputs, pids):
            features[fname] = output
            
        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'.format(i + 1, len(data_loader)))
    return features

def pairwise_distance(features, query=None, gallery=None):
    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist

def mean_ap(distmat, query_ids=None, gallery_ids=None,
            query_cams=None, gallery_cams=None):
    distmat = to_numpy(distmat)
    m, n = distmat.shape
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    aps = []
    for i in range(m):
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        
        y_true = matches[i, valid]
        y_score = -distmat[i][indices[i]][valid]
        if not np.any(y_true): continue
        aps.append(average_precision_score(y_true, y_score))
    if len(aps) == 0:
        raise RuntimeError("No valid query")
    return np.mean(aps)

class CascadeEvaluator(object):
    def __init__(self, dataset, data_loader, encoder_model):
        super(CascadeEvaluator, self).__init__()
        self.dataset = dataset
        self.data_loader = data_loader
        self.encoder_model = encoder_model

    def evaluate(self):
        # Extract features image by image
        features = extract_features(self.encoder_model, self.data_loader)
        
        # Compute pairwise distance and evaluate for the first stage
        distmat = pairwise_distance(features, self.dataset.query, self.dataset.gallery)
        print("Evaluating...")

        query_ids = np.asarray([pid for _, pid, _ in self.dataset.query])
        gallery_ids = np.asarray([pid for _, pid, _ in self.dataset.gallery])
        query_cams = np.asarray([cam for _, _, cam in self.dataset.query])
        gallery_cams = np.asarray([cam for _, _, cam in self.dataset.gallery])
        
        # Compute mean AP
        mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)

        return mAP
