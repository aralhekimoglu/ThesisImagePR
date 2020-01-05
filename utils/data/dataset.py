from __future__ import print_function
import os.path as osp

import numpy as np

from utils.serialization import read_json

def _pluck(identities, indices, relabel=False):
    ret = []
    query = {}
    for index, pid in enumerate(indices):
        pid_images = identities[pid]
        if relabel:
            if index not in query.keys():
                query[index] = []
        else:
            if pid not in query.keys():
                query[pid] = []
        for camid, cam_images in enumerate(pid_images):
            for fname in cam_images:
                name = osp.splitext(fname)[0]
                x, y, _ = map(int, name.split('_'))
                assert pid == x and camid == y
                if relabel:
                    ret.append((fname, index, camid))
                    query[index].append(fname)
                else:
                    ret.append((fname, pid, camid))
                    query[pid].append(fname)
    return ret, query

class Dataset(object):
    def __init__(self, root, split_id=0,data_portion = 1.0):
        ## Set dataset properties
        self.root = root
        self.split_id = split_id
        self.meta = None
        self.split = None
        self.train, self.val, self.trainval = [], [], []
        self.query, self.gallery = [], []
        self.num_train_ids, self.num_val_ids, self.num_trainval_ids = 0, 0, 0

        ## Check integrity of dataset
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. " +
                               "Please follow README.md to prepare Market1501 dataset.")

        ## Load dataset
        self.load(data_portion = data_portion)

    def __len__(self):
        return

    @property
    def images_dir(self):
        return osp.join(self.root, 'images')

    @property
    def poses_dir(self):
        return osp.join(self.root, 'poses')

    def load(self, data_portion=1.0, verbose=True):
        splits = read_json(osp.join(self.root, 'splits.json'))
        if self.split_id >= len(splits):
            raise ValueError("split_id exceeds total splits {}"
                             .format(len(splits)))
        self.split = splits[self.split_id]
        
        train_portion = int(round(len(self.split['trainval']) * data_portion))
        query_portion = int(round(len(self.split['query']) * data_portion))
        gallery_portion = int(round(len(self.split['gallery']) * data_portion))
        
        train_pids = sorted(self.split['trainval'][:train_portion])
        query_pids = sorted(self.split['query'][:query_portion])
        gallery_pids = sorted(self.split['gallery'][:gallery_portion])
        
        self.meta = read_json(osp.join(self.root, 'meta.json'))
        identities = self.meta['identities']
        self.train, self.train_query = _pluck(identities, train_pids, relabel=True)
        self.query, self.query_query = _pluck(identities, query_pids)
        self.gallery, self.gallery_query = _pluck(identities, gallery_pids)
        
        self.num_train_ids = len(train_pids)
        self.num_query_ids = len(query_pids)
        self.num_gallery_ids = len(gallery_pids)
        
        if verbose:
            print(self.__class__.__name__, "dataset loaded")
            print("  subset   | # ids | # images")
            print("  ---------------------------")
            print("  train    | {:5d} | {:8d}"
                  .format(self.num_train_ids, len(self.train)))
            print("  query    | {:5d} | {:8d}"
                  .format(self.num_query_ids, len(self.query)))
            print("  gallery  | {:5d} | {:8d}"
                  .format(self.num_gallery_ids, len(self.gallery)))

    def _check_integrity(self):
        return osp.isdir(osp.join(self.root, 'images')) and \
               osp.isfile(osp.join(self.root, 'meta.json')) and \
               osp.isfile(osp.join(self.root, 'splits.json')) and \
               osp.isdir(osp.join(self.root, 'poses'))
