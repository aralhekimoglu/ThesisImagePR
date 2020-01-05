from __future__ import absolute_import
import os
import sys
import GPUtil as gpu
import nvidia_smi

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def save_imgs(img_dir,epoch,orig_imgs,targ_imgs,fake_imgs):
    orig_imgs = orig_imgs.numpy().transpose([0,2,3,1])
    targ_imgs = targ_imgs.numpy().transpose([0,2,3,1])
    fake_imgs = fake_imgs.numpy().transpose([0,2,3,1])

    save_path = os.path.join(img_dir,'epoch'+str(epoch))
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    
    for b in range(orig_imgs.shape[0]):
        fig, ax = plt.subplots(1, 3)
        orig_img = orig_imgs[b]
        targ_img = targ_imgs[b]
        fake_img = fake_imgs[b]
        orig_img = (orig_img-np.min(orig_img))/(np.max(orig_img)-np.min(orig_img))
        targ_img = (targ_img-np.min(targ_img))/(np.max(targ_img)-np.min(targ_img))
        fake_img = (fake_img-np.min(fake_img))/(np.max(fake_img)-np.min(fake_img))
        ax[0].imshow(orig_img) 
        ax[1].imshow(targ_img) 
        ax[2].imshow(fake_img) 
        plt.savefig(os.path.join(save_path,'img'+str(b)+'.jpg'))
        plt.close(fig)

class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

class GPULogger(object):
    def __init__(self,print_limit=100):
        nvidia_smi.nvmlInit()
        self.deviceCount = nvidia_smi.nvmlDeviceGetCount()
        print("Detected",self.deviceCount,"GPUs")
        self.handles = []
        for i in range(self.deviceCount):
            self.handles.append( nvidia_smi.nvmlDeviceGetHandleByIndex(i) )
        
        self.print_limit = print_limit 
        self.printed = 0 
    
    def show(self):
        if self.printed >self.print_limit:return
        print("GPU")
        self.printed +=1
        for i in range(self.deviceCount):
            mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handles[i])
            gb = mem_res.used / (1024 ** 3)
            perc = 100 * (mem_res.used / mem_res.total)
            print("ID %d [%.1f GB : %.1f Percent ]"%(i,gb,perc))
