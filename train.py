from __future__ import print_function, absolute_import

## Arguments
from utils.arguments import Arguments

## Data utils
from utils.data.dataloader import get_data

## Model
from model.reid_model import REIDModel
from model.fdgan_model import FDGANModel

## Evaluator
from utils.evaluators import CascadeEvaluator as Evaluator

from utils.logging import Logger, GPULogger,save_imgs
from tensorboardX import SummaryWriter
import sys
import os
from tqdm import tqdm
import time
from datetime import datetime
import torch

def main(args):
    print("Starting")
    torch.backends.cudnn.benchmark = True

    ## Set loggers
    sys.stdout = Logger(args.log_dir)
    writer = SummaryWriter(args.summary_dir)
    gpu_logger = GPULogger()
    
    ## Create model and get data loaders of each stage
    if args.stage == 1:
        dataset,train_loader,test_loader = get_data(args)
        model = REIDModel(args)
    else:
        dataset,train_loader,test_loader,generator_loader = get_data(args)
        generator_loader = iter(generator_loader)
        model = FDGANModel(args)
        print("Model")
    
    ## Create evaluator for the model
    evaluator = Evaluator(dataset = dataset,
                          data_loader = test_loader,
                          encoder_model = model.net_E)

    if args.test_baseline:
        print("Testing baseline")
        eval_init_time = time.time()
        mAP = evaluator.evaluate()
        print("Baseline mAP",mAP*100)
        print("Eval finished in",time.time()-eval_init_time)
                   
    for e in range(1,args.epochs):
        i = 0
        epoch_loss = 0
        print("Epoch",e)
        for data in tqdm(train_loader):
            i+=1
            if i>5 and args.show_gpu:break
            model.set_input(data)
            model.optimize_parameters()
            if args.show_gpu:gpu_logger.show()
            epoch_loss += model.get_loss()
            model.clear_memory()
            
        print("Epoch finished with loss",epoch_loss)
        print("Current time",datetime.now().strftime("%I:%M%p"))

        ## Also generate test images for stage 2/3
        if (args.stage != 1) and (e % 1 == 0):
            print("Generating images for epoch",e)
            generator_data = next(generator_loader)
            model.set_eval_status()
            model.set_input(generator_data)
            model.forward()
            orig_imgs = model.origin.cpu().detach()
            targ_imgs = model.target.cpu().detach()
            fake_imgs = model.fake.cpu().detach()
            model.clear_memory()
            save_imgs(args.gen_img_dir,e,orig_imgs,targ_imgs,fake_imgs)
        
        if (e % args.eval_step == 0) and (args.stage != 2) :
            mAP = evaluator.evaluate()
            if args.show_gpu:gpu_logger.show()
            print("Epoch",e,"mAP",mAP*100)
        
        if e % args.save_step == 0:
            model.save_model(e)
            
        model.update_lr()
    
if __name__ == '__main__':
    print("Starting")
    args = Arguments().parse()
    main(args)