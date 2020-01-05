#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:p40:2 -c8
#SBATCH --time=20:00:00
#SBATCH --mem=16GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahh335@nyu.edu
#SBATCH --job-name=st3_debug
#SBATCH --output=st3_debug.out

source /scratch/ahh335/thesis/.env/bin/activate

nvidia-smi

python train.py \
--show-gpu \
--stage 3 \
--exp-dir /scratch/ahh335/experiments/thesis/stage3/st3_debug \
--gen-img-dir ./generated_images/st3_debug \
--data-dir /scratch/ahh335/data/reid \
--dataset market1501 \
--data-portion 1.0 \
--num-class 751 \
--train-batch-size 8 \
--test-batch-size 256 \
--gen-batch-size 8 \
--workers 8 \
--np-ratio 3 \
--height 384 \
--width 128 \
--pose-aug gauss \
--arch resnet50 \
--num-parts 6 \
--part-dims 256 \
--no-downsampling \
--classifier-type class \
--smooth-label \
--lambda-gan-di 1.0 \
--lambda-gan-dp 1.0 \
--lambda-recon 100.0 \
--lambda-veri 0.0 \
--lambda-sp 10.0 \
--gen-net unet \
--drop 0 \
--pose-dim 18 \
--noise-dim 256 \
--e-lr  1e-5 \
--v-lr 1e-5 \
--pcb-lr 1e-5 \
--g-lr 1e-5 \
--di-lr 1e-4 \
--dp-lr 1e-4 \
--momentum 0.9 \
--weight-decay 5e-4 \
--epochs 100 \
--lr-step 40 \
--save-step 1 \
--eval-step 1 \
--netE-pretrain /scratch/ahh335/experiments/thesis/stage2/st2_unet/ckpt/7_net_E.pth \
--netV-pretrain /scratch/ahh335/experiments/thesis/stage2/st2_unet/ckpt/7_net_V.pth \
--netG-pretrain /scratch/ahh335/experiments/thesis/stage2/st2_unet/ckpt/7_net_G.pth \
--netDi-pretrain /scratch/ahh335/experiments/thesis/stage2/st2_unet/ckpt/7_net_Di.pth \
--netDp-pretrain /scratch/ahh335/experiments/thesis/stage2/st2_unet/ckpt/7_net_Dp.pth \

## Debug arguments
#--debug \
#--test-baseline \
#--show-gpu \
