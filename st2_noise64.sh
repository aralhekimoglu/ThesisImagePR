#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:p40:2 -c8
#SBATCH --time=20:00:00
#SBATCH --mem=16GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahh335@nyu.edu
#SBATCH --job-name=st2_noise64
#SBATCH --output=st2_noise64.out

source /scratch/ahh335/thesis/.env/bin/activate

nvidia-smi

python train.py \
--stage 2 \
--exp-dir /scratch/ahh335/experiments/thesis/stage2/st2_noise64 \
--gen-img-dir ./generated_images/st2_noise64 \
--data-dir /scratch/ahh335/data/reid \
--dataset market1501 \
--data-portion 1.0 \
--num-class 751 \
--train-batch-size 4 \
--test-batch-size 12 \
--gen-batch-size 4 \
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
--noise-dim 64 \
--e-lr  1e-5 \
--v-lr 1e-5 \
--pcb-lr 1e-5 \
--g-lr 1e-4 \
--di-lr 1e-5 \
--dp-lr 1e-3 \
--momentum 0.9 \
--weight-decay 5e-4 \
--epochs 100 \
--lr-step 40 \
--save-step 1 \
--eval-step 1 \
--netE-pretrain /scratch/ahh335/experiments/thesis/stage1/r50_pcb_class/ckpt/90_net_E.pth \
--netV-pretrain /scratch/ahh335/experiments/thesis/stage1/r50_pcb_class/ckpt/90_net_V.pth \
--netDi-pretrain /scratch/ahh335/experiments/thesis/stage1/r50_pcb/ckpt/66_net_V.pth \

## Debug arguments
#--debug \
#--test-baseline \
#--show-gpu \
