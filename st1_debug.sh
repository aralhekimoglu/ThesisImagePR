#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2 -c8
#SBATCH --time=20:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=st1_debug
#SBATCH --output=st1_debug.out

source /scratch/ahh335/thesis/.env/bin/activate

nvidia-smi

python train.py \
--show-gpu \
--stage 1 \
--exp-dir /scratch/ahh335/experiments/thesis/stage1/st1_debug \
--data-dir /scratch/ahh335/data/reid \
--dataset market1501 \
--data-portion 1.0 \
--num-class 751 \
--train-batch-size 8 \
--test-batch-size 256 \
--workers 8 \
--np-ratio 3 \
--height 384 \
--width 128 \
--arch resnet50 \
--num-parts 6 \
--part-dims 256 \
--no-downsampling \
--classifier-type combined \
--e-lr  1e-2 \
--v-lr 1e-2 \
--pcb-lr 1e-1 \
--momentum 0.9 \
--weight-decay 5e-4 \
--epochs 100 \
--lr-step 40 \
--save-step 300 \
--eval-step 3 \