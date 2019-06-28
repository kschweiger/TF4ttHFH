#!/bin/bash
#
#SBATCH --job-name=trainAutoencoder
#SBATCH --account=gpu_gres               # to access gpu resources
#SBATCH --partition=gpu                                           
#SBATCH --nodes=1                        # request to run job on single node                                       
##SBATCH --ntasks=10                     # request 10 CPU's (t3gpu01/02: balance between CPU and GPU : 5CPU/1GPU)      
#SBATCH --gres=gpu:1                     # request  for two GPU's on machine, this is total  amount of GPUs for job        
##SBATCH --mem=4000M                     # memory (per node)
#SBATCH --time=0-02:00                   # time  in format DD-HH:MM
#SBATCH -e output/trainAutoencoder/gpu-%A.err
#SBATCH -o output/trainAutoencoder/gpu-%A.out

echo CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES
cd $TF4TTHFHDIR
source activate TFGPU
python train_autoencoder.py --config data/autoencoder_Run2Dev_v1_test1.cfg --device GPU:$CUDA_VISIBLE_DEVICES --batchMode
