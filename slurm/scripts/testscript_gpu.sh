#!/bin/bash
#
#SBATCH --job-name=test_job 
#SBATCH --account=gpu_gres               # to access gpu resources
#SBATCH --partition=gpu                                           
#SBATCH --nodes=1                        # request to run job on single node                                       
##SBATCH --ntasks=10                     # request 10 CPU's (t3gpu01/02: balance between CPU and GPU : 5CPU/1GPU)      
#SBATCH --gres=gpu:1                     # request  for two GPU's on machine, this is total  amount of GPUs for job        
##SBATCH --mem=4000M                     # memory (per node)
#SBATCH --time=0-00:30                   # time  in format DD-HH:MM
#SBATCH -e output/slurm-gpu-%A.err
#SBATCH -o output/slurm-gpu-%A.out

# Slurm reserves two GPU's (according to requirement above), those ones that are recorded in shell variable CUDA_VISIBLE_DEVICES
echo CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES
echo "--------------------------------------------------------------------"
env
echo "--------------------------------------------------------------------"
cd $TF4TTHFHDIR
source activate TFGPU
pyenv versions
echo "--------------------------------------------------------------------"
nvidia-smi
echo "--------------------------------------------------------------------"

python test/tfGPUCheck.py
