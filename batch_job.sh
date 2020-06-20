#!/bin/bash
#SBATCH --job-name=ocr_nmt
#SBATCH --account=project_2002413
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2000
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=END
#SBATCH --mail-user=quan.duong@helsinki.fi


module load python-data/3.7.6-1 
module load pytorch/1.4

srun python -m src.main --data yle-train-100.txt --batch 128 --epoch 2