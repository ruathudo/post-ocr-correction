#!/bin/bash
#SBATCH --job-name=train_nctx
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2000
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=END
#SBATCH --mail-user=quan.duong@helsinki.fi


# module load python-data/3.7.6-1 
# module load pytorch/1.4

# python -m src.main --data yle-train-100.txt --model tf_mix_ctx_test --batch 256 --epoch 2

# python -m src.main --data yle-all.txt --model tf_ctx_trained_3_full --batch 256 --epoch 3 --rand 0 --window 3 --resume

# python -m src.main --data yle-all.txt --model tf_ctx_trained_3_full --batch 256 --epoch 3 --rand 0 --window 3 --resume
# python -m src.main --data yle-all.txt --model tf_ctx_trained_5_full --batch 256 --epoch 3 --rand 0 --window 5 --resume
# python -m src.main --data yle-all.txt --model tf_nctx_trained_full --batch 256 --epoch 3 --rand 0 --window 1 --resume

# python -m src.main --data yle-all.txt --model tf_ctx_rand_3_full --batch 256 --epoch 3 --rand 1 --window 3 --resume
# python -m src.main --data yle-all.txt --model tf_ctx_rand_5_full --batch 256 --epoch 2 --rand 1 --window 5 --resume
# python -m src.main --data yle-all.txt --model tf_nctx_rand_full --batch 256 --epoch 3 --rand 1 --window 1 --resume

conda activate
srun python -m src.main --data yle-all.txt --model tf_nctx_trained_full --batch 256 --epoch 3 --rand 0 --window 1 --resume

# srun -p gpu --gres=gpu:1 -c 16 --mem=32 -t 1-0:0 python -m src.main --data yle-all.txt --model tf_nctx_trained_full --batch 256 --epoch 5 --rand 0 --window 1
