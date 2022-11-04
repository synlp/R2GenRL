#!/bin/bash
#SBATCH -J cmm+rl
#SBATCH -p p-V100
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1

source /cm/shared/apps/anaconda3/etc/profile.d/conda.sh

conda activate a100

seed=32103
# seed=${RANDOM}
noamopt_warmup=1000

# python train.py \
#     --image_dir data/mimic_cxr/images/ \
#     --ann_path data/mimic_cxr/annotation.json \
#     --dataset_name mimic_cxr \
#     --max_seq_length 100 \
#     --threshold 10 \
#     --batch_size 16 \
#     --epochs 30 \
#     --save_dir results/mimic_cxr/base_cmn_${seed} \
#     --step_size 1 \
#     --gamma 0.8 \
#     --seed ${seed} \
#     --topk 32 \
#     --noamopt_warmup ${noamopt_warmup}

RESUME=results/mimic_cxr/base_cmn_seed_${seed}

seed=${RANDOM}
noamopt_warmup=1000
save_dir=results/mimic_cxr/rl_cmn_seed_${seed}

python train_rl.py \
    --image_dir data/mimic_cxr/images/ \
    --ann_path data/mimic_cxr/annotation.json \
    --dataset_name mimic_cxr \
    --max_seq_length 100 \
    --threshold 10 \
    --batch_size 6 \
    --epochs 50 \
    --save_dir ${save_dir} \
    --step_size 1 \
    --gamma 0.8 \
    --seed ${seed} \
    --topk 32 \
    --sc_eval_period 3000 \
    --resume ${RESUME}/model_best.pth