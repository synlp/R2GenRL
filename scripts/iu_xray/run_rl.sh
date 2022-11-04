seed=${RANDOM}
noamopt_warmup=1000

RESUME=${1}

python train_rl.py \
    --image_dir data/iu_xray/images/ \
    --ann_path data/iu_xray/annotation.json \
    --dataset_name iu_xray \
    --max_seq_length 60 \
    --threshold 3 \
    --batch_size 10 \
    --epochs 200 \
    --save_dir ${RESUME} \
    --step_size 1 \
    --gamma 0.8 \
    --seed ${seed} \
    --topk 32 \
    --beam_size 3 \
    --log_period 100 \
    --resume ${RESUME}/model_best.pth
