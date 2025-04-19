#!/bin/bash

SCRIPT=./gen_sem_ddp.py

BASE_DIR="/home/hanshengliang/hm3d-data/train-data/hm3d-train-habitat-v0.2"
#BASE_DIR="/home/hanshengliang/hm3d-data/val-data/hm3d-val-habitat-v0.2"

# 启动 8 个批次，每个绑定不同 GPU
for i in {0..7}
do
    echo "启动 batch $i on GPU $i"
    CUDA_VISIBLE_DEVICES=$i python $SCRIPT --batch-id $i --total-batch 8 --base-dir $BASE_DIR > log_batch$i.txt 2>&1 &
done

echo "所有批次已启动，请使用 'htop' 或 'nvidia-smi' 监控运行状态。"