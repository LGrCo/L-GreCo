#!/usr/bin/env bash

NUM_NODES=8
batch_size=$(( 128 / $NUM_NODES ))
log_dir="logs"
mkdir -p $log_dir
rank=4
seed=42
method='powerSGD'
python -m torch.distributed.launch --nproc_per_node=$NUM_NODES \
cifar_train.py --epochs 200 --dataset-dir ~/Datasets/cifar100 \
--batch-size $batch_size --powersgd-rank $rank --method $method
