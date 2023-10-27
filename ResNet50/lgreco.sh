NUM_NODES=8
dataset_path=#path to imagenet dataset
BATCH_SIZE=256
lr=`echo "print($BATCH_SIZE * $NUM_NODES * 0.001)" | python`

workspace="./logs/"
mkdir -p $workspace

python -m torch.distributed.launch --nproc_per_node=$NUM_NODES \
./main.py $dataset_path --data-backend dali-cpu --raport-file raport.json -j8 -p 100 --lr $lr --optimizer-batch-size $(( BATCH_SIZE * NUM_NODES )) --warmup 8 --arch resnet50 --label-smoothing 0.1 --lr-schedule cosine --mom 0.875 --wd 3.0517578125e-05 --workspace $workspace -b $BATCH_SIZE --amp --static-loss-scale 128 --epochs 90 --no-checkpoints --lgreco-rank 4