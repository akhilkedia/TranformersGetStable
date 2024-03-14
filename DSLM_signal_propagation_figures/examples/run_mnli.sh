#!/bin/bash

PRETRAINED_PATH="$HOME/checkpoint_bert"
VOCAB_FILE="$HOME/dataset_bert/vocab.txt"

TRAIN_DATA="$HOME/MNLI/train.tsv"
VALID_DATA="$HOME/MNLI/dev_matched.tsv \
            $HOME/MNLI/dev_mismatched.tsv"

mkdir -p $CHECKPOINT_PATH

export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=16

DISTRIBUTED_ARGS="--nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 1234"

for i in $(seq 1 5)
do
    for j in $(seq 1 5)
    do
        SEED=$(( ( RANDOM % 1000 )  + 1000 ))
        echo $i $SEED
        CHECKPOINT_PATH="$HOME/checkpoint_mnli_$i_$j"
        mkdir -p $CHECKPOINT_PATH

        python -m torch.distributed.launch $DISTRIBUTED_ARGS \
            tasks/main.py \
            --task MNLI \
            --vocab-file $VOCAB_FILE \
            --train-data $TRAIN_DATA \
            --valid-data $VALID_DATA \
            --save-interval 30000 \
            --save $CHECKPOINT_PATH \
            --log-interval 100 \
            --eval-interval 1000 \
            --eval-iters 10 \
            --weight-decay 1.0e-2 \
            --micro-batch-size 32 \
            --global-batch-size 32 \
            --seq-length 512 \
            --max-position-embeddings 512 \
            --tokenizer-type BertWordPieceLowerCase \
            --epochs 3 \
            --lr-warmup-fraction 0.01 \
            --lr-decay-style linear \
            --min-lr 0 \
            --clip-grad 1.0 \
            --lr-warmup-fraction 0.1 \
            --hidden-dropout 0.1 \
            --attention-dropout 0.1 \
            --no-masked-softmax-fusion \
            --pretrained-checkpoint $PRETRAINED_PATH  --num-layers 12 --hidden-size 768 --num-attention-heads 12 --bf16  --apply-residual-connection-post-layernorm --lr $i.0e-5 --seed $SEED \
            --tensorboard-dir $CHECKPOINT_PATH/tensorboard
    done
done


