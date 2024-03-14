#!/bin/bash

CHECKPOINT_PATH="$HOME/checkpoint_bert"
VOCAB_FILE="$HOME/dataset_bert/vocab.txt"
DATA_PATH="$HOME/dataset_bert/wikipedia-en_text_sentence"

mkdir -p $CHECKPOINT_PATH

export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=16

DISTRIBUTED_ARGS="--nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 1234"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_bert.py \
       --num-layers 12 \
       --hidden-size 768 \
       --num-attention-heads 12 \
       --micro-batch-size 32 \
       --global-batch-size 256 \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --train-iters 488281 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --vocab-file $VOCAB_FILE \
       --data-impl mmap \
       --split 980,10,10 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --min-lr 0 \
       --lr-decay-iters 488281 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 400 \
       --save-interval 50000 \
       --eval-interval 4000 \
       --apply-residual-connection-post-layernorm \
       --data-path $DATA_PATH \
       --eval-iters 50 \
       --bf16 \
       --hidden-dropout 0.1 \
       --attention-dropout 0.1 \
       --tensorboard-dir $CHECKPOINT_PATH/tensorboard \
       --no-masked-softmax-fusion