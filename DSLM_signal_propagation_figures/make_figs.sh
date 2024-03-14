#!/bin/bash

source ~/vscode_setup_env.sh

CHECKPOINT_PATH="$HOME/checkpoint_plots"
rm -rf $CHECKPOINT_PATH
mkdir -p $CHECKPOINT_PATH

VOCAB_FILE="$HOME/dataset_bert/vocab.txt"
DATA_PATH="$HOME/dataset_bert/wikipedia-en_text_sentence"


export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=16

DISTRIBUTED_ARGS="--nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 1234"

#PRE
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.launch $DISTRIBUTED_ARGS        pretrain_bert.py        --num-layers 192        --hidden-size 1024        --num-attention-heads 16        --micro-batch-size 8       --global-batch-size 64        --seq-length 256        --max-position-embeddings 256        --train-iters 1000        --save $CHECKPOINT_PATH        --load $CHECKPOINT_PATH        --data-path $DATA_PATH        --vocab-file $VOCAB_FILE        --data-impl mmap        --split 949,50,1        --distributed-backend nccl        --lr 0        --lr-decay-style linear        --min-lr 0       --lr-decay-iters 495000        --weight-decay 0        --clip-grad 1.0        --lr-warmup-fraction .01        --log-interval 20        --save-interval 100        --eval-interval 1000        --eval-iters 100        --tensorboard-dir $CHECKPOINT_PATH/tensorboard --no-masked-softmax-fusion --moment-control-sigma --moment-control-lambda-beta --attention-dropout 0.1 --hidden-dropout 0.1


# #POST
rm -rf $CHECKPOINT_PATH
mkdir -p $CHECKPOINT_PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.launch $DISTRIBUTED_ARGS        pretrain_bert.py        --num-layers 192        --hidden-size 1024        --num-attention-heads 16        --micro-batch-size 8       --global-batch-size 64        --seq-length 256        --max-position-embeddings 256        --train-iters 1000        --save $CHECKPOINT_PATH        --load $CHECKPOINT_PATH        --data-path $DATA_PATH        --vocab-file $VOCAB_FILE        --data-impl mmap        --split 949,50,1        --distributed-backend nccl        --lr 0        --lr-decay-style linear        --min-lr 0       --lr-decay-iters 495000        --weight-decay 0        --clip-grad 1.0        --lr-warmup-fraction .01        --log-interval 20        --save-interval 100        --eval-interval 1000        --eval-iters 100        --tensorboard-dir $CHECKPOINT_PATH/tensorboard --no-masked-softmax-fusion --moment-control-sigma --moment-control-lambda-beta --attention-dropout 0.1 --hidden-dropout 0.1 --apply-residual-connection-post-layernorm


