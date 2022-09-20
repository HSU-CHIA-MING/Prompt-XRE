#!/bin/sh

MAX_UPDATE=40000 # 200000
WARMUP=2000
# DECAY_STEP=200000
INTERVAL=1000 # 625 or 10000

# effective batch size (8 GPUs) = 8*32*8 = 2048
SEQ_LEN=512
# BS=8
FREQ=8 # 128 or 8
LR=4e-4
# MIN_LR=5e-5

# max_positions=512

ARCH=roberta_tiny
DATA_PATH=/export/c01/haoranxu/oscar/en-zh-databin
STORE_PATH=/export/c01/haoranxu/LMs/tiny_LMs/en-zh
# RESTORE_CKPT=/srv/local2/shijie/pretrained/xlmr.large/model.pt
DATABINS=$(python third_party/get_databin_list.py --start 1 --end 2 --data_path $DATA_PATH)
echo $DATABINS
# mkdir -p /srv/local2/shijie/checkpoints/$CODENAME

CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train $DATABINS \
--save-dir ${STORE_PATH} \
--train-subset train \
--fp16 \
--memory-efficient-fp16 \
--num-workers 4 \
--task masked_lm \
--criterion masked_lm \
--arch $ARCH \
--sample-break-mode complete \
--tokens-per-sample $SEQ_LEN \
--optimizer adam \
--adam-betas "(0.9, 0.999)" \
--adam-eps 1e-6 \
--clip-norm 1.0 \
--lr-scheduler polynomial_decay \
--lr $LR \
--warmup-updates $WARMUP \
--dropout 0.1 \
--attention-dropout 0.1 \
--weight-decay 0.01 \
--max-tokens 4096 \
--update-freq $FREQ \
--max-update $MAX_UPDATE \
--total-num-update $MAX_UPDATE \
--required-batch-size-multiple 8 \
--empty-cache-freq 100 \
--skip-invalid-size-inputs-valid-test \
--log-format json \
--log-interval 5 \
--fast-stat-sync \
--seed 1 \
--validate-interval $INTERVAL \
--save-interval-updates $INTERVAL \
--no-epoch-checkpoints \
--distributed-world-size 4 \
| tee ${STORE_PATH}/train.log

# --end-learning-rate $MIN_LR \
# --total-num-update $DECAY_STEP \
#--max-sentences $BS \
# --restore-file $RESTORE_CKPT \
# --multilang-sampling-alpha 0.3 \
# --tensorboard-logdir ${STORE_PATH} \