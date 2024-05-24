#!/bin/bash

# 设置单机、多机环境变量
# sensecore启动多机时，会注册WORLD_SIZE、RANK、MASTER_PORT、MASTER_ADDR这几个环境变量，其中前两个都是节点的node数和rank数
# 如果WORLD_SIZE没有定义或者为空，则设置其为1, 这就是单机模式了
if [ -z "$WORLD_SIZE" ]; then  
    WORLD_SIZE=1  
    RANK=0
    MASTER_ADDR=127.0.0.1
    MASTER_PORT=6000
fi
export WANDB_PROJECT="Qwen"
export WANDB_MODE="offline"
GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
NUM_PROCESS=$(($GPUS_PER_NODE*$WORLD_SIZE))
DATA_PROCESS=$(($GPUS_PER_NODE*4))
# Hyper Parameter Start
MODEL_NAME=Qwen1.5-1.8B
TRAIN_DATA=/mnt/lustre/tangyang2/hjq/gritlm/work_dir/train_data/all_v3_unified/
JOB_NAME=Ve3_BS24_E1-36_GROUP3
LOGPATH=/mnt/lustre/tangyang2/hjq/gritlm/work_dir/logs/$MODEL_NAME.$JOB_NAME.gpu${NUM_PROCESS}.log
CONFIG=${MODEL_NAME}_fsdp_${NUM_PROCESS}gpu.yml
# Hyper Parameter End

echo "TOTAL PROCESS NUM: "$NUM_PROCESS
echo "NODE RANK IS: "$RANK
echo "NNODES IS: "$WORLD_SIZE
echo "MASTER ADDR: MASTER PORT: "$MASTER_ADDR.$MASTER_PORT

cd /mnt/lustre/tangyang2/hjq/gritlm/gritlm/

LAUNCHER="accelerate launch \
    --config_file /mnt/lustre/tangyang2/hjq/gritlm/work_dir/configs_jk/$CONFIG \
    --num_machines $WORLD_SIZE \
    --num_processes $NUM_PROCESS \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $RANK \
    --rdzv_conf rdzv_backend=c10d \
    --max_restarts 0 \
    --tee 3 \
    "

export CMD=" \
    -m training.run \
    --lora False \
    --loss_gen_factor 0.1 \
    --output_dir /mnt/lustre/tangyang2/hjq/gritlm/work_dir/outputs/$JOB_NAME \
    --model_name_or_path /mnt/lustre/tangyang2/hjq/model/$MODEL_NAME/ \
    --train_data $TRAIN_DATA \
    --dataloader_num_workers $DATA_PROCESS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --num_train_epoch 1 \
    --per_device_train_batch_size 24 \
    --gradient_accumulation_steps 1 \
    --dataloader_drop_last \
    --normalized \
    --temperature 0.02 \
    --train_group_size 2 \
    --negatives_cross_device \
    --query_max_len 128 \
    --passage_max_len 512 \
    --mode unified \
    --logging_steps 5 \
    --bf16 \
    --pooling_method mean \
    --attn bbcc \
    --attn_implementation sdpa \
    --save_steps 500000 \
    --gradient_checkpointing
    "


bash -c "$LAUNCHER $CMD" 2>&1 | tee -a $LOGPATH
