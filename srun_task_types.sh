
# 通过指定task types的方式进行提交

# 定义任务列表
MODEL_NAME=framework_test
MODEL_PATH=/mnt/lustre/jingzihao/tmp4/$MODEL_NAME
OUTPUT_FOLDER=/mnt/lustre/jingzihao/mteb_results/$MODEL_NAME

all_task_types=(STS PairClassification Classification)
gpu_counts=(2 2 2)
batch_sizes=(32 16 16)

for ((i=0; i<${#all_task_types[@]}; i++)); do
    (
        task="${all_task_types[i]}"
        batch_size=${batch_sizes[i]}
        gpu_count=${gpu_counts[i]}

        # 获取对应任务的 GPU 数量
        job_name=$task.eval_$MODEL_NAME

        # 输出当前提交的任务和 GPU 数量
        echo "Submitted task: $task with GPU count: $gpu_count, batch size: $batch_size"
        # cd /mnt/lustre/jingzihao/gritlm
        # 在子 shell 中执行任务
            cmd="cp /mnt/lustre/jingzihao/gritlm/scripts/modeling_qwen2_gritlm.py /opt/conda/lib/python3.10/site-packages/transformers/models/qwen2/modeling_qwen2.py && \
            cd /mnt/lustre/jingzihao/gritlm/ && export PYTHONPATH=$PYTHONPATH:$(pwd) && \
            export PYTHONPATH=/mnt/lustre/huangjunqin/mteb_benchmark_zh/:$PYTHONPATH && export WANDB_PROJECT="gritlm" && pwd && python "evaluation/eval_mteb.py" \
            --model_name_or_path $MODEL_PATH \
            --output_folder $OUTPUT_FOLDER \
            --instruction_set e5 \
            --instruction_format gritlm \
            --task_types $task \
            --attn_implementation eager \
            --pooling_method mean \
            --batch_size $batch_size"
        srun -p model-1986-v100-32g --workspace-id c6f2207b-5358-47b5-85b6-9a050f5bcff9 -f pt -r N1lS.Ib.I00.$gpu_count -N 1 \
            --container-image registry.st-sh-01.sensecore.cn/ccr_gm_1/jinkin_gritlm:v1.0.2 -j $job_name \
            --container-mounts=0aa9c1fa-bb4f-11ee-a844-ee00953f5cc3:/mnt/lustre --output /mnt/lustre/jingzihao/logs \
            bash -c "$cmd"
    ) &
    
    # 在每次提交任务后暂停1秒
    sleep 1
done