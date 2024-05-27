#!/bin/bash

# 模型名
# model_name="cluster_512"
# model_root="/mnt/lustre/share_data/huangjunqin/to/cxx/model_xlm/"

# 创建一个存储日志的目录，以及子目录，子目录的名称为模型名
# mkdir -p logs/$model_name/chinese

# 定义任务列表
# tasks="STS PairClassification Reranking Retrieval Classification"
clustering_tasks="CLSClusteringS2S CLSClusteringP2P ThuNewsClusteringS2S ThuNewsClusteringP2P"
retrieval_tasks="T2Retrieval MMarcoRetrieval DuRetrieval CovidRetrieval CmedqaRetrieval EcomRetrieval MedicalRetrieval VideoRetrieval"
rerank_tasks="T2Reranking MmarcoReranking CMedQAv1 CMedQAv2"
sts_tasks="ATEC BQ LCQMC PAWSX STSB AFQMC QBQTC STS22_zh"
classification_tasks="TNews IFlyTek JDReview OnlineShopping Waimai AmazonReviewsClassificationZh MassiveIntentClassificationZh MassiveScenarioClassificationZh"
pairClassification_tasks="Ocnli Cmnli"

# 选择任务
# selected_tasks=$rerank_tasks
MODEL_NAME=Qwen1.5-1.8B.e1.lr1e-5.B64.Neg1.G1.Qwen_Test
MODEL_PATH=/mnt/lustre/jingzihao/tmp4/$MODEL_NAME
OUTPUT_FOLDER=/mnt/lustre/jingzihao/mteb_results/$MODEL_NAME

all_tasks=(clustering_tasks retrieval_tasks rerank_tasks)
gpu_counts=(2 2 2)
batch_sizes=(32 8 8)

for ((i=0; i<${#all_tasks[@]}; i++)); do
    task_name="${all_tasks[i]}"
    selected_tasks=$(eval echo "\$$task_name")  # 使用eval命令间接引用数组
    batch_size=${batch_sizes[i]}
    gpu_count=${gpu_counts[i]}
    for task in $selected_tasks; do
        (
            # 获取对应任务的 GPU 数量
            job_name=$task.eval_$MODEL_NAME

            # 输出当前提交的任务和 GPU 数量
            echo "Submitted task: $task with GPU count: $gpu_count, batch size: $batch_size"
            # cd /mnt/lustre/jingzihao/gritlm
            # 在子 shell 中执行任务
            cmd="cp /mnt/lustre/jingzihao/gritlm/scripts/modeling_qwen2_gritlm.py /opt/conda/lib/python3.10/site-packages/transformers/models/qwen2/modeling_qwen2.py && \
                cp /mnt/lustre/jingzihao/gritlm/scripts/modeling_mistral_gritlm.py /opt/conda/lib/python3.10/site-packages/transformers/models/mistral/modeling_mistral.py && \
                cd /mnt/lustre/jingzihao/gritlm/ && export PYTHONPATH=$PYTHONPATH:$(pwd) && \
                export PYTHONPATH=/mnt/lustre/huangjunqin/mteb_benchmark_zh/:$PYTHONPATH && export WANDB_PROJECT="gritlm" && pwd && python "evaluation/eval_mteb.py" \
                --model_name_or_path $MODEL_PATH \
                --output_folder $OUTPUT_FOLDER \
                --instruction_set e5 \
                --instruction_format gritlm \
                --task_names $task \
                --attn_implementation eager \
                --pooling_method mean \
                --batch_size $batch_size"
            srun -p model-1986-v100-32g --workspace-id c6f2207b-5358-47b5-85b6-9a050f5bcff9 -f pt -r N1lS.Ib.I00.$gpu_count -N 1 \
                --container-image registry.st-sh-01.sensecore.cn/ccr_gm_1/jinkin_gritlm:v1.0.2 -j $job_name \
                --container-mounts=0aa9c1fa-bb4f-11ee-a844-ee00953f5cc3:/mnt/lustre --output /mnt/lustre/jingzihao/logs \
                bash -c "$cmd"
        ) &
        
        # 在每次提交任务后暂停2秒
        sleep 2
    done
done