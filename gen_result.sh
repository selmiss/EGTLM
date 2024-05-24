MODEL=framework_test

python /home/mnt/huangjunqin/mteb_benchmark_zh/scripts/generate_results_zh.py \
/home/mnt/jingzihao/mteb_results/$MODEL/ --task-types Clustering

python /home/mnt/huangjunqin/mteb_benchmark_zh/scripts/generate_results_zh.py \
/home/mnt/jingzihao/mteb_results/$MODEL/ --task-types STS

python /home/mnt/huangjunqin/mteb_benchmark_zh/scripts/generate_results_zh.py \
/home/mnt/jingzihao/mteb_results/$MODEL/ --task-types Classification

python /home/mnt/huangjunqin/mteb_benchmark_zh/scripts/generate_results_zh.py \
/home/mnt/jingzihao/mteb_results/$MODEL/ --task-types Pairclassification

python /home/mnt/huangjunqin/mteb_benchmark_zh/scripts/generate_results_zh.py \
/home/mnt/jingzihao/mteb_results/$MODEL/ --task-types Retrieval

python /home/mnt/huangjunqin/mteb_benchmark_zh/scripts/generate_results_zh.py \
/home/mnt/jingzihao/mteb_results/$MODEL/ --task-types Reranking