#!/bin/bash
# nohup bash megnet_orig.sh > top.log 2>&1 &
subset=(
    "matbench_jdft2d"
    "matbench_phonons"
    "matbench_dielectric"
    "matbench_log_gvrh"
    "matbench_log_kvrh"
    "matbench_perovskites"
)

fold=$1
dim=$2
task_num=$3
cuda_device=$4
#fold=0
#dim=4
#task_num="6"
#cuda_device=1
# 计算起始索引
start_index=$(( ${#subset[@]} - task_num ))

# 使用起始索引和数组长度来选择后 task_num 个子集
selected_subset=("${subset[@]:start_index}")
# 将选择的子集列表转换为逗号分隔的字符串
selected_subset_str=$(IFS=, ; echo "${selected_subset[*]}")

echo "Embedding Running with dim_node_embed = $dim fold = $fold task = $selected_subset_str cuda_device = $cuda_device"
nohup python megnet_orig.py --dim_node_embed "$dim" --fold "$fold" --subsets "${selected_subset[@]}" --cuda_devices "$cuda_device" > "fold${fold}_dim${dim}_redress.log" 2>&1
# python megnet_orig.py --dim_node_embed $dim --fold $fold --subsets "${selected_subset[@]}" --cuda_devices $cuda_device

## 存储P