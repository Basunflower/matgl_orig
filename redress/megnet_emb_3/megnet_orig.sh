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
dim_node_embed_values=("${@:2}")
task_num=$3
#fold=0
#dim_node_embed_values=(128)
#task_num=4
# 计算起始索引
start_index=$(( ${#subset[@]} - task_num ))

# 使用起始索引和数组长度来选择后 task_num 个子集
selected_subset=("${subset[@]:start_index}")
# 将选择的子集列表转换为逗号分隔的字符串
selected_subset_str=$(IFS=, ; echo "${selected_subset[*]}")
for dim in "${dim_node_embed_values[@]}"
do
      echo "Embedding Running with dim_node_embed = $dim fold = $fold task = $selected_subset_str"

      nohup python megnet_orig.py --dim_node_embed $dim --fold $fold --subsets "${selected_subset[@]}" > "fold${fold}_dim${dim}_redress.log" 2>&1
      # python megnet_orig.py --dim_node_embed $dim --fold $fold --subsets "${selected_subset[@]}"

      echo
done
