#!/bin/bash
# 检查还有哪些任务没有完成

# 日志文件路径
log_="check_task.log"

# 清空日志文件
> $log_

# 定义要搜索的字符
search_string="test_MAE"
total_unfinish_tasks=0
# 遍历megnet_emb_fold文件夹
for fold_index in {0..4}; do
  for emb_dim in 1 2 4 8 16 32 64 128; do
    if [ "$emb_dim" -eq 1 ] || [ "$emb_dim" -eq 2 ] || [ "$emb_dim" -eq 4 ] || [ "$emb_dim" -eq 8 ]; then
      log_file="megnet_emb_${fold_index}/fold${fold_index}_dim${emb_dim}.log"
    else
      log_file="megnet_emb_${fold_index}_1/fold${fold_index}_dim${emb_dim}.log"
    fi

    # 检查日志文件是否存在
    if [ ! -f "$log_file" ]; then
        echo "'$log_file' not found.Need 6 task." >> $log_
        total_unfinish_tasks=$((total_unfinish_tasks + 6))
    else
      # 统计task数量
      count=$(grep -o "$search_string" "$log_file" | wc -l)
      unfinish_tasks=$((6-count))
      total_unfinish_tasks=$((total_unfinish_tasks + unfinish_tasks))
      if [ "$unfinish_tasks" -eq 0 ]; then
        echo "$log_file 已完成."
      else
        echo "$log_file Need $((6-count)) task." >> $log_
      fi
    fi
  done
done


for fold_index in {0..4}; do
  for emb_dim in 1 2 4 8 16 32 64 128; do
    if [ "$emb_dim" -eq 1 ] || [ "$emb_dim" -eq 2 ] || [ "$emb_dim" -eq 4 ] || [ "$emb_dim" -eq 8 ]; then
      log_file="megnet_mds_${fold_index}/fold${fold_index}_dim${emb_dim}_mds.log"
    else
      log_file="megnet_mds_${fold_index}_1/fold${fold_index}_dim${emb_dim}_mds.log"
    fi
    # 检查日志文件是否存在
    if [ ! -f "$log_file" ]; then
        echo "'$log_file' not found.Need 6 task." >> $log_
        total_unfinish_tasks=$((total_unfinish_tasks + 6))
    else
      # 统计task数量
      count=$(grep -o "$search_string" "$log_file" | wc -l)
      unfinish_tasks=$((6-count))
      total_unfinish_tasks=$((total_unfinish_tasks + unfinish_tasks))
      if [ "$unfinish_tasks" -eq 0 ]; then
        echo "$log_file 已完成."
      else
        echo "$log_file Need $((6-count)) task." >> $log_
      fi
    fi
  done
done


echo "Total unfinished tasks: $total_unfinish_tasks"