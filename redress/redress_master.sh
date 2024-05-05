#!/bin/bash

# 文件路径
file_path="check_task.log"

# 二维数组
arg=()

# 逐行读取文件
while IFS= read -r line; do
    # 使用正则表达式匹配行中的所需信息
    if [[ $line =~ fold([0-9]+)_dim([0-9]+)(_mds)?\.log\ Need\ ([0-9]+)\ task\. ]]; then
        # 匹配的信息
        fold="${BASH_REMATCH[1]}"
        dim="${BASH_REMATCH[2]}"
        task="${BASH_REMATCH[4]}"
        # 判断是否包含_mds
        if [[ ${BASH_REMATCH[3]} ]]; then
            mds=1
        else
            mds=0
        fi
        arg+=("$fold" "$dim" "$task" "$mds")
    fi
done < "$file_path"

# 保存脚本原始工作目录
original_dir=$(pwd)

# 设置最大并行数量
max_parallel=8
# 计数器
parallel_count=0

# 循环处理数组，一行四个参数
for ((i = 0; i < ${#arg[@]}; i += 4)); do
    fold="${arg[i]}"
    dim="${arg[i+1]}"
    task="${arg[i+2]}"
    mds="${arg[i+3]}"

    # 根据mds标记复制并运行
    if ((mds == 0)); then
        cp -r megnet_emb "emb_${fold}_${dim}"
        cd "emb_${fold}_${dim}" || exit 1
        bash megnet_orig.sh "$fold" "$dim" "$task" "$fold" &
    else
        cp -r megnet_mds "mds_${fold}_${dim}"
        cd "mds_${fold}_${dim}" || exit 1
        bash megnet_pre.sh "$fold" "$dim" "$task" "$fold" &
    fi

    cd "$original_dir" || exit 1
    # 更新并行计数器
    ((parallel_count++))

    if ((parallel_count >= max_parallel)); then
        # 等待任一后台进程执行完成
        wait -n
        ((parallel_count--))
    fi

done
echo "Finish."