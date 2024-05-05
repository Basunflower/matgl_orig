models=(
    # "--model_type llama --model_name baffo32/decapoda-research-llama-7B-hf"
    "--model_type auto --model_name bert-large-uncased"
    # "--model_type llama --model_name NousResearch/Llama-2-7b-hf"
    "--model_type auto --model_name bert-base-cased"
    # "--model_type auto --model_name Undi95/Meta-Llama-3-8B-hf"
    "--model_type auto --model_name bert-large-cased"
)

tasks=(
    "matbench_jdft2d"
    "matbench_phonons"
    "matbench_dielectric"
    "matbench_log_gvrh"
    "matbench_log_kvrh"
    "matbench_perovskites"
    "matbench_mp_gap"
    "matbench_mp_e_form"
    # "matbench_mp_is_metal"
)
fold=(0 1 2 3 4)
opt_llm=("True" "False")
train_cnt=(
508 509 509 509 509
1012 1012 1012 1012 1012
3811 3811 3811 3811 3812
8789 8789 8790 8790 8790
8789 8789 8790 8790 8790
15142 15142 15142 15143 15143
84890 84890 84890 84891 84891
106201 106201 106202 106202 106202
# 84890 84890 84890 84891 84891
)

test_cnt=(
128 127 127 127 127
253 253 253 253 253
953 953 953 953 952
2198 2198 2197 2197 2197
2198 2198 2197 2197 2197
3786 3786 3786 3785 3785
21223 21223 21223 21222 21222
26551 26551 26550 26550 26550
# 21223 21223 21223 21222 21222
)

jobs=()

index=0
for i in "${tasks[@]}"; do
    for j in "${fold[@]}"; do
        for k in "${opt_llm[@]}"; do
            idx=$((index * 5 + j))
            jobs+=("--task_name $i --fold $j --opt_llm $k --train_cnt ${train_cnt[$idx]} --test_cnt ${test_cnt[$idx]}")
        done
    done
    ((index++))
done

# echo "${jobs[@]}"

function check_gpu {
    local gpu_id=$1
    nvidia-smi -i $gpu_id --query-compute-apps=used_memory --format=csv,noheader | wc -l
}

for t in "${jobs[@]}"; do
    for (( i=0; i<${#models[@]}; i+=1 )); do
        m1=${models[i]}
        # m2=${models[i+1]}
        finish=0
        while [ $finish -eq 0 ]; do
            for gpu_id in {2..7}; do
                if [[ $(check_gpu $gpu_id) -eq 0 ]]; then
                    echo "$(date '+%Y-%m-%d %H:%M:%S') Running task on GPU $gpu_id: $t $m1"
                    source ~/lxt/venv/bin/activate
                    export CUDA_VISIBLE_DEVICES=$gpu_id
                    python train_llama_pl.py $t $m1 > /dev/null 2>&1 &
                    sleep 60
                    # echo "$(date '+%Y-%m-%d %H:%M:%S') Running task on GPU $gpu_id: $t $m2"
                    # python train_llama_pl.py $t $m2 > /dev/null 2>&1 &
                    # sleep 60
                    ((finish++))
                    break
                fi
            done
            if [ $finish -eq 0 ]; then
                sleep 60
            fi
        done
    done
done
