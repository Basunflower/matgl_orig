#! /bin/bash
# bash megnet_orig.sh
# nohup bash megnet_orig.sh > bash422.log 2>&1 &
dim_node_embed_values=(1 2 4 8 16 32 64 128)

for dim in "${dim_node_embed_values[@]}"
do
      echo "Running with dim_node_embed = $dim"

      nohup python ../megnet_emb_0/megnet_orig.py --dim_node_embed $dim --fold 0 > "fold0_dim${dim}.log" 2>&1 &
      nohup python ../megnet_emb_1/megnet_orig.py --dim_node_embed $dim --fold 1 > "fold1_dim${dim}.log" 2>&1 &
      nohup python ../megnet_emb_2/megnet_orig.py --dim_node_embed $dim --fold 2 > "fold2_dim${dim}.log" 2>&1 &
      nohup python ../megnet_emb_3/megnet_orig.py --dim_node_embed $dim --fold 3 > "fold3_dim${dim}.log" 2>&1 &
      nohup python ../megnet_emb_4/megnet_orig.py --dim_node_embed $dim --fold 4 > "fold4_dim${dim}.log" 2>&1 &

      # 等待所有后台进程结束
      wait

      echo
done
