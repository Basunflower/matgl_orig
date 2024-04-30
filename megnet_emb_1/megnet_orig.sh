#! /bin/bash
# bash megnet_orig.sh
# nohup bash megnet_orig.sh > bash422.log 2>&1 &
dim_node_embed_values=(1 2 4 8)

for dim in "${dim_node_embed_values[@]}"
do
      echo "Embedding Running with dim_node_embed = $dim fold = 1"

      nohup python megnet_orig.py --dim_node_embed $dim --fold 1 > "fold1_dim${dim}.log" 2>&1

      echo
done
