#! /bin/bash
# bash megnet_orig.sh
# nohup bash megnet_pre.sh > top.log 2>&1 &
dim_node_embed_values=(1 2 4 8)

for dim in "${dim_node_embed_values[@]}"
do
      echo "MDS Running with dim_node_embed = $dim fold = 1"

      nohup python fix_embedding.py --dim_node_embed $dim --fold 1 > "fold1_dim${dim}_mds.log" 2>&1

      echo
done