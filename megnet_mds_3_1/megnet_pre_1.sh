#! /bin/bash
# bash megnet_orig.sh
# nohup bash megnet_pre.sh > top.log 2>&1 &
dim_node_embed_values=(16 32 64 128)

for dim in "${dim_node_embed_values[@]}"
do
      echo "MDS Running with dim_node_embed = $dim fold = 3"

      nohup python fix_embedding.py --dim_node_embed $dim --fold 3 > "fold3_dim${dim}_mds.log" 2>&1

      echo
done