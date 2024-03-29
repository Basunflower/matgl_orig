#! /bin/bash
# bash megnet_orig.sh
# nohup bash megnet_orig.sh > perovskites_different_dim_2481632.log 2>&1 &
dim_node_embed_values=(128) #(2 4 8 16 32)  # 1,64)

for dim in "${dim_node_embed_values[@]}"
do
      echo "Running with dim_node_embed = $dim"

      python megnet_orig.py --dim_node_embed $dim

      echo
done