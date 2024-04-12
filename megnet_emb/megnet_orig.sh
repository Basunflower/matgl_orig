#! /bin/bash
# bash megnet_orig.sh
# nohup bash megnet_orig.sh > matbench.log 2>&1 &
dim_node_embed_values=(1 2 4 8 16 32 64 128)

for dim in "${dim_node_embed_values[@]}"
do
      echo "Running with dim_node_embed = $dim"

      python megnet_orig.py --dim_node_embed $dim

      echo
done