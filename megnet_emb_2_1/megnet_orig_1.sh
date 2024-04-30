#! /bin/bash
# bash megnet_orig.sh
# nohup bash megnet_orig.sh > bash422.log 2>&1 &
dim_node_embed_values=(16 32 64 128)

for dim in "${dim_node_embed_values[@]}"
do
      echo "Embedding Running with dim_node_embed = $dim fold = 2"

      nohup python megnet_orig.py --dim_node_embed $dim --fold 2 > "fold2_dim${dim}.log" 2>&1

      echo
done
