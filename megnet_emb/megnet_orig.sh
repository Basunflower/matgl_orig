#! /bin/bash

dim_node_embed_values=(1 2 4 8 16 32 64)

for dim in "${dim_node_embed_values[@]}"
do
      echo "Running with dim_node_embed = $dim"

      python megnet_orig.py --dim_node_embed $dim

      echo
done