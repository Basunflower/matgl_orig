#!/bin/bash
# nohup bash redress_master.sh > redress1.log 2>&1 &
# emb
cd megnet_emb
bash megnet_orig.sh 0 1 1&

cd ../megnet_emb_1
bash megnet_orig.sh 0 4 3&

cd ../megnet_emb_2
bash megnet_orig.sh 0 16 1&

cd ../megnet_emb_3
bash megnet_orig.sh 0 32 4&
# mds
cd ../megnet_mds
bash megnet_pre.sh 0 1 1&

cd ../megnet_mds_1
bash megnet_pre.sh 0 2 1&

cd ../megnet_mds_2
bash megnet_pre.sh 0 4 4&

cd ../megnet_mds_3
bash megnet_pre.sh 0 8 3&

wait

cd ../megnet_emb
bash megnet_orig.sh 0 64 4&

cd ../megnet_emb_1
bash megnet_orig.sh 0 128 6&

cd ../megnet_emb_2
bash megnet_orig.sh 1 1 1&

cd ../megnet_emb_3
bash megnet_orig.sh 1 2 3&
# mds
cd ../megnet_mds
bash megnet_pre.sh 0 16 2&

cd ../megnet_mds_1
bash megnet_pre.sh 0 32 1&

cd ../megnet_mds_2
bash megnet_pre.sh 0 64 3&

cd ../megnet_mds_3
bash megnet_pre.sh 1 1 1&

wait

cd ../megnet_emb
bash megnet_orig.sh 1 4 6&

cd ../megnet_emb_1
bash megnet_orig.sh 1 8 6&

cd ../megnet_emb_2
bash megnet_orig.sh 1 16 1&

cd ../megnet_emb_3
bash megnet_orig.sh 1 32 4&
# mds
cd ../megnet_mds
bash megnet_pre.sh 1 2 1&

cd ../megnet_mds_1
bash megnet_pre.sh 1 4 3&

cd ../megnet_mds_2
bash megnet_pre.sh 1 32 3&

cd ../megnet_mds_3
bash megnet_pre.sh 1 64 4&

echo "Finish."
