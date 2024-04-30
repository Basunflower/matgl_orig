#!/bin/bash
# nohup bash master.sh > master.log 2>&1 &
# emb
cd megnet_emb_0
bash megnet_orig.sh &

cd ../megnet_emb_0_1
bash megnet_orig_1.sh &

cd ../megnet_emb_1
bash megnet_orig.sh &

cd ../megnet_emb_1_1
bash megnet_orig_1.sh &

cd ../megnet_emb_2
bash megnet_orig.sh &

cd ../megnet_emb_2_1
bash megnet_orig_1.sh &

cd ../megnet_emb_3
bash megnet_orig.sh &

cd ../megnet_emb_3_1
bash megnet_orig_1.sh &

cd ../megnet_emb_4
bash megnet_orig.sh &

cd ../megnet_emb_4_1
bash megnet_orig_1.sh &

# mds
cd ../megnet_mds_0
bash megnet_pre.sh &

cd ../megnet_mds_0_1
bash megnet_pre_1.sh &

cd ../megnet_mds_1
bash megnet_pre.sh &

cd ../megnet_mds_1_1
bash megnet_pre_1.sh &

cd ../megnet_mds_2
bash megnet_pre.sh &

cd ../megnet_mds_2_1
bash megnet_pre_1.sh &

cd ../megnet_mds_3
bash megnet_pre.sh &

cd ../megnet_mds_3_1
bash megnet_pre_1.sh &

cd ../megnet_mds_4
bash megnet_pre.sh &

cd ../megnet_mds_4_1
bash megnet_pre_1.sh &

wait

echo "Finish."
