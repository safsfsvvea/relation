#!/bin/bash
START_INDEX=0
END_INDEX=19

# 循环执行命令，递增 index 参数
for i in $(seq $START_INDEX $END_INDEX)
do
    python main.py \
    --pretrained /cluster/home/clin/clin/relation/results/test_20_gt_focal/checkpoint.pth.tar \
    --hoi_path /cluster/home/clin/clin/RLIPv2/data/hico_20160224_det \
    --hico_det_file /cluster/home/clin/clin/RLIPv2/data/hico_20160224_det/internimage/hico_det.json \
    --output_dir /cluster/home/clin/clin/relation/results/test \
    --hoi \
    --dataset_file hico_det_gt \
    --num_queries 200 \
    --relation_threshold 0.0 \
    --num_workers 1 \
    --batch_size 4 \
    --use_correct_subject_category_hico \
    --epochs 20 \
    --verb_loss_type bce \
    --eval \
    --subset_size 20 \
    --index $i
done

    