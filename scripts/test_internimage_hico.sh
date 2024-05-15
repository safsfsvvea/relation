#!/bin/bash
python evaluate_det.py \
    --hoi_path "/cluster/home/clin/clin/RLIPv2/data/hico_20160224_det" \
    --output_dir /cluster/home/clin/clin/relation/results/test \
    --hoi \
    --dataset_file hico \
    --num_workers 1 \
    --batch_size 4 \
    --use_correct_subject_category_hico \
    --eval \
    # --subset_size 20 \