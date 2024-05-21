#!/bin/bash
python evaluate_det.py \
    --hoi_path "../RLIPv2/data/hico_20160224_det" \
    --output_dir results/test \
    --hoi \
    --dataset_file hico \
    --num_workers 1 \
    --batch_size 4 \
    --use_correct_subject_category_hico \
    --eval \
    # --subset_size 20 \