# python models/ops/setup.py build install;
# pip install -I transformers==4.5.1 --no-cache-dir --force-reinstall
# pip install -r requirements_ParSeDETRHOI.txt;
# pip install pkgs/pycocotools-2.0.2.tar.gz;
# pip install submitit==1.3.0;
# pip install timm;
# export NCCL_DEBUG=INFO;
# export NCCL_IB_HCA=mlx5;
# export NCCL_IB_TC=136;
# export NCCL_IB_SL=5;
# export NCCL_IB_GID_INDEX=3;
# export TORCH_DISTRIBUTED_DETAIL=DEBUG;
# # Pay attention to the learning rate if channging #nodes
python main.py \
    --hoi_path /cluster/home/clin/clin/RLIPv2/data/hico_20160224_det \
    --hico_det_file /cluster/home/clin/clin/RLIPv2/data/hico_20160224_det/internimage/hico_det.json \
    --output_dir /cluster/home/clin/clin/relation/results/50 \
    --hoi \
    --dataset_file hico_det \
    --num_queries 200 \
    --relation_threshold 0.20 \
    --num_workers 8 \
    --batch_size 4 \
    --use_correct_subject_category_hico \
    --epochs 100 \
    --verb_loss_type bce \
    --subset_size 100 \
    --do_cross_validation
    # --load_backbone supervised \
    # --backbone DINOv2 \
    # --set_cost_bbox 2.5 \
    # --set_cost_giou 1 \
    # --bbox_loss_coef 2.5 \
    # --giou_loss_coef 1 \
    # --lr_drop 15 \
    # --epochs 20 \
    # --lr 1e-4 \
    # --lr_backbone 0 \
    # --text_encoder_lr 1e-5 \
    # --schedule step \
    # --num_workers 4 \
    # --batch_size 2 \
    # --obj_loss_type cross_entropy \
    # --verb_loss_type focal \
    # --enc_layers 6 \
    # --dec_layers 3 \
    # --num_queries 200 \
    # --use_nms_filter \
    # --cross_modal_pretrain \
    # --save_ckp \
    # --RLIP_ParSeDA_v2 \
    # --with_box_refine \
    # --num_feature_levels 4 \
    # --num_patterns 0 \
    # --pe_temperatureH 20 \
    # --pe_temperatureW 20 \
    # --dim_feedforward 2048 \
    # --dropout 0.0 \
    # --drop_path_rate 0.5 \
    # --subject_class \
    # --use_no_obj_token \
    # --negative_text_sampling 500 \
    # --sampling_stategy freq \
    # --fusion_type GLIP_attn \
    # --gating_mechanism VXAc \
    # --verb_query_tgt_type vanilla_MBF \
    # --fusion_interval 2 \
    # --fusion_last_vis \
    # --lang_aux_loss \
    # --giou_verb_label \
    # --pseudo_verb \
    # --relation_threshold 0.20 \


    # COCO + VG
    # --pretrained /Path/To/params/swin_large_drop_path0.5_dp0_mqs_lft_dab_deformable_detr_plus_iterative_bbox_refinement_50eps_converted.pth \
    # --output_dir /Path/To/logs/RLIP_PDA_v2_SwinL_VGCOCO_BLIP_Nu10_thr20_RQL_LSE_RPL_20e \
    