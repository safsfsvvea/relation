from models.denoise_backbone import DenoisingVitBackbone
from models.backbone import DINOv2Backbone
from models.hoi import HOIModel, CriterionHOI, PostProcessHOI, backbone_time
from models.matcher import HungarianMatcherHOI_det
from engine import train_one_epoch, evaluate_hoi, train_one_epoch_with_profiler, train_one_epoch_with_time, train_backbone_time
import datasets
from datasets.hico import CustomSubset
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset, BatchIterativeDistributedSampler
# from engine import evaluate, train_one_epoch, evaluate_hoi, evaluate_hoi_with_text, evaluate_hoi_with_text_matching_uniformity, \
#         evaluate_sgg_with_text
import json       
import argparse
from datetime import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import os
from torch.utils.data import Subset
from sklearn.model_selection import KFold
import copy
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import MultiStepLR

# from ultralytics import YOLO
def get_args_parser():
    parser = argparse.ArgumentParser('relation training and evaluation script', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_detector', default=1e-5, type=float)
    parser.add_argument('--lr_backbone', default=0, type=float) #1e-5
    parser.add_argument('--text_encoder_lr', default=5e-5, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--accumulation_steps', default=32, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument(
        "--schedule",
        default = None,
        type=str,
        choices=("step", "multistep", "linear_with_warmup", "all_linear_with_warmup"),
    )
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.9998)
    parser.add_argument("--fraction_warmup_steps", default=0.01, type=float, help="Fraction of total number of steps")
    parser.add_argument('--use_checkpoint', action='store_true') # This is for Swin-transformer to save memory.

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--frozen_vision', action = 'store_true',
                        help='Freeze vision model.')
    parser.add_argument('--unfrozen_params', action = 'store_true',
                        help='Unfreeze partial parameters.')
    parser.add_argument('--frozen_detection', action = 'store_true',
                        help='Freeze object detection part for RLIP.')
    #mlp
    parser.add_argument('--use_LN', action='store_true', help='use LayerNorm.')
    parser.add_argument('--add_negative_category', action='store_true', help='add negative category.')
    parser.add_argument('--positive_negative', action='store_true', help='add positive negative mlp.')

    # * Backbone
    parser.add_argument('--denoised', action='store_true', help='use denoised vit.')
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--load_backbone', default='supervised', type=str, choices=['swav', 'supervised'])

    # * Transformer
    parser.add_argument('--use_CLS', action='store_true', help='use CLS token.')
    parser.add_argument('--use_attention', action='store_true', help='use attention.')
    parser.add_argument(
        "--position_encoding_type",
        default = None,
        type=str,
        choices=("default", "HW")
    )
    parser.add_argument('--attention_layers', default=1, type=int,
                        help="Number of attention layers")
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer and mlp")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true') # False
    parser.add_argument('--stochastic_context_transformer', action = 'store_true',
                        help='Enable the stochastic context transformer')
    parser.add_argument('--semantic_hidden_dim', default=256, type=int,
                        help="Size of the embeddings for semantic reasoning")
    parser.add_argument('--gru_hidden_dim', default=256, type=int,
                        help="Size of the embeddings GRU")
    parser.add_argument('--dec_layers_hopd', default=3, type=int,
                        help="For CDNHOI: Number of hopd decoding layers in the transformer")
    parser.add_argument('--dec_layers_interaction', default=3, type=int,
                        help="For CDNHOI: Number of interaction decoding layers in the transformer")

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")


    # HOI
    # Only one of --coco, --hoi and --cross_modal_pretrain can be True.
    parser.add_argument('--topK', default=15, type=int,
                        help="iou top K filter human object pairs.")
    parser.add_argument('--iou_threshold', default=0.0, type=float,
                        help="iou threshold to filter human object pairs.")
    parser.add_argument('--hoi', action='store_true',
                        help="Train for HOI if the flag is provided")
    parser.add_argument('--sgg', action='store_true',
                        help="Train for HOI if the flag is provided")
    parser.add_argument('--coco', action='store_true',
                        help="Train for COCO if the flag is provided")
    parser.add_argument('--num_obj_classes', type=int, default=80,
                        help="Number of object classes")
    parser.add_argument('--num_verb_classes', type=int, default=117,
                        help="Number of verb classes")
    parser.add_argument('--pretrained', type=str, default='',
                        help='Pretrained model path')
    parser.add_argument('--pretrained_backbone', type=str, default='',
                        help='Pretrained backbone path')
    parser.add_argument('--pretrained_swin', type=str, default='',
                        help='Pretrained model path for the swin backbone only!!!')
    parser.add_argument('--drop_path_rate', default=0.2, type=float,
                        help="drop_path_rate applied in the Swin transformer.")
    parser.add_argument('--subject_category_id', default=0, type=int)
    parser.add_argument('--use_correct_subject_category_hico', action = 'store_true', 
                        help='We use the correct subject category. \
                              Previously, in HICOdetection class, the default subject category is set to 1. \
                              It is actually 0 (subject_category_id).')
    parser.add_argument('--verb_loss_type', type=str, default='focal',
                        help='Loss type for the verb classification')
    parser.add_argument('--obj_loss_type', type=str, default='cross_entropy',
                        help='Loss type for the obj (sub) classification')
    parser.add_argument('--matching_symmetric', action = 'store_true',
                        help='Whether to use symmetric cross-modal matching loss')
    parser.add_argument('--HOICVAE', action = 'store_true',
                        help='Enable the CVAE model for DETRHOI')
    parser.add_argument('--SemanticDETRHOI', action = 'store_true',
                        help='Enable the Semantic model for DETRHOI')
    parser.add_argument('--IterativeDETRHOI', action = 'store_true',
                        help='Enable the Iterative Refining model for DETRHOI')
    parser.add_argument('--DETRHOIhm', action = 'store_true',
                        help='Enable the verb heatmap query prediction for DETRHOI')
    parser.add_argument('--OCN', action = 'store_true',
                        help='Augment DETRHOI with Cross-Modal Calibrated Semantics.')
    parser.add_argument('--SeqDETRHOI', action = 'store_true',
                        help='Sequential decoding by DETRHOI')
    parser.add_argument('--SepDETRHOI', action = 'store_true',
                        help='SepDETRHOI: Fully disentangled decoding by DETRHOI')
    parser.add_argument('--ParSe', action = 'store_true',
                        help='ParSe: Fully disentangled decoding by DETRHOI')
    parser.add_argument('--SepDETRHOIv3', action = 'store_true',
                        help='SepDETRHOIv3: Fully disentangled decoding by DETRHOI')
    parser.add_argument('--CDNHOI', action = 'store_true',
                        help='CDNHOI')
    parser.add_argument('--RLIP_ParSe', action = 'store_true',
                        help='Parallel Detection and Sequential Relation Inferring.')
    parser.add_argument('--RLIP_ParSeD_v2', action = 'store_true',
                        help='RLIP_ParSeD_v2.') 
    parser.add_argument('--RLIP_ParSeDA_v2', action = 'store_true',
                        help='RLIP_ParSeDA_v2.') 
    parser.add_argument('--RLIP_ParSe_v2', action = 'store_true',
                        help='RLIP_ParSe_v2.') 
    parser.add_argument('--ParSeDABDDETR', action = 'store_true',
                        help='Parallel Detection and Sequential Relation Inferring using DAB-Deformable DETR.')
    parser.add_argument('--ParSeDABDETR', action = 'store_true',
                        help='Parallel Detection and Sequential Relation Inferring using DAB-DETR.')
    parser.add_argument('--RLIPParSeDABDETR', action = 'store_true',
                        help='RLIP-Parallel Detection and Sequential Relation Inferring using DAB-DETR.')
    parser.add_argument('--save_ckp', action = 'store_true', help='Save model for the last 5 epoches')
    
    # DDETRHOI
    parser.add_argument('--DDETRHOI', action = 'store_true',
                        help='Deformable DETR for HOI detection.')
    parser.add_argument('--ParSeD', action = 'store_true',
                        help='ParSeD: Fully disentangled decoding by DDETRHOI.')
    parser.add_argument('--RLIP_ParSeD', action = 'store_true',
                        help='Cross-modal Parallel Detection and Sequential Relation Inferring using DDETR.')
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    ## DABDETR
    parser.add_argument('--transformer_activation', default='prelu', type=str)
    parser.add_argument('--num_patterns', default=0, type=int,
                        help='number of pattern embeddings. See Anchor DETR for more details.')
    parser.add_argument('--random_refpoints_xy', action='store_true', 
                        help="Random init the x,y of anchor boxes and freeze them.")
    parser.add_argument('--pe_temperatureH', default=20, type=int, 
                        help="Temperature for height positional encoding.")
    parser.add_argument('--pe_temperatureW', default=20, type=int, 
                        help="Temperature for width positional encoding.")

    # Cross-Modal Fusion
    parser.add_argument('--use_checkpoint_fusion', default=False, action='store_true', help = 'Use checkpoint to save memory.')
    parser.add_argument('--fusion_type', default = "no_fusion", choices = ("MDETR_attn", "GLIP_attn", "no_fusion"), )
    parser.add_argument('--fusion_interval', default=1, type=int, help="Fusion interval in VLFuse.")
    parser.add_argument('--fusion_last_vis', default=False, action='store_true', help = 'Whether to fuse the last layer of the vision features.')
    parser.add_argument('--lang_aux_loss', default=False, action='store_true', help = 'Whether to use aux loss to calculate the loss functions.')
    parser.add_argument('--separate_bidirectional', default=False, action='store_true', help = 'For GLIP_attn, we perform separate attention for different levels of features.')
    parser.add_argument('--do_lang_proj_outside_checkpoint', default=False, action='store_true', help = 'Use feature resizer to project the concatenation of interactive language features to the dimension of language embeddings.')
    parser.add_argument('--stable_softmax_2d', default=False, action='store_true', help = 'Use "attn_weights = attn_weights - attn_weights.max()" during BiMultiHeadAttention in VLFuse.')
    parser.add_argument('--clamp_min_for_underflow', default=False, action='store_true', help = 'Clamp attention weights (before softmax) during BiMultiHeadAttention in VLFuse.')
    parser.add_argument('--clamp_max_for_overflow', default=False, action='store_true', help = 'Clamp attention weights (before softmax) during BiMultiHeadAttention in VLFuse.')


    # Cross-Modal Pretraining parameters
    parser.add_argument(
        "--cross_modal_pretrain",
        dest="cross_modal_pretrain",
        action="store_true",
        help="Whether to perform cross modal pretraining on VG",
    ) # Only one of --coco, --hoi and --cross_modal_pretrain can be True.
    parser.add_argument(
        "--verb_tagger",
        dest="verb_tagger",
        action="store_true",
        help="Whether to perform verb tagging pre-training",
    )
    parser.add_argument('--label_noise_scale', default=0.2, type=float,
                        help="label noise ratio to flip")
    parser.add_argument('--box_noise_scale', default=0.4, type=float,
                        help="box noise scale to shift and scale")
    parser.add_argument('--pos_neg_ratio', default=0.5, type=float,
                        help="box noise scale to shift and scale")
    parser.add_argument('--relation_threshold', default=0., type=float,
                        help="This is used in the postprocessor.")
    parser.add_argument('--pair_overlap', default=False, action='store_true', 
                        help = "Whether to use 'overlap' as prior knowledge to filter relations.")
    parser.add_argument(
        "--subject_class",
        dest="subject_class",
        action="store_true",
        help="Whether to classify the subject in a triplet",
    )
    parser.add_argument(
        "--use_no_verb_token",
        dest="use_no_verb_token",
        action="store_true",
        help="Whether to use No_verb_token",
    )
    parser.add_argument(
        "--use_no_obj_token",
        dest="use_no_obj_token",
        action="store_true",
        help="Whether to use No_obj_token",
    )
    parser.add_argument(
        "--postprocess_no_sigmoid",
        dest="postprocess_no_sigmoid",
        action="store_true",
        help="Whether to use sigmoid function for postprocessing on verb scores",
    )
    parser.add_argument(
        "--use_aliases",
        dest="use_aliases",
        action="store_true",
        help="Whether to use aliases to reduce label redundancy.",
    )
    parser.add_argument(
        "--use_all_text_labels",
        dest="use_all_text_labels",
        action="store_true",
        help="Whether to use all text labels as input.",
    )
    parser.add_argument(
        "--zero_shot_eval",
        default = None,
        choices = ("hico", "v-coco"),
    )
    parser.add_argument(
        '--negative_text_sampling', 
        default=0, 
        type=int)
    parser.add_argument(
        '--sampling_stategy', 
        type=str, 
        default=None,
        help="String to be parsed as sampling strategies for object and verb sampling.")
    parser.add_argument(
        "--giou_verb_label",
        dest="giou_verb_label",
        action="store_true",
        help="Whether to use sub's and obj's giou as an indicator for a verb soft label.",
    )
    parser.add_argument(
        "--verb_curing",
        dest="verb_curing",
        action="store_true",
        help="Whether to use curing score to suppress verb results.",
    )
    parser.add_argument(
        "--pseudo_verb",
        dest="pseudo_verb",
        action="store_true",
        help="Whether to use pseudo labels to overcome semantic ambiguity.",
    )
    parser.add_argument(
        '--naive_obj_smooth', 
        default=0, 
        type=float,
        help="Use the most naive version of label smoothing for subs and objs."
    )
    parser.add_argument(
        '--naive_verb_smooth', 
        default=0, 
        type=float,
        help="Use the most naive version of label smoothing for verbs."
    )
    parser.add_argument(
        "--triplet_filtering",
        dest="triplet_filtering",
        action="store_true",
        help="Whether to use triplet_filtering to filter out untrustworthy triplets during pre-training.",
    )
    parser.add_argument(
        "--masked_entity_modeling",
        default = None,
        choices = ("subobj"),
    )
    parser.add_argument(
        "--contrastive_loss_hdim",
        type=int,
        default=64,
        help="Projection head output size before computing normalized temperature-scaled cross entropy loss",
    )
    parser.add_argument(
        "--temperature_NCE", type=float, default=0.07, help="Temperature in the  temperature-scaled cross entropy loss"
    )
    parser.add_argument(
        "--no_pass_pos_and_query",
        dest="pass_pos_and_query",
        action="store_false",
        help="Disables passing the positional encodings to each attention layers",
    )
    parser.add_argument(
        "--text_encoder_type",
        default="roberta-base",
        choices=("roberta-base", "distilroberta-base", "roberta-large", "bert-base-uncased", "bert-base-cased"),
    )
    parser.add_argument(
        "--freeze_text_encoder", action="store_true", help="Whether to freeze the weights of the text encoder"
    )

    # Zero-shot setting
    parser.add_argument(
        '--few_shot_transfer',
        default=100,
        type=int,
        choices=[1, 10, 100],
    )
    parser.add_argument(
        '--zero_shot_setting',
        default=None,
        type=str,
        choices=[None, 'UC-RF', 'UC-NF', 'UO', 'UV'],
    )
    parser.add_argument(
        '--relation_label_noise',
        default=0,
        type=int,
        choices=[0, 10, 30, 50],
    )



    # HOI eval parameters
    parser.add_argument('--use_nms_filter', action='store_true', help='Use pair nms filter, default not use')
    parser.add_argument('--thres_nms', default=0.7, type=float, help='Threshold for the pairwise NMS')
    parser.add_argument('--nms_alpha', default=1.0, type=float)
    parser.add_argument('--nms_beta', default=0.5, type=float)
    parser.add_argument('--thres_nms_phr', default=0.5, type=float, help='Threshold for the phrase NMS, only available when the task includes relation detection.')


    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--entropy_bound', action = 'store_true',
                        help='Enable the loss to bound the entropy for the gaussian distribution')
    parser.add_argument('--kl_divergence', action = 'store_true',
                        help='Enable the loss to bound the shape for the gaussian distribution')
    parser.add_argument('--verb_gt_recon', action = 'store_true',
                        help='Enable the loss for recondtructing the gt labels.')
    parser.add_argument('--ranking_verb', action = 'store_true',
                        help='Enable the loss for ranking verbs.')
    parser.add_argument('--no_verb_bce_focal', action = 'store_true',
                        help='Disable the loss for loss_verb_labels.')
    parser.add_argument('--verb_hm', action = 'store_true',
                        help='Enable the heatmap loss DETRHOIhm.')
    parser.add_argument('--semantic_similar', action = 'store_true',
                        help='Enable the loss for semantic similarity.')
    parser.add_argument('--verb_threshold', action = 'store_true',
                        help='Enable the loss for verb similarity.')


    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_cost_obj_class', default=1, type=float,
                        help="Object class coefficient in the matching cost")
    parser.add_argument('--set_cost_verb_class', default=1, type=float,
                        help="Verb class coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--obj_loss_coef', default=1, type=float)
    parser.add_argument('--verb_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--entropy_bound_coef', default=0.01, type=float)
    parser.add_argument('--kl_divergence_coef', default=0.01, type=float)
    parser.add_argument('--verb_gt_recon_coef', default=1, type=float)
    parser.add_argument('--ranking_verb_coef', default=1, type=float)
    parser.add_argument('--verb_hm_coef', default=1, type=float)
    parser.add_argument('--exponential_hyper', default=0.8, type=float)
    parser.add_argument('--exponential_loss', action = 'store_true',
                        help='Enable the exponentially increasing loss coef.')
    parser.add_argument('--semantic_similar_coef', default=1, type=float)
    parser.add_argument('--verb_threshold_coef', default=1, type=float)
    parser.add_argument('--masked_loss_coef', default=1, type=float)
    

    # dataset parameters
    parser.add_argument('--notransform', action='store_true', help='not use transform in debug')
    parser.add_argument('--trainset_val', action='store_true', help='Use trainset validation')
    parser.add_argument('--index', default=None, type=int, help='index for test')
    parser.add_argument('--num_folds', default=5, type=int, help='Number of folds for cross validation')
    parser.add_argument('--do_cross_validation', action='store_true', help='Use cross validation')
    parser.add_argument('--subset_size', default=0, type=int, help='Number of samples to use for training; 0 means use the whole dataset')
    parser.add_argument('--random_subset', action='store_true', help='Select a random subset of the dataset')
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--hoi_path', type=str)
    parser.add_argument('--vg_path', type=str)
    parser.add_argument('--o365_path', type=str)
    parser.add_argument('--hico_path', type=str)
    parser.add_argument('--oi_sgg_path', type=str)
    parser.add_argument('--mixed_anno_file', type=str, 
                        help = "The mixed annotation file. One json for multiple datasets.")
    parser.add_argument('--keep_names_freq_file', type=str, 
                        help = "The mixed keep_names_freq file. One json for multiple datasets.")
    parser.add_argument('--hico_rel_anno_file', type=str, 
                        help = "The annotation file for Objects365 relation detection, used in the ConcatDataset mode.")
    parser.add_argument('--o365_rel_anno_file', type=str, 
                        help = "The annotation file for Objects365 relation detection, used in the ConcatDataset mode.")
    parser.add_argument('--coco_rel_anno_file', type=str, 
                        help = "The annotation file for COCO relation detection, used in the ConcatDataset mode.")
    
    parser.add_argument('--coco_det_file', type=str, 
                        help = "The detection result file for COCO")
    parser.add_argument('--hico_det_file', type=str, 
                        help = "The detection result file for HICO")
    
    parser.add_argument('--vg_rel_anno_file', type=str, 
                        help = "The annotation file for VG relation detection, used in the ConcatDataset mode.")
    parser.add_argument('--vg_keep_names_freq_file', type=str, 
                        help = "The keep_names_freq file for VG relation detection.")
    parser.add_argument('--iterative_paradigm', type=str,
                        help = "Enable pre-training on multiple datasets using gradient accumulation.")
    parser.add_argument('--gradient_strategy', default="vanilla", type=str,
                        choices=["vanilla", "gradient_accumulation"],
                        help = "Enable pre-training on multiple datasets using gradient accumulation.")
    parser.add_argument('--gating_mechanism', default="GLIP", type=str,
                        choices=["GLIP", "Vtanh", "Etanh", "Stanh", "SDFtanh", "SFtanh", "SOtanh", "SXAc", "SDFXAc", "VXAc", "SXAcLN", "SDFXAcLN", "SDFOXAcLN", "MBF", "XGating"],
                        help = "The gating mechanism used to perform language-vision feature fusion.")
    parser.add_argument('--verb_query_tgt_type', default="vanilla", type=str,
                        choices=["vanilla", "MBF", "vanilla_MBF"],
                        help = "The method used to generate queries.")


    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='logs/test',
                        help='path for tensdorboard logs')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    
    return parser

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    train_backbone = False
    if args.lr_backbone > 0:
        train_backbone = True
    backbone = DenoisingVitBackbone(
        model_type="vit_base_patch14_dinov2.lvd142m", device=device, checkpoint_path=args.pretrained_backbone,
        denoised=True, train_backbone=train_backbone
    )
    # detector = YOLO("checkpoints/yolov8m.pt").to(device)
    # print("detector: ", detector)
    
    # print(backbone)
    model = HOIModel(backbone, device=device, use_LN=args.use_LN, iou_threshold=args.iou_threshold, add_negative_category=args.add_negative_category, topK=args.topK, positive_negative=args.positive_negative, num_layers=args.attention_layers, dropout=args.dropout, denoised=args.denoised, position_encoding_type=args.position_encoding_type, use_attention=args.use_attention, use_CLS=args.use_CLS)
    # model = backbone_time(backbone, device=device, use_LN=args.use_LN, iou_threshold=args.iou_threshold, add_negative_category=args.add_negative_category, topK=args.topK, positive_negative=args.positive_negative, num_layers=args.attention_layers, dropout=args.dropout)
    matcher = HungarianMatcherHOI_det(
        device=device,
        cost_obj_class=args.set_cost_obj_class,
        cost_verb_class=args.set_cost_verb_class,
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou,
        add_negative_category=args.add_negative_category,
        args=args
    )
    criterion = CriterionHOI(matcher=matcher, device=device, loss_type=args.verb_loss_type, add_negative_category=args.add_negative_category, positive_negative=args.positive_negative)
    # optimizer = torch.optim.AdamW([
    # {'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr},
    # {'params': filter(lambda p: p.requires_grad, detector.parameters()), 'lr': args.lr_detector},
    # ], weight_decay=args.weight_decay)
    if train_backbone == True: 
        print("Number of backbone parameters before optimizer:", sum(p.numel() for p in model.backbone.parameters()))
        print("Number of mlp parameters before optimizer:", sum(p.numel() for p in model.mlp.parameters()))
        print("Number of positive_negative mlp parameters before optimizer:", sum(p.numel() for p in model.positive_negative.parameters()))
        optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': args.lr_backbone},
        {'params': model.mlp.parameters(), 'lr': args.lr},
        {'params': model.positive_negative.parameters(), 'lr': args.lr},
        ], weight_decay=args.weight_decay)
        print("Number of parameter tensors in optimizer backbone:", len(list(optimizer.param_groups[0]['params'])))
        print("Number of parameter tensors in optimizer mlp:", len(list(optimizer.param_groups[1]['params'])))
        print("Number of parameter tensors in optimizer positive_negative mlp:", len(list(optimizer.param_groups[2]['params'])))
        # 打印优化器中backbone参数的总数
        backbone_params_count = sum(p.numel() for p in optimizer.param_groups[0]['params'])
        print("Total number of parameters in optimizer backbone:", backbone_params_count)

        # 打印优化器中mlp参数的总数
        mlp_params_count = sum(p.numel() for p in optimizer.param_groups[1]['params'])
        print("Total number of parameters in optimizer mlp:", mlp_params_count)
        
        # 打印优化器中positive_negative mlp参数的总数
        positive_negative_params_count = sum(p.numel() for p in optimizer.param_groups[2]['params'])
        print("Total number of parameters in optimizer positive_negative mlp:", positive_negative_params_count)
    else:
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                    weight_decay=args.weight_decay)
        print("Number of parameter tensors in optimizer mlp:", len(list(optimizer.param_groups[0]['params'])))
        # 打印优化器中mlp参数的总数
        mlp_params_count = sum(p.numel() for p in optimizer.param_groups[0]['params'])
        print("Total number of parameters in optimizer mlp:", mlp_params_count)
    # if args.schedule is None:
    #     lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=1e-5, min_lr=1e-6)
    
    postprocessors = PostProcessHOI(relation_threshold=args.relation_threshold, positive_negative=args.positive_negative, device=device)
    if args.pretrained:
        print(f"Loading pretrained model from {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        if train_backbone:
            model.load_state_dict(checkpoint['state_dict'])
            print("Loaded entire model state dict")
        else:
            relation_state_dict = checkpoint['state_dict']
            model_state_dict = model.state_dict()
            
            filtered_relation_state_dict = {k: v for k, v in relation_state_dict.items() if k not in [
            "positive_negative.0.weight", "positive_negative.0.bias", 
            "positive_negative.2.weight", "positive_negative.2.bias", 
            "positive_negative.4.weight", "positive_negative.4.bias", 
            "positive_negative.6.weight", "positive_negative.6.bias", 
            "positive_negative.8.weight", "positive_negative.8.bias", 
            "positive_negative.10.weight", "positive_negative.10.bias"
        ]}
            model_state_dict.update(filtered_relation_state_dict)
            # model_state_dict.update(relation_state_dict)  # 只更新模型中不包含 `backbone` 的参数
            model.load_state_dict(model_state_dict)
            print("Loaded relation state dict")

        # optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {start_epoch}")
    image_set_key = 'train'

    dataset_train = build_dataset(image_set = image_set_key, args=args)
    if args.do_cross_validation:
        dataset_val = build_dataset(image_set='trainset_val', args=args)
        # args.rare_triplets = dataset_val.dataset.rare_triplets
    else:
        if not args.trainset_val:
            dataset_val = build_dataset(image_set='val', args=args)
        else:
            print("using trainset val!!!!!!")
            dataset_val = build_dataset(image_set='trainset_val', args=args)
    if args.subset_size > 0:
        # print("args.index", args.index)
        if args.random_subset:
            subset_indices = np.random.choice(len(dataset_train), args.subset_size, replace=False)
        elif args.index is not None:
            print("args.index here!!!")
            subset_indices = [args.index]
        else:
            subset_indices = list(range(min(args.subset_size, len(dataset_train))))
            # subset_indices = [42, 49]
            # subset_indices = [43, 49, 56, 57]
            # excluded_indices = {43, 47, 49, 56, 57, 59}
            # subset_indices = [i for i in range(2 * args.subset_size, min(3 * args.subset_size, len(dataset_train))) if i not in excluded_indices]
            # subset_indices = list(range(2*args.subset_size, min(3 * args.subset_size, len(dataset_train))))
        print("subset_indices: ", subset_indices)
        dataset_train = CustomSubset(dataset_train, subset_indices)
        dataset_val = CustomSubset(dataset_val, subset_indices)
        
    if args.iterative_paradigm is None:
        # if args.distributed:
        #     sampler_train = DistributedSampler(dataset_train)
        # else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        # print("it is here train!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)
    else:
        assert args.distributed
        batch_sampler_train = BatchIterativeDistributedSampler(dataset_train,
                                                               args.batch_size,
                                                               args.iterative_paradigm,
                                                               drop_last=False)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers, pin_memory=True)

    
    # we do not need eval during pretraining.
    
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                    drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers, pin_memory=True)
    
    # if args.distributed:
    #     print("it is here val!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #     sampler_val = DistributedSampler(dataset_val, shuffle=False)
    # else:
    if args.do_cross_validation:
        # 使用交叉验证
        kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=42)
        results1 = []
        results2 = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset_train)):
            train_subset = CustomSubset(dataset_train, train_idx)
            val_subset1 = CustomSubset(dataset_val, val_idx)
            val_subset2 = CustomSubset(dataset_val, train_idx)
            
            sampler_train = torch.utils.data.RandomSampler(train_subset)
            batch_sampler_train = torch.utils.data.BatchSampler(
                sampler_train, args.batch_size, drop_last=True)
            
            train_loader = DataLoader(train_subset, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
            
            sampler_val1 = torch.utils.data.SequentialSampler(val_subset1)
            val_loader1 = DataLoader(val_subset1, args.batch_size, sampler=sampler_val1,
                        drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
            sampler_val2 = torch.utils.data.SequentialSampler(val_subset2)
            val_loader2 = DataLoader(val_subset2, args.batch_size, sampler=sampler_val2,
                        drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

            # 重新初始化模型和优化器以避免泄露前一轮的信息
            model = HOIModel(backbone, device=device)
            matcher = HungarianMatcherHOI_det(
                device=device,
                cost_obj_class=args.set_cost_obj_class,
                cost_verb_class=args.set_cost_verb_class,
                cost_bbox=args.set_cost_bbox,
                cost_giou=args.set_cost_giou
            )
            criterion = CriterionHOI(matcher=matcher, device=device, loss_type=args.verb_loss_type)
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                        weight_decay=args.weight_decay)
            postprocessors = PostProcessHOI(relation_threshold=args.relation_threshold, device=device)

            print(f"Starting fold {fold+1}/{args.num_folds}")
            for epoch in range(args.epochs):
                train_loss = train_one_epoch(model, criterion, optimizer, train_loader, device, epoch)
                # 这里可以添加验证逻辑和TensorBoard记录

            # 评估这个折叠
            test_stats1 = evaluate_hoi(args.dataset_file, model, postprocessors, val_loader1, args.subject_category_id, device, args)
            test_stats2 = evaluate_hoi(args.dataset_file, model, postprocessors, val_loader2, args.subject_category_id, device, args)
            results1.append(test_stats1['mAP'])
            results2.append(test_stats2['mAP'])
            
            print(f"Fold {fold+1} mAP val: {test_stats1['mAP']}")
            print(f"Fold {fold+1} mAP train: {test_stats2['mAP']}")

        print("Cross-validation val results:", results1)
        print("Mean AP across val folds:", np.mean(results1))
        
        print("Cross-validation train results:", results2)
        print("Mean AP across train folds:", np.mean(results2))
    else:
        if args.output_dir:
            ensure_dir(args.output_dir)
        if args.hoi and args.eval:
            test_stats = evaluate_hoi(args.dataset_file, model, postprocessors, data_loader_val, args.subject_category_id, device, args, criterion=criterion)
            return
        
        tensorboard_writer = SummaryWriter(log_dir=args.log_dir)
        num_epochs = args.epochs
        lr_scheduler =None
        if args.schedule == "linear_with_warmup":
            # 计算总的训练步骤数
            total_steps = len(data_loader_train) * num_epochs // args.accumulation_steps
            warmup_steps = total_steps // 10  # 例如，使用总步骤数的10%作为warmup
            lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        elif args.schedule == "multistep":
            milestones = [int(0.2 * num_epochs), int(0.4 * num_epochs), int(0.6 * num_epochs)]
            gamma = 0.1
            lr_scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        print("lr_scheduler: ", lr_scheduler)
        best_loss = float('inf')
        best_epoch = -1
        best_map = 0
        start_time = time.time()
        train_loss = 0
        for epoch in range(num_epochs):
            train_loss, relation_loss, binary_loss = train_one_epoch(model, criterion, optimizer, data_loader_train, device, epoch, lr_scheduler=lr_scheduler, accumulation_steps=args.accumulation_steps, grad_clip_val=args.clip_max_norm, tensorboard_writer=tensorboard_writer, args=args)
            state_dict_to_save = model.state_dict() if train_backbone else {k: v for k, v in model.state_dict().items() if not k.startswith('backbone.')}
            if args.output_dir and epoch % 100 == 0:
                current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                checkpoint_filename = f"checkpoint_epoch_{epoch}_{current_time}.pth.tar"
                checkpoint_filename = os.path.join(args.output_dir, checkpoint_filename)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': state_dict_to_save,
                    'optimizer': optimizer.state_dict(),
                    'loss': train_loss,
                }, filename=checkpoint_filename)
                
                print(f"Checkpoint saved to {checkpoint_filename}")
            if train_loss < best_loss:
                best_loss = train_loss
                best_epoch = epoch
                
                # 保存最小损失模型
                best_model_filename = os.path.join(args.output_dir, "best_model.pth.tar")
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': state_dict_to_save,
                    'optimizer': optimizer.state_dict(),
                    'loss': train_loss,
                }, filename=best_model_filename)
                
                print(f"Best model saved to {best_model_filename} at epoch {epoch}")
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                test_stats = evaluate_hoi(args.dataset_file, model, postprocessors, data_loader_val, args.subject_category_id, device, args, criterion=criterion)
                tensorboard_writer.add_scalar('Test/mAP', test_stats['mAP'], epoch)
                
                if test_stats['mAP'] > best_map:
                    best_map = test_stats['mAP']
                    best_map_model_filename = os.path.join(args.output_dir, "best_map_model.pth.tar")
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': state_dict_to_save,
                        'optimizer': optimizer.state_dict(),
                        'loss': train_loss,
                        'mAP': test_stats['mAP'],
                    }, filename=best_map_model_filename)
                    print(f"Best mAP model saved to {best_map_model_filename} with mAP {best_map} at epoch {epoch}")
            checkpoint_filename = f"checkpoint.pth.tar"
            checkpoint_filename = os.path.join(args.output_dir, checkpoint_filename)
            save_checkpoint({
                'epoch': num_epochs,
                'state_dict': state_dict_to_save,
                'optimizer': optimizer.state_dict(),
                'loss': train_loss,
            }, filename=checkpoint_filename)
            
            print(f"Checkpoint saved to {checkpoint_filename}")
        
        elapsed_time = time.time() - start_time
        print(f"Training completed in {elapsed_time:.2f} seconds")
        tensorboard_writer.close()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser('relation training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)