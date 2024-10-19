import torch
from torch import nn, Tensor
from typing import Dict, List, Optional, Tuple
import torch.nn.functional as F
from util.misc import NestedTensor
import time
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align, sigmoid_focal_loss
import concurrent.futures
import torch.profiler
from util.box_ops import box_cxcywh_to_xyxy
from torchvision.ops import box_iou
from models.position_encoding import PositionEmbeddingSine, PositionEmbeddingSineHW
import torchvision.ops.boxes as box_ops
import math
import torch.distributed as dist
from models.transformers import (
    TransformerEncoder,
    TransformerDecoder,
    TransformerDecoderLayer,
    SwinTransformer,
)
from models.ops import (
    binary_focal_loss_with_logits,
    compute_spatial_encodings,
    prepare_region_proposals,
    associate_with_ground_truth,
    compute_prior_scores,
    compute_sinusoidal_pe
)
from torchvision.ops import FeaturePyramidNetwork
from collections import OrderedDict

class Permute(nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims
    def forward(self, x: Tensor) -> Tensor:
        return x.permute(self.dims)

class FeatureHead(nn.Module):
    def __init__(self, dim, dim_backbone, return_layer, num_layers):
        super().__init__()
        self.dim = dim
        self.dim_backbone = dim_backbone
        self.return_layer = return_layer

        in_channel_list = [
            int(dim_backbone * 2 ** i)
            for i in range(return_layer + 1, 1)
        ]
        self.fpn = FeaturePyramidNetwork(in_channel_list, dim)
        self.layers = nn.Sequential(
            Permute([0, 2, 3, 1]),
            SwinTransformer(dim, num_layers)
        )
    def forward(self, x):
        pyramid = OrderedDict(
            (f"{i}", x[i].tensors)
            for i in range(self.return_layer, 0)
        )
        mask = x[self.return_layer].mask
        x = self.fpn(pyramid)[f"{self.return_layer}"]
        x = self.layers(x)
        return x, mask

class MultiModalFusion(nn.Module):
    def __init__(self, fst_mod_size, scd_mod_size, repr_size):
        super().__init__()
        self.fc1 = nn.Linear(fst_mod_size, repr_size)
        self.fc2 = nn.Linear(scd_mod_size, repr_size)
        self.ln1 = nn.LayerNorm(repr_size)
        self.ln2 = nn.LayerNorm(repr_size)

        mlp = []
        repr_size = [2 * repr_size, int(repr_size * 1.5), repr_size]
        for d_in, d_out in zip(repr_size[:-1], repr_size[1:]):
            mlp.append(nn.Linear(d_in, d_out))
            mlp.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = self.ln1(self.fc1(x))
        y = self.ln2(self.fc2(y))
        z = F.relu(torch.cat([x, y], dim=-1))
        z = self.mlp(z)
        return z

class HumanObjectMatcher(nn.Module):
    def __init__(self, repr_size, num_verbs, obj_to_verb, dropout=.1, human_idx=0):
        super().__init__()
        self.repr_size = repr_size
        self.num_verbs = num_verbs
        self.human_idx = human_idx
        self.obj_to_verb = obj_to_verb

        self.ref_anchor_head = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 2)
        )
        self.spatial_head = nn.Sequential(
            nn.Linear(36, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, repr_size), nn.ReLU(),
        )
        self.encoder = TransformerEncoder(num_layers=2, dropout=dropout)
        self.mmf = MultiModalFusion(512, repr_size, repr_size)

    def check_human_instances(self, labels):
        is_human = labels == self.human_idx
        n_h = torch.sum(is_human)
        if not torch.all(labels[:n_h]==self.human_idx):
            raise AssertionError("Human instances are not permuted to the top!")
        return n_h

    def compute_box_pe(self, boxes, embeds, image_size):
        bx_norm = boxes / image_size[[1, 0, 1, 0]]
        bx_c = (bx_norm[:, :2] + bx_norm[:, 2:]) / 2
        b_wh = bx_norm[:, 2:] - bx_norm[:, :2]

        c_pe = compute_sinusoidal_pe(bx_c[:, None], 20).squeeze(1)
        wh_pe = compute_sinusoidal_pe(b_wh[:, None], 20).squeeze(1)

        box_pe = torch.cat([c_pe, wh_pe], dim=-1)

        # Modulate the positional embeddings with box widths and heights by
        # applying different temperatures to x and y
        ref_hw_cond = self.ref_anchor_head(embeds).sigmoid()    # n_query, 2
        # Note that the positional embeddings are stacked as [pe(y), pe(x)]
        c_pe[..., :128] *= (ref_hw_cond[:, 1] / b_wh[:, 1]).unsqueeze(-1)
        c_pe[..., 128:] *= (ref_hw_cond[:, 0] / b_wh[:, 0]).unsqueeze(-1)

        return box_pe, c_pe

    def forward(self, region_props, image_sizes, device):
        ho_queries = []
        paired_indices = []
        prior_scores = []
        object_types = []
        positional_embeds = []
        for i, rp in enumerate(region_props):
            if not rp:
                ho_queries.append(torch.zeros(0, self.repr_size, device=device))
                paired_indices.append(torch.zeros(0, 2, device=device, dtype=torch.int64))
                prior_scores.append(torch.zeros(0, 2, self.num_verbs, device=device))
                object_types.append(torch.zeros(0, device=device, dtype=torch.int64))
                positional_embeds.append({})
                continue
            boxes, scores, labels, embeds = rp.values()
            # print("boxes: ", boxes.shape)
            # print("scores shape: ", scores.shape)
            # print("scores: ", scores)
            # print("labels: ", labels)
            # print("labels shape: ", labels.shape)
            # print("embeds: ", embeds.shape)
            # print("labels check: ", labels)
            nh = self.check_human_instances(labels)
            n = len(boxes)
            # Enumerate instance pairs
            x, y = torch.meshgrid(
                torch.arange(n, device=device),
                torch.arange(n, device=device)
            )
            x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < nh)).unbind(1)
            # Skip image when there are no valid human-object pairs
            if len(x_keep) == 0:
                ho_queries.append(torch.zeros(0, self.repr_size, device=device))
                paired_indices.append(torch.zeros(0, 2, device=device, dtype=torch.int64))
                prior_scores.append(torch.zeros(0, 2, self.num_verbs, device=device))
                object_types.append(torch.zeros(0, device=device, dtype=torch.int64))
                positional_embeds.append({})
                continue
            x = x.flatten(); y = y.flatten()
            # Compute spatial features
            pairwise_spatial = compute_spatial_encodings(
                [boxes[x],], [boxes[y],], [image_sizes[i],]
            )
            pairwise_spatial = self.spatial_head(pairwise_spatial)
            pairwise_spatial_reshaped = pairwise_spatial.reshape(n, n, -1)

            box_pe, c_pe = self.compute_box_pe(boxes, embeds, image_sizes[i])
            embeds, _ = self.encoder(embeds.unsqueeze(1), box_pe.unsqueeze(1))
            embeds = embeds.squeeze(1)
            # Compute human-object queries
            ho_q = self.mmf(
                torch.cat([embeds[x_keep], embeds[y_keep]], dim=1),
                pairwise_spatial_reshaped[x_keep, y_keep]
            )
            # Append matched human-object pairs
            ho_queries.append(ho_q)
            paired_indices.append(torch.stack([x_keep, y_keep], dim=1))
            prior_scores.append(compute_prior_scores(
                x_keep, y_keep, scores, labels, self.num_verbs, self.training,
                self.obj_to_verb
            ))
            object_types.append(labels[y_keep])
            positional_embeds.append({
                "centre": torch.cat([c_pe[x_keep], c_pe[y_keep]], dim=-1).unsqueeze(1),
                "box": torch.cat([box_pe[x_keep], box_pe[y_keep]], dim=-1).unsqueeze(1)
            })

        return ho_queries, paired_indices, prior_scores, object_types, positional_embeds

class PViC(nn.Module):
    def __init__(self, backbone, device, num_objects=None, feature_dim=768, person_category_id=1, patch_size=14, num_heads=8, num_layers=1, dropout=0.1, denoised=True, roi_size=7, raw_lambda=2.8, args=None):
        super(PViC, self).__init__()
        self.backbone = backbone
        self.device = device
        self.num_objects = num_objects
        self.feature_dim = feature_dim
        self.person_category_id = person_category_id
        self.patch_size = patch_size
        self.no_pairs_count = 0  # 用于统计 "No pairs found for this batch." 的计数
        self.no_results_count = 0  # 用于统计 "No results found for this image." 的计数
        self.num_category = 117
        self.denoised = denoised
        self.position_encoding = None
        self.roi_size= roi_size
        self.raw_lambda = raw_lambda
        self.alpha = args.alpha
        self.gamma = args.gamma
        self.ref_anchor_head = nn.Sequential(
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 2)
            )
        self.ho_matcher = HumanObjectMatcher(
            repr_size=384,
            num_verbs=117,
            obj_to_verb=args.object_to_verb,
            dropout=dropout,
            human_idx = 0
        )
        self.feature_head = FeatureHead(
            256, feature_dim,
            -1, 1
        )
        self.kv_pe = PositionEmbeddingSine(128, 20, normalize=True)
        decoder_layer = TransformerDecoderLayer(
            q_dim=384, kv_dim=256,
            ffn_interm_dim=384 * 4,
            num_heads=num_heads, dropout=dropout
        )
        self.decoder = TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=2
        )
        self.binary_classifier = nn.Linear(384, self.num_category)
        self.attention_pool = AttentionPool2d(
            spacial_dim=roi_size,      # 7x7 spatial dimensions
            embed_dim=feature_dim,      # embedding dimension (channels of input feature)
            num_heads=num_heads,        # number of attention heads
            output_dim=256     # output dimension, default to 768
        )
        
        self.to(self.device)

    def compute_box_pe(self, boxes, embeds, image_size):
        bx_norm = boxes / image_size[[1, 0, 1, 0]]
        bx_c = (bx_norm[:, :2] + bx_norm[:, 2:]) / 2
        b_wh = bx_norm[:, 2:] - bx_norm[:, :2]

        c_pe = compute_sinusoidal_pe(bx_c[:, None], 20).squeeze(1)
        wh_pe = compute_sinusoidal_pe(b_wh[:, None], 20).squeeze(1)

        box_pe = torch.cat([c_pe, wh_pe], dim=-1)

        # Modulate the positional embeddings with box widths and heights by
        # applying different temperatures to x and y
        ref_hw_cond = self.ref_anchor_head(embeds).sigmoid()    # n_query, 2
        # Note that the positional embeddings are stacked as [pe(y), pe(x)]
        c_pe[..., :128] *= (ref_hw_cond[:, 1] / b_wh[:, 1]).unsqueeze(-1)
        c_pe[..., 128:] *= (ref_hw_cond[:, 0] / b_wh[:, 0]).unsqueeze(-1)

        return box_pe, c_pe
    
    def separate_pooled_features(self, pooled_features, detection_counts):
        separated_features = []
        # separated_additional_info = []
        current_idx = 0

        for count in detection_counts:
            if count > 0:
                separated_features.append(pooled_features[current_idx:current_idx + count])
                # separated_additional_info.append(additional_info[current_idx:current_idx + count])
            else:
                separated_features.append(torch.empty(0, pooled_features.size(1)))
                # separated_additional_info.append([])
            current_idx += count

        return separated_features
    
    def generate_attention_mask(self, num_human=49, num_object=49, num_cls=1):
        total_features = num_human + num_object + num_cls
        attn_mask = torch.zeros((total_features, total_features), dtype=torch.bool, device=self.device)

        # Mask掉human内部、object内部以及cls token之间的注意力
        attn_mask[:num_human, :num_human] = True
        attn_mask[num_human:num_human+num_object, num_human:num_human+num_object] = True
        if num_cls > 0:
            attn_mask[-num_cls:, -num_cls:] = True

        return attn_mask
    
    def compute_classification_loss(self, logits, prior, labels):
        # print("----------------")
        # print("prior1 len: ", len(prior))
        # print("logits1 shape: ", logits.shape)
        # print("labels1 shape: ", labels.shape)
        prior = torch.cat(prior, dim=0).prod(1)
        # print("prior2 shape: ", prior.shape)
        x, y = torch.nonzero(prior).unbind(1)
        # print("x shape: ", x.shape)
        # print("y shape: ", y.shape)
        logits = logits[:, x, y]
        # print("logits2 shape: ", logits.shape)
        prior = prior[x, y]
        # print("prior3 shape: ", prior.shape)
        labels = labels[None, x, y].repeat(len(logits), 1)
        # print("labels2 shape: ", labels.shape)

        n_p = labels.sum()
        if dist.is_initialized():
            print("multi-gpu!!!")
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()
        
        # print("prior: ", prior)
        # print("logits: ", logits)
        # print("labels: ", labels)
        # print("prior shape: ", prior.shape)
        # print("logits shape: ", logits.shape)
        # print("labels shape: ", labels.shape)
        
        loss = binary_focal_loss_with_logits(
            torch.log(
                prior / (1 + torch.exp(-logits) - prior) + 1e-8
            ), labels, reduction='mean',
            alpha=self.alpha, gamma=self.gamma
        )
        # print("loss: ", loss)
        # print("----------------")
        return loss / n_p
    
    def postprocessing(self,
            boxes, paired_inds, object_types,
            logits, prior, image_sizes
        ):
        n = [len(p_inds) for p_inds in paired_inds]
        logits = logits.split(n)

        detections = []
        for bx, p_inds, objs, lg, pr, size in zip(
            boxes, paired_inds, object_types,
            logits, prior, image_sizes
        ):
            pr = pr.prod(1)
            x, y = torch.nonzero(pr).unbind(1)
            scores = lg[x, y].sigmoid() * pr[x, y].pow(self.raw_lambda)
            detections.append(dict(
                boxes=bx, pairing=p_inds[x], scores=scores,
                labels=y, objects=objs[x], size=size, x=x
            ))

        return detections
    def convert_output(self, logits, paired_inds, object_types, additional_info):
        output = []
        # print("len(paired_inds): ", len(paired_inds))
        cumulative_pairs = 0
        for b in range(len(paired_inds)):
            batch_output = []
            if len(paired_inds[b]) == 0:  # 如果没有配对
                output.append(batch_output)  # 直接添加空的 batch_output
                continue  # 跳过后续处理
            for i, pair_idx in enumerate(paired_inds[b]):
                subject_idx, object_idx = pair_idx
                
                # 创建每个配对的字典
                subject_bbox = additional_info[b]['boxes'][subject_idx]
                object_bbox = additional_info[b]['boxes'][object_idx]
                subject_score = additional_info[b]['scores'][subject_idx]
                object_score = additional_info[b]['scores'][object_idx]
                relation_score = logits[-1, cumulative_pairs + i, :]
                # print("object_idx: ", object_idx)
                # print("object_types[b][object_idx]: ", object_types[b][object_idx])
                entry = {
                    'subject_category': 1,
                    'subject_score': subject_score.item(),  
                    'subject_bbox': subject_bbox.tolist(),
                    'object_category': object_types[b][i].item() + 1,
                    'object_score': object_score.item(),  
                    'object_bbox': object_bbox.tolist(),
                    'relation_score': relation_score,
                    'binary_score': None
                }
                batch_output.append(entry)
            output.append(batch_output)
            cumulative_pairs += len(paired_inds[b])
    
        return output
    def forward(self, nested_tensor: NestedTensor, rois_tensor, additional_info, detection_counts, size, targets=None):
        images = nested_tensor.tensors
        # print("image size", images.size())
        mask = nested_tensor.mask
        batch_size = images.size(0)

        batch_denoised_features, raw_features, CLS_token = self.backbone(images)
        # print("batch_denoised_features: ", batch_denoised_features.shape)
        # print("raw_features: ", raw_features.shape)
        if self.denoised:
            backbone_features = batch_denoised_features
        else:
            backbone_features = raw_features
        backbone_features = backbone_features.permute(0, 3, 1, 2)
        
        # 将 mask 调整到与 backbone_features 一样的尺度
        # mask 原始大小是 [B, H, W]，需要先增加一个维度变为 [B, 1, H, W] 以便于插值
        mask = mask.unsqueeze(1).float()  # 增加一个维度并转换为 float 类型

        # 进行最近邻插值，将 mask 调整到与 backbone_features 一样的尺度
        mask = F.interpolate(mask, size=backbone_features.shape[-2:], mode='nearest')

        # 调整后的 mask 变回 [B, H', W'] 形式
        mask = mask.squeeze(1).bool()
        
        nested_backbone_feature = NestedTensor(backbone_features, mask)
        dino_features = [nested_backbone_feature]
        # print("nested_backbone_feature tensors: ", nested_backbone_feature.tensors.shape) # torch.Size([1, 256, 200, 300])
        # print("nested_backbone_feature mask: ", nested_backbone_feature.mask.shape) # torch.Size([1, 200, 300])
        
        output_size = (self.roi_size, self.roi_size)

        rois_tensor = [tensor.to(self.device) for tensor in rois_tensor]
        if all(roi.numel() == 0 for roi in rois_tensor):
            print("No pairs found for this batch.")
            return [[] for _ in range(batch_size)]  # Return a list of empty lists, one per image in batch
        pooled_features = roi_align(backbone_features, rois_tensor, output_size)
        # print("pooled_features shape: ", pooled_features.shape)
        pooled_features = self.attention_pool(pooled_features)
        # print("pooled_features out shape: ", pooled_features.shape)
        separated_features = self.separate_pooled_features(pooled_features, detection_counts)
        # print("separated_features: ", len(separated_features))
        # print("additional_info: ", additional_info)
        for info in additional_info:
            if not info:  # 如果 info 为空，则跳过当前循环
                # print("info is empty, skipping...")
                continue  # 继续下一次循环
            # print("info: ", info)
            info['boxes'] = torch.tensor(info['boxes']).to(self.device)  
            info['scores'] = torch.tensor(info['scores']).to(self.device) 
            info['labels'] = torch.tensor(info['labels']).to(self.device) - 1
        for i in range(len(additional_info)):
            if not additional_info[i]:
                # print("additional_info[i] is empty, skipping...")
                continue  # 继续下一次循环
            additional_info[i]['embeds'] = separated_features[i]
        # boxes = [r['boxes'] for r in additional_info]
        # scores = [r['scores'] for r in additional_info]
        # print("boxes[0].shape: ", boxes[0].shape)
        (
            ho_queries,
            paired_inds, prior_scores,
            object_types, positional_embeds
        ) = self.ho_matcher(additional_info, size, self.device)
        # print("len ho_queries: ", len(ho_queries)) # 1
        # print("ho_queries: ", ho_queries[0].shape) # torch.Size([15, 384])
        # print("paired_inds: ", paired_inds[0].shape) # torch.Size([15, 2])
        # print("prior_scores: ", prior_scores[0].shape) # torch.Size([15, 2, 117])
        # print("object_types: ", object_types[0].shape) # torch.Size([15])
        # Compute keys/values for triplet decoder.
        memory, mask = self.feature_head(dino_features)
        # print("memory: ", memory.shape) # torch.Size([1, 25, 38, 256])
        # print("mask: ", mask.shape) # torch.Size([1, 25, 38])
        b, h, w, c = memory.shape
        memory = memory.reshape(b, h * w, c)
        kv_p_m = mask.reshape(-1, 1, h * w)
        k_pos = self.kv_pe(NestedTensor(memory, mask)).permute(0, 2, 3, 1).reshape(b, h * w, 1, c)
        # Enhance visual context with triplet decoder.
        query_embeds = []
        for i, (ho_q, mem) in enumerate(zip(ho_queries, memory)):
            query_embeds.append(self.decoder(
                ho_q.unsqueeze(1),              # (n, 1, q_dim)
                mem.unsqueeze(1),               # (hw, 1, kv_dim)
                kv_padding_mask=kv_p_m[i],      # (1, hw)
                q_pos=positional_embeds[i],     # centre: (n, 1, 2*kv_dim), box: (n, 1, 4*kv_dim)
                k_pos=k_pos[i]                  # (hw, 1, kv_dim)
            ).squeeze(dim=2))
        # Concatenate queries from all images in the same batch.
        # print("query_embeds len: ", len(query_embeds)) # 1
        query_embeds = torch.cat(query_embeds, dim=1)   # (ndec, \sigma{n}, q_dim)
        # print("query_embeds: ", query_embeds.shape) # torch.Size([2, 15, 384])  2：# of decoder layers
        logits = self.binary_classifier(query_embeds)
        # print("logits: ", logits.shape) # torch.Size([2, 15, 117])
        # print("targets: ", targets)
        # if self.training:
        #     labels = associate_with_ground_truth(
        #         boxes, paired_inds, targets, self.num_category
        #     )
        #     print("labels: ", labels)
        #     cls_loss = self.compute_classification_loss(logits, prior_scores, labels)
        #     # loss_dict = dict(cls_loss=cls_loss)
        #     return cls_loss

        # detections = self.postprocessing(
        #     boxes, paired_inds, object_types,
        #     logits[-1], prior_scores, size
        # )
        
        # for i, result in enumerate(detections):
        #     print(f"Result {i+1}:")
        #     for key, value in result.items():
        #         if isinstance(value, torch.Tensor):
        #             print(f"  Key: {key}, Shape: {value.shape}")
        #         else:
        #             print(f"  Key: {key}, Value: {value}")
        #     print("-" * 50)
        # Result 1:
        # Key: boxes, Shape: torch.Size([6, 4])
        # Key: pairing, Shape: torch.Size([138, 2])
        # Key: scores, Shape: torch.Size([138])
        # Key: labels, Shape: torch.Size([138])
        # Key: objects, Shape: torch.Size([138])
        # Key: size, Shape: torch.Size([2])
        # Key: x, Shape: torch.Size([138])
        final_output = self.convert_output(logits, paired_inds, object_types, additional_info)
        return final_output 
class HOIModel(nn.Module):
    def __init__(self, backbone, device, num_objects=None, feature_dim=768, person_category_id=1, patch_size=14, use_LN=True, iou_threshold = 0.0, topK = 15, positive_negative = False, num_heads=8, num_layers=1, dropout=0.1, denoised=True, position_encoding_type=None, use_attention=True, use_CLS=True, roi_size=7, use_self_attention=True, position_bbox = True, position_relative_bbox=True, position_bbox_dim=512):
        super(HOIModel, self).__init__()
        self.backbone = backbone
        self.device = device
        self.num_objects = num_objects
        self.feature_dim = feature_dim
        self.person_category_id = person_category_id
        self.patch_size = patch_size
        self.no_pairs_count = 0  # 用于统计 "No pairs found for this batch." 的计数
        self.no_results_count = 0  # 用于统计 "No results found for this image." 的计数
        self.iou_threshold = iou_threshold
        self.num_category = 117
        self.topK = topK
        self.positive_negative = positive_negative
        self.denoised = denoised
        self.position_encoding = None
        self.use_CLS = use_CLS
        self.roi_size= roi_size
        self.use_self_attention = use_self_attention
        self.position_relative_bbox = position_relative_bbox
        self.position_bbox = position_bbox
        self.ref_anchor_head = nn.Sequential(
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 2)
            )
        if self.position_bbox:
            self.attention_pool = AttentionPool2d(
                spacial_dim=roi_size,      # 7x7 spatial dimensions
                embed_dim=feature_dim,      # embedding dimension (channels of input feature)
                num_heads=num_heads,        # number of attention heads
                output_dim=None     # output dimension, default to 768
            )
        if self.position_relative_bbox:
            # Map spatial encodings to a specific dimension
            self.spatial_head = nn.Sequential(
                nn.Linear(36, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, position_bbox_dim),
                nn.ReLU(),
            )
            self.mbf = MultiBranchFusion(
                feature_dim * 2,
                position_bbox_dim, feature_dim * 2,
                cardinality=16
            )
        if use_attention:
            if use_CLS or self.use_self_attention:
                self.attention = MultiheadAttentionStack(embed_dim=feature_dim, num_heads=num_heads, num_layers=num_layers, dropout=dropout)
            else:
                self.attention = BidirectionalCrossAttentionStack(embed_dim=feature_dim, num_heads=num_heads, num_layers=num_layers, dropout=dropout)
            if position_encoding_type == 'default':
                self.position_encoding = PositionEmbeddingSine(num_pos_feats=feature_dim // 2)
            elif position_encoding_type == 'HW':
                self.position_encoding = PositionEmbeddingSineHW(num_pos_feats=feature_dim // 2)
            # if self.position_bbox:
            self.mlp = nn.Sequential(
            nn.Linear(feature_dim*2, 256),
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(256, self.num_category)
            )
        else:
            self.attention = None      
            if not use_LN:
                self.mlp = nn.Sequential(
                    nn.Linear(2 * feature_dim, 1024),
                    nn.ReLU(),
                    nn.Dropout(dropout),  
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Dropout(dropout),  
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(dropout),  
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(dropout),  
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout), 
                    nn.Linear(64, self.num_category)  
                )
                if self.positive_negative:
                    self.positive_negative_mlp = nn.Sequential(
                        nn.Linear(2 * feature_dim, 1024),
                        nn.ReLU(),
                        nn.Dropout(dropout),   
                        nn.Linear(1024, 512),
                        nn.ReLU(),
                        nn.Dropout(dropout),   
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Dropout(dropout), 
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Dropout(dropout),   
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Dropout(dropout), 
                        nn.Linear(64, 2)  
                    )
            else:
                print("use LN")
                if self.positive_negative:
                    self.positive_negative_mlp = nn.Sequential(
                        nn.Linear(2 * feature_dim, 1024),
                        nn.LayerNorm(1024),  
                        nn.ReLU(),
                        nn.Dropout(dropout), 
                        nn.Linear(1024, 512),
                        nn.LayerNorm(512),  
                        nn.ReLU(),
                        nn.Dropout(dropout), 
                        nn.Linear(512, 256),
                        nn.LayerNorm(256),  
                        nn.ReLU(),
                        nn.Dropout(dropout),   
                        nn.Linear(256, 128),
                        nn.LayerNorm(128),  
                        nn.ReLU(),
                        nn.Dropout(dropout),  
                        nn.Linear(128, 64),
                        nn.LayerNorm(64),  
                        nn.ReLU(),
                        nn.Dropout(dropout), 
                        nn.Linear(64, 2)  
                    )
                self.mlp = nn.Sequential(
                    nn.Linear(2 * feature_dim, 1024),
                    nn.LayerNorm(1024),  
                    nn.ReLU(),
                    nn.Dropout(dropout), 
                    nn.Linear(1024, 512),
                    nn.LayerNorm(512),  
                    nn.ReLU(),
                    nn.Dropout(dropout),  
                    nn.Linear(512, 256),
                    nn.LayerNorm(256),  
                    nn.ReLU(),
                    nn.Dropout(dropout), 
                    nn.Linear(256, 128),
                    nn.LayerNorm(128),  
                    nn.ReLU(),
                    nn.Dropout(dropout), 
                    nn.Linear(128, 64),
                    nn.LayerNorm(64),  
                    nn.ReLU(),
                    nn.Dropout(dropout),  
                    nn.Linear(64, self.num_category)  
                )
        print("self.roi_size: ", self.roi_size)
        print("self.use_self_attention: ", self.use_self_attention)
        print("self.use_CLS: ", self.use_CLS)
        print("self.attention: ", self.attention)
        print("position_encoding_type: ", position_encoding_type)
        print("denoised: ", denoised)
        print("num_heads: ", num_heads)
        print("num_layers: ", num_layers)
        print("dropout: ", dropout)
        print("position_bbox: ", position_bbox)
        print("position_relative_bbox: ", position_relative_bbox)
        print("position_bbox_dim: ", position_bbox_dim)
        self.to(self.device)

    def compute_box_pe(self, boxes, embeds, image_size):
        bx_norm = boxes / image_size[[1, 0, 1, 0]]
        bx_c = (bx_norm[:, :2] + bx_norm[:, 2:]) / 2
        b_wh = bx_norm[:, 2:] - bx_norm[:, :2]

        c_pe = compute_sinusoidal_pe(bx_c[:, None], 20).squeeze(1)
        wh_pe = compute_sinusoidal_pe(b_wh[:, None], 20).squeeze(1)

        box_pe = torch.cat([c_pe, wh_pe], dim=-1)

        # Modulate the positional embeddings with box widths and heights by
        # applying different temperatures to x and y
        ref_hw_cond = self.ref_anchor_head(embeds).sigmoid()    # n_query, 2
        # Note that the positional embeddings are stacked as [pe(y), pe(x)]
        c_pe[..., :128] *= (ref_hw_cond[:, 1] / b_wh[:, 1]).unsqueeze(-1)
        c_pe[..., 128:] *= (ref_hw_cond[:, 0] / b_wh[:, 0]).unsqueeze(-1)

        return box_pe, c_pe
    
    def separate_pooled_features(self, pooled_features, detection_counts):
        separated_features = []
        # separated_additional_info = []
        current_idx = 0

        for count in detection_counts:
            if count > 0:
                separated_features.append(pooled_features[current_idx:current_idx + count])
                # separated_additional_info.append(additional_info[current_idx:current_idx + count])
            else:
                if self.position_bbox:
                    separated_features.append(torch.empty(0, pooled_features.size(1)))
                else:
                    separated_features.append(torch.empty(0, pooled_features.size(1), pooled_features.size(2), pooled_features.size(3)))
                # separated_additional_info.append([])
            current_idx += count

        return separated_features
    
    def generate_attention_mask(self, num_human=49, num_object=49, num_cls=1):
        total_features = num_human + num_object + num_cls
        attn_mask = torch.zeros((total_features, total_features), dtype=torch.bool, device=self.device)

        # Mask掉human内部、object内部以及cls token之间的注意力
        attn_mask[:num_human, :num_human] = True
        attn_mask[num_human:num_human+num_object, num_human:num_human+num_object] = True
        if num_cls > 0:
            attn_mask[-num_cls:, -num_cls:] = True

        return attn_mask
    
    def forward(self, nested_tensor: NestedTensor, rois_tensor, additional_info, detection_counts, size):
        images = nested_tensor.tensors
        # print("image size", images.size())
        mask = nested_tensor.mask
        batch_size = images.size(0)

        batch_denoised_features, raw_features, CLS_token = self.backbone(images)
        # print("batch_denoised_features: ", batch_denoised_features.shape)
        # print("raw_features: ", raw_features.shape)
        if self.denoised:
            backbone_features = batch_denoised_features
        else:
            backbone_features = raw_features
        backbone_features = backbone_features.permute(0, 3, 1, 2)
        
        # 将 mask 调整到与 backbone_features 一样的尺度
        # mask 原始大小是 [B, H, W]，需要先增加一个维度变为 [B, 1, H, W] 以便于插值
        mask = mask.unsqueeze(1).float()  # 增加一个维度并转换为 float 类型

        # 进行最近邻插值，将 mask 调整到与 backbone_features 一样的尺度
        mask = F.interpolate(mask, size=backbone_features.shape[-2:], mode='nearest')

        # 调整后的 mask 变回 [B, H', W'] 形式
        mask = mask.squeeze(1).bool()
        
        nested_backbone_feature = NestedTensor(backbone_features, mask)

        if self.position_encoding is not None:
            pos_encoding = self.position_encoding(nested_backbone_feature)
            backbone_features = backbone_features + pos_encoding
        
        # rois, additional_info, detection_counts= self.prepare_rois_cpu(detections_batch)
        if self.attention is not None:
            output_size = (self.roi_size, self.roi_size)
        else:
            output_size = (1, 1)
        rois_tensor = [tensor.to(self.device) for tensor in rois_tensor]
        pooled_features = roi_align(backbone_features, rois_tensor, output_size)
        # print("pooled_features shape: ", pooled_features.shape)
        if self.position_bbox:
            pooled_features = self.attention_pool(pooled_features)
        # print("pooled_features out shape: ", pooled_features.shape)
        separated_features = self.separate_pooled_features(pooled_features, detection_counts)
        # 从 additional_info 中提取并区分 human 和 object 的 features
        all_pairs = []
        query_list = []
        key_list = []
        pair_start_indices = []
        hoi_results = []
        current_index = 0
        bbox_human = []
        bbox_object = []
        for i, (features, info) in enumerate(zip(separated_features, additional_info)):
            human_features = []
            object_features = []

            for feat, det_info in zip(features, info):
                # print("feat size: ", feat.shape)
                flattened_feat = feat.view(self.feature_dim, -1)  # [768, 49]
                # if self.use_CLS:
                #     flattened_feat = torch.cat([flattened_feat, CLS_token[i].unsqueeze(1)], dim=1)  # [768, 50]
                if det_info['label'] == self.person_category_id:
                    human_features.append((flattened_feat, det_info['score'], det_info['label'], det_info['bbox']))
                else:
                    object_features.append((flattened_feat, det_info['score'], det_info['label'], det_info['bbox']))

            # 排序并限制 human 和 object 的数量
            human_features.sort(key=lambda x: -x[1])
            object_features.sort(key=lambda x: -x[1])
            if self.num_objects is not None:
                human_features = human_features[:self.num_objects]
                object_features = object_features[:self.num_objects]
            
            pair_start_indices.append(current_index)
            
            H, W = size[i]
            for human in human_features:
                for obj in object_features:
                    human_bbox = human[3]  # human 的边界框 [x_min, y_min, x_max, y_max]
                    obj_bbox = obj[3]  # object 的边界框 [x_min, y_min, x_max, y_max]

                    # 转换成 0-1 的 xyxy 格式
                    human_bbox_normalized = [human_bbox[i] / (W if i % 2 == 0 else H) for i in range(4)]
                    obj_bbox_normalized = [obj_bbox[i] / (W if i % 2 == 0 else H) for i in range(4)]

                    bbox_human.append(human_bbox_normalized)
                    bbox_object.append(obj_bbox_normalized)

                    if self.attention is not None:
                        if self.use_CLS:
                            combined_feature = torch.cat([human[0].unsqueeze(1), obj[0].unsqueeze(1), CLS_token[i].unsqueeze(1).unsqueeze(2)], dim=-1)
                            query_list.append(combined_feature)
                        elif self.use_self_attention:
                            combined_feature = torch.cat([human[0].unsqueeze(1), obj[0].unsqueeze(1)], dim=-1)
                            query_list.append(combined_feature)
                        else:
                            query_list.append(human[0].unsqueeze(1))  # [768, 49] -> [768, 1, 49]
                            key_list.append(obj[0].unsqueeze(1))
                    else:
                        combined_feature = torch.cat([human[0].squeeze(1), obj[0].squeeze(1)], dim=0).unsqueeze(0)
                        all_pairs.append(combined_feature)
                    
                    hoi_results.append({
                        'subject_category': human[2],
                        'subject_score': human[1],
                        'subject_bbox': human[3],
                        'object_category': obj[2],
                        'object_score': obj[1],
                        'object_bbox': obj[3],
                        'relation_score': None,
                        'binary_score': None
                    })
                    current_index += 1
        # print("len(all_pairs): ", len(all_pairs))
        bbox_human = [torch.tensor(bbox_human)]
        bbox_object = [torch.tensor(bbox_object)]
        pair_feature_out = 0
        if self.attention is not None:
            if query_list:
                if self.use_CLS or self.use_self_attention:
                    query = torch.cat(query_list, dim=1)  # [768, 1, 99] -> [768, N, 99]
                    query = query.permute(2, 1, 0)  # [768, N, 99] -> [99, N, 768]
                    if self.use_self_attention:
                        attn_mask = self.generate_attention_mask(num_human=self.roi_size**2, num_object=self.roi_size**2, num_cls=0)
                    else:
                        attn_mask = self.generate_attention_mask(num_human=self.roi_size**2, num_object=self.roi_size**2, num_cls=1)
                    query_out = self.attention(query=query, key=query ,value=query, attn_mask=attn_mask)
                    pair_feature_out = torch.cat([query_out[48, :, :], query_out[97, :, :]], dim=-1)
                else:
                    query = torch.cat(query_list, dim=1)  # [768, 1, 49] -> [768, N, 49]
                    key = torch.cat(key_list, dim=1)      # [768, 1, 49] -> [768, N, 49]

                    # 将张量转换为 [49, N, 768] 的形状
                    query = query.permute(2, 1, 0)  # [768, N, 49] -> [49, N, 768]
                    key = key.permute(2, 1, 0)      # [768, N, 49] -> [49, N, 768]

                    # value 可以与 key 相同
                    value = key.clone()
                    
                    query_out, key_out = self.attention(query, key ,value)
                    pair_feature_out = torch.cat([query_out[-1, :, :], key_out[-1, :, :]], dim=-1)
                if self.position_relative_bbox:
                    # Compute spatial features
                    box_pair_spatial = compute_spatial_encodings(
                        bbox_human, bbox_object, [(1, 1)]
                    ).to(self.device)
                    box_pair_spatial = self.spatial_head(box_pair_spatial)
                    # print("pair_feature_out: ", pair_feature_out.shape)
                    # print("box_pair_spatial: ", box_pair_spatial.shape)
                    mlp_input_feature = self.mbf(pair_feature_out, box_pair_spatial)
                    # print("mlp_input_feature: ", mlp_input_feature.shape)
                else:
                    mlp_input_feature = pair_feature_out
                relation_scores = self.mlp(mlp_input_feature)

                for i, score in enumerate(relation_scores):
                    hoi_results[i]['relation_score'] = score
                if torch.isnan(relation_scores).any():
                        print("NaN detected in relation_scores.")
                # positive_pairs = all_pairs_tensor[positive_labels]
                # if positive_pairs.size(0) > 0:
                #     relation_scores = self.mlp(positive_pairs)
                #     if torch.isnan(relation_scores).any():
                #         raise ValueError("NaN detected in relation_scores.")
                #     # 保存关系分数到正样本 HOI 结果中
                #     positive_indices = positive_labels.nonzero(as_tuple=True)[0]
                #     for idx, score in zip(positive_indices, relation_scores):
                #         hoi_results[idx]['relation_score'] = score
            else:
                self.no_pairs_count += 1  # 统计 "No pairs found for this batch."
                print("No pairs found for this batch.")
                return [[] for _ in range(batch_size)]  # Return a list of empty lists, one per image in batch
        else:
            if all_pairs:
                all_pairs_tensor = torch.cat(all_pairs, dim=0)
                if self.positive_negative:
                    positive_logits = self.positive_negative_mlp(all_pairs_tensor)
                    for i, logit in enumerate(positive_logits):
                        hoi_results[i]['binary_score'] = logit
                relation_scores = self.mlp(all_pairs_tensor)
                for i, score in enumerate(relation_scores):
                    hoi_results[i]['relation_score'] = score
            else:
                self.no_pairs_count += 1  # 统计 "No pairs found for this batch."
                print("No pairs found for this batch.")
                return [[] for _ in range(batch_size)]  # Return a list of empty lists, one per image in batch


        for i, score in enumerate(relation_scores):
            hoi_results[i]['relation_score'] = score
        # Organize results per image
        image_hoi_results  = []
        # assert len(pair_start_indices) == batch_size, "len(pair_start_indices) != batch_size"
        for i in range(len(pair_start_indices)):
            start_idx = pair_start_indices[i]
            end_idx = pair_start_indices[i+1] if i+1 < len(pair_start_indices) else len(relation_scores)
            if start_idx == end_idx:
                # If there are no results for this image, append an empty list or a placeholder
                image_hoi_results.append([])
                # print("No results found for this image.")
                # print(f"GT for image {i}: {targets[i]}")
                self.no_results_count += 1  # 统计 "No results found for this image."
            else:
                image_hoi_results.append(hoi_results[start_idx:end_idx])
        return image_hoi_results 

class MultiBranchFusion(nn.Module):
    """
    Multi-branch fusion module

    Parameters:
    -----------
    appearance_size: int
        Size of the appearance features
    spatial_size: int
        Size of the spatial features
    hidden_state_size: int
        Size of the intermediate representations
    cardinality: int
        The number of homogeneous branches
    """
    def __init__(self,
        appearance_size: int, spatial_size: int,
        hidden_state_size: int, cardinality: int
    ) -> None:
        super().__init__()
        self.cardinality = cardinality

        sub_repr_size = int(hidden_state_size / cardinality)
        assert sub_repr_size * cardinality == hidden_state_size, \
            "The given representation size should be divisible by cardinality"

        self.fc_1 = nn.ModuleList([
            nn.Linear(appearance_size, sub_repr_size)
            for _ in range(cardinality)
        ])
        self.fc_2 = nn.ModuleList([
            nn.Linear(spatial_size, sub_repr_size)
            for _ in range(cardinality)
        ])
        self.fc_3 = nn.ModuleList([
            nn.Linear(sub_repr_size, hidden_state_size)
            for _ in range(cardinality)
        ])
    def forward(self, appearance: Tensor, spatial: Tensor) -> Tensor:
        return F.relu(torch.stack([
            fc_3(F.relu(fc_1(appearance) * fc_2(spatial)))
            for fc_1, fc_2, fc_3
            in zip(self.fc_1, self.fc_2, self.fc_3)
        ]).sum(dim=0))


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)
    
class MultiheadAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiheadAttentionBlock, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        attn_output, _ = self.multihead_attn(query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        query = query + self.dropout(attn_output)  # Add & Norm
        query = self.norm1(query)
        return query

class BidirectionalCrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(BidirectionalCrossAttentionBlock, self).__init__()
        self.multihead_attn_qkv = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.multihead_attn_kvq = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        # Cross Attention: query -> key, value
        attn_output_qkv, _ = self.multihead_attn_qkv(query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        query = query + self.dropout(attn_output_qkv)  # Add & Norm
        query = self.norm1(query)

        # Cross Attention: key, value -> query
        attn_output_kvq, _ = self.multihead_attn_kvq(key, query, query, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        key = key + self.dropout(attn_output_kvq)  # Add & Norm
        key = self.norm2(key)

        return query, key
    
class MultiheadAttentionStack(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout=0.1):
        super(MultiheadAttentionStack, self).__init__()
        self.layers = nn.ModuleList([
            MultiheadAttentionBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        # 将输入依次通过每个 MultiheadAttentionBlock
        for layer in self.layers:
            query = layer(query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return query

class BidirectionalCrossAttentionStack(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout=0.1):
        super(BidirectionalCrossAttentionStack, self).__init__()
        self.layers = nn.ModuleList([
            BidirectionalCrossAttentionBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        # 依次通过每个 BidirectionalCrossAttentionBlock
        for layer in self.layers:
            query, key = layer(query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return query, key
   
class PostProcessHOI(nn.Module):
    def __init__(self, relation_threshold=0.0, positive_negative=False, device='cuda'):
        super().__init__()
        self.relation_threshold = relation_threshold
        print("self.relation_threshold: ", self.relation_threshold)
        self.positive_negative = positive_negative
        print("self.positive_negative: ", self.positive_negative)
        self.device = device

    def forward(self, batch_hoi_results, size_batch, orig_size_batch):
        # 存储最终的处理结果
        processed_results = []

        # 对批量数据进行处理
        for b, image_results in enumerate(batch_hoi_results):
            # 初始化列表和字典来存储处理后的结果
            subject_boxes = []
            object_boxes = []
            verb_scores = []
            sub_ids = []
            subject_labels = []
            object_labels = []
            obj_ids = []
            idx = 0
            orig_size = orig_size_batch[b].to(self.device)
            current_size = size_batch[b].to(self.device)
            scale_w = orig_size[1].float() / current_size[1].float()  # 宽度比例
            scale_h = orig_size[0].float() / current_size[0].float()  # 高度比例
            # 对每个图像的HOI结果进行处理
            for hoi in image_results:
                # print("hoi['binary_score']: ", hoi['binary_score'])
                if self.positive_negative and hoi['binary_score'][1] > hoi['binary_score'][0]: 
                    # print("negative!!!!!!!!")
                    # print("hoi['binary_score']: ", hoi['binary_score'])
                    relation_probs = torch.zeros_like(hoi['relation_score']).to(self.device)
                else:
                    # print("positive!!!!!!!!")
                    # print("hoi['binary_score']: ", hoi['binary_score'])
                    relation_probs = torch.sigmoid(hoi['relation_score'].to(self.device))

                if torch.any(relation_probs > self.relation_threshold):
                    # 更新列表和字典
                    
                    subject_labels.append(hoi['subject_category']-1)
                    object_labels.append(hoi['object_category']-1)
                    
                    subject_box = torch.tensor(hoi['subject_bbox'], device=self.device) * torch.tensor([scale_w, scale_h, scale_w, scale_h], device=self.device)
                    object_box = torch.tensor(hoi['object_bbox'], device=self.device) * torch.tensor([scale_w, scale_h, scale_w, scale_h], device=self.device)
                    subject_box = subject_box.unsqueeze(0)
                    object_box = object_box.unsqueeze(0)
                    subject_boxes.append(subject_box)
                    object_boxes.append(object_box)
                    
                    verb_scores.append(relation_probs)
                    sub_ids.append(idx)
                    obj_ids.append(idx)
                    idx += 1  # 更新下一个主体和客体的索引

            # 检查是否有有效的HOI对
            if sub_ids:
                verb_scores = torch.stack(verb_scores)
                labels = torch.tensor(subject_labels + object_labels, dtype=torch.int64, device=self.device)
                boxes = torch.cat(subject_boxes + object_boxes)
                image_dict = {
                    'labels': labels,
                    'boxes': boxes,
                    'verb_scores': verb_scores,
                    'sub_ids': torch.tensor(sub_ids, dtype=torch.int64, device=self.device),
                    'obj_ids': torch.tensor([n + len(sub_ids) for n in obj_ids], dtype=torch.int64, device=self.device)
                }
                # print("-----------------")
                # print("image_dict['labels']: ", image_dict['labels'].shape)
                # print("image_dict['boxes']: ", image_dict['boxes'].shape)
                # print("image_dict['verb_scores']: ", image_dict['verb_scores'].shape)
                # print("image_dict['sub_ids']: ", image_dict['sub_ids'].shape)
                # print("image_dict['obj_ids']: ", image_dict['obj_ids'].shape)
                # print("-----------------")
            else:
                image_dict = {}
            
            processed_results.append(image_dict)
        # print("-----------------")
        # print("processed_results[0]: ", processed_results[0])
        # print("processed_results[0]['labels']: ", processed_results[0]['labels'].shape)
        # print("processed_results[0]['boxes']: ", processed_results[0]['boxes'].shape)
        # print("processed_results[0]['verb_scores']: ", processed_results[0]['verb_scores'].shape)
        # print("processed_results[0]['sub_ids']: ", processed_results[0]['sub_ids'].shape)
        # print("processed_results[0]['obj_ids']: ", processed_results[0]['obj_ids'].shape)
        # print("-----------------")
        return processed_results

class CriterionHOI(nn.Module):
    def __init__(self, matcher, device, alpha=0.25, gamma=2.0, loss_type='focal', add_negative_category=False, positive_negative=False, negative_sample_num = 10):
        super().__init__()
        self.matcher = matcher
        self.device = device
        self.alpha = alpha
        self.gamma = gamma
        self.loss_type = loss_type
        self.negative_sample_num = negative_sample_num
        print("add_negative_category: ", add_negative_category)
        print("negative_sample_num: ", negative_sample_num)
        print("loss type: ", self.loss_type)
        print("self.alpha: ", self.alpha)
        print("self.gamma: ", self.gamma)
        # 初始化不匹配计数器
        self.subject_label_mismatch = 0
        self.object_label_mismatch = 0
        self.subject_box_mismatch = 0
        self.object_box_mismatch = 0
        self.total_pairs = 0
        self.add_negative_category = add_negative_category
        self.num_category = 117
        self.positive_negative = positive_negative
    def forward(self, outputs, targets):
        """
        计算模型输出与目标之间的Focal Loss。
        Args:
            outputs (list): 模型输出的列表，每个元素包含该图像的所有人物-物体对的预测。
            targets (list): 目标列表，每个元素包含该图像的真实标签信息。
        
        Returns:
            torch.Tensor: 该批次的平均Focal Loss。
        """
        # 使用匹配器找到最优匹配
        matched_indices = self.matcher(outputs, targets)
        # print("matched_indices: ", matched_indices)
        relation_loss = torch.tensor(0.0, device=self.device)
        binary_loss= torch.tensor(0.0, device=self.device)
        num_processed = 0  # 记录参与损失计算的图像数
        # print("len(outputs)", len(outputs))
        # print("len(targets)", len(targets))
        
        matched_pred_verb_scores = []
        matched_pred_binary_scores = []
        matched_tgt_verb_labels = []
        matched_binary_labels = []
        for idx, (pred, tgt) in enumerate(zip(outputs, targets)):
            if len(pred) == 0:
                # 如果当前图像没有预测结果，跳过此图像的损失计算
                continue
            # print("pred: ", pred)
            # print("tgt: ", tgt)
            sub_inds, obj_inds = matched_indices[idx]
            if len(sub_inds) == 0 or len(obj_inds) == 0:
                # 如果没有有效匹配，也跳过此图像
                continue

            for sub_idx, obj_idx in zip(sub_inds, obj_inds):
                # 收集所有匹配对的预测logits和目标labels
                # print(f"sub_idx: {sub_idx}, obj_idx: {obj_idx}")
                matched_pred_verb_scores.append(pred[sub_idx]['relation_score'])
                if self.positive_negative:
                    matched_pred_binary_scores.append(pred[sub_idx]['binary_score'])
                # 使用 [1, 0] 作为正类标签
                matched_binary_labels.append(torch.tensor([1, 0], device=self.device).unsqueeze(0))
                
                matched_tgt_verb_labels.append(tgt['verb_labels'][obj_idx].unsqueeze(0))
                # pred_subject = pred[sub_idx]

                # tgt_subject = tgt['sub_labels'][obj_idx]
                # tgt_object = tgt['obj_labels'][obj_idx]
                # H, W = tgt['size']
                # tgt_sub_boxes = box_cxcywh_to_xyxy(tgt['sub_boxes'][obj_idx]) * torch.tensor([W, H, W, H], device=self.device)
                # tgt_obj_boxes = box_cxcywh_to_xyxy(tgt['obj_boxes'][obj_idx]) * torch.tensor([W, H, W, H], device=self.device)
                # self.total_pairs += 1
                # try:
                #     assert pred_subject['subject_category'] - 1 == tgt_subject.item(), f"Subject labels do not match: pred {pred_subject['subject_category']-1}, tgt {tgt_subject.item()}"
                # except AssertionError as e:
                #     print(f"AssertionError: {e}")
                #     print(f"sub_idx: {sub_idx}, obj_idx: {obj_idx}")
                #     print(tgt['filename'])
                #     self.subject_label_mismatch += 1
                # try:
                #     assert pred_subject['object_category'] - 1 == tgt_object.item(), f"Object labels do not match: pred {pred_subject['object_category']}, tgt {tgt_object.item()}"
                # except AssertionError as e:
                #     print(f"AssertionError: {e}")
                #     print(f"sub_idx: {sub_idx}, obj_idx: {obj_idx}")
                #     print(tgt['filename'])
                #     self.object_label_mismatch += 1
                # try:
                #     assert torch.allclose(torch.tensor(pred_subject['subject_bbox'], device=self.device), tgt_sub_boxes), f"Subject boxes do not match: pred {pred_subject['subject_bbox']}, tgt {tgt_sub_boxes}"
                # except AssertionError as e:
                #     print(f"AssertionError: {e}")
                #     print(f"sub_idx: {sub_idx}, obj_idx: {obj_idx}")
                #     print(tgt['filename'])
                #     self.subject_box_mismatch += 1
                # try:
                #     assert torch.allclose(torch.tensor(pred_subject['object_bbox'], device=self.device), tgt_obj_boxes), f"Object boxes do not match: pred {pred_subject['object_bbox']}, tgt {tgt_obj_boxes}"
                # except AssertionError as e:
                #     print(f"AssertionError: {e}")
                #     print(f"sub_idx: {sub_idx}, obj_idx: {obj_idx}")
                #     print(tgt['filename'])
                #     self.object_box_mismatch += 1
            if self.add_negative_category:
                unmatched_pred_verb_scores = []
                unmatched_pred_scores = []  # 用于保存 pred[i]['subject_score'] 和 pred[i]['object_score'] 的乘积
                for i in range(len(pred)):
                    if i not in sub_inds:
                        unmatched_pred_verb_scores.append(pred[i]['relation_score'])
                        unmatched_pred_scores.append(pred[i]['subject_score'] * pred[i]['object_score'])

                if unmatched_pred_verb_scores:
                    if len(unmatched_pred_verb_scores) > self.negative_sample_num:
                        top_indices = sorted(range(len(unmatched_pred_scores)), key=lambda i: unmatched_pred_scores[i], reverse=True)[:10]
                        # 筛选出对应的 unmatched_pred_verb_scores
                        unmatched_pred_verb_scores = [unmatched_pred_verb_scores[i] for i in top_indices]
                    matched_pred_verb_scores += unmatched_pred_verb_scores
                    negative_labels = torch.zeros((len(unmatched_pred_verb_scores), self.num_category), device=self.device)
                    negative_labels[:, 57] = 1  # 将标签设置为第57类, no_interaction
                    matched_tgt_verb_labels += [negative_labels]

            if self.positive_negative:
                unmatched_pred_binary_scores = []
                for i in range(len(pred)):
                    if i not in sub_inds:
                        unmatched_pred_binary_scores.append(pred[i]['binary_score'])
                # print("unmatched_pred_binary_scores len: ", len(unmatched_pred_binary_scores))
                if unmatched_pred_binary_scores:
                    matched_pred_binary_scores += unmatched_pred_binary_scores
                    negative_labels = torch.tensor([[0, 1]] * len(unmatched_pred_binary_scores), device=self.device)
                    matched_binary_labels.append(negative_labels)
        if self.positive_negative and matched_pred_binary_scores:
            matched_pred_binary_scores = torch.stack([tensor for tensor in matched_pred_binary_scores]).requires_grad_(True)
            matched_binary_labels = torch.cat(matched_binary_labels, dim=0) #.requires_grad_(True)
            if self.loss_type == 'focal':
                binary_loss = self.focal_loss(matched_pred_binary_scores, matched_binary_labels.float())
                # binary_loss = F.cross_entropy(matched_pred_binary_scores, matched_binary_labels.argmax(dim=1))
                # print("matched_pred_binary_scores: ", matched_pred_binary_scores)
                # print("matched_binary_labels: ", matched_binary_labels)
                # binary_loss = self.bce_loss(matched_pred_binary_scores, matched_binary_labels.float()) 
                # print("binary_loss: ", binary_loss)
            elif self.loss_type == 'bce':
                binary_loss = self.bce_loss(matched_pred_binary_scores, matched_binary_labels.float())      
              
        if matched_pred_verb_scores:
            matched_pred_verb_scores = torch.stack([tensor for tensor in matched_pred_verb_scores]).requires_grad_(True)
            matched_tgt_verb_labels = torch.cat(matched_tgt_verb_labels, dim=0) #.requires_grad_(True)

            if self.loss_type == 'focal':
                relation_loss = self.focal_loss(matched_pred_verb_scores, matched_tgt_verb_labels)
                # print("relation_loss: ", relation_loss)
            elif self.loss_type == 'bce':
                relation_loss = self.bce_loss(matched_pred_verb_scores, matched_tgt_verb_labels)

        return relation_loss, binary_loss

    def focal_loss(self, inputs, targets):
        """ 计算Focal Loss，用于处理类别不平衡问题 """
        loss = sigmoid_focal_loss(inputs, targets, alpha=self.alpha, gamma=self.gamma, reduction='mean')
        # print("focal_loss: ", loss)
        return loss
        # print("inputs: ", inputs)
        # print("targets: ", targets)
        # bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # print("bce_loss: ", bce_loss)
        # probas = torch.sigmoid(inputs)
        # # print("probas: ", probas)
        # loss = self.alpha * (1 - probas) ** self.gamma * bce_loss
        # return loss.mean()
    
    def bce_loss(self, inputs, targets):
        """ 计算二元交叉熵损失 """
        return F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')

class CriterionPVic(nn.Module):
    def __init__(self, matcher, device, alpha=0.25, gamma=2.0, loss_type='focal', add_negative_category=False, positive_negative=False, negative_sample_num = 10):
        super().__init__()
        self.matcher = matcher
        self.device = device
        self.alpha = alpha
        self.gamma = gamma
        self.loss_type = loss_type
        self.negative_sample_num = negative_sample_num
        print("negative_sample_num: ", negative_sample_num)
        print("loss type: ", self.loss_type)
        print("self.alpha: ", self.alpha)
        print("self.gamma: ", self.gamma)
        # 初始化不匹配计数器
        self.subject_label_mismatch = 0
        self.object_label_mismatch = 0
        self.subject_box_mismatch = 0
        self.object_box_mismatch = 0
        self.total_pairs = 0
        self.add_negative_category = add_negative_category
        self.num_category = 117
        self.positive_negative = positive_negative
    def forward(self, outputs, targets):
        """
        计算模型输出与目标之间的Focal Loss。
        Args:
            outputs (list): 模型输出的列表，每个元素包含该图像的所有人物-物体对的预测。
            targets (list): 目标列表，每个元素包含该图像的真实标签信息。
        
        Returns:
            torch.Tensor: 该批次的平均Focal Loss。
        """
        # 使用匹配器找到最优匹配
        matched_indices = self.matcher(outputs, targets)
        # print("matched_indices: ", matched_indices)
        relation_loss = torch.tensor(0.0, device=self.device)
        binary_loss= torch.tensor(0.0, device=self.device)
        num_processed = 0  # 记录参与损失计算的图像数
        # print("len(outputs)", len(outputs))
        # print("len(targets)", len(targets))
        
        matched_pred_verb_scores = []
        matched_pred_binary_scores = []
        matched_tgt_verb_labels = []
        matched_binary_labels = []
        for idx, (pred, tgt) in enumerate(zip(outputs, targets)):
            if len(pred) == 0:
                # 如果当前图像没有预测结果，跳过此图像的损失计算
                continue
            # print("pred: ", pred)
            # print("tgt: ", tgt)
            sub_inds, obj_inds = matched_indices[idx]
            if len(sub_inds) == 0 or len(obj_inds) == 0:
                # 如果没有有效匹配，也跳过此图像
                continue

            for sub_idx, obj_idx in zip(sub_inds, obj_inds):
                # 收集所有匹配对的预测logits和目标labels
                # print(f"sub_idx: {sub_idx}, obj_idx: {obj_idx}")
                matched_pred_verb_scores.append(pred[sub_idx]['relation_score'])
                if self.positive_negative:
                    matched_pred_binary_scores.append(pred[sub_idx]['binary_score'])
                # 使用 [1, 0] 作为正类标签
                matched_binary_labels.append(torch.tensor([1, 0], device=self.device).unsqueeze(0))
                
                matched_tgt_verb_labels.append(tgt['verb_labels'][obj_idx].unsqueeze(0))
                # pred_subject = pred[sub_idx]

                # tgt_subject = tgt['sub_labels'][obj_idx]
                # tgt_object = tgt['obj_labels'][obj_idx]
                # H, W = tgt['size']
                # tgt_sub_boxes = box_cxcywh_to_xyxy(tgt['sub_boxes'][obj_idx]) * torch.tensor([W, H, W, H], device=self.device)
                # tgt_obj_boxes = box_cxcywh_to_xyxy(tgt['obj_boxes'][obj_idx]) * torch.tensor([W, H, W, H], device=self.device)
                # self.total_pairs += 1
                # try:
                #     assert pred_subject['subject_category'] - 1 == tgt_subject.item(), f"Subject labels do not match: pred {pred_subject['subject_category']-1}, tgt {tgt_subject.item()}"
                # except AssertionError as e:
                #     print(f"AssertionError: {e}")
                #     print(f"sub_idx: {sub_idx}, obj_idx: {obj_idx}")
                #     print(tgt['filename'])
                #     self.subject_label_mismatch += 1
                # try:
                #     assert pred_subject['object_category'] - 1 == tgt_object.item(), f"Object labels do not match: pred {pred_subject['object_category']}, tgt {tgt_object.item()}"
                # except AssertionError as e:
                #     print(f"AssertionError: {e}")
                #     print(f"sub_idx: {sub_idx}, obj_idx: {obj_idx}")
                #     print(tgt['filename'])
                #     self.object_label_mismatch += 1
                # try:
                #     assert torch.allclose(torch.tensor(pred_subject['subject_bbox'], device=self.device), tgt_sub_boxes), f"Subject boxes do not match: pred {pred_subject['subject_bbox']}, tgt {tgt_sub_boxes}"
                # except AssertionError as e:
                #     print(f"AssertionError: {e}")
                #     print(f"sub_idx: {sub_idx}, obj_idx: {obj_idx}")
                #     print(tgt['filename'])
                #     self.subject_box_mismatch += 1
                # try:
                #     assert torch.allclose(torch.tensor(pred_subject['object_bbox'], device=self.device), tgt_obj_boxes), f"Object boxes do not match: pred {pred_subject['object_bbox']}, tgt {tgt_obj_boxes}"
                # except AssertionError as e:
                #     print(f"AssertionError: {e}")
                #     print(f"sub_idx: {sub_idx}, obj_idx: {obj_idx}")
                #     print(tgt['filename'])
                #     self.object_box_mismatch += 1
            if self.add_negative_category:
                unmatched_pred_verb_scores = []
                unmatched_pred_scores = []  # 用于保存 pred[i]['subject_score'] 和 pred[i]['object_score'] 的乘积
                for i in range(len(pred)):
                    if i not in sub_inds:
                        unmatched_pred_verb_scores.append(pred[i]['relation_score'])
                        unmatched_pred_scores.append(pred[i]['subject_score'] * pred[i]['object_score'])

                if unmatched_pred_verb_scores:
                    if len(unmatched_pred_verb_scores) > self.negative_sample_num:
                        top_indices = sorted(range(len(unmatched_pred_scores)), key=lambda i: unmatched_pred_scores[i], reverse=True)[:10]
                        # 筛选出对应的 unmatched_pred_verb_scores
                        unmatched_pred_verb_scores = [unmatched_pred_verb_scores[i] for i in top_indices]
                    matched_pred_verb_scores += unmatched_pred_verb_scores
                    negative_labels = torch.zeros((len(unmatched_pred_verb_scores), self.num_category), device=self.device)
                    negative_labels[:, 57] = 1  # 将标签设置为第57类, no_interaction
                    matched_tgt_verb_labels += [negative_labels]

            if self.positive_negative:
                unmatched_pred_binary_scores = []
                for i in range(len(pred)):
                    if i not in sub_inds:
                        unmatched_pred_binary_scores.append(pred[i]['binary_score'])
                # print("unmatched_pred_binary_scores len: ", len(unmatched_pred_binary_scores))
                if unmatched_pred_binary_scores:
                    matched_pred_binary_scores += unmatched_pred_binary_scores
                    negative_labels = torch.tensor([[0, 1]] * len(unmatched_pred_binary_scores), device=self.device)
                    matched_binary_labels.append(negative_labels)
        if self.positive_negative and matched_pred_binary_scores:
            matched_pred_binary_scores = torch.stack([tensor for tensor in matched_pred_binary_scores]).requires_grad_(True)
            matched_binary_labels = torch.cat(matched_binary_labels, dim=0) #.requires_grad_(True)
            if self.loss_type == 'focal':
                binary_loss = self.focal_loss(matched_pred_binary_scores, matched_binary_labels.float())
                # binary_loss = F.cross_entropy(matched_pred_binary_scores, matched_binary_labels.argmax(dim=1))
                # print("matched_pred_binary_scores: ", matched_pred_binary_scores)
                # print("matched_binary_labels: ", matched_binary_labels)
                # binary_loss = self.bce_loss(matched_pred_binary_scores, matched_binary_labels.float()) 
                # print("binary_loss: ", binary_loss)
            elif self.loss_type == 'bce':
                binary_loss = self.bce_loss(matched_pred_binary_scores, matched_binary_labels.float())      
              
        if matched_pred_verb_scores:
            matched_pred_verb_scores = torch.stack([tensor for tensor in matched_pred_verb_scores]).requires_grad_(True)
            matched_tgt_verb_labels = torch.cat(matched_tgt_verb_labels, dim=0) #.requires_grad_(True)

            if self.loss_type == 'focal':
                relation_loss = self.focal_loss(matched_pred_verb_scores, matched_tgt_verb_labels)
                # print("relation_loss: ", relation_loss)
            elif self.loss_type == 'bce':
                relation_loss = self.bce_loss(matched_pred_verb_scores, matched_tgt_verb_labels)

        return relation_loss, binary_loss

    def focal_loss(self, inputs, targets):
        """ 计算Focal Loss，用于处理类别不平衡问题 """
        loss = sigmoid_focal_loss(inputs, targets, alpha=self.alpha, gamma=self.gamma, reduction='mean')
        # print("focal_loss: ", loss)
        return loss
        # print("inputs: ", inputs)
        # print("targets: ", targets)
        # bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # print("bce_loss: ", bce_loss)
        # probas = torch.sigmoid(inputs)
        # # print("probas: ", probas)
        # loss = self.alpha * (1 - probas) ** self.gamma * bce_loss
        # return loss.mean()
    
    def bce_loss(self, inputs, targets):
        """ 计算二元交叉熵损失 """
        return F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
