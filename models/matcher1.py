import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou

def compute_loss_hungarian(model_outputs, targets):
    total_loss = 0.0
    criterion = torch.nn.BCEWithLogitsLoss()

    for output, target in zip(model_outputs, targets):
        # Prepare tensors from model outputs
        sub_cats = torch.tensor([o['subject_category'] for o in output])
        obj_cats = torch.tensor([o['object_category'] for o in output])
        sub_scores = torch.tensor([o['subject_score'] for o in output])
        obj_scores = torch.tensor([o['object_score'] for o in output])
        relations = torch.stack([torch.tensor(o['relation_score']) for o in output])
        sub_boxes = torch.stack([torch.tensor(o['subject_bbox']) for o in output])
        obj_boxes = torch.stack([torch.tensor(o['object_bbox']) for o in output])

        # Prepare tensors from targets
        tgt_sub_labels = target['sub_labels']
        tgt_obj_labels = target['obj_labels']
        tgt_verb_labels = target['verb_labels']
        tgt_sub_boxes = target['sub_boxes']
        tgt_obj_boxes = target['obj_boxes']

        # Calculate costs
        iou_subs = box_iou(sub_boxes, tgt_sub_boxes)
        iou_objs = box_iou(obj_boxes, tgt_obj_boxes)
        iou_scores = (iou_subs + iou_objs) / 2  # Mean IOU for subject and object

        class_match_subs = (sub_cats[:, None] == tgt_sub_labels[None, :]).float()
        class_match_objs = (obj_cats[:, None] == tgt_obj_labels[None, :]).float()
        class_scores = (class_match_subs + class_match_objs) / 2  # Mean class match for subject and object

        conf_scores = (sub_scores[:, None] * obj_scores[:, None])  # Product of confidences

        # Create cost matrix (-log to convert to cost, higher is better so we negate)
        costs = -torch.log(iou_scores + 1e-6) - torch.log(class_scores + 1e-6) - torch.log(conf_scores + 1e-6)

        # Hungarian algorithm to find the minimum cost matching
        row_idx, col_idx = linear_sum_assignment(costs.cpu().detach().numpy())

        # Select matched relation scores and target verb labels
        matched_relations = relations[row_idx]
        matched_targets = tgt_verb_labels[col_idx]

        # Compute BCE loss for the matched pairs
        loss = criterion(matched_relations, matched_targets.float())
        total_loss += loss

    # Average loss over all images in the batch
    total_loss /= len(model_outputs)

    return total_loss


import torch
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
from torchvision.ops import box_iou, generalized_box_iou

class HungarianMatcherHOI(nn.Module):
    def __init__(self, cost_class: float = 1, cost_score: float = 1, cost_relation: float = 1, cost_iou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_score = cost_score
        self.cost_relation = cost_relation
        self.cost_iou = cost_iou

    @torch.no_grad()
    def forward(self, model_outputs, targets):
        batch_size = len(model_outputs)
        batch_indices = []

        for b in range(batch_size):
            outputs = model_outputs[b]
            tgt_labels = targets[b]['labels']
            tgt_verb_labels = targets[b]['verb_labels']
            tgt_sub_labels = targets[b]['sub_labels']
            tgt_obj_labels = targets[b]['obj_labels']
            tgt_sub_boxes = targets[b]['sub_boxes']
            tgt_obj_boxes = targets[b]['obj_boxes']

            # Prepare cost matrix
            num_out = len(outputs)
            num_tgt = tgt_labels.size(0)
            cost_matrix = torch.zeros(num_out, num_tgt)

            for i, out in enumerate(outputs):
                for j in range(num_tgt):
                    # Class cost
                    cost_class = (out['subject_category'] != tgt_sub_labels[j]).float() * self.cost_class
                    cost_class += (out['object_category'] != tgt_obj_labels[j]).float() * self.cost_class

                    # Score cost
                    cost_score = 1 - (out['subject_score'] + out['object_score']) / 2
                    cost_score *= self.cost_score

                    # Relation cost
                    relation_scores = torch.tensor(out['relation_score'], dtype=torch.float32)
                    relation_scores = torch.sigmoid(relation_scores)  # Convert logits to probabilities
                    cost_relation = F.binary_cross_entropy(relation_scores, tgt_verb_labels[j], reduction='sum')
                    cost_relation *= self.cost_relation

                    # Box IoU cost
                    sub_iou = 1 - box_iou(torch.tensor([out['subject_box']]), tgt_sub_boxes[j:j+1]).item()
                    obj_iou = 1 - box_iou(torch.tensor([out['object_box']]), tgt_obj_boxes[j:j+1]).item()
                    cost_iou = (sub_iou + obj_iou) / 2 * self.cost_iou

                    # Total cost
                    cost_matrix[i, j] = cost_class + cost_score + cost_relation + cost_iou

            # Perform matching using the Hungarian algorithm
            row_indices, col_indices = linear_sum_assignment(cost_matrix.cpu())
            batch_indices.append((row_indices, col_indices))

        return batch_indices


import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class HungarianMatcherHOI(nn.Module):
    def __init__(self, cost_class=1, cost_conf=1, cost_bbox=1, cost_action=1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_conf = cost_conf
        self.cost_bbox = cost_bbox
        self.cost_action = cost_action

    def forward(self, outputs, targets):
        """
        Perform matching between model outputs and targets using Hungarian algorithm.
        
        Args:
            outputs (list of dicts): Each dict contains:
                'subject_category', 'object_category', 'subject_score', 'object_score', 'relation_score', 'subject_bbox', 'object_bbox'
            targets (list of dicts): Each dict contains:
                'sub_labels', 'obj_labels', 'verb_labels', 'sub_boxes', 'obj_boxes'
        
        Returns:
            list of tuples: Each tuple contains matched indices for model outputs and targets for each image.
        """
        matches = []
        for output, target in zip(outputs, targets):
            if not output:
                matches.append([])
                continue

            cost_matrix = []
            for out in output:
                out_costs = []
                for sub_label, obj_label, verb_label, sub_box, obj_box in zip(
                        target['sub_labels'], target['obj_labels'], target['verb_labels'],
                        target['sub_boxes'], target['obj_boxes']):

                    # Calculate costs
                    class_cost = self.cost_class * ((out['subject_category'] != sub_label) + (out['object_category'] != obj_label))
                    conf_cost = self.cost_conf * (2 - out['subject_score'] - out['object_score'])
                    bbox_cost = self.cost_bbox * (1 - self.bbox_iou(out['subject_bbox'], sub_box) + 1 - self.bbox_iou(out['object_bbox'], obj_box))
                    action_cost = self.cost_action * F.binary_cross_entropy_with_logits(
                        torch.tensor(out['relation_score']), verb_label, reduction='none').sum()

                    total_cost = class_cost + conf_cost + bbox_cost + action_cost
                    out_costs.append(total_cost)

                cost_matrix.append(out_costs)

            cost_matrix = torch.tensor(cost_matrix)
            row_indices, col_indices = linear_sum_assignment(cost_matrix.cpu().numpy())
            matches.append((row_indices, col_indices))

        return matches

    @staticmethod
    def bbox_iou(bbox1, bbox2):
        """Compute the IoU of two bounding boxes."""
        # Implement the IoU calculation
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        inter_area = max(x2 - x1, 0) * max(y2 - y1, 0)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = bbox1_area + bbox2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

class HungarianMatcherHOI(nn.Module):
    def __init__(self, cost_obj_class=1.0, cost_verb_class=1.0, cost_bbox=1.0, cost_giou=1.0):
        super().__init__()
        self.cost_obj_class = cost_obj_class
        self.cost_verb_class = cost_verb_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        batch_size = len(outputs)
        all_indices = []

        for idx in range(batch_size):
            pred = outputs[idx]
            tgt = targets[idx]

            # 将输出和目标的 bbox, logits 转化为 Tensor 并计算概率或使用已有分数
            pred_sub_boxes = torch.tensor([p['subject_bbox'] for p in pred])
            pred_obj_boxes = torch.tensor([p['object_bbox'] for p in pred])
            pred_obj_scores = torch.tensor([p['object_score'] for p in pred])
            pred_verb_scores = torch.stack([torch.tensor(p['relation_score']) for p in pred])

            tgt_sub_boxes = tgt['sub_boxes']
            tgt_obj_boxes = tgt['obj_boxes']
            tgt_obj_labels = tgt['obj_labels']
            tgt_verb_labels = tgt['verb_labels']

            # 计算成本
            cost_class = F.cross_entropy(pred_obj_scores.unsqueeze(0), tgt_obj_labels.unsqueeze(0), reduction='none')
            cost_verb = F.binary_cross_entropy_with_logits(pred_verb_scores, tgt_verb_labels, reduction='none').sum(1)
            cost_bbox = torch.cdist(pred_sub_boxes, tgt_sub_boxes, p=1) + torch.cdist(pred_obj_boxes, tgt_obj_boxes, p=1)
            cost_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(pred_sub_boxes), box_cxcywh_to_xyxy(tgt_sub_boxes))) + \
                        1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(pred_obj_boxes), box_cxcywh_to_xyxy(tgt_obj_boxes)))

            # 整合成本
            C = self.cost_obj_class * cost_class + self.cost_verb_class * cost_verb + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
            C = C.view(-1).cpu()

            # 使用匈牙利算法进行匹配
            sub_ind, obj_ind = linear_sum_assignment(C.numpy())
            all_indices.append((sub_ind, obj_ind))

        return all_indices

# Helper function to convert boxes from cx, cy, w, h to x1, y1, x2, y2
def box_cxcywh_to_xyxy(boxes):
    x_c, y_c, w, h = boxes.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

# Function to calculate generalized IoU
def generalized_box_iou(boxes1, boxes2):
    # Code to calculate generalized IoU between two sets of boxes
    pass

