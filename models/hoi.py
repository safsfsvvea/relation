import torch
import torch.nn as nn
import torch.nn.functional as F
from util.misc import NestedTensor

class HOIModel(nn.Module):
    def __init__(self, backbone, device, num_objects=None, feature_dim=768, person_category_id=1, patch_size=14, relation_score_threshold=0.5, combined_score_threshold=0.1):
        super(HOIModel, self).__init__()
        self.backbone = backbone
        self.device = device
        self.num_objects = num_objects
        self.feature_dim = feature_dim
        self.person_category_id = person_category_id
        self.patch_size = patch_size
        self.relation_score_threshold = relation_score_threshold
        self.combined_score_threshold = combined_score_threshold
        self.mlp = nn.Sequential(
            nn.Linear(2 * feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 117)  # Assuming 117 relation classes
        )
        self.to(self.device)
    def forward(self, nested_tensor: NestedTensor , detections_batch):
        images = nested_tensor.tensors
        mask = nested_tensor.mask
        print("images: ", images.shape)
        print("mask: ", mask.shape)
        B, C, H, W = images.shape
        # Process the whole batch through the backbone
        batch_denoised_features, _, scales = self.backbone(images)
        # print("batch_denoised_features: ", batch_denoised_features.shape)
        batch_size = images.size(0)
        all_pairs = []
        pair_start_indices = []
        hoi_results = []
        current_index = 0

        for b in range(batch_size):
            detections = detections_batch[b]
            denoised_features = batch_denoised_features[b]
            # print("denoised_features: ", denoised_features.shape)
            h, w, _ = denoised_features.shape
            objects_features = []
            human_features = []

            # Extract features for detected humans and objects
            for det in detections:
                bbox = det['bbox']
                a, b, c, d = bbox
                bbox_scaled = [int(bbox[i] * scales[i % 2] / self.patch_size) for i in range(4)]
                x1, y1, x2, y2 = bbox_scaled
                # 确保边界框不超出图像边界
                # x1 = max(0, min(x1, w - 1))
                # x2 = max(0, min(x2, w))
                # y1 = max(0, min(y1, h - 1))
                # y2 = max(0, min(y2, h))

                if y2 <= y1 or x2 <= x1:  # 检查是否有效
                    print("Invalid bbox scaled:", x1, y1, x2, y2)
                    continue
                if y2 > h or y1 < 0 or x1 < 0 or x2 > w:  # 检查是否超出范围
                    print("Bbox scaled out of range:", x1, y1, x2, y2)
                    print("feature shape:", h, w)
                    print("Original bbox:", a, b, c, d)
                    print("image shape:", H, W)
                    continue
                obj_feature = denoised_features[y1:y2, x1:x2, :].mean(dim=[0, 1])
                if torch.isnan(obj_feature).any():
                    print("NaN detected in obj_feature")
                    continue
                # print("obj_feature: ", obj_feature.shape)
                if det['category_id'] == self.person_category_id:
                    human_features.append((obj_feature, det['score'], det['category_id']))
                else:
                    objects_features.append((obj_feature, det['score'], det['category_id']))

            # Sort and limit the number of humans and objects if num_objects is set
            human_features.sort(key=lambda x: -x[1])
            objects_features.sort(key=lambda x: -x[1])
            if self.num_objects is not None:
                human_features = human_features[:self.num_objects]
                objects_features = objects_features[:self.num_objects]

            # Record the start index for this image's pairs
            pair_start_indices.append(current_index)

            # Generate all valid human-object pairs
            for human in human_features:
                for obj in objects_features:
                    combined_feature = torch.cat([human[0], obj[0]], dim=0)
                    # print("combined_feature: ", combined_feature.shape)
                    all_pairs.append(combined_feature.unsqueeze(0))
                    hoi_results.append({
                        'subject_category': human[2],
                        'subject_score': human[1],
                        'object_category': obj[2],
                        'object_score': obj[1]
                    })
                    current_index += 1
        # print("all_pairs[0]: ", all_pairs[0].shape)
        # Concatenate all pairs into a single batch for MLP processing
        if all_pairs:
            all_pairs_tensor = torch.cat(all_pairs, dim=0)
            # print("all_pairs_tensor: ", all_pairs_tensor.shape)
            relation_scores = self.mlp(all_pairs_tensor)
            if torch.isnan(relation_scores).any():
                    raise ValueError("NaN detected in relation_scores.")
            # relation_probs = F.softmax(relation_scores, dim=1)
            # top_probs, top_indices = relation_probs.max(dim=1)
            # for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            #     if prob > self.relation_score_threshold:
            #         combined_score = prob.item() * hoi_results[i]['subject_score'] * hoi_results[i]['object_score']
            #         if combined_score > self.combined_score_threshold:
            #             hoi_results[i]['relation_category'] = idx.item()
            #             hoi_results[i]['relation_score'] = prob.item()
            #         else:
            #             hoi_results[i] = None
            #     else:
            #         hoi_results[i] = None
        else:
            # relation_scores = torch.empty(0, 117, device=images.device)
            print("No valid pairs formed.")
            return [[] for _ in range(batch_size)]  # Return a list of empty lists, one per image in batch
        # hoi_results = [result for result in hoi_results if result is not None]
        for i, score in enumerate(relation_scores):
            # print("score: ", score.shape)
            # print("score: ", score)
            hoi_results[i]['relation_score'] = score.cpu().detach().numpy()
            # print("hoi_results[i]['relation_score']: ", hoi_results[i]['relation_score'])
        # Organize results per image
        image_hoi_results  = []
        for i in range(len(pair_start_indices)):
            start_idx = pair_start_indices[i]
            end_idx = pair_start_indices[i+1] if i+1 < len(pair_start_indices) else len(relation_scores)
            if start_idx == end_idx:
                # If there are no results for this image, append an empty list or a placeholder
                image_hoi_results.append([])
            else:
                image_hoi_results.append(hoi_results[start_idx:end_idx])
                
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print("image_hoi_results: ", image_hoi_results)
        return image_hoi_results 

    
