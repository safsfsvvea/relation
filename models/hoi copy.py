import torch
import torch.nn as nn
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

class HOIModel(nn.Module):
    def __init__(self, backbone, device, num_objects=None, feature_dim=768, person_category_id=1, patch_size=14, use_LN=True, iou_threshold = 0.0, add_negative_category = False, topK = 30):
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
        self.num_category = 117 if not add_negative_category else 118
        self.topK = topK
        if not use_LN:
        #     self.mlp = nn.Sequential(
        #     nn.Linear(2 * feature_dim, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 117)  
        # )
            self.mlp = nn.Sequential(
                nn.Linear(2 * feature_dim, 1024),
                nn.ReLU(),
                # nn.Dropout(p=0.5),  
                nn.Linear(1024, 512),
                nn.ReLU(),
                # nn.Dropout(p=0.5),  
                nn.Linear(512, 256),
                nn.ReLU(),
                # nn.Dropout(p=0.5),  
                nn.Linear(256, 128),
                nn.ReLU(),
                # nn.Dropout(p=0.5),  
                nn.Linear(128, 64),
                nn.ReLU(),
                # nn.Dropout(p=0.5), 
                nn.Linear(64, self.num_category)  
            )
            self.postive_negative = nn.Sequential(
                nn.Linear(2 * feature_dim, 1024),
                nn.ReLU(),
                # nn.Dropout(p=0.5),  
                nn.Linear(1024, 512),
                nn.ReLU(),
                # nn.Dropout(p=0.5),  
                nn.Linear(512, 256),
                nn.ReLU(),
                # nn.Dropout(p=0.5),  
                nn.Linear(256, 128),
                nn.ReLU(),
                # nn.Dropout(p=0.5),  
                nn.Linear(128, 64),
                nn.ReLU(),
                # nn.Dropout(p=0.5), 
                nn.Linear(64, 2)  
            )
        else:
            print("use LN")
            self.postive_negative = nn.Sequential(
                nn.Linear(2 * feature_dim, 1024),
                nn.LayerNorm(1024),  
                nn.ReLU(),
                # nn.Dropout(p=0.5),  
                nn.Linear(1024, 512),
                nn.LayerNorm(512),  
                nn.ReLU(),
                # nn.Dropout(p=0.5),  
                nn.Linear(512, 256),
                nn.LayerNorm(256),  
                nn.ReLU(),
                # nn.Dropout(p=0.5),  
                nn.Linear(256, 128),
                nn.LayerNorm(128),  
                nn.ReLU(),
                # nn.Dropout(p=0.5),  
                nn.Linear(128, 64),
                nn.LayerNorm(64),  
                nn.ReLU(),
                # nn.Dropout(p=0.5),  
                nn.Linear(64, 2)  
            )
            self.mlp = nn.Sequential(
                nn.Linear(2 * feature_dim, 1024),
                nn.LayerNorm(1024),  
                nn.ReLU(),
                # nn.Dropout(p=0.5),  
                nn.Linear(1024, 512),
                nn.LayerNorm(512),  
                nn.ReLU(),
                # nn.Dropout(p=0.5),  
                nn.Linear(512, 256),
                nn.LayerNorm(256),  
                nn.ReLU(),
                # nn.Dropout(p=0.5),  
                nn.Linear(256, 128),
                nn.LayerNorm(128),  
                nn.ReLU(),
                # nn.Dropout(p=0.5),  
                nn.Linear(128, 64),
                nn.LayerNorm(64),  
                nn.ReLU(),
                # nn.Dropout(p=0.5),  
                nn.Linear(64, self.num_category)  
            )

        self.to(self.device)

    def prepare_rois_cpu(self, detections_batch, targets):
        """
        准备 ROI Align 所需的 RoIs 列表。

        :param detections_batch: 每张图片的检测结果列表，每个元素是一个字典，包含 'boxes' 和 'labels'
        :param targets: 每张图片的原始尺寸信息，通常是 {'height': ..., 'width': ...}
        :return: ROIs 张量，形状为 [N, 5]，其中 N 是所有图片中检测框的总数， 额外的附加信息以及每张图片的检测框数量
        """
        all_boxes = []
        all_labels = []
        all_scores = []
        image_indices = []

        scale_factor = self.patch_size / 2  

        
        def process_image(image_index, detections, target):
            H, W = target['size'].cpu()
            if not detections['boxes'].nelement(): 
                return None


            boxes = detections['boxes'].cpu()
            labels = detections['labels'].cpu()
            scores = detections['scores'].cpu()

            cx, cy, bw, bh = boxes.T
            xmin = (cx - bw / 2) * W
            ymin = (cy - bh / 2) * H
            xmax = (cx + bw / 2) * W
            ymax = (cy + bh / 2) * H
            
            scaled_xmin = xmin / scale_factor
            scaled_ymin = ymin / scale_factor
            scaled_xmax = xmax / scale_factor
            scaled_ymax = ymax / scale_factor
            
            rois = torch.stack([torch.full_like(scaled_ymin, image_index), scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax], dim=1)

            return rois, labels, scores, torch.full((boxes.size(0),), image_index, dtype=torch.int64)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_image, image_index, detections, target) 
                    for image_index, (detections, target) in enumerate(zip(detections_batch, targets))]
            results = [f.result() for f in futures]

        for result in results:
            if result is not None:
                rois, labels, scores, img_indices = result
                all_boxes.append(rois)
                all_labels.append(labels)
                all_scores.append(scores)
                image_indices.append(img_indices)

        if not all_boxes:
            return torch.empty((0, 5), device=self.device), [], []
        
        rois_tensor = torch.cat(all_boxes)
        labels_tensor = torch.cat(all_labels)
        scores_tensor = torch.cat(all_scores)
        image_indices_tensor = torch.cat(image_indices)

        detection_counts = [len(detections['boxes']) for detections in detections_batch]

        additional_info = [{'label': label.item(), 'bbox': [roi[1].item() * scale_factor, roi[2].item() * scale_factor, roi[3].item() * scale_factor, roi[4].item() * scale_factor], 'image_index': idx.item(), 'score': score.item()}
                        for label, roi, idx, score in zip(labels_tensor, rois_tensor, image_indices_tensor, scores_tensor)]

        return rois_tensor.float().to(self.device), additional_info, detection_counts

    def prepare_rois(self, detections_batch, targets):
        """
        准备 ROI Align 所需的 RoIs 列表。

        :param detections_batch: 每张图片的检测结果列表，每个元素是一个字典，包含 'boxes' 和 'labels'
        :param targets: 每张图片的原始尺寸信息，通常是 {'height': ..., 'width': ...}
        :return: ROIs 张量，形状为 [N, 5]，其中 N 是所有图片中检测框的总数， 额外的附加信息以及每张图片的检测框数量
        """
        all_boxes = []
        all_labels = []
        all_scores = []
        image_indices = []

        scale_factor = self.patch_size / 2  

        for image_index, (detections, target) in enumerate(zip(detections_batch, targets)):
            H, W = target['size']
            if detections['boxes'].numel() == 0:  
                continue

            boxes = detections['boxes'].to(self.device)
            labels = detections['labels'].to(self.device)
            scores = detections['scores'].to(self.device)

            cx, cy, bw, bh = boxes.T
            xmin = (cx - bw / 2) * W
            ymin = (cy - bh / 2) * H
            xmax = (cx + bw / 2) * W
            ymax = (cy + bh / 2) * H

            scaled_xmin = xmin / scale_factor
            scaled_ymin = ymin / scale_factor
            scaled_xmax = xmax / scale_factor
            scaled_ymax = ymax / scale_factor

            rois = torch.stack([torch.full_like(scaled_ymin, image_index), scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax], dim=1)

            all_boxes.append(rois)
            all_labels.append(labels)
            all_scores.append(scores)
            image_indices.append(torch.full((boxes.size(0),), image_index, dtype=torch.int64, device=self.device))

        if not all_boxes:
            return torch.empty((0, 5), device=self.device), [], []

        rois_tensor = torch.cat(all_boxes)
        labels_tensor = torch.cat(all_labels)
        scores_tensor = torch.cat(all_scores)
        image_indices_tensor = torch.cat(image_indices)

        detection_counts = [len(detections['boxes']) for detections in detections_batch]

        additional_info = [{'label': label.item(), 'bbox': [roi[1].item() * scale_factor, roi[2].item() * scale_factor, roi[3].item() * scale_factor, roi[4].item() * scale_factor], 'image_index': idx.item(), 'score': score.item()}
                        for label, roi, idx, score in zip(labels_tensor, rois_tensor, image_indices_tensor, scores_tensor)]

        return rois_tensor.float(), additional_info, detection_counts
    def separate_pooled_features(self, pooled_features, additional_info, detection_counts):
        separated_features = []
        separated_additional_info = []
        current_idx = 0

        for count in detection_counts:
            if count > 0:
                separated_features.append(pooled_features[current_idx:current_idx + count])
                separated_additional_info.append(additional_info[current_idx:current_idx + count])
            else:
                separated_features.append(torch.empty(0, pooled_features.size(1), pooled_features.size(2), pooled_features.size(3)))
                separated_additional_info.append([])
            current_idx += count

        return separated_features, separated_additional_info
    
    def forward(self, nested_tensor: NestedTensor , targets, detections_batch):
        images = nested_tensor.tensors
        mask = nested_tensor.mask
        batch_size = images.size(0)
        # print("detections_batch: ", detections_batch)
        batch_denoised_features, _, scales = self.backbone(images)

        batch_denoised_features = batch_denoised_features.permute(0, 3, 1, 2)
        batch_denoised_features = F.interpolate(batch_denoised_features, scale_factor=2, mode='bilinear', align_corners=True)

        
        rois, additional_info, detection_counts= self.prepare_rois_cpu(detections_batch, targets)

        output_size = (1, 1)

        pooled_features = roi_align(batch_denoised_features, rois, output_size)

        separated_features, separated_additional_info = self.separate_pooled_features(pooled_features, additional_info, detection_counts)

        # 从 additional_info 中提取并区分 human 和 object 的 features
        all_pairs = []
        pair_start_indices = []
        hoi_results = []
        current_index = 0
        for i, (features, info) in enumerate(zip(separated_features, separated_additional_info)):
            human_features = []
            object_features = []

            for feat, det_info in zip(features, info):
                if det_info['label'] == self.person_category_id:
                    human_features.append((feat.view(-1), det_info['score'], det_info['label'], det_info['bbox']))
                else:
                    object_features.append((feat.view(-1), det_info['score'], det_info['label'], det_info['bbox']))

            # 排序并限制 human 和 object 的数量
            human_features.sort(key=lambda x: -x[1])
            object_features.sort(key=lambda x: -x[1])
            if self.num_objects is not None:
                human_features = human_features[:self.num_objects]
                object_features = object_features[:self.num_objects]
            
            pair_start_indices.append(current_index)
            # 生成所有有效的 human-object pairs
            print("len(human_features): ", len(human_features))
            print("len(object_features): ", len(object_features))
            
            # for human1 in human_features:
            #     for human2 in human_features:
            #         human_feature = human1[0]
            #         obj_feature = human2[0]
            #         cosine_similarity = F.cosine_similarity(human_feature.unsqueeze(0), obj_feature.unsqueeze(0))
            #         print("human human Cosine Similarity:", cosine_similarity.item())
            # for object1 in object_features:
            #     for object2 in object_features:
            #         human_feature = object1[0]
            #         obj_feature = object2[0]
            #         cosine_similarity = F.cosine_similarity(human_feature.unsqueeze(0), obj_feature.unsqueeze(0))
            #         print("object object Cosine Similarity:", cosine_similarity.item())
            
            # if len(human_features) > 0 and len(object_features) > 0:
            #     human_boxes = torch.tensor([human[3] for human in human_features], dtype=torch.float32)
            #     object_boxes = torch.tensor([obj[3] for obj in object_features], dtype=torch.float32)

            #     # 计算 IoU 矩阵
            #     iou_matrix = box_iou(human_boxes, object_boxes)
            #     # print("iou_matrix: ", iou_matrix)
                
            #     positive_iou_values = iou_matrix[iou_matrix >= 0].reshape(-1)
            #     positive_indices = torch.nonzero(iou_matrix >= 0, as_tuple=False)

            #     # 直接使用 topk，如果 positive_iou_values 的元素个数少于 K，torch.topk 会返回所有元素
            #     topk_values, topk_indices_in_pos = torch.topk(positive_iou_values, min(self.topK, len(positive_iou_values)))
            #     topk_indices = positive_indices[topk_indices_in_pos]

            #     # print("topk_indices: ", topk_indices)

            #     # 遍历筛选后的配对
            #     for pair in topk_indices:
            #         human_idx, obj_idx = pair
            #         human = human_features[human_idx]
            #         obj = object_features[obj_idx]
                    
            #         combined_feature = torch.cat([human[0], obj[0]], dim=0)
            #         # human_feature = human[0]
            #         # obj_feature = obj[0]
            #         # cosine_similarity = F.cosine_similarity(human_feature.unsqueeze(0), obj_feature.unsqueeze(0))
            #         # print("human object Cosine Similarity:", cosine_similarity.item())
            #         all_pairs.append(combined_feature.unsqueeze(0))
            #         hoi_results.append({
            #             'subject_category': human[2],
            #             'subject_score': human[1],
            #             'subject_bbox': human[3],
            #             'object_category': obj[2],
            #             'object_score': obj[1],
            #             'object_bbox': obj[3],
            #         })
            #         current_index += 1               
             
            for human in human_features:
                for obj in object_features:
                    combined_feature = torch.cat([human[0], obj[0]], dim=0)
                    # human_feature = human[0]
                    # obj_feature = obj[0]
                    # cosine_similarity = F.cosine_similarity(human_feature.unsqueeze(0), obj_feature.unsqueeze(0))
                    # print("human object Cosine Similarity:", cosine_similarity.item())
                    all_pairs.append(combined_feature.unsqueeze(0))
                    hoi_results.append({
                        'subject_category': human[2],
                        'subject_score': human[1],
                        'subject_bbox': human[3],
                        'object_category': obj[2],
                        'object_score': obj[1],
                        'object_bbox': obj[3],
                    })
                    current_index += 1
        print("len(all_pairs): ", len(all_pairs))
        if all_pairs:
            all_pairs_tensor = torch.cat(all_pairs, dim=0)
            positive_logits = self.positive_negative(all_pairs_tensor)
            positive_probs = F.softmax(positive_logits, dim=-1)  # 转换为概率分布
            positive_labels = (positive_probs[:, 1] > 0.5).squeeze().bool()  # 选择正类概率大于0.5的样本
            for i, logit in enumerate(positive_logits):
                hoi_results[i]['binary_score'] = logit
            positive_pairs = all_pairs_tensor[positive_labels]
            if positive_pairs.size(0) > 0:
                relation_scores = self.mlp(positive_pairs)
                if torch.isnan(relation_scores).any():
                    raise ValueError("NaN detected in relation_scores.")
                # 保存关系分数到正样本 HOI 结果中
                positive_indices = positive_labels.nonzero(as_tuple=True)[0]
                for idx, score in zip(positive_indices, relation_scores):
                    hoi_results[idx]['relation_score'] = score
        else:
            self.no_pairs_count += 1  # 统计 "No pairs found for this batch."
            print("No pairs found for this batch.")
            return [[] for _ in range(batch_size)]  # Return a list of empty lists, one per image in batch

        for i, score in enumerate(relation_scores):
            hoi_results[i]['relation_score'] = score
        # Organize results per image
        image_hoi_results  = []
        assert len(pair_start_indices) == batch_size, "len(pair_start_indices) != batch_size"
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
    
class HOIModel_time(nn.Module):
    def __init__(self, backbone, device, num_objects=10, feature_dim=768, person_category_id=1, patch_size=14):
        super(HOIModel_time, self).__init__()
        self.backbone = backbone
        self.device = device
        self.num_objects = num_objects
        self.feature_dim = feature_dim
        self.person_category_id = person_category_id
        self.patch_size = patch_size
        self.mlp = nn.Sequential(
            nn.Linear(2 * feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 117)  # Assuming 117 relation classes
        )
        self.to(self.device)

    def prepare_rois_cpu(self, detections_batch, targets):
        all_boxes = []
        all_labels = []
        all_scores = []
        image_indices = []

        scale_factor = self.patch_size / 2
        timings = []

        # torch.cuda.synchronize()

        # # 方法二：张量操作
        # start_event = torch.cuda.Event(enable_timing=True)
        # end_event = torch.cuda.Event(enable_timing=True)
        
        # start_event.record()
        # sizes = torch.stack([target['size'] for target in targets])
        # sizes_on_cpu_2 = sizes.cpu()
        # end_event.record()
        
        # torch.cuda.synchronize()
        # method_2_time = start_event.elapsed_time(end_event) / 1000.0
        # timings.append(f"方法二耗时: {method_2_time} 秒")
        
        # torch.cuda.synchronize()
        
        # # 方法一：列表解析
        # start_event.record()
        # sizes_on_cpu_1 = [target['size'].cpu() for target in targets]
        # end_event.record()
        
        # torch.cuda.synchronize()
        # method_1_time = start_event.elapsed_time(end_event) / 1000.0
        # timings.append(f"方法一耗时: {method_1_time} 秒")

        def process_image(image_index, detections, target):
            process_timings = []
            
            # start_event = torch.cuda.Event(enable_timing=True)
            # end_event = torch.cuda.Event(enable_timing=True)
            
            # start_event.record()
            H, W = target['size'].cpu()
            # end_event.record()
            # torch.cuda.synchronize()
            # process_image_time1 = start_event.elapsed_time(end_event) / 1000.0
            # process_timings.append(f"Process image time1: {process_image_time1}")
            
            if not detections['boxes'].nelement():
                return None, process_timings
            
            # start_event.record()
            boxes = detections['boxes'].cpu()
            labels = detections['labels'].cpu()
            scores = detections['scores'].cpu()
            # end_event.record()
            # torch.cuda.synchronize()
            # process_image_time2 = start_event.elapsed_time(end_event) / 1000.0
            # process_timings.append(f"Process image time2: {process_image_time2}")
            
            # start_event.record()
            cx, cy, bw, bh = boxes.T
            xmin = (cx - bw / 2) * W
            ymin = (cy - bh / 2) * H
            xmax = (cx + bw / 2) * W
            ymax = (cy + bh / 2) * H
            # end_event.record()
            # torch.cuda.synchronize()
            # process_image_time3 = start_event.elapsed_time(end_event) / 1000.0
            # process_timings.append(f"Process image time3: {process_image_time3}")
            
            # start_event.record()
            scaled_xmin = xmin / scale_factor
            scaled_ymin = ymin / scale_factor
            scaled_xmax = xmax / scale_factor
            scaled_ymax = ymax / scale_factor
            # end_event.record()
            # torch.cuda.synchronize()
            # process_image_time4 = start_event.elapsed_time(end_event) / 1000.0
            # process_timings.append(f"Process image time4: {process_image_time4}")
            
            # start_event.record()
            rois = torch.stack([torch.full_like(scaled_ymin, image_index), scaled_ymin, scaled_xmin, scaled_ymax, scaled_xmax], dim=1)
            # end_event.record()
            # torch.cuda.synchronize()
            # process_image_time5 = start_event.elapsed_time(end_event) / 1000.0
            # process_timings.append(f"Process image time5: {process_image_time5}")
            
            return (rois, labels, scores, torch.full((boxes.size(0),), image_index, dtype=torch.int64)), process_timings

        # multi_process_start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_image, image_index, detections, target) 
                    for image_index, (detections, target) in enumerate(zip(detections_batch, targets))]
            results = [f.result() for f in futures]
        # multi_process_time = time.time() - multi_process_start_time
        # timings.append(f"Multi process time: {multi_process_time}")

        # inside_for_start_time = time.time()
        for result, process_timings in results:
            if result is not None:
                rois, labels, scores, img_indices = result
                all_boxes.append(rois)
                all_labels.append(labels)
                all_scores.append(scores)
                image_indices.append(img_indices)
                timings.extend(process_timings)
        # inside_for_time = time.time() - inside_for_start_time

        # if_start_time = time.time()
        # torch.cuda.synchronize()
        if not all_boxes:
            return torch.empty((0, 5), device=self.device), [], []
        # torch.cuda.synchronize()
        # if_time = time.time() - if_start_time
        # timings.append(f"If time: {if_time}")

        # cat_start_event = torch.cuda.Event(enable_timing=True)
        # cat_end_event = torch.cuda.Event(enable_timing=True)
        
        # cat_start_event.record()
        rois_tensor = torch.cat(all_boxes)
        labels_tensor = torch.cat(all_labels)
        scores_tensor = torch.cat(all_scores)
        image_indices_tensor = torch.cat(image_indices)
        # cat_end_event.record()
        
        # torch.cuda.synchronize()
        # cat_time = cat_start_event.elapsed_time(cat_end_event) / 1000.0
        # timings.append(f"Cat time: {cat_time}")

        # rois_result_start_time = time.time()
        # torch.cuda.synchronize()
        detection_counts = [len(detections['boxes']) for detections in detections_batch]

        additional_info = [{'label': label.item(),
                            'bbox': [roi[2].item() * scale_factor,
                                    roi[1].item() * scale_factor,
                                    roi[4].item() * scale_factor,
                                    roi[3].item() * scale_factor],
                            'image_index': idx.item(),
                            'score': score.item()}
                        for label, roi, idx, score in zip(labels_tensor, rois_tensor, image_indices_tensor, scores_tensor)]
        # torch.cuda.synchronize()
        # rois_result_time = time.time() - rois_result_start_time
        # timings.append(f"RoIs result time: {rois_result_time}")

        # 打印所有计时信息
        # for timing in timings:
        #     print(timing)
        
        return rois_tensor.float().to(self.device), additional_info, detection_counts

    def separate_pooled_features(self, pooled_features, additional_info, detection_counts):
        separated_features = []
        separated_additional_info = []
        current_idx = 0

        for count in detection_counts:
            if count > 0:
                separated_features.append(pooled_features[current_idx:current_idx + count])
                separated_additional_info.append(additional_info[current_idx:current_idx + count])
            else:
                separated_features.append(torch.empty(0, pooled_features.size(1), pooled_features.size(2), pooled_features.size(3)))
                separated_additional_info.append([])
            current_idx += count

        return separated_features, separated_additional_info

    def forward(self, nested_tensor: NestedTensor, targets, detections_batch):
        images = nested_tensor.tensors
        mask = nested_tensor.mask
        batch_size = images.size(0)
        # print("backbone device: ", self.backbone.device)
        timings = []

        # 计算backbone时间
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        batch_denoised_features, _, scales = self.backbone(images)
        end_event.record()
        torch.cuda.synchronize()
        backbone_time = start_event.elapsed_time(end_event) / 1000.0
        timings.append(f"Backbone time: {backbone_time} 秒")

        # 计算downsampling时间
        downsampling_start_time = time.time()
        batch_denoised_features = batch_denoised_features.permute(0, 3, 1, 2)
        batch_denoised_features = F.interpolate(batch_denoised_features, scale_factor=2, mode='bilinear', align_corners=True)
        downsampling_time = time.time() - downsampling_start_time
        timings.append(f"Downsampling time: {downsampling_time} 秒")

        # 计算ROI时间
        roi_start_time = time.time()
        rois, additional_info, detection_counts = self.prepare_rois_cpu(detections_batch, targets)
        roi_time = time.time() - roi_start_time
        timings.append(f"ROI time: {roi_time} 秒")

        # 计算pool时间
        pool_start_event = torch.cuda.Event(enable_timing=True)
        pool_end_event = torch.cuda.Event(enable_timing=True)

        pool_start_event.record()
        output_size = (1, 1)
        pooled_features = roi_align(batch_denoised_features, rois, output_size)
        pool_end_event.record()
        torch.cuda.synchronize()
        pool_time = pool_start_event.elapsed_time(pool_end_event) / 1000.0
        timings.append(f"Pool time: {pool_time} 秒")

        # 计算separate时间
        separate_start_time = time.time()
        separated_features, separated_additional_info = self.separate_pooled_features(pooled_features, additional_info, detection_counts)
        separate_time = time.time() - separate_start_time
        timings.append(f"Separate time: {separate_time} 秒")

        # 计算for循环时间
        for_start_time = time.time()
        all_pairs = []
        pair_start_indices = []
        hoi_results = []
        current_index = 0
        for i, (features, info) in enumerate(zip(separated_features, separated_additional_info)):
            human_features = []
            object_features = []

            for feat, det_info in zip(features, info):
                if det_info['label'] == self.person_category_id:
                    human_features.append((feat.view(-1), det_info['score'], det_info['label'], det_info['bbox']))
                else:
                    object_features.append((feat.view(-1), det_info['score'], det_info['label'], det_info['bbox']))

            human_features.sort(key=lambda x: -x[1])
            object_features.sort(key=lambda x: -x[1])
            if self.num_objects is not None:
                human_features = human_features[:self.num_objects]
                object_features = object_features[:self.num_objects]

            pair_start_indices.append(current_index)
            for human in human_features:
                for obj in object_features:
                    combined_feature = torch.cat([human[0], obj[0]], dim=0)
                    all_pairs.append(combined_feature.unsqueeze(0))
                    hoi_results.append({
                        'subject_category': human[2],
                        'subject_score': human[1],
                        'subject_bbox': human[3],
                        'object_category': obj[2],
                        'object_score': obj[1],
                        'object_bbox': obj[3],
                    })
                    current_index += 1
        for_time = time.time() - for_start_time
        timings.append(f"for loop time: {for_time} 秒")

        # 计算MLP时间
        if all_pairs:
            mlp_start_event = torch.cuda.Event(enable_timing=True)
            mlp_end_event = torch.cuda.Event(enable_timing=True)

            mlp_start_event.record()
            all_pairs_tensor = torch.cat(all_pairs, dim=0)
            relation_scores = self.mlp(all_pairs_tensor)
            mlp_end_event.record()
            torch.cuda.synchronize()
            mlp_time = mlp_start_event.elapsed_time(mlp_end_event) / 1000.0
            timings.append(f"MLP time: {mlp_time} 秒")

            if torch.isnan(relation_scores).any():
                raise ValueError("NaN detected in relation_scores.")
        else:
            timings.append("No pairs found for this batch.")
            return [[] for _ in range(batch_size)]

        # 计算result时间
        result_start_time = time.time()
        for i, score in enumerate(relation_scores):
            hoi_results[i]['relation_score'] = score

        image_hoi_results = []
        for i in range(len(pair_start_indices)):
            start_idx = pair_start_indices[i]
            end_idx = pair_start_indices[i + 1] if i + 1 < len(pair_start_indices) else len(relation_scores)
            if start_idx == end_idx:
                image_hoi_results.append([])
                timings.append("No results found for this image.")
            else:
                image_hoi_results.append(hoi_results[start_idx:end_idx])
        result_time = time.time() - result_start_time
        timings.append(f"Result time: {result_time} 秒")

        # 打印所有计时信息
        for timing in timings:
            print(timing)

        return image_hoi_results
    
class HOIModel_profiler(nn.Module):
    def __init__(self, backbone, device, num_objects=10, feature_dim=768, person_category_id=1, patch_size=14):
        super(HOIModel_profiler, self).__init__()
        self.backbone = backbone
        self.device = device
        self.num_objects = num_objects
        self.feature_dim = feature_dim
        self.person_category_id = person_category_id
        self.patch_size = patch_size
        self.mlp = nn.Sequential(
            nn.Linear(2 * feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 117)  # Assuming 117 relation classes
        )
        self.to(self.device)

    def prepare_rois_cpu(self, detections_batch, targets):
        all_boxes = []
        all_labels = []
        all_scores = []
        image_indices = []

        scale_factor = self.patch_size / 2
        timings = []

        # torch.cuda.synchronize()

        # # 方法二：张量操作
        # start_event = torch.cuda.Event(enable_timing=True)
        # end_event = torch.cuda.Event(enable_timing=True)
        
        # start_event.record()
        # sizes = torch.stack([target['size'] for target in targets])
        # sizes_on_cpu_2 = sizes.cpu()
        # end_event.record()
        
        # torch.cuda.synchronize()
        # method_2_time = start_event.elapsed_time(end_event) / 1000.0
        # timings.append(f"方法二耗时: {method_2_time} 秒")
        
        # torch.cuda.synchronize()
        
        # # 方法一：列表解析
        # start_event.record()
        # sizes_on_cpu_1 = [target['size'].cpu() for target in targets]
        # end_event.record()
        
        # torch.cuda.synchronize()
        # method_1_time = start_event.elapsed_time(end_event) / 1000.0
        # timings.append(f"方法一耗时: {method_1_time} 秒")

        def process_image(image_index, detections, target):
            process_timings = []
            
            # start_event = torch.cuda.Event(enable_timing=True)
            # end_event = torch.cuda.Event(enable_timing=True)
            
            # start_event.record()
            H, W = target['size'].cpu()
            # end_event.record()
            # torch.cuda.synchronize()
            # process_image_time1 = start_event.elapsed_time(end_event) / 1000.0
            # process_timings.append(f"Process image time1: {process_image_time1}")
            
            if not detections['boxes'].nelement():
                return None, process_timings
            
            # start_event.record()
            boxes = detections['boxes'].cpu()
            labels = detections['labels'].cpu()
            scores = detections['scores'].cpu()
            # end_event.record()
            # torch.cuda.synchronize()
            # process_image_time2 = start_event.elapsed_time(end_event) / 1000.0
            # process_timings.append(f"Process image time2: {process_image_time2}")
            
            # start_event.record()
            cx, cy, bw, bh = boxes.T
            xmin = (cx - bw / 2) * W
            ymin = (cy - bh / 2) * H
            xmax = (cx + bw / 2) * W
            ymax = (cy + bh / 2) * H
            # end_event.record()
            # torch.cuda.synchronize()
            # process_image_time3 = start_event.elapsed_time(end_event) / 1000.0
            # process_timings.append(f"Process image time3: {process_image_time3}")
            
            # start_event.record()
            scaled_xmin = xmin / scale_factor
            scaled_ymin = ymin / scale_factor
            scaled_xmax = xmax / scale_factor
            scaled_ymax = ymax / scale_factor
            # end_event.record()
            # torch.cuda.synchronize()
            # process_image_time4 = start_event.elapsed_time(end_event) / 1000.0
            # process_timings.append(f"Process image time4: {process_image_time4}")
            
            # start_event.record()
            rois = torch.stack([torch.full_like(scaled_ymin, image_index), scaled_ymin, scaled_xmin, scaled_ymax, scaled_xmax], dim=1)
            # end_event.record()
            # torch.cuda.synchronize()
            # process_image_time5 = start_event.elapsed_time(end_event) / 1000.0
            # process_timings.append(f"Process image time5: {process_image_time5}")
            
            return (rois, labels, scores, torch.full((boxes.size(0),), image_index, dtype=torch.int64)), process_timings

        # multi_process_start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_image, image_index, detections, target) 
                    for image_index, (detections, target) in enumerate(zip(detections_batch, targets))]
            results = [f.result() for f in futures]
        # multi_process_time = time.time() - multi_process_start_time
        # timings.append(f"Multi process time: {multi_process_time}")

        # inside_for_start_time = time.time()
        for result, process_timings in results:
            if result is not None:
                rois, labels, scores, img_indices = result
                all_boxes.append(rois)
                all_labels.append(labels)
                all_scores.append(scores)
                image_indices.append(img_indices)
                timings.extend(process_timings)
        # inside_for_time = time.time() - inside_for_start_time

        # if_start_time = time.time()
        # torch.cuda.synchronize()
        if not all_boxes:
            return torch.empty((0, 5), device=self.device), [], []
        # torch.cuda.synchronize()
        # if_time = time.time() - if_start_time
        # timings.append(f"If time: {if_time}")

        # cat_start_event = torch.cuda.Event(enable_timing=True)
        # cat_end_event = torch.cuda.Event(enable_timing=True)
        
        # cat_start_event.record()
        rois_tensor = torch.cat(all_boxes)
        labels_tensor = torch.cat(all_labels)
        scores_tensor = torch.cat(all_scores)
        image_indices_tensor = torch.cat(image_indices)
        # cat_end_event.record()
        
        # torch.cuda.synchronize()
        # cat_time = cat_start_event.elapsed_time(cat_end_event) / 1000.0
        # timings.append(f"Cat time: {cat_time}")

        # rois_result_start_time = time.time()
        # torch.cuda.synchronize()
        detection_counts = [len(detections['boxes']) for detections in detections_batch]

        additional_info = [{'label': label.item(),
                            'bbox': [roi[2].item() * scale_factor,
                                    roi[1].item() * scale_factor,
                                    roi[4].item() * scale_factor,
                                    roi[3].item() * scale_factor],
                            'image_index': idx.item(),
                            'score': score.item()}
                        for label, roi, idx, score in zip(labels_tensor, rois_tensor, image_indices_tensor, scores_tensor)]
        # torch.cuda.synchronize()
        # rois_result_time = time.time() - rois_result_start_time
        # timings.append(f"RoIs result time: {rois_result_time}")

        # 打印所有计时信息
        # for timing in timings:
        #     print(timing)
        
        return rois_tensor.float().to(self.device), additional_info, detection_counts

    def separate_pooled_features(self, pooled_features, additional_info, detection_counts):
        separated_features = []
        separated_additional_info = []
        current_idx = 0

        for count in detection_counts:
            if count > 0:
                separated_features.append(pooled_features[current_idx:current_idx + count])
                separated_additional_info.append(additional_info[current_idx:current_idx + count])
            else:
                separated_features.append(torch.empty(0, pooled_features.size(1), pooled_features.size(2), pooled_features.size(3)))
                separated_additional_info.append([])
            current_idx += count

        return separated_features, separated_additional_info   

    def forward(self, nested_tensor: NestedTensor, targets, detections_batch):
        images = nested_tensor.tensors
        mask = nested_tensor.mask
        batch_size = images.size(0)

        timings = []

        # Initialize profiler
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/log_4'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            # Backbone time
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            batch_denoised_features, _, scales = self.backbone(images)
            end_event.record()
            torch.cuda.synchronize()
            backbone_time = start_event.elapsed_time(end_event) / 1000.0
            timings.append(f"Backbone time: {backbone_time} 秒")
            prof.step()

            # Downsampling time
            downsampling_start_time = time.time()
            batch_denoised_features = batch_denoised_features.permute(0, 3, 1, 2)
            batch_denoised_features = F.interpolate(batch_denoised_features, scale_factor=2, mode='bilinear', align_corners=True)
            downsampling_time = time.time() - downsampling_start_time
            timings.append(f"Downsampling time: {downsampling_time} 秒")
            prof.step()

            # ROI time
            roi_start_time = time.time()
            rois, additional_info, detection_counts = self.prepare_rois_cpu(detections_batch, targets)
            roi_time = time.time() - roi_start_time
            timings.append(f"ROI time: {roi_time} 秒")
            prof.step()

            # Pool time
            pool_start_event = torch.cuda.Event(enable_timing=True)
            pool_end_event = torch.cuda.Event(enable_timing=True)

            pool_start_event.record()
            output_size = (1, 1)
            pooled_features = roi_align(batch_denoised_features, rois, output_size)
            pool_end_event.record()
            torch.cuda.synchronize()
            pool_time = pool_start_event.elapsed_time(pool_end_event) / 1000.0
            timings.append(f"Pool time: {pool_time} 秒")
            prof.step()

            # Separate time
            separate_start_time = time.time()
            separated_features, separated_additional_info = self.separate_pooled_features(pooled_features, additional_info, detection_counts)
            separate_time = time.time() - separate_start_time
            timings.append(f"Separate time: {separate_time} 秒")
            prof.step()

            # For loop time
            for_start_time = time.time()
            all_pairs = []
            pair_start_indices = []
            hoi_results = []
            current_index = 0
            for i, (features, info) in enumerate(zip(separated_features, separated_additional_info)):
                human_features = []
                object_features = []

                for feat, det_info in zip(features, info):
                    if det_info['label'] == self.person_category_id:
                        human_features.append((feat.view(-1), det_info['score'], det_info['label'], det_info['bbox']))
                    else:
                        object_features.append((feat.view(-1), det_info['score'], det_info['label'], det_info['bbox']))

                human_features.sort(key=lambda x: -x[1])
                object_features.sort(key=lambda x: -x[1])
                if self.num_objects is not None:
                    human_features = human_features[:self.num_objects]
                    object_features = object_features[:self.num_objects]

                pair_start_indices.append(current_index)
                for human in human_features:
                    for obj in object_features:
                        combined_feature = torch.cat([human[0], obj[0]], dim=0)
                        all_pairs.append(combined_feature.unsqueeze(0))
                        hoi_results.append({
                            'subject_category': human[2],
                            'subject_score': human[1],
                            'subject_bbox': human[3],
                            'object_category': obj[2],
                            'object_score': obj[1],
                            'object_bbox': obj[3],
                        })
                        current_index += 1
            for_time = time.time() - for_start_time
            timings.append(f"for loop time: {for_time} 秒")
            prof.step()

            # MLP time
            if all_pairs:
                mlp_start_event = torch.cuda.Event(enable_timing=True)
                mlp_end_event = torch.cuda.Event(enable_timing=True)

                mlp_start_event.record()
                all_pairs_tensor = torch.cat(all_pairs, dim=0)
                relation_scores = self.mlp(all_pairs_tensor)
                mlp_end_event.record()
                torch.cuda.synchronize()
                mlp_time = mlp_start_event.elapsed_time(mlp_end_event) / 1000.0
                timings.append(f"MLP time: {mlp_time} 秒")
                prof.step()

                if torch.isnan(relation_scores).any():
                    raise ValueError("NaN detected in relation_scores.")
            else:
                timings.append("No pairs found for this batch.")
                return [[] for _ in range(batch_size)]

            # Result time
            result_start_time = time.time()
            for i, score in enumerate(relation_scores):
                hoi_results[i]['relation_score'] = score

            image_hoi_results = []
            for i in range(len(pair_start_indices)):
                start_idx = pair_start_indices[i]
                end_idx = pair_start_indices[i + 1] if i + 1 < len(pair_start_indices) else len(relation_scores)
                if start_idx == end_idx:
                    image_hoi_results.append([])
                    timings.append("No results found for this image.")
                else:
                    image_hoi_results.append(hoi_results[start_idx:end_idx])
            result_time = time.time() - result_start_time
            timings.append(f"Result time: {result_time} 秒")
            prof.step()

        # 打印所有计时信息
        for timing in timings:
            print(timing)

        return image_hoi_results
        
class HOIModel_old(nn.Module):
    def __init__(self, backbone, device, num_objects=10, feature_dim=768, person_category_id=1, patch_size=14):
        super(HOIModel_old, self).__init__()
        self.backbone = backbone
        self.device = device
        self.num_objects = num_objects
        self.feature_dim = feature_dim
        self.person_category_id = person_category_id
        self.patch_size = patch_size
        self.mlp = nn.Sequential(
            nn.Linear(2 * feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 117)  # Assuming 117 relation classes
        )
        self.to(self.device)
    def forward(self, nested_tensor: NestedTensor , targets, detections_batch):
        # print("targets[0]: ", targets[0])
        images = nested_tensor.tensors
        mask = nested_tensor.mask
        backbone_start_time = time.time()
        batch_denoised_features, _, scales = self.backbone(images)
        backbone_time = time.time() - backbone_start_time
        print("Backbone time: ", backbone_time)
        
        downsampling_start_time = time.time()
        batch_denoised_features = batch_denoised_features.permute(0, 3, 1, 2)
        batch_denoised_features = F.interpolate(batch_denoised_features, scale_factor=2, mode='bilinear', align_corners=True)
        batch_denoised_features = batch_denoised_features.permute(0, 2, 3, 1)

        downsampled_mask = F.interpolate(mask.unsqueeze(1).float(), size=(mask.shape[1] // self.patch_size * 2, mask.shape[2] // self.patch_size * 2), mode='nearest').bool().squeeze(1)

        downsampling_time = time.time() - downsampling_start_time
        print("Downsampling time: ", downsampling_time)
        
        batch_size = images.size(0)
        all_pairs = []
        pair_start_indices = []
        hoi_results = []
        current_index = 0
        
        for_start_time = time.time()
        for b in range(batch_size):
            H, W = targets[b]["size"]
            input_detections = detections_batch[b]
            detections = []
            
            detection_transfer_start_time = time.time()
            boxes = input_detections['boxes']
            if len(boxes) == 0:
                print("No boxes found for this image.")
                # continue
            labels = input_detections['labels']
            scores = input_detections['scores']
            
            for i, box in enumerate(boxes):
                cx, cy, bw, bh = box
                xmin = (cx - bw / 2) * W
                ymin = (cy - bh / 2) * H
                xmax = (cx + bw / 2) * W
                ymax = (cy + bh / 2) * H
                
                detections.append({
                    'category_id': labels[i].item(),
                    'bbox': [xmin, ymin, xmax, ymax],
                    'score': scores[i].item()
                })
            detection_transfer_time = time.time() - detection_transfer_start_time
            print("Detection transfer time: ", detection_transfer_time)
            
            denoised_features = batch_denoised_features[b]
            print("device: ", denoised_features.device)
            h, w, _ = denoised_features.shape
            objects_features = []
            human_features = []
            
            if len(detections) == 0: 
                print("No detections found for this image.")
                continue
            # # Extract features for detected humans and objects
            # bboxes = torch.tensor([det['bbox'] for det in detections], device=self.device)
            # bboxes_scaled = (bboxes * torch.tensor(scales + scales, device=self.device).repeat((bboxes.size(0), 1)) / self.patch_size * 2).int()
            detection_loop_start_time = time.time()
            for det in detections:
                det_start_time = time.time()
                bbox = det['bbox']
                a, b, c, d = bbox
                bbox_scaled = [int(bbox[i] * scales[i % 2] / self.patch_size * 2) for i in range(4)]
                x1, y1, x2, y2 = bbox_scaled
                # 确保边界框不超出图像边界
                # x1 = max(0, min(x1, w - 1))
                # x2 = max(0, min(x2, w))
                # y1 = max(0, min(y1, h - 1))
                # y2 = max(0, min(y2, h))

                if y2 <= y1 or x2 <= x1:  # 检查是否有效
                    y2 = max(y2, y1 + 1)  # 确保边界框至少有1个像素的高度
                    x2 = max(x2, x1 + 1)  # 确保边界框至少有1个像素的宽度
                    print("Invalid bbox:", x1, y1, x2, y2)
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

                if det['category_id'] == self.person_category_id:
                    human_features.append((obj_feature, det['score'], det['category_id'], det['bbox']))
                else:
                    objects_features.append((obj_feature, det['score'], det['category_id'], det['bbox']))
                det_time = time.time() - det_start_time
                print("Det time: ", det_time)
            detection_loop_time = time.time() - detection_loop_start_time
            print("-----------------")
            print("Detection loop time: ", detection_loop_time)
            # print("-----------------")
            # print("human_features: ", len(human_features))
            # print("objects_features: ", len(objects_features))
            # Sort and limit the number of humans and objects if num_objects is set
            sort_start_time = time.time()
            human_features.sort(key=lambda x: -x[1])
            objects_features.sort(key=lambda x: -x[1])
            if self.num_objects is not None:
                human_features = human_features[:self.num_objects]
                objects_features = objects_features[:self.num_objects]
            sort_time = time.time() - sort_start_time
            print("Sort time: ", sort_time)
            # Record the start index for this image's pairs
            pair_start_indices.append(current_index)

            pair_start_time = time.time()
            # Generate all valid human-object pairs
            for human in human_features:
                for obj in objects_features:
                    combined_feature = torch.cat([human[0], obj[0]], dim=0)
                    # print("combined_feature: ", combined_feature.shape)
                    all_pairs.append(combined_feature.unsqueeze(0))
                    hoi_results.append({
                        'subject_category': human[2],
                        'subject_score': human[1],
                        'subject_bbox': human[3],
                        'object_category': obj[2],
                        'object_score': obj[1],
                        'object_bbox': obj[3],
                    })
                    current_index += 1
            pair_time = time.time() - pair_start_time
            print("Pair time: ", pair_time)
            print("-----------------")
            # print("hoi_results: ", len(hoi_results))
            # print("-----------------")
        for_time = time.time() - for_start_time
        print("for loop time: ", for_time)
        
        if all_pairs:
            mlp_start_time = time.time()
            all_pairs_tensor = torch.cat(all_pairs, dim=0)
            relation_scores = self.mlp(all_pairs_tensor)
            mlp_time = time.time() - mlp_start_time
            print("MLP time: ", mlp_time)
            if torch.isnan(relation_scores).any():
                    raise ValueError("NaN detected in relation_scores.")
        else:
            print("No pairs found for this batch.")
            return [[] for _ in range(batch_size)]  # Return a list of empty lists, one per image in batch
        
        result_start_time = time.time()
        for i, score in enumerate(relation_scores):
            hoi_results[i]['relation_score'] = score
        # Organize results per image
        image_hoi_results  = []
        for i in range(len(pair_start_indices)):
            start_idx = pair_start_indices[i]
            end_idx = pair_start_indices[i+1] if i+1 < len(pair_start_indices) else len(relation_scores)
            if start_idx == end_idx:
                # If there are no results for this image, append an empty list or a placeholder
                image_hoi_results.append([])
                print("No results found for this image.")
            else:
                image_hoi_results.append(hoi_results[start_idx:end_idx])
        result_time = time.time() - result_start_time
        print("Result time: ", result_time)
        return image_hoi_results 

class PostProcessHOI(nn.Module):
    def __init__(self, relation_threshold=0.0, device='cuda'):
        super().__init__()
        self.relation_threshold = relation_threshold
        print("self.relation_threshold: ", self.relation_threshold)
        self.device = device

    def forward(self, batch_hoi_results, targets):
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
            orig_size = targets[b]['orig_size'].to(self.device)
            current_size = targets[b]['size'].to(self.device)
            scale_w = orig_size[1].float() / current_size[1].float()  # 宽度比例
            scale_h = orig_size[0].float() / current_size[0].float()  # 高度比例
            # 对每个图像的HOI结果进行处理
            for hoi in image_results:
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
    def __init__(self, matcher, device, alpha=0.25, gamma=2.0, loss_type='focal', add_negative_category=False):
        super().__init__()
        self.matcher = matcher
        self.device = device
        self.alpha = alpha
        self.gamma = gamma
        self.loss_type = loss_type
        # 初始化不匹配计数器
        self.subject_label_mismatch = 0
        self.object_label_mismatch = 0
        self.subject_box_mismatch = 0
        self.object_box_mismatch = 0
        self.total_pairs = 0
        self.add_negative_category = add_negative_category
        self.num_category = 117 if not add_negative_category else 118
        
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
        batch_loss = 0.0
        num_processed = 0  # 记录参与损失计算的图像数
        # print("len(outputs)", len(outputs))
        # print("len(targets)", len(targets))
        
        matched_pred_verb_scores = []
        matched_tgt_verb_labels = []
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
                
                if self.add_negative_category:
                    matched_tgt_verb_labels.append(torch.cat([tgt['verb_labels'][obj_idx], torch.tensor([0.0], device=self.device)], dim=0).unsqueeze(0))
                else:
                    matched_tgt_verb_labels.append(tgt['verb_labels'][obj_idx].unsqueeze(0))
                pred_subject = pred[sub_idx]

                tgt_subject = tgt['sub_labels'][obj_idx]
                tgt_object = tgt['obj_labels'][obj_idx]
                H, W = tgt['size']
                tgt_sub_boxes = box_cxcywh_to_xyxy(tgt['sub_boxes'][obj_idx]) * torch.tensor([W, H, W, H], device=self.device)
                tgt_obj_boxes = box_cxcywh_to_xyxy(tgt['obj_boxes'][obj_idx]) * torch.tensor([W, H, W, H], device=self.device)
                self.total_pairs += 1
                try:
                    assert pred_subject['subject_category'] - 1 == tgt_subject.item(), f"Subject labels do not match: pred {pred_subject['subject_category']-1}, tgt {tgt_subject.item()}"
                except AssertionError as e:
                    print(f"AssertionError: {e}")
                    print(f"sub_idx: {sub_idx}, obj_idx: {obj_idx}")
                    print(tgt['filename'])
                    self.subject_label_mismatch += 1
                try:
                    assert pred_subject['object_category'] - 1 == tgt_object.item(), f"Object labels do not match: pred {pred_subject['object_category']}, tgt {tgt_object.item()}"
                except AssertionError as e:
                    print(f"AssertionError: {e}")
                    print(f"sub_idx: {sub_idx}, obj_idx: {obj_idx}")
                    print(tgt['filename'])
                    self.object_label_mismatch += 1
                try:
                    assert torch.allclose(torch.tensor(pred_subject['subject_bbox'], device=self.device), tgt_sub_boxes), f"Subject boxes do not match: pred {pred_subject['subject_bbox']}, tgt {tgt_sub_boxes}"
                except AssertionError as e:
                    print(f"AssertionError: {e}")
                    print(f"sub_idx: {sub_idx}, obj_idx: {obj_idx}")
                    print(tgt['filename'])
                    self.subject_box_mismatch += 1
                try:
                    assert torch.allclose(torch.tensor(pred_subject['object_bbox'], device=self.device), tgt_obj_boxes), f"Object boxes do not match: pred {pred_subject['object_bbox']}, tgt {tgt_obj_boxes}"
                except AssertionError as e:
                    print(f"AssertionError: {e}")
                    print(f"sub_idx: {sub_idx}, obj_idx: {obj_idx}")
                    print(tgt['filename'])
                    self.object_box_mismatch += 1
            if self.add_negative_category:
                unmatched_pred_verb_scores = []
                for i in range(len(pred)):
                    if i not in sub_inds:
                        unmatched_pred_verb_scores.append(pred[i]['relation_score'])

                if unmatched_pred_verb_scores:
                    matched_pred_verb_scores += unmatched_pred_verb_scores
                    negative_labels = torch.zeros((len(unmatched_pred_verb_scores), self.num_category), device=self.device)
                    negative_labels[:, -1] = 1  # 将标签设置为第118类
                    matched_tgt_verb_labels += [negative_labels]

        if matched_pred_verb_scores:
            # matched_pred_verb_scores = torch.stack(matched_pred_verb_scores)
            # matched_tgt_verb_labels = torch.stack(matched_tgt_verb_labels)
            matched_pred_verb_scores = torch.stack([tensor for tensor in matched_pred_verb_scores]).requires_grad_(True)
            matched_tgt_verb_labels = torch.cat(matched_tgt_verb_labels, dim=0) #.requires_grad_(True)
            # print("matched_tgt_verb_labels: ", matched_tgt_verb_labels)
            # 确保tensor中的元素只包含0或1
            # assert torch.all((matched_tgt_verb_labels == 0) | (matched_tgt_verb_labels == 1)), "Tensor should only contain 0s and 1s."
            
            # 确保每个样本的编码中只有一个1
            # assert torch.all(torch.sum(matched_tgt_verb_labels, dim=-1) == 1), "Each sample should have exactly one category set to 1."
            if self.loss_type == 'focal':
                # 计算Focal Loss
                # print("matched_pred_verb_scores: ", matched_pred_verb_scores)
                # print("matched_tgt_verb_labels: ", matched_tgt_verb_labels)
                batch_loss = self.focal_loss(matched_pred_verb_scores, matched_tgt_verb_labels)
                print("batch_loss: ", batch_loss)
            elif self.loss_type == 'bce':
                # 计算二元交叉熵损失
                batch_loss = self.bce_loss(matched_pred_verb_scores, matched_tgt_verb_labels)
                # print("bce_loss: ", loss)
            # batch_loss += loss
            # num_processed += 1
        # if batch_loss == 0.0:
        #     batch_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        return batch_loss

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

