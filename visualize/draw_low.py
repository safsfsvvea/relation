import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm

def draw_bbox(ax, bbox, label, color):
    # bbox 是 xyxy 形式的
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min
    rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    ax.text(x_min, y_min, label, verticalalignment='top', color=color, fontsize=8, weight='bold', bbox=dict(facecolor='white', alpha=0.5))

def draw_line_between_centers(ax, sub_bbox, obj_bbox, label, color):
    # 计算边界框的中心点
    sub_center = ((sub_bbox[0] + sub_bbox[2]) / 2, (sub_bbox[1] + sub_bbox[3]) / 2)
    obj_center = ((obj_bbox[0] + obj_bbox[2]) / 2, (obj_bbox[1] + obj_bbox[3]) / 2)
    # 绘制连接线
    ax.plot([sub_center[0], obj_center[0]], [sub_center[1], obj_center[1]], color=color, linewidth=2, linestyle='--')
    # 在连线中间标注动作类别
    mid_point = ((sub_center[0] + obj_center[0]) / 2, (sub_center[1] + obj_center[1]) / 2)
    # ax.text(mid_point[0], mid_point[1], label, verticalalignment='top', color=color, fontsize=8, weight='bold', bbox=dict(facecolor='white', alpha=0.5))
    ax.text(mid_point[0], mid_point[1], label, verticalalignment='top', color=color, fontsize=12, weight='bold', bbox=dict(facecolor='white', alpha=0.7))  # 增加字体大小和透明度

def visualize_gt(info, output_dir, dataset_dir, draw_relation=True):
    gt = info['ground_truth']
    filename = gt['filename']
    image_path = os.path.join(dataset_dir, filename)
    if not os.path.exists(image_path):
        print(f"Image {filename} not found at {image_path}. Skipping...")
        return
    
    # 打开图像
    image = Image.open(image_path).convert('RGB')
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)
    gt_boxes = gt['boxes']
    gt_labels = gt['labels']
    for bbox, label in zip(gt_boxes, gt_labels):
        draw_bbox(ax, bbox, f'GT: {label}', 'green')
    # 绘制 GT 的边界框和类别标签
    # annotations = gt['annotations']
    # for annotation in annotations:
    #     bbox = annotation['bbox']
    #     category_id = annotation['category_id']
    #     draw_bbox(ax, bbox, f'Cat: {category_id}', 'green')

    # 绘制 HOI 信息
    if draw_relation:
        hois = gt['hois']
        for hoi in hois:
            sub_idx = hoi[0]
            obj_idx = hoi[1]
            category_id = hoi[2]
            sub_bbox = gt_boxes[sub_idx]
            obj_bbox = gt_boxes[obj_idx]
            draw_line_between_centers(ax, sub_bbox, obj_bbox, f'Action: {category_id}', 'blue')

    # 保存输出图像
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.close()

def visualize_pred(info, output_dir, dataset_dir, draw_relation=True):
    pred = info['prediction']
    filename = info['ground_truth']['filename']
    image_path = os.path.join(dataset_dir, filename)
    if not os.path.exists(image_path):
        print(f"Image {filename} not found at {image_path}. Skipping...")
        return
    
    # 打开图像
    image = Image.open(image_path).convert('RGB')
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    # 绘制 Pred 的边界框和类别标签
    boxes = pred['boxes']
    labels = pred['labels']
    for bbox, label in zip(boxes, labels):
        draw_bbox(ax, bbox, f'Cat: {label}', 'red')

    # 绘制 Pred 的 HOI 信息
    if draw_relation:
        sub_ids = pred['sub_ids']
        obj_ids = pred['obj_ids']
        top_labels = pred['top_labels']
        for sub_idx, obj_idx, labels in zip(sub_ids, obj_ids, top_labels):
            sub_bbox = boxes[sub_idx]
            obj_bbox = boxes[obj_idx]
            label_str = ', '.join(map(str, labels))
            draw_line_between_centers(ax, sub_bbox, obj_bbox, f'Top Actions: {label_str}', 'orange')

    # 保存输出图像
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.close()

def load_and_visualize(json_file, output_dir_gt, output_dir_pred, dataset_dir, draw_relation=True):
    with open(json_file, 'r') as f:
        data = json.load(f)

    if not os.path.exists(output_dir_gt):
        os.makedirs(output_dir_gt)
    if not os.path.exists(output_dir_pred):
        os.makedirs(output_dir_pred)

    for info in tqdm(data, desc="Processing images"):
        visualize_gt(info, output_dir_gt, dataset_dir, draw_relation)
        visualize_pred(info, output_dir_pred, dataset_dir, draw_relation)

# 使用示例
json_file = '/bd_targaryen/users/clin/relation/results/pvic/gt/40_60/no_add_negative_category/low_mAP_info.json'
output_dir_gt = '/bd_byt4090i1/users/clin/relation/draws/pvic/gt/40_60/no_add_negative_category/gt'
output_dir_pred = '/bd_byt4090i1/users/clin/relation/draws/pvic/gt/40_60/no_add_negative_category/pred'
dataset_dir = '/bd_byt4090i1/users/clin/RLIPv2/hico_20160224_det/images/train2015'
load_and_visualize(json_file, output_dir_gt, output_dir_pred, dataset_dir, draw_relation=True)
