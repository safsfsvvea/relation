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
    ax.text(mid_point[0], mid_point[1], label, verticalalignment='top', color=color, fontsize=8, weight='bold', bbox=dict(facecolor='white', alpha=0.5))

def visualize_annotations(info, output_dir):
    filename = info['file_name']
    image_path = os.path.join('/bd_byt4090i1/users/clin/RLIPv2/hico_20160224_det/images/test2015', filename)
    image = Image.open(image_path).convert('RGB')
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    # 绘制所有注释中的边界框和类别标签
    annotations = info['annotations']
    for annotation in annotations:
        bbox = annotation['bbox']
        category_id = annotation['category_id']
        draw_bbox(ax, bbox, f'Cat: {category_id}', 'green')

    # 绘制 HOI 信息，通过连线和动作标签表示
    hois = info['hoi_annotation']
    for hoi in hois:
        sub_idx = hoi['subject_id']
        obj_idx = hoi['object_id']
        category_id = hoi['category_id']
        sub_bbox = annotations[sub_idx]['bbox']
        obj_bbox = annotations[obj_idx]['bbox']
        draw_line_between_centers(ax, sub_bbox, obj_bbox, f'Action: {category_id}', 'blue')

    # 保存输出图像
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.close()

def load_and_visualize(json_file, output_dir):
    with open(json_file, 'r') as f:
        data = json.load(f)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    visualize_annotations(data[8195], output_dir)
    # for info in tqdm(data, desc="Processing images"):
    #     visualize_annotations(info, output_dir)

# 使用示例
json_file = '/bd_byt4090i1/users/clin/RLIPv2/hico_20160224_det/annotations/test_hico.json'
output_dir = '/bd_byt4090i1/users/clin/relation/results/test_annotations_test'
load_and_visualize(json_file, output_dir)
