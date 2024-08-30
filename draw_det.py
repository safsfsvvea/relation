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

def visualize_annotations(filename, annotations, output_dir):
    # 获取图像路径
    image_path = os.path.join('/bd_byt4090i1/users/clin/RLIPv2/hico_20160224_det/images/train2015', filename)
    if not os.path.exists(image_path):
        print(f"Image {filename} not found at {image_path}. Skipping...")
        return
    
    # 打开图像
    image = Image.open(image_path).convert('RGB')
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    # 绘制所有注释中的边界框和类别标签
    for annotation in annotations:
        bbox = annotation['bbox']
        category_id = annotation['category_id']
        draw_bbox(ax, bbox, f'Cat: {category_id}', 'green')

    # 保存输出图像
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.close()

def load_and_visualize(json_file, output_dir):
    with open(json_file, 'r') as f:
        data = json.load(f)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename, annotations in tqdm(data.items(), desc="Processing images"):
        visualize_annotations(filename, annotations, output_dir)

# 使用示例
json_file = '/bd_targaryen/users/clin/RLIPv2/data/hico_20160224_det/internimage/hico_det.json'
output_dir = '/bd_targaryen/users/clin/relation/results/internimage_train_detection'
load_and_visualize(json_file, output_dir)
