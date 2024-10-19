import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm
def draw_bbox(ax, bbox, label, color):
    x_min, y_min, x_max, y_max = bbox
    rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    ax.text(x_min, y_min, label, verticalalignment='top', color=color, fontsize=12, weight='bold')

def visualize_gt_and_pred(info, output_dir):
    gt = info['ground_truth']
    pred = info['prediction']

    filename = gt['filename']
    image_path = os.path.join('/bd_byt4090i1/users/clin/RLIPv2/hico_20160224_det/images/test2015', filename)
    # image_path = '/bd_targaryen/users/clin/relation/results/test_single/gt/HICO_test2015_00000064.jpg'
    image = Image.open(image_path).convert('RGB')
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    # Draw GT bboxes and labels
    gt_boxes = gt['boxes']
    gt_labels = gt['labels']
    for bbox, label in zip(gt_boxes, gt_labels):
        draw_bbox(ax, bbox, f'GT: {label}', 'green')

    # Draw Predicted bboxes and labels
    pred_boxes = pred['boxes']
    pred_labels = pred['labels']
    for bbox, label in zip(pred_boxes, pred_labels):
        draw_bbox(ax, bbox, f'Pred: {label}', 'red')

    # Draw HOI information
    gt_hois = gt['hois']
    for hoi in gt_hois:
        sub_idx, obj_idx, verb_label = hoi
        sub_bbox = gt_boxes[sub_idx]
        obj_bbox = gt_boxes[obj_idx]
        draw_bbox(ax, sub_bbox, f'GT HOI Sub: {verb_label}', 'blue')
        draw_bbox(ax, obj_bbox, f'GT HOI Obj: {verb_label}', 'blue')

    pred_hois = zip(pred['sub_ids'], pred['obj_ids'], pred['top_labels'], pred['top_probabilities'])
    for sub_idx, obj_idx, verbs, probs in pred_hois:
        sub_bbox = pred_boxes[sub_idx]
        obj_bbox = pred_boxes[obj_idx]
        for verb, prob in zip(verbs, probs):
            draw_bbox(ax, sub_bbox, f'Pred HOI Sub: {verb} ({prob:.2f})', 'orange')
            draw_bbox(ax, obj_bbox, f'Pred HOI Obj: {verb} ({prob:.2f})', 'orange')

    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.close()

def load_and_visualize(json_file, output_dir):
    with open(json_file, 'r') as f:
        data = json.load(f)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for info in tqdm(data, desc="Processing images"):
        visualize_gt_and_pred(info, output_dir)

# Example usage
json_file = '/bd_targaryen/users/clin/relation/results/test_single/low_mAP_info.json'
output_dir = '/bd_targaryen/users/clin/relation/results/test_single/images'
load_and_visualize(json_file, output_dir)
