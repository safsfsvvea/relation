import time
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import util.misc as utils
import itertools
from datasets.hico_eval import HICOEvaluator
import numpy as np
import copy
from torchvision.ops import box_iou
import torch.nn.functional as F

def train_one_epoch(model, detector, criterion, optimizer, data_loader, device, epoch, print_freq=100, tensorboard_writer=None):
    # TODO: 优化成RLIP的样子
    model.train()
    detector.model.train()
    start_time = time.time()
    epoch_loss = 0.0
    num_batches = len(data_loader)
    
    progress_bar = tqdm(enumerate(data_loader), total=num_batches, desc=f"Epoch {epoch}")
    
    for batch_idx, (images, targets, detections) in progress_bar:
        step_times = {}
        step_start_time = time.time()
        
        transfer_start_time = time.time()
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items() if k not in ['filename', 'image_id', 'obj_classes', 'verb_classes']} for t in targets]
        step_times['data_transfer'] = time.time() - transfer_start_time
        
        forward_start_time = time.time()
        optimizer.zero_grad()
        tensors = images.tensors
        new_height = ((tensors.shape[2] - 1) // 32 + 1) * 32
        new_width = ((tensors.shape[3] - 1) // 32 + 1) * 32
        resized_images = F.interpolate(tensors, size=(new_height, new_width), mode='bilinear', align_corners=False)

        yolo_out = detector.model(resized_images)
        print(f"yolo_out: {yolo_out[0]}")
        out = model(images, targets, detections)
        step_times['forward'] = time.time() - forward_start_time
        
        
        if not all(len(output) == 0 for output in out):
            loss_start_time = time.time()
            loss = criterion(out, targets)
            step_times['loss_calc'] = time.time() - loss_start_time
            # print(loss)
            backward_start_time = time.time()
            loss.backward()
            step_times['backward'] = time.time() - backward_start_time
            
            optimize_start_time = time.time()
            optimizer.step()
            step_times['optimize'] = time.time() - optimize_start_time
        else:
            loss_start_time = time.time()
            loss = torch.tensor(0.0, device=device)
            step_times['loss_calc'] = time.time() - loss_start_time
            step_times['optimize'] = 0.0
        epoch_loss += loss.item()
        
        # 打印每个batch的时间
        print(f"Batch {batch_idx + 1}:")
        print(f"  Data transfer: {step_times['data_transfer']:.4f} seconds")
        print(f"  Forward pass: {step_times['forward']:.4f} seconds")
        print(f"  Loss calculation: {step_times['loss_calc']:.4f} seconds")
        print(f"  Backward pass: {step_times['backward']:.4f} seconds")
        print(f"  Optimization: {step_times['optimize']:.4f} seconds")
        
        if (batch_idx + 1) % print_freq == 0:
            avg_loss = epoch_loss / (batch_idx + 1)
            elapsed_time = time.time() - start_time
            eta = elapsed_time / (batch_idx + 1) * (num_batches - batch_idx - 1)
            
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}", time=f"{elapsed_time:.2f}s", eta=f"{eta:.2f}s")
        
        # 保存训练中间信息到tensorboard
        if tensorboard_writer is not None:
            global_step = epoch * num_batches + batch_idx
            tensorboard_writer.add_scalar('train_loss', loss.item(), global_step=global_step)
    
    # 打印训练总时间
    elapsed_time = time.time() - start_time
    print(f"Epoch {epoch} completed in {elapsed_time:.2f} seconds. Average loss: {epoch_loss / num_batches:.4f}")
    
    return epoch_loss / num_batches

@torch.no_grad()
def evaluate_hoi(dataset_file, model, postprocessors, data_loader, subject_category_id, device, args):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    preds = []
    gts = []
    indices = []
    # filename_list = []
    print_freq = 500
    for samples, targets, detections in metric_logger.log_every(data_loader, print_freq, header):
        # targets: tuple, len(tuple) = batch_size
        #          element in tuple: a dict, whose keys are ['orig_size', 'size', 'boxes', 'labels', 'id', 'hois']
                 
        # print(targets[0]['orig_size'])
        # print(targets[0]['size'])
        # print('')
        samples = samples.to(device)

        outputs = model(samples, targets, detections)
        
        results = postprocessors(outputs, targets)
        # print(results)
        # print(len(list(itertools.chain.from_iterable(utils.all_gather(results)))))
        # print(list(itertools.chain.from_iterable(utils.all_gather(results)))[0])
        
        # preds: merge predicted batch data
        preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
        # For avoiding a runtime error, the copy is used
        # gts: merge ground truth batch data
        gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets)))))
        
        # break
        # # Add for evaluation
        # filename_list += [t['filename'] for t in targets]

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    img_ids = [img_gts['id'] for img_gts in gts]
    _, indices = np.unique(img_ids, return_index=True)
    preds = [img_preds for i, img_preds in enumerate(preds) if i in indices]
    gts = [img_gts for i, img_gts in enumerate(gts) if i in indices]
    
    if dataset_file == 'hico' or dataset_file == 'hico_det' or dataset_file == 'hico_det_gt':
        evaluator = HICOEvaluator(preds, gts, subject_category_id, data_loader.dataset.rare_triplets,
                                  data_loader.dataset.non_rare_triplets, data_loader.dataset.correct_mat, args)
    # elif dataset_file == 'vcoco':
    #     evaluator = VCOCOEvaluator(preds, gts, subject_category_id, data_loader.dataset.correct_mat, args)

    stats = evaluator.evaluate()
    # for i, pred in enumerate(preds):
    #     print(f"-----------------")
    #     print(f"preds[{i}]: ", pred)
    #     print(f"预测的标签索引: ", torch.argmax(pred['verb_scores'], dim=1))
    #     probabilities = torch.nn.functional.softmax(pred['verb_scores'], dim=1)
    #     max_probabilities, predicted_labels = torch.max(probabilities, dim=1)
    #     print(f"预测的概率: ", probabilities)
    #     print(f"每个样本最高的概率值: ", max_probabilities)
    #     print(f"预测的标签索引: ", predicted_labels)
    #     print(f"preds[{i}]['labels']: ", pred['labels'].shape)
    #     print(f"preds[{i}]['boxes']: ", pred['boxes'].shape)
    #     print(f"preds[{i}]['verb_scores']: ", pred['verb_scores'].shape)
    #     print(f"preds[{i}]['sub_ids']: ", pred['sub_ids'].shape)
    #     print(f"preds[{i}]['obj_ids']: ", pred['obj_ids'].shape)
    #     print(f"-----------------")
    #     print(f"gts[{i}]: ", gts[i]) 
    # print("len(preds): ", len(preds))
    # print("len(gts): ", len(gts))
    if preds[0]:
        import torch.nn.functional as F

        # 假设 preds[0]['verb_scores'] 是 PyTorch 张量
        logits = preds[0]['verb_scores']
        # 转换 logits 为概率
        probabilities = F.softmax(logits, dim=1)

        sorted_probabilities, sorted_indices = torch.sort(probabilities, dim=1, descending=True)
        num_hois = len(gts[0]['hois'])
        top_probabilities = sorted_probabilities[:, :num_hois]
        top_labels = sorted_indices[:, :num_hois]
        # 获取每个样本的预测标签索引
        # max_probabilities, predicted_labels = torch.max(probabilities, dim=1)

        print("-----------------")
        print("preds[0]: ", preds[0])
        print("预测的概率: ", probabilities)
        print("每个样本最高的概率值: ", top_probabilities)
        print("预测的标签索引: ", top_labels)
        print("preds[0]['labels']: ", preds[0]['labels'].shape)
        print("preds[0]['boxes']: ", preds[0]['boxes'].shape)
        print("preds[0]['verb_scores']: ", preds[0]['verb_scores'].shape)
        print("preds[0]['sub_ids']: ", preds[0]['sub_ids'].shape)
        print("preds[0]['obj_ids']: ", preds[0]['obj_ids'].shape)
        print("-----------------")
    # if preds[1]:
    #     print("-----------------")
    #     print("preds[1]: ", preds[1])
    #     print("preds[1]['labels']: ", preds[1]['labels'].shape)
    #     print("preds[1]['boxes']: ", preds[1]['boxes'].shape)
    #     print("preds[1]['verb_scores']: ", preds[1]['verb_scores'].shape)
    #     print("preds[1]['sub_ids']: ", preds[1]['sub_ids'].shape)
    #     print("preds[1]['obj_ids']: ", preds[1]['obj_ids'].shape)
    #     print("-----------------")
    # print("len(gts): ", len(gts))
    print("gts[0]: ", gts[0]) 
    return stats

def evaluate_det_hico(model, data_loader, device, args):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    preds = []
    gts = []
    indices = []
    # filename_list = []
    iou_threshold = 0.5
    total_gt_boxes = 0
    correct_detections = 0
    print_freq = 500
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        # targets: tuple, len(tuple) = batch_size
        #          element in tuple: a dict, whose keys are ['orig_size', 'size', 'boxes', 'labels', 'id', 'hois']
                 
        # print(targets[0]['orig_size'])
        # print(targets[0]['size'])
        # print('')
        samples = samples.to(device)
        
        with torch.no_grad():
            
            predictions = model(samples)

        pred_boxes = predictions['boxes']
        gt_boxes = targets['boxes']

        iou = box_iou(pred_boxes, gt_boxes)

        # 计算IoU > 0.5的检测框数量
        max_iou, _ = iou.max(dim=0)
        correct_detections += (max_iou >= iou_threshold).sum().item()
        total_gt_boxes += gt_boxes.size(0)

    accuracy = correct_detections / total_gt_boxes
    print(f"Detection Accuracy: {accuracy:.2f}")
    return accuracy

