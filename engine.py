import time
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import util.misc as utils
import itertools
from datasets.hico_eval import HICOEvaluator
import numpy as np
import copy

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq=100, tensorboard_writer=None):
    # TODO: 优化成RLIP的样子
    model.train()
    start_time = time.time()
    epoch_loss = 0.0
    num_batches = len(data_loader)
    
    progress_bar = tqdm(enumerate(data_loader), total=num_batches, desc=f"Epoch {epoch}")
    
    for batch_idx, (images, targets, detections) in progress_bar:
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items() if k not in ['filename', 'image_id', 'obj_classes', 'verb_classes']} for t in targets]
        
        optimizer.zero_grad()
        out = model(images, targets, detections)
        if not all(len(output) == 0 for output in out):
            loss = criterion(out, targets)
            # print(loss)
            loss.backward()
            optimizer.step()
        else:
            loss = torch.tensor(0.0, device=device)
        epoch_loss += loss.item()
        
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
    
    if dataset_file == 'hico' or dataset_file == 'hico_det':
        evaluator = HICOEvaluator(preds, gts, subject_category_id, data_loader.dataset.rare_triplets,
                                  data_loader.dataset.non_rare_triplets, data_loader.dataset.correct_mat, args)
    # elif dataset_file == 'vcoco':
    #     evaluator = VCOCOEvaluator(preds, gts, subject_category_id, data_loader.dataset.correct_mat, args)

    stats = evaluator.evaluate()
    
    # print("len(preds): ", len(preds))
    # if preds[0]:
    #     print("-----------------")
    #     print("preds[0]: ", preds[0])
    #     print("preds[0]['labels']: ", preds[0]['labels'].shape)
    #     print("preds[0]['boxes']: ", preds[0]['boxes'].shape)
    #     print("preds[0]['verb_scores']: ", preds[0]['verb_scores'].shape)
    #     print("preds[0]['sub_ids']: ", preds[0]['sub_ids'].shape)
    #     print("preds[0]['obj_ids']: ", preds[0]['obj_ids'].shape)
    #     print("-----------------")
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
    # print("gts[0]: ", gts[0]) 
    return stats