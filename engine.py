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
import torchvision.transforms as T
import torch.profiler
from torch.cuda.amp import GradScaler, autocast  # 添加混合精度相关库
from torch.nn.utils import clip_grad_norm_
import json
from pathlib import Path
import os
def tensor_to_list(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.tolist()
    elif isinstance(tensor, list):
        return [tensor_to_list(t) for t in tensor]
    elif isinstance(tensor, dict):
        return {k: tensor_to_list(v) for k, v in tensor.items()}
    else:
        return tensor
def train_one_epoch_with_profiler(model, criterion, optimizer, data_loader, device, epoch, print_freq=100, tensorboard_writer=None):
    model.train()
    start_time = time.time()
    epoch_loss = 0.0
    num_batches = len(data_loader)
    
    progress_bar = tqdm(enumerate(data_loader), total=num_batches, desc=f"Epoch {epoch}")
    
    # Initialize profiler
    schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    )
    
    profiler.start()

    for batch_idx, (images, targets, detections) in progress_bar:
        step_times = {}
        
        # Data transfer to GPU
        transfer_start_event = torch.cuda.Event(enable_timing=True)
        transfer_end_event = torch.cuda.Event(enable_timing=True)
        
        transfer_start_event.record()
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items() if k not in ['filename', 'image_id', 'obj_classes', 'verb_classes']} for t in targets]
        transfer_end_event.record()
        
        torch.cuda.synchronize()
        step_times['data_transfer'] = transfer_start_event.elapsed_time(transfer_end_event) / 1000.0
        
        # Forward pass
        forward_start_event = torch.cuda.Event(enable_timing=True)
        forward_end_event = torch.cuda.Event(enable_timing=True)
        
        forward_start_event.record()
        optimizer.zero_grad()
        out = model(images, targets, detections)
        forward_end_event.record()
        
        torch.cuda.synchronize()
        step_times['forward'] = forward_start_event.elapsed_time(forward_end_event) / 1000.0
        
        if not all(len(output) == 0 for output in out):
            # Loss calculation
            loss_start_event = torch.cuda.Event(enable_timing=True)
            loss_end_event = torch.cuda.Event(enable_timing=True)
            
            loss_start_event.record()
            loss = criterion(out, targets)
            loss_end_event.record()
            
            torch.cuda.synchronize()
            step_times['loss_calc'] = loss_start_event.elapsed_time(loss_end_event) / 1000.0
            
            # Backward pass
            backward_start_event = torch.cuda.Event(enable_timing=True)
            backward_end_event = torch.cuda.Event(enable_timing=True)
            
            backward_start_event.record()
            loss.backward()
            backward_end_event.record()
            
            torch.cuda.synchronize()
            step_times['backward'] = backward_start_event.elapsed_time(backward_end_event) / 1000.0
            
            # Optimization step
            optimize_start_event = torch.cuda.Event(enable_timing=True)
            optimize_end_event = torch.cuda.Event(enable_timing=True)
            
            optimize_start_event.record()
            optimizer.step()
            optimize_end_event.record()
            
            torch.cuda.synchronize()
            step_times['optimize'] = optimize_start_event.elapsed_time(optimize_end_event) / 1000.0
        else:
            loss = torch.tensor(0.0, device=device)
            step_times['loss_calc'] = 0.0
            step_times['backward'] = 0.0
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
        
        profiler.step()
    
    profiler.stop()

    # 打印训练总时间
    elapsed_time = time.time() - start_time
    print(f"Epoch {epoch} completed in {elapsed_time:.2f} seconds. Average loss: {epoch_loss / num_batches:.4f}")
    
    return epoch_loss / num_batches
def train_one_epoch_with_time(model, criterion, optimizer, data_loader, device, epoch, lr_scheduler=None, print_freq=100, accumulation_steps=32, grad_clip_val=1.0, tensorboard_writer=None, args=None):
    # TODO: 优化成RLIP的样子
    model.train()
    # detector.model.train()
    start_time = time.time()
    
    epoch_relation_loss  = 0.0
    epoch_binary_loss = 0.0  # 新增，用于记录二分类损失
    epoch_loss  = 0.0
    
    num_batches = len(data_loader)
    
    no_pairs_batches = 0  # 用于统计 "No pairs found for this batch." 出现的 batch 数量
    no_results_images = 0  # 用于统计 "No results found for this image." 出现的图片数量
    total_images = 0  # 用于统计总的图片数量
    subject_label_mismatches = 0
    object_label_mismatches = 0
    subject_box_mismatches = 0
    object_box_mismatches = 0
    total_pairs = 0  # 总的 matcher pairs 数量
    progress_bar = tqdm(enumerate(data_loader), total=num_batches, desc=f"Epoch {epoch}", mininterval=1.0)
    
    scaler = GradScaler()  # 初始化 GradScaler
    optimizer.zero_grad()
    has_non_zero_loss = False  # 标志位，跟踪是否有非零损失的累积

    forward_total_time = 0
    dataloarder_total_time = 0
    criterion_total_time = 0
    backward_total_time = 0
    optimizer_total_time = 0
    tensorboard_total_time = 0
    scheduler_total_time = 0
    progress_bar_total_time = 0
    
    dataloarder_start_time = time.time()
    for batch_idx, (images, targets, rois_tensor, additional_info, detection_counts, _) in progress_bar:
        dataloarder_time = time.time() - dataloarder_start_time
        dataloarder_total_time += dataloarder_time
        images = images.to(device)
        # input.append((images, targets, detections))
        targets = [{k: v.to(device) for k, v in t.items() if k not in ['filename', 'image_id', 'obj_classes', 'verb_classes']} for t in targets]
        # targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items() if k not in ['image_id', 'obj_classes', 'verb_classes']} for t in targets]
        # print("filename: ", targets[0]['filename'])
        
        with autocast():  # 使用 autocast 进行混合精度前向传播
            forward_start_time = time.time()
            out = model(images, rois_tensor, additional_info, detection_counts)
            forward_time = time.time() - forward_start_time
            forward_total_time += forward_time
            # print("out: ", out)
            # for image_results in out:
            #     for hoi in image_results:
            #         relation_score = torch.sigmoid(hoi['relation_score'])
            #         max_score, max_index = torch.max(F.softmax(relation_score, dim=0), dim=0)
            #         print(f"Highest probability relation score: {max_score.item()}, Index: {max_index.item()}")
        if not all(len(output) == 0 for output in out):
            with autocast():  # 使用 autocast 进行混合精度损失计算
                criterion_start_time = time.time()
                relation_loss, binary_loss = criterion(out, targets)
                criterion_time = time.time() - criterion_start_time
                criterion_total_time += criterion_time
                
                loss = relation_loss + binary_loss * 0.1
            if loss > 0:
                relation_loss = relation_loss / accumulation_steps  # 损失均摊到梯度累积步骤数
                binary_loss = binary_loss / accumulation_steps  # 损失均摊到梯度累积步骤数
                loss = loss / accumulation_steps  # 损失均摊到梯度累积步骤数
                
                backward_start_time = time.time()
                scaler.scale(loss).backward()  
                backward_time = time.time() - backward_start_time
                backward_total_time += backward_time

                has_non_zero_loss = True  # 设置标志位
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer_start_time = time.time()
                    scaler.unscale_(optimizer)  # 取消缩放
                    total_norm = clip_grad_norm_(model.parameters(), grad_clip_val)  # 计算梯度模并进行裁剪
                    
                    scaler.step(optimizer)  # 使用 GradScaler 进行优化步骤
                    scaler.update()
                    optimizer.zero_grad()  # 清空梯度
                    optimizer_time = time.time() - optimizer_start_time
                    optimizer_total_time += optimizer_time
                    
                    tensorboard_start_time = time.time()
                    if tensorboard_writer is not None and batch_idx % 10 == 0:
                        global_step = epoch * num_batches + batch_idx
                        tensorboard_writer.add_scalar('grad_norm', total_norm, global_step)  # 将梯度模写入TensorBoard
                    tensorboard_time = time.time() - tensorboard_start_time
                    tensorboard_total_time += tensorboard_time
                    
                    scheduler_start_time = time.time()
                    if args.schedule == "linear_with_warmup":
                        lr_scheduler.step()
                    scheduler_time = time.time() - scheduler_start_time
                    scheduler_total_time += scheduler_time
        else:
            relation_loss = torch.tensor(0.0, device=device, requires_grad=True)  # 确保需要梯度
            binary_loss = torch.tensor(0.0, device=device, requires_grad=True)  # 确保需要梯度
            loss = relation_loss + binary_loss * 0.1
            
            backward_start_time = time.time()
            scaler.scale(loss).backward() 
            backward_time = time.time() - backward_start_time
            backward_total_time += backward_time
            
        epoch_binary_loss += binary_loss.item() * accumulation_steps  # 修改，记录二分类损失
        epoch_relation_loss += relation_loss.item() * accumulation_steps  # 修改，记录多分类损失
        epoch_loss += loss.item() * accumulation_steps
        
        no_pairs_batches += model.no_pairs_count
        no_results_images += model.no_results_count
        total_images += len(targets)  # 更新总的图片数量
        
        # subject_label_mismatches += criterion.subject_label_mismatch
        # object_label_mismatches += criterion.object_label_mismatch
        # subject_box_mismatches += criterion.subject_box_mismatch
        # object_box_mismatches += criterion.object_box_mismatch
        # total_pairs += criterion.total_pairs
        progress_bar_start_time = time.time()
        if (batch_idx + 1) % print_freq == 0:
            avg_binary_loss = epoch_binary_loss / (batch_idx + 1)  # 修改，计算平均二分类损失
            avg_relation_loss = epoch_relation_loss / (batch_idx + 1)  # 修改，计算平均多分类损失
            avg_loss = epoch_loss / (batch_idx + 1)
            elapsed_time = time.time() - start_time
            eta = elapsed_time / (batch_idx + 1) * (num_batches - batch_idx - 1)
            progress_bar.set_postfix(total_loss=f"{avg_loss:.4f}", binary_loss=f"{avg_binary_loss:.4f}", relation_loss=f"{avg_relation_loss:.4f}", time=f"{elapsed_time:.2f}s", eta=f"{eta:.2f}s")  # 修改，显示二分类和多分类损失
        progress_bar_time = time.time() - progress_bar_start_time
        progress_bar_total_time += progress_bar_time
        
        model.no_pairs_count = 0  # 重置计数器
        model.no_results_count = 0  # 重置计数器
        
        # criterion.subject_label_mismatch = 0
        # criterion.object_label_mismatch = 0
        # criterion.subject_box_mismatch = 0
        # criterion.object_box_mismatch = 0
        # criterion.total_pairs = 0
        
        tensorboard_start_time = time.time()
        if tensorboard_writer is not None and batch_idx % 10 == 0:
            global_step = epoch * num_batches + batch_idx
            tensorboard_writer.add_scalar('train_total_loss', loss.item(), global_step=global_step)  # 修改，记录总损失到Tensorboard
            tensorboard_writer.add_scalar('train_binary_loss', binary_loss.item(), global_step=global_step)  # 修改，记录二分类损失到Tensorboard
            tensorboard_writer.add_scalar('train_relation_loss', relation_loss.item(), global_step=global_step)  # 修改，记录多分类损失到Tensorboard
        tensorboard_time = time.time() - tensorboard_start_time
        tensorboard_total_time += tensorboard_time    
        
        dataloarder_start_time = time.time()
        
    if (batch_idx + 1) % accumulation_steps != 0 and has_non_zero_loss:
        optimizer_start_time = time.time()
        scaler.unscale_(optimizer)  # 取消缩放
        total_norm = clip_grad_norm_(model.parameters(), grad_clip_val)  # 计算梯度模并进行裁剪

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        optimizer_time = time.time() - optimizer_start_time
        optimizer_total_time += optimizer_time
        
        tensorboard_start_time = time.time()
        if tensorboard_writer is not None:
            global_step = epoch * num_batches + batch_idx
            tensorboard_writer.add_scalar('grad_norm', total_norm, global_step)  # 将梯度模写入TensorBoard
        tensorboard_time = time.time() - tensorboard_start_time
        tensorboard_total_time += tensorboard_time 
        
        scheduler_start_time = time.time()
        if args.schedule == "linear_with_warmup":
            lr_scheduler.step()
        scheduler_time = time.time() - scheduler_start_time
        scheduler_total_time += scheduler_time
     
    scheduler_start_time = time.time()   
    if args.schedule == "multistep":
        lr_scheduler.step()
    scheduler_time = time.time() - scheduler_start_time
    scheduler_total_time += scheduler_time
    # if lr_scheduler is not None:
    #     lr_scheduler.step(epoch_loss  / num_batches)  # 根据当前 epoch 的平均损失更新学习率
    # 打印训练总时间
    elapsed_time = time.time() - start_time
    print(f"Epoch {epoch} completed in {elapsed_time:.2f} seconds. Average total loss: {epoch_loss  / num_batches:.4f}. Average relation loss: {epoch_relation_loss  / num_batches:.4f}. Average binary loss: {epoch_binary_loss  / num_batches:.4f}")
    print("forward_total_time: ", forward_total_time)
    print("dataloarder_total_time: ", dataloarder_total_time)
    print("criterion_total_time: ", criterion_total_time)
    print("backward_total_time: ", backward_total_time)
    print("optimizer_total_time: ", optimizer_total_time)
    print("tensorboard_total_time: ", tensorboard_total_time)
    print("scheduler_total_time: ", scheduler_total_time)
    print("progress_bar_total_time: ", progress_bar_total_time)
    no_pairs_ratio = no_pairs_batches / num_batches
    no_results_ratio = no_results_images / total_images
    
    # subject_label_mismatch_ratio = subject_label_mismatches / total_pairs if total_pairs else 0
    # object_label_mismatch_ratio = object_label_mismatches / total_pairs if total_pairs else 0
    # subject_box_mismatch_ratio = subject_box_mismatches / total_pairs if total_pairs else 0
    # object_box_mismatch_ratio = object_box_mismatches / total_pairs if total_pairs else 0
    
    print(f"No pairs found ratio: {no_pairs_ratio:.4f}")
    print(f"No results found ratio: {no_results_ratio:.4f}")
    
    # print(f"Subject label mismatch ratio: {subject_label_mismatch_ratio:.4f}")
    # print(f"Object label mismatch ratio: {object_label_mismatch_ratio:.4f}")
    # print(f"Subject box mismatch ratio: {subject_box_mismatch_ratio:.4f}")
    # print(f"Object box mismatch ratio: {object_box_mismatch_ratio:.4f}")

    if tensorboard_writer is not None:
        tensorboard_writer.add_scalar('no_pairs_ratio', no_pairs_ratio, epoch)
        tensorboard_writer.add_scalar('no_results_ratio', no_results_ratio, epoch)
        
        # tensorboard_writer.add_scalar('subject_label_mismatch_ratio', subject_label_mismatch_ratio, epoch)
        # tensorboard_writer.add_scalar('object_label_mismatch_ratio', object_label_mismatch_ratio, epoch)
        # tensorboard_writer.add_scalar('subject_box_mismatch_ratio', subject_box_mismatch_ratio, epoch)
        # tensorboard_writer.add_scalar('object_box_mismatch_ratio', object_box_mismatch_ratio, epoch)
    return epoch_loss / num_batches, epoch_relation_loss / num_batches, epoch_binary_loss  / num_batches

def train_backbone_time(model, criterion, optimizer, data_loader, device, epoch, lr_scheduler=None, print_freq=100, accumulation_steps=32, grad_clip_val=1.0, tensorboard_writer=None, args=None):
    # TODO: 优化成RLIP的样子
    model.train()
    model.backbone.eval()
    epoch_relation_loss  = 0.0
    epoch_binary_loss = 0.0  # 新增，用于记录二分类损失
    epoch_loss  = 0.0
    
    num_batches = len(data_loader)

    progress_bar = tqdm(enumerate(data_loader), total=num_batches, desc=f"Epoch {epoch}", mininterval=1.0)

    optimizer.zero_grad()

    dataloarder_total_time = 0
    forward_total_time = 0
    data_transfer_total_time = 0
    start_time = time.time()
    dataloarder_start_time = time.time()
    for batch_idx, (images, targets, rois_tensor, additional_info, detection_counts, _) in progress_bar:
        dataloarder_time = time.time() - dataloarder_start_time
        dataloarder_total_time += dataloarder_time
        
        data_transfer_start_time = time.time()
        images = images.to(device)
        
        # targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items() if k not in ['image_id', 'obj_classes', 'verb_classes']} for t in targets]
        data_transfer_time = time.time() - data_transfer_start_time
        data_transfer_total_time += data_transfer_time
        
        with autocast():  # 使用 autocast 进行混合精度前向传播
            forward_start_time = time.time()
            out = model(images, rois_tensor, additional_info, detection_counts)
            forward_time = time.time() - forward_start_time
            forward_total_time += forward_time
            
        dataloarder_start_time = time.time()
    elapsed_time = time.time() - start_time
    print(f"Epoch {epoch} completed in {elapsed_time:.2f} seconds.")
    print("forward_total_time: ", forward_total_time)    
    print("dataloarder_total_time: ", dataloarder_total_time)  
    print("data_transfer_total_time: ", data_transfer_total_time)  
    return epoch_loss / num_batches, epoch_relation_loss / num_batches, epoch_binary_loss  / num_batches

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, lr_scheduler=None, print_freq=100, accumulation_steps=32, grad_clip_val=1.0, tensorboard_writer=None, args=None):
    # TODO: 优化成RLIP的样子
    model.train()
    model.backbone.eval()
    
    start_time = time.time()
    
    epoch_relation_loss  = 0.0
    epoch_binary_loss = 0.0  # 新增，用于记录二分类损失
    epoch_loss  = 0.0
    
    num_batches = len(data_loader)
    
    no_pairs_batches = 0  # 用于统计 "No pairs found for this batch." 出现的 batch 数量
    no_results_images = 0  # 用于统计 "No results found for this image." 出现的图片数量
    total_images = 0  # 用于统计总的图片数量
    # subject_label_mismatches = 0
    # object_label_mismatches = 0
    # subject_box_mismatches = 0
    # object_box_mismatches = 0
    # total_pairs = 0  # 总的 matcher pairs 数量
    progress_bar = tqdm(enumerate(data_loader), total=num_batches, desc=f"Epoch {epoch}", mininterval=1.0)
    
    scaler = GradScaler()  # 初始化 GradScaler
    optimizer.zero_grad()
    has_non_zero_loss = False  # 标志位，跟踪是否有非零损失的累积

    # debug
    # input = []
    # forward_total_time = 0
    for batch_idx, (images, targets, rois_tensor, additional_info, detection_counts, _) in progress_bar:
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items() if k not in ['filename', 'image_id', 'obj_classes', 'verb_classes']} for t in targets]
        # targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items() if k not in ['image_id', 'obj_classes', 'verb_classes']} for t in targets]
        
        with autocast():  # 使用 autocast 进行混合精度前向传播
            # forward_start_time = time.time()
            out = model(images, rois_tensor, additional_info, detection_counts)
            # forward_time = time.time() - forward_start_time
            # forward_total_time += forward_time

        if not all(len(output) == 0 for output in out):
            with autocast():  # 使用 autocast 进行混合精度损失计算
                relation_loss, binary_loss = criterion(out, targets)
                loss = relation_loss + binary_loss * 0.1
            if loss > 0:
                relation_loss = relation_loss / accumulation_steps  # 损失均摊到梯度累积步骤数
                binary_loss = binary_loss / accumulation_steps  # 损失均摊到梯度累积步骤数
                loss = loss / accumulation_steps  # 损失均摊到梯度累积步骤数
                scaler.scale(loss).backward()  

                has_non_zero_loss = True  # 设置标志位
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)  # 取消缩放
                    total_norm = clip_grad_norm_(model.parameters(), grad_clip_val)  # 计算梯度模并进行裁剪
                    
                    scaler.step(optimizer)  # 使用 GradScaler 进行优化步骤
                    scaler.update()
                    optimizer.zero_grad()  # 清空梯度
                    if tensorboard_writer is not None:
                        global_step = epoch * num_batches + batch_idx
                        tensorboard_writer.add_scalar('grad_norm', total_norm, global_step)  # 将梯度模写入TensorBoard
                    if args.schedule == "linear_with_warmup":
                        lr_scheduler.step()
        else:
            relation_loss = torch.tensor(0.0, device=device, requires_grad=True)  # 确保需要梯度
            binary_loss = torch.tensor(0.0, device=device, requires_grad=True)  # 确保需要梯度
            loss = relation_loss + binary_loss * 0.1
            scaler.scale(loss).backward() 
        epoch_binary_loss += binary_loss.item() * accumulation_steps  # 修改，记录二分类损失
        epoch_relation_loss += relation_loss.item() * accumulation_steps  # 修改，记录多分类损失
        epoch_loss += loss.item() * accumulation_steps
        
        no_pairs_batches += model.no_pairs_count
        no_results_images += model.no_results_count
        total_images += len(targets)  # 更新总的图片数量
        
        # subject_label_mismatches += criterion.subject_label_mismatch
        # object_label_mismatches += criterion.object_label_mismatch
        # subject_box_mismatches += criterion.subject_box_mismatch
        # object_box_mismatches += criterion.object_box_mismatch
        # total_pairs += criterion.total_pairs
        
        if (batch_idx + 1) % print_freq == 0:
            avg_binary_loss = epoch_binary_loss / (batch_idx + 1)  # 修改，计算平均二分类损失
            avg_relation_loss = epoch_relation_loss / (batch_idx + 1)  # 修改，计算平均多分类损失
            avg_loss = epoch_loss / (batch_idx + 1)
            elapsed_time = time.time() - start_time
            eta = elapsed_time / (batch_idx + 1) * (num_batches - batch_idx - 1)
            progress_bar.set_postfix(total_loss=f"{avg_loss:.4f}", binary_loss=f"{avg_binary_loss:.4f}", relation_loss=f"{avg_relation_loss:.4f}", time=f"{elapsed_time:.2f}s", eta=f"{eta:.2f}s")  # 修改，显示二分类和多分类损失
        
        model.no_pairs_count = 0  # 重置计数器
        model.no_results_count = 0  # 重置计数器
        
        # criterion.subject_label_mismatch = 0
        # criterion.object_label_mismatch = 0
        # criterion.subject_box_mismatch = 0
        # criterion.object_box_mismatch = 0
        # criterion.total_pairs = 0
        
        # 保存训练中间信息到tensorboard
        if tensorboard_writer is not None:
            global_step = epoch * num_batches + batch_idx
            tensorboard_writer.add_scalar('train_total_loss', loss.item(), global_step=global_step)  # 修改，记录总损失到Tensorboard
            tensorboard_writer.add_scalar('train_binary_loss', binary_loss.item(), global_step=global_step)  # 修改，记录二分类损失到Tensorboard
            tensorboard_writer.add_scalar('train_relation_loss', relation_loss.item(), global_step=global_step)  # 修改，记录多分类损失到Tensorboard

    if (batch_idx + 1) % accumulation_steps != 0 and has_non_zero_loss:
        scaler.unscale_(optimizer)  # 取消缩放
        total_norm = clip_grad_norm_(model.parameters(), grad_clip_val)  # 计算梯度模并进行裁剪

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if tensorboard_writer is not None:
            global_step = epoch * num_batches + batch_idx
            tensorboard_writer.add_scalar('grad_norm', total_norm, global_step)  # 将梯度模写入TensorBoard
        if args.schedule == "linear_with_warmup":
            lr_scheduler.step()
    if args.schedule == "multistep":
        lr_scheduler.step()
    # if lr_scheduler is not None:
    #     lr_scheduler.step(epoch_loss  / num_batches)  # 根据当前 epoch 的平均损失更新学习率
    # 打印训练总时间
    elapsed_time = time.time() - start_time
    print(f"Epoch {epoch} completed in {elapsed_time:.2f} seconds. Average total loss: {epoch_loss  / num_batches:.4f}. Average relation loss: {epoch_relation_loss  / num_batches:.4f}. Average binary loss: {epoch_binary_loss  / num_batches:.4f}")
    # print("forward_total_time: ", forward_total_time)
    no_pairs_ratio = no_pairs_batches / num_batches
    no_results_ratio = no_results_images / total_images
    
    # subject_label_mismatch_ratio = subject_label_mismatches / total_pairs if total_pairs else 0
    # object_label_mismatch_ratio = object_label_mismatches / total_pairs if total_pairs else 0
    # subject_box_mismatch_ratio = subject_box_mismatches / total_pairs if total_pairs else 0
    # object_box_mismatch_ratio = object_box_mismatches / total_pairs if total_pairs else 0
    
    print(f"No pairs found ratio: {no_pairs_ratio:.4f}")
    print(f"No results found ratio: {no_results_ratio:.4f}")
    
    # print(f"Subject label mismatch ratio: {subject_label_mismatch_ratio:.4f}")
    # print(f"Object label mismatch ratio: {object_label_mismatch_ratio:.4f}")
    # print(f"Subject box mismatch ratio: {subject_box_mismatch_ratio:.4f}")
    # print(f"Object box mismatch ratio: {object_box_mismatch_ratio:.4f}")

    if tensorboard_writer is not None:
        tensorboard_writer.add_scalar('no_pairs_ratio', no_pairs_ratio, epoch)
        tensorboard_writer.add_scalar('no_results_ratio', no_results_ratio, epoch)
        
        # tensorboard_writer.add_scalar('subject_label_mismatch_ratio', subject_label_mismatch_ratio, epoch)
        # tensorboard_writer.add_scalar('object_label_mismatch_ratio', object_label_mismatch_ratio, epoch)
        # tensorboard_writer.add_scalar('subject_box_mismatch_ratio', subject_box_mismatch_ratio, epoch)
        # tensorboard_writer.add_scalar('object_box_mismatch_ratio', object_box_mismatch_ratio, epoch)
    return epoch_loss / num_batches, epoch_relation_loss / num_batches, epoch_binary_loss  / num_batches


@torch.no_grad()
def evaluate_hoi(dataset_file, model, postprocessors, data_loader, subject_category_id, device, args, criterion=None):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    preds = []
    gts = []
    indices = []
    # filename_list = []
    print_freq = 500
    
    #debug
    # input = []
    
    for samples, targets, rois_tensor, additional_info, detection_counts, detections in metric_logger.log_every(data_loader, print_freq, header):
        # targets: tuple, len(tuple) = batch_size
        #          element in tuple: a dict, whose keys are ['orig_size', 'size', 'boxes', 'labels', 'id', 'hois']
                
        # print(targets[0]['orig_size'])
        # print(targets[0]['size'])
        # print('')
        samples = samples.to(device)

        # input.append((samples, targets, detections))
        
        outputs = model(samples, rois_tensor, additional_info, detection_counts)
        # print("outputs: ", outputs)
        # loss = criterion(outputs, targets)
        # print("loss: ", loss)
        results = postprocessors(outputs, detections)
        # print(results)
        # print(len(list(itertools.chain.from_iterable(utils.all_gather(results)))))
        # print(list(itertools.chain.from_iterable(utils.all_gather(results)))[0])
        
        preds.extend(results)
        gts.extend(copy.deepcopy(targets))
        
        # preds: merge predicted batch data
        # preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results)))) #用于多GPU训练
        # For avoiding a runtime error, the copy is used
        # gts: merge ground truth batch data
        # gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets))))) #用于多GPU训练
            
            # break
            # # Add for evaluation
            # filename_list += [t['filename'] for t in targets]

    # gather the stats from all processes
    # metric_logger.synchronize_between_processes() # 用于多GPU训练

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

        # # 假设 preds[0]['verb_scores'] 是 PyTorch 张量
        logits = preds[0]['verb_scores']
        # verb_labels = torch.zeros((1,117), device='cuda:0')
        # verb_labels[0][gts[0]['hois'][:, 2]] = 1

        # print("One-hot encoded verb_labels:", verb_labels)
        # print("verb_labels shape: ", verb_labels.shape)
        # print("logits shape: ", logits.shape)
        # try:
        #     focal_loss = criterion.focal_loss(logits, verb_labels)
        # except Exception as e:
        #     focal_loss = None
        #     print(f"Error calculating focal loss: {e}")

        # try:
        #     bce_loss = criterion.bce_loss(logits, verb_labels)
        # except Exception as e:
        #     bce_loss = None
        #     print(f"Error calculating bce loss: {e}")

        # if focal_loss is not None:
        #     print("focal loss: ", focal_loss)

        # if bce_loss is not None:
        #     print("bce loss: ", bce_loss)
        # 转换 logits 为概率
        # probabilities = F.softmax(logits, dim=1)

        sorted_probabilities, sorted_indices = torch.sort(logits, dim=1, descending=True)
        num_hois = len(gts[0]['hois'])
        top_probabilities = sorted_probabilities[:, :num_hois]
        top_labels = sorted_indices[:, :num_hois]
        # 获取每个样本的预测标签索引
        # max_probabilities, predicted_labels = torch.max(probabilities, dim=1)

        print("-----------------")
        print("preds[0]: ", preds[0])
        print("预测的概率: ", logits)
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

@torch.no_grad()
def evaluate_hoi_single(model, postprocessors, data_loader, subject_category_id, device, args, threshold=0.5):
    model.eval()
    low_mAP_info = []

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    for samples, targets, rois_tensor, additional_info, detection_counts, detections in metric_logger.log_every(data_loader, 500, header):
        samples = samples.to(device)
        outputs = model(samples, rois_tensor, additional_info, detection_counts)
        
        results = postprocessors(outputs, detections)
        preds = results
        gts = copy.deepcopy(targets)

        # Evaluate using HICOEvaluator
        evaluator = HICOEvaluator(preds, gts, subject_category_id, data_loader.dataset.rare_triplets,
                                  data_loader.dataset.non_rare_triplets, data_loader.dataset.correct_mat, args)
        stats = evaluator.evaluate()
        
        # Check mAP and save low mAP info
        if stats['mAP'] < threshold:
            log_info = {
                    'ground_truth': tensor_to_list(gts[0]),
                    'stats': stats,
                }
            if preds[0]:
                logits = preds[0]['verb_scores']
                sorted_probabilities, sorted_indices = torch.sort(logits, dim=1, descending=True)
                num_hois = len(gts[0]['hois'])
                top_probabilities = sorted_probabilities[:, :num_hois]
                top_labels = sorted_indices[:, :num_hois]
                pred_info = {
                    'labels': tensor_to_list(preds[0]['labels']),
                    'boxes': tensor_to_list(preds[0]['boxes']),
                    'verb_scores': tensor_to_list(preds[0]['verb_scores']),
                    'sub_ids': tensor_to_list(preds[0]['sub_ids']),
                    'obj_ids': tensor_to_list(preds[0]['obj_ids']),
                    'top_probabilities': tensor_to_list(top_probabilities),
                    'top_labels': tensor_to_list(top_labels),
                }
                log_info.update({
                    'prediction': pred_info
                })
                low_mAP_info.append(log_info)

    # Save low mAP info to JSON
    low_mAP_file = os.path.join(args.output_dir, "low_mAP_info.json")
    with open(low_mAP_file, 'w') as f:
        json.dump(low_mAP_info, f, indent=4)


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

