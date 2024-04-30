import time
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq=100, tensorboard_writer=None):
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
