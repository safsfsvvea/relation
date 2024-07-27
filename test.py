import os
import sys
import torch
import torchvision.transforms as transforms
import time
import argparse
from tqdm import tqdm
from datetime import datetime
from accelerate import Accelerator
import timm
from accelerate.utils import set_seed
from torch.utils.data import Dataset, DataLoader
import gc


class RandomImageDataset(Dataset):
    def __init__(self, size=(518, 518), transform=None):
        self.size = size
        self.transform = transform

    def __len__(self):
        return 1000000  # Large number to ensure we don't run out of data

    def __getitem__(self, idx):
        img = torch.rand(3, *self.size)
        if self.transform:
            img = self.transform(img)
        return img


def log_results(log_file, message):
    print(message)
    with open(log_file, 'a') as f:
        f.write(message + '\n')


def main(args):
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    log_file = f"acce_flash_bf{current_time}.txt"

    log_results(log_file, f"Backbone Timing Test - {current_time}")
    log_results(log_file, f"Batch sizes: {args.batch_sizes}")
    log_results(log_file, f"Pretrained backbone: {args.pretrained_backbone}")
    log_results(log_file, "")
    enable_acce = True
    if enable_acce:
        accelerator = Accelerator()
    set_seed(42)
    if enable_acce:
        device = accelerator.device
    else:
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
    dtype = torch.bfloat16  # flash atten doesn't support fp32.
    log_results(log_file, f"Accelerator Device: {device}")
    if torch.cuda.is_available():
        log_results(log_file, f"CUDA Device: {torch.cuda.get_device_name(device.index)}")

    backbone = timm.create_model("hf_hub:timm/vit_base_patch14_dinov2.lvd142m", pretrained=True)
    backbone.eval()
    if enable_acce:
        accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 4
        backbone = accelerator.prepare(backbone)  # only handles devices, need cast dtype manually
    else:
        backbone = backbone.to(device)
    backbone = backbone.to(dtype)
    # print(f"Accelerator mixed precision: {accelerator.mixed_precision}")

    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.518, 0.225])
    dataset = RandomImageDataset(transform=transform)

    batch_sizes = args.batch_sizes
    results = []

    for batch_size in batch_sizes:
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
        if enable_acce:

            dataloader = accelerator.prepare(dataloader)

        print(f"Backbone dtype: {next(backbone.parameters()).dtype}")

        if args.num_steps != 0:
            num_batches = args.num_steps
        else:
            num_batches = args.total_images // batch_size

        total_images = num_batches * batch_size
        log_results(log_file, f"\nProcessing with batch size {batch_size}")
        log_results(log_file, f"Total images: {total_images}")

        total_time = 0
        pbar = tqdm(enumerate(dataloader), total=num_batches, desc=f"Batch size {batch_size}")
        for i, batch in pbar:
            # batch = batch.to(dtype)
            print(f"batch dtype:{batch.dtype()}")
            if not enable_acce:
                batch = batch.to(device)
            if i >= num_batches:
                break
            if enable_acce:
                accelerator.wait_for_everyone()
            start_time = time.perf_counter()
            flash_attention_config = {
                'enable_flash': True,  # flash attn v2
                'enable_math': False,  # vanilla attn
                'enable_mem_efficient': False
            }

            # Use the context manager to enable Flash Attention
            with torch.backends.cuda.sdp_kernel(**flash_attention_config):
                with torch.no_grad():
                    final_feat, middle_features = backbone.forward_intermediates(batch)
            if enable_acce:
                accelerator.wait_for_everyone()
            end_time = time.perf_counter()

            batch_time = end_time - start_time
            total_time += batch_time
            pbar.set_postfix({"Avg Time/Image": f"{batch_time / batch_size:.6f}s"})
            # torch.cuda.empty_cache() # will increase time cost
            # gc.collect() # will increase time cost

        avg_time_per_image = total_time / total_images
        fps = 1 / avg_time_per_image
        results.append((batch_size, avg_time_per_image, fps))

        log_results(log_file, f"Batch size: {batch_size}")
        log_results(log_file, f"Total time: {total_time:.6f} seconds")
        log_results(log_file, f"Average time per image: {avg_time_per_image:.6f} seconds")
        log_results(log_file, f"FPS: {fps:.2f}")

    log_results(log_file, "\nSummary:")
    for batch_size, avg_time, fps in results:
        log_results(log_file, f"Batch size {batch_size}: {avg_time:.6f} seconds per image, FPS: {fps:.2f}")

    best_batch_size, best_time, best_fps = max(results, key=lambda x: x[2])
    log_results(log_file,
                f"\nBest performance: Batch size {best_batch_size} with {best_time:.6f} seconds per image, FPS: {best_fps:.2f}")


if __name__ == "__main__":
    #
    parser = argparse.ArgumentParser(description="Test backbone average inference time and FPS")
    parser.add_argument("--pretrained_backbone", default="/path/to/pretrained/backbone.pth", type=str,
                        help="Path to pretrained backbone checkpoint")
    parser.add_argument("--total_images", type=int, default=5000, help="Approximate total number of images to process")
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[8, 16, 32, 64, 128, 256],
                        help="Batch sizes to test")
    parser.add_argument("--num_steps", type=int, default=20, help="steps to test")
    args = parser.parse_args()
    main(args)