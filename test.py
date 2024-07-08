import os
import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
import time
import argparse
import numpy as np

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models.denoise_backbone import DenoisingVitBackbone


def generate_random_image(size=(518, 518)):
    return Image.fromarray(np.random.randint(0, 255, (*size, 3), dtype=np.uint8))


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the backbone
    backbone = DenoisingVitBackbone(
        model_type="vit_base_patch14_dinov2.lvd142m",
        device=device,
        checkpoint_path=args.pretrained_backbone,
        denoised=True,
        train_backbone=False
    )
    backbone.eval()

    # Prepare image transformation
    transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.518, 0.225]),
    ])

    # Generate 10 random images and measure inference time
    total_time = 0
    num_images = 10

    for _ in range(num_images):
        img = generate_random_image()
        img_tensor = transform(img).unsqueeze(0).to(device)

        start_time = time.time()
        with torch.no_grad():
            _ = backbone(img_tensor)
        end_time = time.time()

        total_time += end_time - start_time

    avg_time = total_time / num_images
    print(f"Average inference time over {num_images} images: {avg_time:.4f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test DenoisingVitBackbone average inference time")
    parser.add_argument("--pretrained_backbone", type=str, required=True, help="Path to pretrained backbone checkpoint")
    args = parser.parse_args()
    main(args)