import torch
from PIL import Image
import torchvision.transforms as T
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
import visualize_tools
import numpy as np
from vitWrapper import ViTWrapper
from denoiser import Denoiser

img = Image.open('fig/0000000001.png').convert('RGB')
# single_image_visualize(model,img,1024//14,1280//14)

vit = ViTWrapper('vit_base_patch14_dinov2.lvd142m',stride=14)

model = Denoiser(noise_map_height=37, noise_map_width=37,vit=vit,feature_dim=768)
freevit_model_ckpt = torch.load('/cluster/home/clin/clin/feature_map_visualize/checkpoints/feat_visualize_models/dinov2_v1.pth')["denoiser"]
model.load_state_dict(freevit_model_ckpt,strict=False)

model = model.to('cuda')

print(model)
print("image size: ", img.size)

transform = T.Compose([
    T.Resize((518, 518),T.InterpolationMode.BICUBIC),
    T.CenterCrop((518, 518)),
    T.ToTensor(),
    # T.Normalize(mean=[0.5], std=[0.5]),
    # T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
raw_img = transform(img)[:3].unsqueeze(0)
raw_img = raw_img.to('cuda')

img = transform(img)[:3].unsqueeze(0)
img = img.to('cuda')

features = model.forward(img,return_dict=True)

raw_features = features['raw_vit_feats']
denoised_features = features['pred_denoised_feats']

print("raw_features shape: ", raw_features.shape)
