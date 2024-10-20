import torch
import torchvision.transforms as T
from PIL import Image
from torch import nn
from feature_map_visualize.vitWrapper import ViTWrapper
from feature_map_visualize.denoiser import Denoiser
from util.misc import NestedTensor
import torch.nn.functional as F

class DenoisingVitBackbone(nn.Module):
    def __init__(self, model_type, device, checkpoint_path, train_backbone=False, denoised=True, resize=None):
        super().__init__()
        self.denoised = denoised
        self.model_type = model_type
        self.train_backbone = train_backbone
        self.resize = resize
        self.patch_size = self.get_patch_size(model_type)
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model = self.load_model()

    def get_patch_size(self, model_type):
        import re
        pattern = r"patch(\d+)"
        match = re.search(pattern, model_type)
        return int(match.group(1))

    def load_model(self):
        if self.denoised:
            print("model_type: ", self.model_type)
            vit = ViTWrapper(self.model_type, stride=self.patch_size)
            model = Denoiser(noise_map_height=37, noise_map_width=37, vit=vit, feature_dim=768)
            model.load_state_dict(torch.load(self.checkpoint_path), strict=False)
        else:
            model = ViTWrapper(self.model_type, stride=self.patch_size)
        model.to(self.device)
        if not self.train_backbone:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
        else:
            for param in model.parameters():
                param.requires_grad = True
        return model

    def get_transform(self, img):
        # 如果 resize 参数被设置，使用指定大小，否则使用最接近原图大小的 patch size 的倍数
        if self.resize:
            img_size = (self.resize[0], self.resize[1])
        else:
            _, _, H, W = img.shape
            img_size = (
                (H // self.patch_size) * self.patch_size,
                (W // self.patch_size) * self.patch_size,
            )
        scale_x = img_size[0] / H
        scale_y = img_size[1] / W

        img = F.interpolate(img, size=img_size, mode='bicubic', align_corners=False)
        return img, (scale_x, scale_y)

    def forward(self, img):
        # img = nested_tensor.tensors
        # mask = nested_tensor.mask
        # img, scales = self.get_transform(img)
        img = img.to(self.device)
        features = self.model(img)
        if self.denoised:
            raw_features = features['raw_vit_feats']
            denoised_features = features['pred_denoised_feats']
            CLS_token = features['prefix_tokens']
            return denoised_features, raw_features, CLS_token
        else:
            print("features shape: ", features.shape)
            return features, None, None

    def extract_features(self, img_path):
        img = Image.open(img_path).convert('RGB')
        denoised_features, raw_features = self(img) # need modify
        return denoised_features, raw_features