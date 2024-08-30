import torch
import torchvision.transforms as T
from PIL import Image
from torch import nn
from util.misc import NestedTensor
import torch.nn.functional as F

class DINOv2Backbone(nn.Module):
    def __init__(self, model_type, patch_size=16, pretrained_path=None, num_feature_levels=1, *args, **kwargs):
        super().__init__()
        self.model_type = model_type
        self.patch_size = patch_size
        self.num_feature_levels = num_feature_levels
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(pretrained_path)

    def load_model(self, pretrained_path):
        # 加载 DINOv2 模型
        model = torch.hub.load('../dinov2', 'dinov2_vitb14_reg', source='local', pretrained=False)
        if pretrained_path:
            model.load_state_dict(torch.load(pretrained_path))
            print('Loading DINOv2 from localdir...')
        model.to(self.device)
        model.eval()
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
        with torch.no_grad():
            intermediate_layers = self.model.get_intermediate_layers(img, self.num_feature_levels)
            features = {}
            for i, x in enumerate(intermediate_layers):
                x = x.permute(0, 2, 1)
                x = x.view(x.shape[0], -1, self.new_height // 14, self.new_width // 14)
                features[f'layer_{len(intermediate_layers) - i}'] = x
        return features

    def extract_features(self, img_path):
        img = Image.open(img_path).convert('RGB')
        features, scales = self(img) # need modify
        return features, scales

# 示例用法
# if __name__ == "__main__":
#     backbone = DINOv2Backbone(
#         model_type="dinov2_vitb14_reg",
#         patch_size=14,
#         pretrained_path='../RLIPv2/checkpoints/DINOv2/dinov2_vitb14_reg4_pretrain.pth'
#     )
#     features, scales = backbone.extract_features('/path/to/image.jpg')
#     print("提取的特征:", features.shape, "缩放比例:", scales)
