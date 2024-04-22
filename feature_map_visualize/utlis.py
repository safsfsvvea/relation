from torch import Tensor
from vitWrapper import ViTWrapper
from denoiser import Denoiser
from visualize_tools import get_pca_map, get_cluster_map
import torch

def get_denoised_features(img: Tensor, model:Denoiser, img_size:tuple, num_clusters=5) -> Tensor:
    with torch.no_grad():
        features = model.forward(img,return_dict=True)
    raw_features = features['raw_vit_feats']
    denoised_features = features['pred_denoised_feats']
    pca = get_pca_map(raw_features,img_size=img_size)
    d_pca = get_pca_map(denoised_features,img_size=img_size)
    kmeans = get_cluster_map(raw_features, img_size=img_size,num_clusters=num_clusters)
    d_kmeans = get_cluster_map(denoised_features, img_size=img_size,num_clusters=num_clusters)
    out = torch.cat((img[0].cpu(),torch.from_numpy(pca).permute(2,0,1),torch.from_numpy(d_pca).permute(2,0,1),torch.from_numpy(kmeans).permute(2,0,1),torch.from_numpy(d_kmeans).permute(2,0,1)),dim=2)
    return out

def get_features(img: Tensor, vit:ViTWrapper, img_size:tuple, num_clusters=5) -> Tensor:
    with torch.no_grad():
        vit_outputs = vit.get_intermediate_layers(
            img,
            n=[vit.last_layer_index],
            reshape=True,
            return_prefix_tokens=False,
            return_class_token=False,
            norm=True,
        )
        raw_features = vit_outputs[0].permute(0, 2, 3, 1).detach()
    pca = get_pca_map(raw_features,img_size=img_size)
    # d_pca = get_pca_map(denoised_features,img_size=img_size)
    kmeans = get_cluster_map(raw_features, img_size=img_size,num_clusters=num_clusters)
    # d_kmeans = get_cluster_map(denoised_features, img_size=img_size,num_clusters=num_clusters)
    out = torch.cat((img[0].cpu(),torch.from_numpy(pca).permute(2,0,1),torch.from_numpy(kmeans).permute(2,0,1)),dim=2)
    return out

