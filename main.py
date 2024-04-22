from model.backbone import DenoisingVitBackbone

def main():
    backbone = DenoisingVitBackbone(
        model_type="vit_base_patch14_dinov2.lvd142m",
        denoised=True
    )
    print(backbone)
    img_path = '/cluster/home/clin/clin/feature_map_visualize/img/HICO_test2015_00000001.jpg'
    denoised_features, raw_features, scales = backbone.extract_features(img_path)
    print("denoised_features:", denoised_features.shape)
    print("raw_features:", raw_features.shape)
    print("scales:", scales)
if __name__ == "__main__":
    main()
