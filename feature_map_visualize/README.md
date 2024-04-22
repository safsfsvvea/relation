Run the following command to see the sample.
```
python feature_extract.py --input_path fig --output_dir feat --denoised
```

The runtime depends on the selected model and the input image size. Typically, it takes about 0.3s to process a image of size 518x518 with model`vit_base_patch14_dinov2.lvd142m`.


| Argument | Description |
| ------ | ------ |
| `-t <model type>`, `--vit_type <model type>` | Specify the model type. Default as `vit_base_patch14_dinov2.lvd142m`  |
| `-d`, `--denoised` | Use `Denoised-Vit` to denoise the feature |
| `-o <output path>`, `--output <output path>` | Specify the output folder. Default as the `input path` |
| `-i <input path>`, `--input <input path>` | Specify the input folder / file |
| `-r <size>/<width, height>`, `--resize <size>/<width, height>` | Specify the size of output images. Default as the original size of image |
| `-n <# of clusters>`, `--n_clusters <# of clusters>` | Specify the number of clusters |

Notice that `--denoised` flag only supports `vit_base_patch16_224.dino` and `vit_base_patch14_dinov2.lvd142m`, and the output size will be fixed as 512x512 and 518x518 respectively.

| Supported Models | Description | # of params |
| ------ | ------ | ------ |
| `vit_small_patch8_224.dino` | ViT-S/8| 21M|
| `vit_small_patch16_224.dino`| ViT-S/16 |21M|
| `vit_base_patch8_224.dino`| ViT-B/8 |85M|
| `vit_base_patch16_224.dino`| ViT-B/16 |85M|
| `vit_small_patch14_dinov2.lvd142m`| ViT-S/14 distilled |21 M|
| `vit_base_patch14_dinov2.lvd142m`| ViT-B/14 distilled |86 M|
| `vit_large_patch14_dinov2.lvd142m`| ViT-L/14 distilled |300 M|
| `vit_giant_patch14_dinov2.lvd142m`| ViT-g/14 |1,100 M|
| `vit_small_patch14_reg4_dinov2.lvd142m`| ViT-S/14 distilled with registers|21 M|
| `vit_base_patch14_reg4_dinov2.lvd142m`| ViT-B/14 distilled with registers|86 M|
| `vit_large_patch14_reg4_dinov2.lvd142m`| ViT-L/14 distilled with registers|300 M|
| `vit_giant_patch14_reg4_dinov2.lvd142m`| ViT-g/14 with registers|1,100 M|

All models are put under the folder `/bd_byta6000i0/users/dataset/feat_visualize_models`.

Or you can refer to the `visualize.ipynb` to see how do the functions works.