import torch

checkpoint_filename = '/bd_byt4090i1/users/clin/relation/results/denoised_dinov2/position_encoding_HW/bi_attention/all/attention_layers1/clip_grad_01/multistep/dropout01_batch64_gamma01_alpha05/debug_info.pth'
checkpoint = torch.load(checkpoint_filename)

outputs = checkpoint['outputs']
targets = checkpoint['targets']
cost_matrix = checkpoint['cost_matrix']
cost_obj_class = checkpoint['cost_obj_class']
cost_verb_class = checkpoint['cost_verb_class']
cost_bbox = checkpoint['cost_bbox']
cost_giou = checkpoint['cost_giou']

print("cost_matrix: ", cost_matrix)
print("cost_obj_class: ", cost_obj_class)
print("cost_verb_class: ", cost_verb_class)
print("cost_bbox: ", cost_bbox)
print("cost_giou: ", cost_giou)

print("outputs: ", outputs)
# 检查outputs中的所有键是否包含nan
nan_found = False
for i, output_list in enumerate(outputs):
    print(f"Checking output list {i}")
    for j, output in enumerate(output_list):
        print(f"Checking output {j} in list {i}")
        print(f"Type of output: {type(output)}")
        for key, value in output.items():
            print(f"Checking key: {key}")
            if isinstance(value, torch.Tensor):
                if torch.isnan(value).any():
                    nan_found = True
                    print(f"Output {i}-{j} has NaN in {key}")
                    break
            elif isinstance(value, list):
                for v in value:
                    if isinstance(v, torch.Tensor) and torch.isnan(v).any():
                        nan_found = True
                        print(f"Output {i}-{j} has NaN in list under key {key}")
                        break
            # 处理嵌套的情况
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor) and torch.isnan(sub_value).any():
                        nan_found = True
                        print(f"Output {i}-{j} has NaN in {key}.{sub_key}")
                        break
            if nan_found:
                break
        if nan_found:
            break
    if nan_found:
        break

if not nan_found:
    print("No NaN found in any key of outputs")
# nan_found = False
# for i, output in enumerate(outputs):
#     print(f"Type of output: {type(output)}")
#     relation_score = output['relation_score']
#     if torch.isnan(relation_score).any():
#         nan_found = True
#         print(f"Output {i} has NaN in relation_score")
#         break

# if not nan_found:
#     print("No NaN found in any relation_score")
# print("Outputs:", outputs)
# print("Targets:", targets)
# print("Cost Matrix:", cost_matrix)
