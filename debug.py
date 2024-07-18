import torch

checkpoint_filename = os.path.join('your_output_dir', 'debug_info.pth')
checkpoint = torch.load(checkpoint_filename)

outputs = checkpoint['outputs']
targets = checkpoint['targets']
cost_matrix = checkpoint['cost_matrix']

print("Outputs:", outputs)
print("Targets:", targets)
print("Cost Matrix:", cost_matrix)
