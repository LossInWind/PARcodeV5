import torch
from model.VQCNN import VQE
net = VQE()
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_params = count_parameters(net)
print(f"The number of trainable parameters in the model is {num_params}")

