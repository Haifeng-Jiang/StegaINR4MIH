# -*- coding:utf-8 -*-
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # sovle the multiple copies of libiomp5md.dll problem

import torch
# %matplotlib inline
from models.function_representation import load_function_representation

import torch
import torch.nn.utils.prune as prune

device = torch.device('cpu')

model_file = "./2024-06-21_23-40-13_celebahq128_experiment_INR_prune/inr_model_50000.pt"

print("Load model:", model_file)
function_representation_inr = load_function_representation(device, model_file)  # one model at least

##################################################
###########  在这里剪枝训练后的模型   ##############
###########      训练后剪枝方法         ###########
method = 'l1_us'

# method可以选择ln_s，ln结构化剪枝
# 可以选择l1_us, l1非结构化剪枝

if method == 'ln_s':
    prune_rate = 0.2
    structure_prune_channel = 0

    for name, module in function_representation_inr.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.ln_structured(module, name='weight', amount=prune_rate, n=2, dim=structure_prune_channel)
            module.weight = module.weight_orig * module.weight_mask

elif method == 'l1_us':
    prune_rate = 0.3

    for name, module in function_representation_inr.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=prune_rate)
            module.weight = module.weight_orig * module.weight_mask

##################################################
##################################################
########   新模型命名是原模型名字加上剪枝方法  ######
#########      保存在原模型的文件夹中   ############
state_dict = function_representation_inr.state_dict()
for name, module in function_representation_inr.named_modules():
    if isinstance(module, torch.nn.Linear):
        if name + '.weight_orig' in state_dict and name + '.weight_mask' in state_dict:
            state_dict[name + '.weight'] = state_dict[name + '.weight_orig'] * state_dict[name + '.weight_mask']
            del state_dict[name + '.weight_orig']
            del state_dict[name + '.weight_mask']
            print(name)

torch.save({'config': function_representation_inr._get_config(),
            'state_dict': state_dict}, model_file.replace('.pt', f"_{method}.pt"))