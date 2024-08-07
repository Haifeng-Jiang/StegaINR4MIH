
import torch
import math
import torch
import torch.nn as nn
import copy

def generate_sparse_mask(model, sparse_ratio):
   
    return weight_magnitude_based_sc(model, sparse_ratio, 'lwf')
    
def weight_magnitude_based_sc(model, sr, sc):
    '''
    weight_magnitude_based_selection_criteria.
    '''
    compare = torch.gt if sc == 'lwf' else torch.lt
    sr = sr if sc == 'lwf' else (1-sr)
    
    sparse_masks = []
    for k, m in list(model.named_modules()):
        # if isinstance(m, nn.Linear) or (isinstance(m, nn.Conv2d) and 'shortcut' not in k):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            weight_copy = m.weight.data.abs().clone()
            n = int(torch.prod(torch.as_tensor(weight_copy.shape))*sr)
            
            sorted_values, _ = torch.sort(weight_copy.view(-1), descending=True)

            threshold = sorted_values[n-1]
            
            mask = compare(weight_copy, threshold).float()
            sparse_masks.append(mask)
    return sparse_masks

def random_prune_and_store(func_inr, sr=0.05, random_seed=None):
    if random_seed is not None:
        torch.manual_seed(random_seed)
    device = 'cpu'
    sparse_masks = []
    fixed = []
    i = 0
    for m in func_inr.modules():
        if isinstance(m, nn.Linear):
            weight = m.weight.data.clone()
            n = int(weight.numel()*sr)

            indices = torch.randperm(weight.numel())[:n]

            mask = torch.zeros_like(weight).view(-1)
            mask[indices] = 1.
            mask = mask.view(weight.shape)
            mask = mask == 1.

            sparse_masks.append(mask)

            # 保留被插入的部分, 以便恢复原模型
            indices = torch.stack(torch.where(mask))
            values = weight[mask]
            fixed.append(torch.sparse.FloatTensor(indices, values, weight.size()).coalesce() )
            
            i += 1
    return sparse_masks, fixed


def prune_and_store(func_inr, sr=0.05):
    # 前5%最大响应mask
    device = 'cpu'
    sparse_masks = generate_sparse_mask(func_inr, sr)

    fixed = []
    i = 0
    for m in func_inr.modules():
        if isinstance(m, nn.Linear):
            weight = m.weight.data.clone()
            
            mask = sparse_masks[i] == 1
            # 保留被插入的部分, 以便恢复原模型
            indices = torch.stack(torch.where(mask))
            values = weight[mask]
            fixed.append(torch.sparse.FloatTensor(indices, values, weight.size()).coalesce() )
            
            i += 1
    return sparse_masks, fixed

def prune(func_inr, sr=0.05):
    # 前5%最大响应mask
    device = 'cpu'
    sparse_masks = generate_sparse_mask(func_inr, sr)

    i = 0
    for m in func_inr.modules():
        if isinstance(m, nn.Linear):
            weight = m.weight.data.clone()
            
            mask = sparse_masks[i] == 1
                    
            i += 1
    return sparse_masks


def store(func_inr, sparse_masks):
    # 前5%最大响应mask
    device = 'cpu'
    fixed = []
    i = 0
    for m in func_inr.modules():
        if isinstance(m, nn.Linear):
            weight = m.weight.data.clone()
            
            mask = sparse_masks[i] == 1
            # 保留被插入的部分, 以便恢复原模型
            indices = torch.stack(torch.where(mask))
            values = weight[mask]
            fixed.append(torch.sparse.FloatTensor(indices, values, weight.size()).coalesce() )
                    
            i += 1
    return fixed

def reverse_mask(masks):
    # mask: a list contain several mask of conv or bn layer
    tmp = copy.deepcopy(masks)
    for idx in range(len(masks)):
        tmp[idx] = 1. - masks[idx]
    return tmp


def init_weights(model, random_seed=None, init_method='xavier'):
    if random_seed is not None:
        torch.manual_seed(random_seed)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if init_method == 'xavier':
                nn.init.xavier_normal_(m.weight)
            elif init_method == 'kaiming':
                nn.init.kaiming_normal_(m.weight)
            elif init_method == 'uniform':
                nn.init.uniform_(m.weight)
            elif init_method == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight)
            elif init_method == 'kaiming_uniform':
                nn.init.kaiming_uniform_(m.weight)
            elif init_method == 'orthogonal':
                nn.init.orthogonal_(m.weight)

def generate_key(func_inr, sparse_masks, random_seed=42):
    if random_seed != None:
        torch.manual_seed(random_seed)

    key = []
    i = 0
    for m in func_inr.modules():
        if isinstance(m, nn.Linear):
            weight = m.weight.data.clone()
            
            mask = sparse_masks[i] == 1
            # 分布一致性
            random_weight = weight[mask].clone()
            random_weight = random_weight.normal_(weight[mask].mean(), weight[mask].std())

            weight[mask] = random_weight
            
            indices = torch.stack(torch.where(mask))
            values = weight[mask]
            sparse_tensor = torch.sparse.FloatTensor(indices, values, weight.size()).coalesce() 
            # key是稀疏张量
            key.append(sparse_tensor)
            i += 1
    return key


