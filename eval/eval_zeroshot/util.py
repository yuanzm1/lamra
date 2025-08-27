import torch
# 复用之前定义的加载MLP参数的函数
def load_mlp_parameters(model, load_path):
    """加载单独保存的mlp参数并赋值给模型"""
    mlp_params = torch.load(load_path, map_location='cuda:0')
    loaded_count = 0
    
    for name, param in model.named_parameters():
        if 'modify' in name:
            new_name = 'base_model.model.' + name
            param.data.copy_(mlp_params[new_name])
            loaded_count += 1
    # if loaded_count == 0:
    #     raise ValueError("未找到匹配的MLP参数")
    return loaded_count