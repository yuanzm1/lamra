import torch
import torch.nn as nn
# # 复用之前定义的加载MLP参数的函数
def load_mlp_parameters(model, load_path):
    """加载单独保存的mlp参数并赋值给模型"""
    mlp_params = torch.load(load_path, map_location='cuda')
    loaded_count = 0
    
    for name, param in model.named_parameters():
        if 'modify' in name:
            new_name = 'base_model.model.' + name
            param.data.copy_(mlp_params[new_name])
            loaded_count += 1
    # if loaded_count == 0:
    #     raise ValueError("未找到匹配的MLP参数")
    print(f"[INFO] Loaded {loaded_count} MLP params from {load_path}")
    return loaded_count

# def load_mlp_parameters(model, load_path):
#     """加载单独保存的mlp参数并赋值给模型"""
#     mlp_params = torch.load(load_path, map_location='cpu')  # 先用 CPU 安全加载
#     state_dict_keys = set(mlp_params.keys())

#     loaded_count = 0
#     for name, param in model.named_parameters():
#         if 'modify' in name:
#             # 原始 key
#             new_name = 'base_model.model.' + name

#             # 兼容 DataParallel / DDP 的 key
#             if new_name not in state_dict_keys and ("module." + new_name) in state_dict_keys:
#                 new_name = "module." + new_name

#             if new_name in mlp_params:
#                 param.data.copy_(mlp_params[new_name].to(param.device))
#                 loaded_count += 1
#             else:
#                 print(f"[WARN] Key {new_name} not found in checkpoint.")
#     print(f"[INFO] Loaded {loaded_count} MLP params from {load_path}")
#     return loaded_count

def convert_meta_tensors(model, checkpoint=None, device='cpu'):
    """
    将模型中的所有 meta tensor 转换为实体参数
    
    参数:
        model: 待处理的模型
        checkpoint: 可选，包含参数权重的字典（如 torch.load 加载的 state_dict）
        device: 实体参数的目标设备（如 'cpu' 或 'cuda:0'）
    """
    # 遍历模型所有参数
    for name, param in model.named_parameters():
        if param.device.type == 'meta':  # 检测 meta tensor
            print(f"处理 meta tensor: {name}")
            # 1. 创建同形状、同 dtype 的实体张量（分配内存）
            # 注意：必须指定 device，meta tensor 无实际设备
            new_param = torch.empty(
                param.shape,
                dtype=param.dtype,
                device=device
            )
            # 2. 填充数据（优先从 checkpoint 加载，否则初始化）
            if checkpoint is not None and name in checkpoint:
                # 从 checkpoint 加载（确保设备匹配）
                new_param.data.copy_(checkpoint[name].to(device))
                print(f"  从 checkpoint 加载参数: {name}")
            else:
                # 无 checkpoint 时，根据参数类型初始化
                if 'embedding' in name.lower():
                    # 嵌入层：正态分布初始化（参考 transformers 标准逻辑）
                    nn.init.normal_(new_param, mean=0.0, std=model.config.initializer_range)
                elif 'linear' in name.lower() or 'mlp' in name.lower():
                    # 线性层/MLP：Xavier 均匀初始化
                    nn.init.xavier_uniform_(new_param)
                elif 'bias' in name:
                    # 偏置项：零初始化
                    nn.init.zeros_(new_param)
                print(f"  初始化参数: {name}（无 checkpoint 权重）")
            
            # 3. 替换模型中的 meta tensor
            param.data = new_param.data
            
    # 检查是否还有残留的 meta tensor
    remaining_meta = [name for name, param in model.named_parameters() if param.device.type == 'meta']
    if remaining_meta:
        print(f"警告：仍有未处理的 meta tensor: {remaining_meta}")
    else:
        print("所有 meta tensor 已转换为实体参数")
        
def mbeir_get_mode(data_path):
    mode = ''
    if "mscoco_task3" in data_path:
        mode = 'image->text'
    elif "mscoco_task0" in data_path:
        mode = 'text->image'
    elif "cirr_task7" in data_path:
        mode = 'image+text->image'
    elif "fashioniq_task7" in data_path:
        mode = 'image+text->image'
    elif "webqa_task1" in data_path:
        mode = 'text->text'
    elif "nights_task4" in data_path:
        mode = 'image->image'
    elif "oven_task6" in data_path:
        mode = 'image+text->text'
    elif "infoseek_task6" in data_path:
        mode = 'image+text->text'
    elif "fashion200k_task0" in data_path:
        mode = 'text->image'
    elif "visualnews_task0" in data_path:
        mode = 'text->image'
    elif "webqa_task2" in data_path:
        mode = 'text->image+text'
    elif "oven_task8" in data_path:
        mode = 'image+text->image+text'
    elif "infoseek_task8" in data_path:
        mode = 'image+text->image+text'
    elif "fashion200k_task3" in data_path:
        mode = 'image->text'
    elif "visualnews_task3" in data_path:
        mode = 'image->text'
    elif "edis_task2" in data_path:
        mode = 'text->image+text'
    else:
        raise NotImplementedError
    return mode