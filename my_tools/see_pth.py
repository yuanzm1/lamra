import torch
import pdb

# 加载 .pt 文件
model_states_path = "/home/yuanzm/LamRA/checkpoints/qwen2-vl-2b_LamRA_Ret_Pretrain2/checkpoint-7656/global_step7656/mp_rank_00_model_states.pt"  # 替换为实际的文件路径
model_states = torch.load(model_states_path, map_location="cpu")
# pdb.set_trace()

# 存储找到的keys
matching_keys = []
for key in model_states['module'].keys():
    if "lm_head" in key: #"mlp.0.bias"
        matching_keys.append(key)

if matching_keys:
    print("找到名字中包含'mlp.0.bias'的keys：")
    for key in matching_keys:
        print(key)
else:
    print("未找到名字中包含'mlp.0.bias'的keys")
