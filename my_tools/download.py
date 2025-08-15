# # 1.安装huggingface_hub
# # pip install huggingface_hub
# import os
# from huggingface_hub import snapshot_download
 
# # 使用cache_dir参数，将模型/数据集保存到指定“本地路径”
# snapshot_download(repo_id="code-kunkun/LamRA_Eval", repo_type="dataset",
#                   cache_dir="/mnt/disk2/yuanzm/lamra/",
#                   local_dir_use_symlinks=False, resume_download=True,
#                   token='hf_***')


from transformers import AutoModelForCausalLM, AutoTokenizer

# 模型名称
model_name = "Qwen/Qwen2-VL-2B-Instruct"

# 下载并加载模型（首次运行会自动下载）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True  # Qwen模型需要此参数
)

# 下载并加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

# 查看模型缓存路径（默认在~/.cache/huggingface/hub）
print("模型缓存路径：", model.config._name_or_path)
