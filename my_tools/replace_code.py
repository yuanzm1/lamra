import os

def comment_and_replace_code(folder_path, target_code, new_code, comment_symbol='# '):
    """
    处理文件夹下所有Python文件，注释目标代码并添加新代码
    
    参数:
        folder_path: 要处理的文件夹路径
        target_code: 要被注释的目标代码
        new_code: 要添加的新代码
        comment_symbol: 注释符号
    """
    # 遍历文件夹中的所有文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.py'):  # 只处理Python文件
                file_path = os.path.join(root, file)
                
                try:
                    # 读取文件内容
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 检查目标代码是否存在于文件中
                    if target_code in content:
                        # 对目标代码的每一行添加注释符号
                        lines = target_code.split('\n')
                        commented_lines = [f"{comment_symbol}{line}" for line in lines]
                        commented_code = '\n'.join(commented_lines)
                        
                        # 构建替换内容：注释后的代码 + 新代码
                        replacement = f"{commented_code}\n{new_code}"
                        
                        # 执行替换
                        new_content = content.replace(target_code, replacement)
                        
                        # 写回文件
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        
                        print(f"已处理: {file_path}")
                    else:
                        print(f"未找到目标代码: {file_path}")
                        
                except Exception as e:
                    print(f"处理文件 {file_path} 出错: {str(e)}")

if __name__ == "__main__":
    # 配置参数
    FOLDER_PATH = "/home/yuanzm/LamRA/eval/eval_zeroshot/"  # 替换为你的文件夹路径
    TARGET_CODE = """    model = Qwen2VLRetForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
    )"""  # 要注释的目标代码
    
    # 要添加的新代码（根据需要修改）
    NEW_CODE = """    # 使用新的参数加载模型
    model = Qwen2VLRetForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
        device_map="auto"
    )"""
    
    # 执行处理
    comment_and_replace_code(
        folder_path=FOLDER_PATH,
        target_code=TARGET_CODE,
        new_code=NEW_CODE
    )
    
    print("处理完成！")
