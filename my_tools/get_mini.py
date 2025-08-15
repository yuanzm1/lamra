import json
import random
from collections import defaultdict

def sample_by_qid_prefix(input_file, output_file, sample_ratio=0.1, random_seed=42):
    """
    从JSONL文件中按qid前缀分组，每组抽取指定比例的样本
    
    参数:
        input_file: 输入JSONL文件路径
        output_file: 输出JSONL文件路径
        sample_ratio: 抽样比例，默认10%
        random_seed: 随机种子，保证结果可复现
    """
    # 设置随机种子，确保结果可复现
    random.seed(random_seed)
    
    # 按qid前缀分组存储样本
    groups = defaultdict(list)
    
    # 读取输入文件并分组
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 解析JSON行
            data = json.loads(line.strip())
            # 获取qid并提取前缀（冒号前的部分）
            qid = data.get('qid', '')
            if ':' in qid:
                prefix = qid.split(':', 1)[0]  # 只分割一次，取冒号前的部分
                groups[prefix].append(data)
            else:
                # 处理没有冒号的qid，归入'other'组
                groups['other'].append(data)
    
    # 存储所有选中的样本
    selected_samples = []
    
    # 对每个组进行抽样
    for prefix, samples in groups.items():
        # 计算需要抽取的样本数量
        sample_count = max(1, int(len(samples) * sample_ratio))  # 至少抽取1个样本
        # 随机抽取样本
        selected = random.sample(samples, sample_count)
        selected_samples.extend(selected)
        # 打印分组抽样信息
        print(f"前缀 '{prefix}' 共有 {len(samples)} 个样本，抽取了 {sample_count} 个")
    
    # 将选中的样本写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in selected_samples:
            # 每个样本占一行
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"总抽样完成，共抽取 {len(selected_samples)} 个样本，已保存到 {output_file}")

# 使用示例
if __name__ == "__main__":
    # 输入JSONL文件路径
    input_jsonl = "/mnt/disk2/yuanzm/dataset/lamra_data/M-BEIR/query/union_train/mbeir_union_up_train.jsonl"
    # 输出JSONL文件路径
    output_jsonl = "/mnt/disk2/yuanzm/dataset/lamra_data/M-BEIR/query/union_train/mbeir_union_up_train_mini.jsonl"
    # 执行抽样
    sample_by_qid_prefix(input_jsonl, output_jsonl, sample_ratio=0.1)
