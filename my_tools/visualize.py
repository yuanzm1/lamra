import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# 设置图表样式
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

def load_evaluation_results(save_dir, save_name):
    """加载评估结果文件"""
    results = {}
    
    # 加载查询和候选结果
    with open(f"{save_dir}/{save_name}_query_names.json", 'r') as f:
        results['query_names'] = json.load(f)
    
    with open(f"{save_dir}/{save_name}_cand_names.json", 'r') as f:
        results['cand_names'] = json.load(f)
    
    with open(f"{save_dir}/{save_name}_scores.json", 'r') as f:
        results['scores'] = json.load(f)
    
    return results

def load_qrels(qrels_path):
    """加载相关性判断数据"""
    qrel = {}
    with open(qrels_path, "r") as f:
        for line in f:
            query_id, _, doc_id, relevance_score, _ = line.strip().split()
            if int(relevance_score) > 0:
                if query_id not in qrel:
                    qrel[query_id] = []
                qrel[query_id].append(doc_id)
    return qrel

def analyze_errors(results, qrel, k=10):
    """分析检索错误结果"""
    errors = {
        'total': 0,
        'no_relevant_in_top_k': [],  # 在top k中没有相关文档
        'partial_match': [],         # 有部分相关但不完整
        'queries': results['query_names'],
        'error_types': defaultdict(list)
    }
    
    # 计算每个查询的错误情况
    for i, query_name in enumerate(results['query_names']):
        if query_name not in qrel:
            continue
            
        relevant_docs = set(qrel[query_name])
        retrieved_docs = set(results['cand_names'][i][:k])
        
        # 计算匹配的相关文档数
        matched = len(relevant_docs & retrieved_docs)
        
        if matched == 0:
            # 完全错误：top k中没有相关文档
            errors['total'] += 1
            errors['no_relevant_in_top_k'].append({
                'query': query_name,
                'relevant': list(relevant_docs),
                'retrieved': results['cand_names'][i][:k],
                'scores': results['scores'][i][:k]
            })
            errors['error_types']['complete_error'].append(query_name)
        elif matched < len(relevant_docs):
            # 部分匹配：有相关但不完整
            errors['partial_match'].append({
                'query': query_name,
                'matched': list(relevant_docs & retrieved_docs),
                'missed': list(relevant_docs - retrieved_docs),
                'retrieved': results['cand_names'][i][:k]
            })
            errors['error_types']['partial_error'].append(query_name)
    
    return errors

def visualize_error_distribution(errors, save_path):
    """可视化错误分布"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 错误类型分布
    error_counts = {k: len(v) for k, v in errors['error_types'].items()}
    error_counts['no_error'] = len(errors['queries']) - sum(error_counts.values())
    
    ax.bar(error_counts.keys(), error_counts.values(), color=['#ff9999','#66b3ff','#99ff99'])
    
    plt.title('Distribution of Retrieval Error Types')
    plt.ylabel('Number of Queries')
    plt.xticks(rotation=45)
    
    # 在柱形上添加数值
    for i, v in enumerate(error_counts.values()):
        ax.text(i, v + 5, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/error_distribution.png")
    plt.close()

def visualize_error_scores(errors, results, save_path):
    """可视化错误案例的分数分布"""
    if not errors['no_relevant_in_top_k']:
        return
        
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 收集错误案例的分数
    error_scores = []
    for error in errors['no_relevant_in_top_k']:
        query_idx = results['query_names'].index(error['query'])
        error_scores.extend(results['scores'][query_idx][:10])
    
    # 收集随机正确案例的分数作为对比
    correct_scores = []
    correct_queries = [q for q in results['query_names'] 
                      if q not in errors['error_types']['complete_error'] 
                      and q not in errors['error_types']['partial_error']]
    
    if correct_queries:
        sample_size = min(50, len(correct_queries))
        for q in np.random.choice(correct_queries, sample_size, replace=False):
            query_idx = results['query_names'].index(q)
            correct_scores.extend(results['scores'][query_idx][:10])
    
    # 绘制分数分布
    sns.histplot(error_scores, kde=True, label='Error Cases', color='red', alpha=0.5)
    if correct_scores:
        sns.histplot(correct_scores, kde=True, label='Correct Cases', color='green', alpha=0.5)
    
    plt.title('Distribution of Retrieval Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_path}/score_distribution.png")
    plt.close()

def save_error_examples(errors, save_path, top_n=20):
    """保存错误案例示例"""
    with open(f"{save_path}/error_examples.txt", 'w', encoding='utf-8') as f:
        f.write(f"=== Complete Error Cases (Top {top_n}) ===\n\n")
        for i, error in enumerate(errors['no_relevant_in_top_k'][:top_n]):
            f.write(f"Case {i+1}: Query {error['query']}\n")
            f.write(f"Relevant documents: {', '.join(error['relevant'])}\n")
            f.write("Retrieved results (Top 10):\n")
            for doc, score in zip(error['retrieved'][:10], error['scores'][:10]):
                f.write(f"  {doc} (Score: {score:.4f})\n")
            f.write("\n")
        
        f.write(f"\n=== Partial Error Cases (Top {top_n}) ===\n\n")
        for i, error in enumerate(errors['partial_match'][:top_n]):
            f.write(f"Case {i+1}: Query {error['query']}\n")
            f.write(f"Matched relevant documents: {', '.join(error['matched'])}\n")
            f.write(f"Missed relevant documents: {', '.join(error['missed'])}\n")
            f.write("\n")

def visualize_retrieval_errors(model_name, qrels_path, save_dir="./LamRA_Ret_eval_results", k=10):
    """主函数：可视化检索错误结果"""
    # 构建保存文件名
    save_name = qrels_path.split('/')[-1].replace('_qrels.txt', '_')
    save_name = f"{save_name}"
    
    # 创建可视化结果目录
    vis_dir = f"{save_dir}/visualization_{model_name}"
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # 加载数据
    print("Loading evaluation results...")
    results = load_evaluation_results(save_dir, save_name)
    qrel = load_qrels(qrels_path)
    
    # 分析错误
    print("Analyzing error cases...")
    errors = analyze_errors(results, qrel, k)
    
    # 输出基本统计信息
    total_queries = len(results['query_names'])
    error_rate = errors['total'] / total_queries * 100 if total_queries > 0 else 0
    
    print(f"Total queries: {total_queries}")
    print(f"Complete errors: {len(errors['no_relevant_in_top_k'])} ({len(errors['no_relevant_in_top_k'])/total_queries*100:.2f}%)")
    print(f"Partial errors: {len(errors['partial_match'])} ({len(errors['partial_match'])/total_queries*100:.2f}%)")
    
    # 生成可视化
    print("Generating visualizations...")
    visualize_error_distribution(errors, vis_dir)
    visualize_error_scores(errors, results, vis_dir)
    save_error_examples(errors, vis_dir)
    
    print(f"Visualization results saved to: {vis_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='qwen2-vl-2b_LamRA-Ret_mini', help='Name of the model')
    parser.add_argument('--qrels_path', type=str, default='/mnt/disk2/yuanzm/dataset/lamra_data/M-BEIR/qrels/test/mbeir_mscoco_task3_test_qrels.txt', help='Path to qrels file')
    parser.add_argument('--save_dir', type=str, default="/home/yuanzm/LamRA/LamRA_Ret_eval_results", help='Directory for saving results')
    parser.add_argument('--k', type=int, default=10, help='Top k value for evaluation')
    
    args = parser.parse_args()
    visualize_retrieval_errors(args.model_name, args.qrels_path, args.save_dir, args.k)
    