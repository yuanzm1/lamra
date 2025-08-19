import json
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.gridspec as gridspec
from collections import defaultdict

# 设置图表样式
plt.rcParams["figure.figsize"] = (15, 10)
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

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

def load_query_data(query_data_path):
    """加载查询数据（包含文本和图像信息）"""
    query_data = {}
    with open(query_data_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            query_id = data['qid']
            query_data[query_id] = {
                'text': data.get('query_text', ''),
                'image': data.get('image', None)  # 图像路径或标识符
            }
    return query_data

def load_candidate_data(cand_pool_path):
    """加载候选文档数据（包含文本和图像信息）"""
    candidate_data = {}
    with open(cand_pool_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            doc_id = data['did']
            candidate_data[doc_id] = {
                'text': data.get('doc_text', ''),
                'image': data.get('image', None)  # 图像路径或标识符
            }
    return candidate_data

def analyze_errors(results, qrel, query_data, candidate_data, k=10):
    """分析检索错误结果，包含详细内容信息"""
    errors = {
        'total': 0,
        'no_relevant_in_top_k': [],  # 在top k中没有相关文档
        'partial_match': [],         # 有部分相关但不完整
    }
    
    # 计算每个查询的错误情况
    for i, query_name in enumerate(results['query_names']):
        if query_name not in qrel or query_name not in query_data:
            continue
            
        # 获取查询的详细信息
        query_info = query_data[query_name]
        
        # 获取相关文档和检索文档
        relevant_docs = set(qrel[query_name])
        retrieved_docs = results['cand_names'][i][:k]
        retrieved_scores = results['scores'][i][:k]
        
        # 获取检索文档的详细信息
        retrieved_info = []
        for doc_id, score in zip(retrieved_docs, retrieved_scores):
            if doc_id in candidate_data:
                cand_info = candidate_data[doc_id].copy()
                cand_info['score'] = score
                cand_info['doc_id'] = doc_id
                retrieved_info.append(cand_info)
        
        # 获取相关文档的详细信息
        relevant_info = []
        for doc_id in relevant_docs:
            if doc_id in candidate_data:
                relevant_info.append({
                    'doc_id': doc_id,
                    **candidate_data[doc_id]
                })
        
        # 计算匹配的相关文档数
        matched = len([d for d in retrieved_docs if d in relevant_docs])
        
        if matched == 0:
            # 完全错误：top k中没有相关文档
            errors['total'] += 1
            errors['no_relevant_in_top_k'].append({
                'query_id': query_name,
                'query_text': query_info['text'],
                'query_image': query_info['image'],
                'relevant_docs': relevant_info,
                'retrieved_docs': retrieved_info,
                'k': k
            })
        elif matched < len(relevant_docs):
            # 部分匹配：有相关但不完整
            matched_docs = [d for d in retrieved_docs if d in relevant_docs]
            missed_docs = [d for d in relevant_docs if d not in retrieved_docs]
            
            # 获取匹配和遗漏文档的信息
            matched_info = [d for d in relevant_info if d['doc_id'] in matched_docs]
            missed_info = [d for d in relevant_info if d['doc_id'] in missed_docs]
            
            errors['partial_match'].append({
                'query_id': query_name,
                'query_text': query_info['text'],
                'query_image': query_info['image'],
                'matched_docs': matched_info,
                'missed_docs': missed_info,
                'retrieved_docs': retrieved_info,
                'k': k
            })
    
    return errors

def load_image(image_path, image_prefix=None, size=(200, 200)):
    """加载图像并调整大小"""
    if not image_path:
        return None
    
    # 构建完整图像路径
    if image_prefix and not os.path.isabs(image_path):
        full_path = os.path.join(image_prefix, image_path)
    else:
        full_path = image_path
    
    try:
        img = Image.open(full_path).convert('RGB')
        img.thumbnail(size)
        return img
    except Exception as e:
        print(f"无法加载图像 {full_path}: {e}")
        return None

def visualize_error_case(error_case, save_path, image_prefix, case_index, error_type):
    """可视化单个错误案例的详细信息"""
    fig = plt.figure()
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 2, 2])
    
    # 1. 显示查询信息
    ax_query = fig.add_subplot(gs[0, :])
    query_text = error_case['query_text'][:200] + ('...' if len(error_case['query_text']) > 200 else '')
    ax_query.text(0.01, 0.5, f"Query: {query_text}", 
                 verticalalignment='center', fontsize=10, wrap=True)
    ax_query.set_title(f"Error Case {case_index} ({error_type}) - Query ID: {error_case['query_id']}")
    ax_query.axis('off')
    
    # 2. 显示查询图像（如果有）
    query_img = load_image(error_case['query_image'], image_prefix)
    if query_img:
        ax_qimg = fig.add_subplot(gs[1, 0])
        ax_qimg.imshow(query_img)
        ax_qimg.set_title("Query Image")
        ax_qimg.axis('off')
    
    # 3. 显示正确的候选文档
    relevant_title = "Relevant Documents (Missed)" if error_type == "partial" else "Relevant Documents"
    ax_relevant = fig.add_subplot(gs[1, 1:])
    relevant_docs = error_case['missed_docs'] if error_type == "partial" else error_case['relevant_docs']
    
    relevant_text = ""
    for i, doc in enumerate(relevant_docs[:3]):  # 最多显示3个相关文档
        doc_text = doc['text'][:150] + ('...' if len(doc['text']) > 150 else '')
        relevant_text += f"Doc {i+1} ({doc['doc_id']}): {doc_text}\n\n"
    
    if not relevant_text:
        relevant_text = "No relevant documents information available"
        
    ax_relevant.text(0.01, 0.99, relevant_text, 
                    verticalalignment='top', fontsize=9, wrap=True)
    ax_relevant.set_title(relevant_title)
    ax_relevant.axis('off')
    
    # 4. 显示检索到的候选文档
    ax_retrieved = fig.add_subplot(gs[2, :])
    retrieved_text = ""
    for i, doc in enumerate(error_case['retrieved_docs'][:3]):  # 最多显示3个检索文档
        doc_text = doc['text'][:150] + ('...' if len(doc['text']) > 150 else '')
        retrieved_text += f"Doc {i+1} (Score: {doc['score']:.4f}): {doc_text}\n\n"
    
    ax_retrieved.text(0.01, 0.99, retrieved_text, 
                     verticalalignment='top', fontsize=9, wrap=True)
    ax_retrieved.set_title(f"Retrieved Documents (Top {error_case['k']})")
    ax_retrieved.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/error_case_{case_index}_{error_type}.png", dpi=300, bbox_inches='tight')
    plt.close()

def save_error_details(errors, save_path, image_prefix, max_cases=5):
    """保存错误案例的详细信息和可视化结果"""
    # 创建可视化目录
    vis_dir = os.path.join(save_path, "error_visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # 保存完全错误案例
    with open(f"{save_path}/complete_error_details.txt", 'w', encoding='utf-8') as f:
        f.write("=== Complete Error Cases ===\n\n")
        for i, case in enumerate(errors['no_relevant_in_top_k'][:max_cases]):
            f.write(f"Case {i+1}: Query ID {case['query_id']}\n")
            f.write(f"Query Text: {case['query_text']}\n")
            f.write(f"Query Image: {case['query_image'] or 'None'}\n\n")
            
            f.write("Relevant Documents:\n")
            for doc in case['relevant_docs']:
                f.write(f"- Doc ID: {doc['doc_id']}\n")
                f.write(f"  Text: {doc['text'][:300]}...\n")
                f.write(f"  Image: {doc['image'] or 'None'}\n\n")
            
            f.write(f"Retrieved Documents (Top {case['k']}):\n")
            for doc in case['retrieved_docs'][:5]:
                f.write(f"- Doc ID: {doc['doc_id']} (Score: {doc['score']:.4f})\n")
                f.write(f"  Text: {doc['text'][:300]}...\n")
                f.write(f"  Image: {doc['image'] or 'None'}\n\n")
            
            f.write("-" * 80 + "\n\n")
            
            # 生成可视化图表
            visualize_error_case(case, vis_dir, image_prefix, i+1, "complete")
    
    # 保存部分错误案例
    with open(f"{save_path}/partial_error_details.txt", 'w', encoding='utf-8') as f:
        f.write("=== Partial Error Cases ===\n\n")
        for i, case in enumerate(errors['partial_match'][:max_cases]):
            f.write(f"Case {i+1}: Query ID {case['query_id']}\n")
            f.write(f"Query Text: {case['query_text']}\n")
            f.write(f"Query Image: {case['query_image'] or 'None'}\n\n")
            
            f.write("Matched Relevant Documents:\n")
            for doc in case['matched_docs']:
                f.write(f"- Doc ID: {doc['doc_id']}\n")
                f.write(f"  Text: {doc['text'][:300]}...\n")
                f.write(f"  Image: {doc['image'] or 'None'}\n\n")
            
            f.write("Missed Relevant Documents:\n")
            for doc in case['missed_docs']:
                f.write(f"- Doc ID: {doc['doc_id']}\n")
                f.write(f"  Text: {doc['text'][:300]}...\n")
                f.write(f"  Image: {doc['image'] or 'None'}\n\n")
            
            f.write(f"Retrieved Documents (Top {case['k']}):\n")
            for doc in case['retrieved_docs'][:5]:
                f.write(f"- Doc ID: {doc['doc_id']} (Score: {doc['score']:.4f})\n")
                f.write(f"  Text: {doc['text'][:300]}...\n")
                f.write(f"  Image: {doc['image'] or 'None'}\n\n")
            
            f.write("-" * 80 + "\n\n")
            
            # 生成可视化图表
            visualize_error_case(case, vis_dir, image_prefix, i+1, "partial")

def visualize_error_summary(errors, save_path):
    """生成错误汇总统计"""
    total_queries = len(errors['no_relevant_in_top_k']) + len(errors['partial_match'])
    complete_error_count = len(errors['no_relevant_in_top_k'])
    partial_error_count = len(errors['partial_match'])
    
    # 错误类型分布饼图
    plt.figure()
    labels = ['Complete Errors', 'Partial Errors']
    sizes = [complete_error_count, partial_error_count]
    colors = ['#ff9999', '#66b3ff']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Distribution of Error Types')
    plt.savefig(f"{save_path}/error_type_distribution.png")
    plt.close()
    
    # 保存统计信息
    with open(f"{save_path}/error_summary.txt", 'w', encoding='utf-8') as f:
        f.write("Error Summary Statistics\n")
        f.write("========================\n")
        f.write(f"Total error cases: {total_queries}\n")
        f.write(f"Complete errors: {complete_error_count} ({complete_error_count/total_queries*100:.2f}%)\n")
        f.write(f"Partial errors: {partial_error_count} ({partial_error_count/total_queries*100:.2f}%)\n")

def visualize_retrieval_errors(model_name, qrels_path, query_data_path, cand_pool_path, 
                             image_prefix, save_dir="./LamRA_Ret_eval_results", k=10, max_cases=5):
    """主函数：可视化检索错误结果，包含详细的文本和图像信息"""
    # 构建保存文件名
    save_name = qrels_path.split('/')[-1].replace('_qrels.txt', '_')
    save_name = f"{save_name}"
    
    # 创建结果保存目录
    result_dir = f"{save_dir}/error_details_{model_name}"
    os.makedirs(result_dir, exist_ok=True)
    
    # 加载数据
    print("加载评估结果...")
    results = load_evaluation_results(save_dir, save_name)
    qrel = load_qrels(qrels_path)
    
    print("加载查询和候选数据...")
    query_data = load_query_data(query_data_path)
    candidate_data = load_candidate_data(cand_pool_path)
    
    # 分析错误
    print("分析错误案例...")
    errors = analyze_errors(results, qrel, query_data, candidate_data, k)
    
    # 生成可视化结果
    print("生成错误案例详情...")
    save_error_details(errors, result_dir, image_prefix, max_cases)
    
    # 生成错误汇总
    print("生成错误统计汇总...")
    visualize_error_summary(errors, result_dir)
    
    print(f"错误案例详情已保存至: {result_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='qwen2-vl-2b_LamRA-Ret_mini', help='模型名称')
    parser.add_argument('--qrels_path', type=str, default='/mnt/disk2/yuanzm/dataset/lamra_data/M-BEIR/qrels/test/mbeir_mscoco_task3_test_qrels.txt', help='qrels文件路径')
    parser.add_argument('--query_data_path', type=str, default='/mnt/disk2/yuanzm/dataset/lamra_data/M-BEIR/query/test/mbeir_mscoco_task0_test.jsonl', help='查询数据文件路径')
    parser.add_argument('--cand_pool_path', type=str, default='/mnt/disk2/yuanzm/dataset/lamra_data/M-BEIR/cand_pool/local/mbeir_mscoco_task0_test_cand_pool.jsonl', help='候选文档池路径')
    parser.add_argument('--image_prefix', type=str, default='/mnt/disk2/yuanzm/dataset/lamra_data/M-BEIR', help='图像文件路径前缀')
    parser.add_argument('--save_dir', type=str, default="/home/yuanzm/LamRA/LamRA_Ret_eval_results", help='结果保存目录')
    parser.add_argument('--k', type=int, default=10, help='评估的top k值')
    parser.add_argument('--max_cases', type=int, default=5, help='最大展示的错误案例数量')
    
    args = parser.parse_args()
    visualize_retrieval_errors(
        args.model_name, 
        args.qrels_path, 
        args.query_data_path, 
        args.cand_pool_path,
        args.image_prefix,
        args.save_dir,)