import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pdb

def load_features(file_path):
    """Load the saved feature data"""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def reduce_dimensions(features, method='pca', n_components=2):
    """Reduce dimensions using PCA or t-SNE"""
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42)
    else:
        raise ValueError("Method must be either 'pca' or 'tsne'")
    
    return reducer.fit_transform(features)

def visualize_random_queries(query_feats, candidate_feats, 
                             query_ids=None, candidate_ids=None,
                             n_samples=500, method='pca'):
    """Visualize randomly sampled queries with all candidates"""
    np.random.seed(42)
    # Randomly sample n_samples queries
    n_queries_total = len(query_feats)
    if n_queries_total <= n_samples:
        # Use all queries if there are fewer than n_samples
        sample_indices = np.arange(n_queries_total)
    else:
        sample_indices = np.random.choice(n_queries_total, size=n_samples, replace=False)
    
    # spdb.set_trace()
    # Get sampled query features and IDs
    query_feats_sampled = query_feats[sample_indices]
    query_ids_sampled = query_ids[sample_indices] if query_ids is not None else None
    candidate_feats_sampled = candidate_feats[sample_indices]
    candidate_ids_sampled = candidate_ids[sample_indices] if query_ids is not None else None
    
    # Combine sampled queries with all candidates for consistent reduction
    all_feats = np.vstack([query_feats_sampled, candidate_feats_sampled])
    reduced_feats = reduce_dimensions(all_feats, method=method)
    
    # Split reduced features
    n_queries = len(query_feats_sampled)
    query_reduced = reduced_feats[:n_queries]
    candidate_reduced = reduced_feats[n_queries:]
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Plot scatter points
    plt.scatter(
        candidate_reduced[:, 0], candidate_reduced[:, 1],
        c='orange', label='Candidate Features', alpha=0.5, s=30, edgecolors='none'
    )
    plt.scatter(
        query_reduced[:, 0], query_reduced[:, 1],
        c='blue', label=f'Sampled Query Features (n={n_samples})', 
        alpha=0.7, s=60, edgecolors='black', linewidth=0.5
    )
    
    # Annotate with IDs if provided (for very small samples)
    if query_ids_sampled is not None and len(query_ids_sampled) <= 15:
        for i, idx in enumerate(query_ids_sampled):
            plt.annotate(idx, (query_reduced[i, 0], query_reduced[i, 1]), 
                         fontsize=8, bbox=dict(facecolor='white', alpha=0.8, pad=2))
    
    plt.title(f'Feature Distribution with {n_samples} Random Queries ({method.upper()})')
    plt.xlabel(f'{method.upper()} Dimension 1')
    plt.ylabel(f'{method.upper()} Dimension 2')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    return plt

def visualize_query_pos_neg_pairs(query_feats, candidate_feats,
                                  query_ids=None, candidate_ids=None,
                                  n_fit_queries=1000,   # 先抽取N个query做降维拟合
                                  m_show_queries=200,   # 再从上述N个里抽取M个进行展示
                                  method='tsne',
                                  random_state=42,
                                  annotate_top_k=0,
                                  save_path=None):
    """
    Two-stage sampling:
      1) 从全部 query 中抽取 n_fit_queries 个 (及其 +/− 样本) 参与降维拟合；
      2) 再从这 N 个里抽取 m_show_queries 个三元组进行可视化展示与连线/标注。

    约定：对于 query_id == i，正样本 candidate_id = 2*i，负样本 candidate_id = 2*i+1。

    Args:
        query_feats:       [Nq, D]
        candidate_feats:   [Nc, D]
        query_ids:         Optional [Nq]
        candidate_ids:     Optional [Nc]
        n_fit_queries:     使用多少个 query 三元组用于降维拟合 (N)
        m_show_queries:    从上述 N 中展示多少个 query 三元组 (M)
        method:            'tsne' or 'pca'
        annotate_top_k:    标注前 K 个点（按加入顺序，注意每个三元组有3个点）
        save_path:         若提供则保存图片

    Returns:
        matplotlib.pyplot handle
    """
    rng = np.random.default_rng(random_state)

    # Basic checks
    Nq = len(query_feats)
    Nc = len(candidate_feats)
    assert Nc >= 2 * Nq, (
        f"candidate_feats is expected to have at least 2 per query. Got Nc={Nc}, Nq={Nq} (need Nc>=2*Nq)."
    )

    # Clamp N and M
    n_fit = min(int(n_fit_queries), Nq)
    m_show = min(int(m_show_queries), n_fit)

    # --- Stage 1: 选择用于降维拟合的 N 个 query 索引 ---
    fit_q_idx = rng.choice(Nq, size=n_fit, replace=False) if Nq > n_fit else np.arange(Nq)
    fit_q_idx = np.sort(fit_q_idx)

    # 收集所有用于降维的点（N 个三元组 -> 3N 个点）
    fit_points = []
    fit_labels = []    # 'query' | 'pos' | 'neg'
    fit_id_texts = []  # 文本标注
    fit_triplet_indices = []  # [(q_i, p_i, n_i)] in the big array

    for qi in fit_q_idx:
        pos_idx = 2 * qi
        neg_idx = 2 * qi + 1
        if pos_idx >= Nc or neg_idx >= Nc:
            continue
        base = len(fit_points)
        fit_points.extend([query_feats[qi], candidate_feats[pos_idx], candidate_feats[neg_idx]])
        fit_labels.extend(['query', 'pos', 'neg'])
        q_id = str(query_ids[qi]) if query_ids is not None else str(qi)
        pos_id = str(candidate_ids[pos_idx]) if candidate_ids is not None else str(pos_idx)
        neg_id = str(candidate_ids[neg_idx]) if candidate_ids is not None else str(neg_idx)
        fit_id_texts.extend([f"q:{q_id}", f"+:{pos_id}", f"-:{neg_id}"])
        fit_triplet_indices.append((base + 0, base + 1, base + 2))

    fit_points = np.asarray(fit_points)

    # --- 降维 ---
    reduced_all = reduce_dimensions(fit_points, method=method)

    # --- Stage 2: 从 N 个三元组中选择 M 个进行展示 ---
    # 在 triplet 粒度上采样，保证每次选到的是整组三元组
    num_triplets = len(fit_triplet_indices)
    if num_triplets == 0:
        raise ValueError("No valid triplets were collected for dimensionality reduction.")

    m = min(m_show, num_triplets)
    show_triplet_sel = rng.choice(num_triplets, size=m, replace=False) if num_triplets > m else np.arange(num_triplets)

    # 要展示的点索引（在 reduced_all 中的索引）
    show_point_indices = []
    pair_links = []  # (q_idx, p_idx, n_idx) for lines
    id_texts = []

    for t in show_triplet_sel:
        q_i, p_i, n_i = fit_triplet_indices[t]
        show_point_indices.extend([q_i, p_i, n_i])
        pair_links.append((q_i, p_i, n_i))
        id_texts.extend([fit_id_texts[q_i], fit_id_texts[p_i], fit_id_texts[n_i]])

    # 根据展示子集生成类别索引
    show_labels = [fit_labels[i] for i in show_point_indices]
    show_coords = reduced_all[show_point_indices]

    idx_query = [i for i, t in enumerate(show_labels) if t == 'query']
    idx_pos   = [i for i, t in enumerate(show_labels) if t == 'pos']
    idx_neg   = [i for i, t in enumerate(show_labels) if t == 'neg']

    plt.figure(figsize=(13, 11))

    # 画连线（先画线再画点）
    for q_i, p_i, n_i in pair_links:
        # 需要把原始索引映射到 show 子集的位置
        # 建立一个原始->show 的映射
        # 为了效率，可在上面一次性构建映射，这里数据量一般也不大，直接 list.index 即可
        q_loc = show_point_indices.index(q_i)
        p_loc = show_point_indices.index(p_i)
        n_loc = show_point_indices.index(n_i)

        x = [show_coords[q_loc, 0], show_coords[p_loc, 0]]
        y = [show_coords[q_loc, 1], show_coords[p_loc, 1]]
        plt.plot(x, y, linestyle='-', linewidth=1.0, alpha=0.5, color='green')

        x = [show_coords[q_loc, 0], show_coords[n_loc, 0]]
        y = [show_coords[q_loc, 1], show_coords[n_loc, 1]]
        plt.plot(x, y, linestyle='--', linewidth=1.0, alpha=0.5, color='red')

    # 画点
    plt.scatter(show_coords[idx_neg, 0], show_coords[idx_neg, 1], marker='x', s=40, alpha=0.8, c='red', label='Negative (2i+1)')
    plt.scatter(show_coords[idx_pos, 0], show_coords[idx_pos, 1], marker='^', s=60, alpha=0.9, c='green', label='Positive (2i)')
    plt.scatter(show_coords[idx_query, 0], show_coords[idx_query, 1], marker='o', s=80, alpha=0.9, edgecolors='black', linewidth=0.6, c='blue', label='Queries (i)')

    # 可选标注（限制在展示子集内）
    if annotate_top_k and annotate_top_k > 0:
        to_annotate = min(annotate_top_k * 3, show_coords.shape[0])
        for i in range(to_annotate):
            plt.annotate(id_texts[i], (show_coords[i, 0], show_coords[i, 1]), fontsize=8,
                         bbox=dict(facecolor='white', alpha=0.75, pad=1))

    plt.title(f'Queries with Positive/Negative Pairs — {method.upper()} (fit N={n_fit}, show M={m})')
    plt.xlabel(f'{method.upper()} Dimension 1')
    plt.ylabel(f'{method.upper()} Dimension 2')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    return plt


if __name__ == "__main__":
    # Configure file path
    model_id = "qwen2-vl-2b_LamRA-Ret_mini_sugar_add_att"  # Replace with actual model ID _modpro_systeml1
    file_path = os.path.join("/mnt/disk2/yuanzm/weights/lamra/checkpoints", f"{model_id}.json")
    
    # Load data
    data = load_features(file_path)
    query_features = data['query_features']
    candidate_features = data['candidate_features']
    query_ids = data.get('query_ids')
    candidate_ids = data.get('candidate_ids')
    
    # Convert to numpy arrays if they are torch tensors
    if hasattr(query_features, 'numpy'):
        query_features = query_features.cpu().detach().numpy()
    if hasattr(candidate_features, 'numpy'):
        candidate_features = candidate_features.cpu().detach().numpy()
    query_ids = np.array(query_ids)
    candidate_ids = np.array(candidate_ids)
    
    # # Visualize with PCA
    # plt_pca = visualize_random_queries(
    #     query_features, candidate_features,
    #     query_ids, candidate_ids,
    #     n_samples=500,
    #     method='pca'
    # )
    # plt_pca.savefig(f"{model_id}_500queries_pca.png", dpi=300)
    # plt_pca.show()
    
    # # Visualize with t-SNE (may take longer for large datasets)
    # plt_tsne = visualize_random_queries(
    #     query_features, candidate_features,
    #     query_ids, candidate_ids,
    #     n_samples=2000,
    #     method='tsne'
    # )
    # plt_tsne.savefig(f"{model_id}_500queries_tsne.png", dpi=300)
    
    out_path = f"{model_id}_pairs_tsne.png"
    plt_tsne = visualize_query_pos_neg_pairs(
        query_features, candidate_features,
        query_ids=query_ids, candidate_ids=candidate_ids,
        n_fit_queries=2000, m_show_queries=20, method='tsne', random_state=42,
        annotate_top_k=0,  # set to 0 to disable annotations
        save_path=out_path,
    )

