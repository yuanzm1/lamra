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

if __name__ == "__main__":
    # Configure file path
    model_id = "qwen2-vl-2b_LamRA-Ret_mini_modpro_systeml1"  # Replace with actual model ID _modpro_systeml1
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
    
    # Visualize with t-SNE (may take longer for large datasets)
    plt_tsne = visualize_random_queries(
        query_features, candidate_features,
        query_ids, candidate_ids,
        n_samples=2000,
        method='tsne'
    )
    plt_tsne.savefig(f"{model_id}_500queries_tsne.png", dpi=300)
    # plt_tsne.show()

