import argparse
import os
import re
import numpy as np
import torch
import faiss
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from collections import Counter


def get_id_from_path(path: str):
    """Extract numeric ID (e.g., image_10 -> 10) from filename/path."""
    match = re.search(r'_(\d+)', path)
    return int(match.group(1)) if match else None


def load_embeddings(paths, key: str):
    """
    Load and merge embeddings from .npz files.

    Args:
        paths: list of npz file paths
        key: npz key where data is stored (e.g. 'train', 'test')

    Returns:
        dict with:
            'embeddings' -> numpy array of shape (N, D)
            'labels' -> list of labels
    """
    embeddings, labels = [], []
    for path in paths:
        # TODO: Adapt this based on your npz file structure
        data = np.load(path, allow_pickle=True)[key].item()
        embeddings.append(data['embeddings'])
        labels.extend(data['labels'])
    return {
        'embeddings': np.vstack(embeddings),
        'labels': labels
    }


def knn_evaluate(train_emb, train_labels, test_emb, test_labels, k=5):
    """
    Evaluate embeddings using kNN (with cosine similarity).
    Returns simple accuracy.
    """
    # Normalize
    train_emb = torch.nn.functional.normalize(torch.from_numpy(train_emb).float(), dim=1)
    test_emb = torch.nn.functional.normalize(torch.from_numpy(test_emb).float(), dim=1)

    # Build Faiss index
    index = faiss.IndexFlatIP(train_emb.shape[1])
    index.add(train_emb.numpy())

    D, I = index.search(test_emb.numpy(), k)
    preds = [train_labels[i[0]] for i in I]  # top-1 prediction
    acc = np.mean([p == t for p, t in zip(preds, test_labels)])
    return acc


def cosine_similarity_matrix(emb):
    """Compute average pairwise cosine similarity among embeddings."""
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    sim_matrix = np.dot(emb, emb.T)
    avg_sim = np.mean(sim_matrix[np.triu_indices(len(emb), k=1)])
    return avg_sim


def visualize_with_labels(embeddings, labels, title, method="pca", top_n_labels=10):
    """
    Visualize embeddings in 2D with PCA, t-SNE, or UMAP.
    Only top-N most frequent labels are shown, rest grouped into "OTHER".
    """
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    embeddings = embeddings / norms

    label_counts = Counter(labels)
    top_labels = set([label for label, _ in label_counts.most_common(top_n_labels)])
    filtered_labels = [label if label in top_labels else "OTHER" for label in labels]
    unique_labels = sorted(set(filtered_labels))

    palette = sns.color_palette("hsv", len(unique_labels))
    label_to_color = {label: palette[i] for i, label in enumerate(unique_labels)}

    if method == "pca":
        reducer = PCA(n_components=2)
    elif method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    elif method == "umap":
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")

    reduced = reducer.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    for label in unique_labels:
        idxs = [i for i, l in enumerate(filtered_labels) if l == label]
        plt.scatter(reduced[idxs, 0], reduced[idxs, 1],
                    c=[label_to_color[label]], label=label, alpha=0.7, s=20)

    plt.title(f"{method.upper()} - {title} (Top {top_n_labels} Labels + OTHER)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_npz', nargs='+', required=True, help="List of training .npz files")
    parser.add_argument('--test_npz', nargs='+', required=True, help="List of testing .npz files")
    parser.add_argument('--key', type=str, required=True, help="Key inside the .npz file (e.g., 'train')")
    args = parser.parse_args()

    # Pair train and test files by extracted ID
    pairs = {}
    for p in args.train_npz:
        d = get_id_from_path(p)
        pairs.setdefault(d, [None, None])[0] = p
    for p in args.test_npz:
        d = get_id_from_path(p)
        pairs.setdefault(d, [None, None])[1] = p
    pairs = {k: v for k, v in pairs.items() if v[0] and v[1]}

    ids = sorted(pairs.keys())
    print("\nLoaded IDs:", ids)

    selected_total = []

    while True:
        user_input = input("\nEnter IDs (or 'q' to quit): ").strip().lower()
        if user_input == 'q':
            break

        try:
            new_ids = list(map(int, user_input.strip().split()))
            for d in new_ids:
                if d not in selected_total:
                    selected_total.append(d)
        except Exception:
            print("Invalid input.")
            continue

        print(f"\nCurrent Ref IDs: {selected_total}")

        # Reference = merged embeddings from all selected
        ref_paths = [pairs[d][0] for d in selected_total]
        ref_data = load_embeddings(ref_paths, args.key)

        # Query = each new ID individually
        for q in new_ids:
            query_data = load_embeddings([pairs[q][1]], args.key)
            acc = knn_evaluate(
                ref_data['embeddings'], ref_data['labels'],
                query_data['embeddings'], query_data['labels']
            )
            print(f"Ref {selected_total} Query {q} => acc: {acc:.4f}")

            # Optional: similarity and visualization
            # sim = cosine_similarity_matrix(query_data['embeddings'])
            # print(f"=> Cosine Similarity: {sim:.4f}")
            # visualize_with_labels(query_data['embeddings'], query_data['labels'], f"Query {q}", method="pca")


if __name__ == "__main__":
    main()
