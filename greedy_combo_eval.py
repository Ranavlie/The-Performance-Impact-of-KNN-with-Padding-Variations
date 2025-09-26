import argparse
import itertools
import faiss
import numpy as np
import torch
import re
import os
from utils import FaissKNN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_npz', nargs='+', type=str, required=True, help='List of train npz files')
    parser.add_argument('--test_npz', nargs='+', type=str, required=True, help='List of test npz files')
    parser.add_argument('--combination_size', type=int, default=2, help='How many npz files to combine')
    parser.add_argument('--company_name', type=str, required=True, help='Company key inside npz')
    return parser.parse_args()


def combine_embeddings(npz_paths, company_name):
    embeddings = []
    class_names = []
    for path in npz_paths:
        data = np.load(path, allow_pickle=True)[company_name].item()
        embeddings.append(data['embeddings'])
        class_names.extend(data['class_names'])
    combined = {
        company_name: {
            'embeddings': np.vstack(embeddings),
            'class_names': class_names
        }
    }
    return combined


def get_id(path):
    match = re.search(r'image_(\d+)', path)
    return int(match.group(1)) if match else None


def main():
    args = parse_args()

    dolt_to_train = {get_id(path): path for path in args.train_npz}
    dolt_to_test  = {get_id(path): path for path in args.test_npz}
    common_dolt_ids = set(dolt_to_train.keys()) & set(dolt_to_test.keys())
    matched_pairs = [(dolt_to_train[i], dolt_to_test[i]) for i in sorted(common_dolt_ids)]

    combos = list(itertools.combinations(matched_pairs, args.combination_size))

    with open("results.txt", "a", encoding="utf-8") as f:
        for combo in combos:
            train_paths = [pair[0] for pair in combo]
            test_paths = [pair[1] for pair in combo]

            train_data = combine_embeddings(train_paths, args.company_name)
            test_data = combine_embeddings(test_paths, args.company_name)

            reference_embeddings = torch.from_numpy(train_data[args.company_name]['embeddings']).float()
            query_embeddings = torch.from_numpy(test_data[args.company_name]['embeddings']).float()

            all_classes = set(train_data[args.company_name]['class_names'] + test_data[args.company_name]['class_names'])
            class_to_idx = {c: i for i, c in enumerate(all_classes)}
            reference_labels = torch.tensor([class_to_idx[c] for c in train_data[args.company_name]['class_names']])
            query_labels = torch.tensor([class_to_idx[c] for c in test_data[args.company_name]['class_names']])

            # FAISS kNN
            knn = FaissKNN(index_init_fn=faiss.IndexFlatIP, device=0, tempmem=1, verbose=False)
            knn.add(torch.nn.functional.normalize(reference_embeddings, dim=1))

            # Basit accuracy hesaplama
            D, I = knn.search(torch.nn.functional.normalize(query_embeddings, dim=1), k=5)
            preds = reference_labels[I[:, 0]]
            acc = (preds == query_labels).float().mean().item()

            ref_str = f"Reference npz files: ({', '.join(train_paths)})"
            query_str = f"Query npz files: ({', '.join(test_paths)})"
            metrics_str = f"Accuracy@1: {acc:.4f}"

            print(f"\n{ref_str}")
            print(query_str)
            print(metrics_str)

            f.write(ref_str + "\n")
            f.write(query_str + "\n")
            f.write(metrics_str + "\n\n")


if __name__ == "__main__":
    main()
