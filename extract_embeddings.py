import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from glob import glob


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Root dataset path')
    parser.add_argument('--folders', nargs='+', required=True, help='Subfolder names to process')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DataLoader')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers')
    parser.add_argument('--log_dir', type=str, default='logs', help='Path to model logs/checkpoints (if needed)')
    return parser.parse_args()


class CustomDataset(Dataset):
    """Generic dataset loader with transforms."""

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label, Path(img_path).stem


def get_data(data_root, label_folder, splits):
    """
    Collect image paths and labels.

    Args:
        data_root: dataset root path
        label_folder: subfolder name
        splits: list of splits to include, e.g. ["train1", "train2"]

    Returns:
        list of (file_path, class_name)
    """
    data = []
    # TODO: Adjust split-to-folder mapping based on your dataset structure
    split_to_folder = {"train": "1", "test": "2"}

    for split in splits:
        subset_folder = split_to_folder[split]
        subset_path = os.path.join(data_root, label_folder, subset_folder)
        jpg_files = glob(os.path.join(subset_path, '**', '*.jpg'), recursive=True)

        for file_path in jpg_files:
            parent_folder_name = os.path.basename(os.path.dirname(file_path))
            data.append((file_path, parent_folder_name))

    return data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def embedding_saver(data_root, label_folder, splits, args):
    """
    Extract embeddings for given dataset splits and save as .npz file.
    """

    # TODO: Replace with your own model
    # Example: model = YourModelClass.load_from_checkpoint(args.log_dir)
    model = None
    if model is None:
        raise NotImplementedError("Please replace 'model' with your own model implementation.")

    # TODO: Replace with your own transforms
    test_transform = None
    if test_transform is None:
        raise NotImplementedError("Please provide a transform (e.g., torchvision transforms).")

    # Load dataset
    data = get_data(data_root, label_folder, splits)
    dataset = CustomDataset(data, transform=test_transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    dataset_name = os.path.basename(args.data_root)

    model.eval()
    model.to(device)

    embeddings_list, labels, ids = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Extracting {label_folder} {splits}"):
            imgs, class_names, stems = batch
            imgs = imgs.to(device)

            # TODO: Replace with your model's embedding extraction function
            emb = model.extract_features(imgs)

            embeddings_list.append(emb)
            labels.extend(class_names)
            ids.extend(stems)

    # Convert to numpy
    embeddings = torch.cat(embeddings_list, dim=0).cpu().numpy()
    n = embeddings.shape[0]

    # Build class mapping
    class_to_idx = {cn: idx for idx, cn in enumerate(sorted(set(labels)))}
    classes = sorted(class_to_idx.keys())
    targets = np.array([class_to_idx[cn] for cn in labels], dtype=np.int32)

    # Prepare dictionary
    out_dict = {
        'dataset_name': dataset_name,
        'classes': classes,
        'embeddings': np.array(embeddings, dtype=np.float32),
        'targets': targets,
        'ids': ids,
    }

    # Save as compressed npz
    splits_str = "_".join(splits)
    out_filename = f'{splits_str}_embeddings.npz'
    out_path = os.path.join(args.data_root, label_folder, out_filename)
    np.savez_compressed(out_path, **{label_folder: out_dict})

    print(f"saved embeddings: {out_path}  (rows: {n})")


def main():
    args = parse_args()
    for label_folder in args.folders:
        for splits in [["train"], ["test"]]:
            embedding_saver(args.data_root, label_folder, splits, args)


if __name__ == '__main__':
    main()
