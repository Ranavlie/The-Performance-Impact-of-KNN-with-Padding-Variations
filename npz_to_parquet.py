import os
import argparse
import glob
import numpy as np
import pandas as pd

def safe_get(obj, key):
    """Safely get a key from a dictionary and convert to a numpy array."""
    if isinstance(obj, dict):
        value = obj.get(key)
        if value is not None:
            return np.asarray(value, dtype=object)
    return None

def process_npz(npz_path):
    """Load a single NPZ file and convert its contents to a DataFrame."""
    data = np.load(npz_path, allow_pickle=True)
    key = data.files[0]
    obj = data[key]
    inner = obj.item() if hasattr(obj, "item") else obj

    # Try to find embeddings under common keys
    embeddings = None
    if isinstance(inner, dict):
        for candidate_key in ("embeddings", "emb", "vectors"):
            tmp = safe_get(inner, candidate_key)
            if tmp is not None:
                embeddings = tmp
                break
    elif isinstance(inner, np.ndarray):
        embeddings = np.asarray(inner)

    if embeddings is None:
        print(f"[SKIP] No embeddings found in: {npz_path}")
        return None

    embeddings = embeddings.astype("float32")
    n_rows = embeddings.shape[0]

    # Optional metadata
    sample_ids = safe_get(inner, "dolt_ids") or safe_get(inner, "sample_ids")
    class_labels = safe_get(inner, "class_names") or safe_get(inner, "labels")
    targets = safe_get(inner, "targets")
    crop_ids = safe_get(inner, "crop_ids")

    # Dataset name / company fallback
    dataset_name = None
    if isinstance(inner, dict):
        dataset_name = inner.get("dataset_name") or inner.get("company_name")
    if dataset_name is None:
        dataset_name = os.path.basename(os.path.dirname(npz_path))
    if isinstance(dataset_name, (list, tuple, np.ndarray)):
        dataset_name = str(dataset_name[0]) if len(dataset_name) > 0 else ""
    else:
        dataset_name = str(dataset_name)

    # Normalize metadata length
    def normalize_column(arr, n):
        if arr is None:
            return [None] * n
        arr = np.asarray(arr, dtype=object)
        if arr.size == 1 and n > 1:
            return [arr.tolist()] * n
        if arr.shape[0] == n:
            return arr.tolist()
        return [None] * n

    sample_ids = normalize_column(sample_ids, n_rows)
    class_labels = normalize_column(class_labels, n_rows)
    targets = normalize_column(targets, n_rows)
    crop_ids = normalize_column(crop_ids, n_rows)

    # Build DataFrame dictionary
    rows = {
        "embedding": [row.tolist() for row in embeddings],
        "dataset_name": [dataset_name] * n_rows,
        "sample_id": sample_ids,
        "crop_id": crop_ids,
        "_source_file": [os.path.basename(npz_path)] * n_rows,
    }

    # Expand class labels into separate columns
    max_classes = max(len(x) if isinstance(x, (list, tuple, np.ndarray)) else 1 for x in class_labels) if class_labels else 0
    for i in range(max_classes):
        rows[f"class_{i}"] = [
            str(x[i]) if isinstance(x, (list, tuple, np.ndarray)) and len(x) > i else str(x) if x is not None else None
            for x in class_labels
        ]

    # Expand targets into separate columns
    max_targets = max(len(x) if isinstance(x, (list, tuple)) else 1 for x in targets) if targets else 0
    for i in range(max_targets):
        rows[f"target_{i}"] = [
            str(x[i]) if isinstance(x, (list, tuple, np.ndarray)) and len(x) > i else str(x) if x is not None else None
            for x in targets
        ]

    df = pd.DataFrame(rows)
    return df


def main(args):
    """Convert all NPZ files under root to a single Parquet file."""
    npz_paths = sorted(glob.glob(os.path.join(args.root, "**", "*.npz"), recursive=True))
    if not npz_paths:
        print("No NPZ files found under root:", args.root)
        return

    print(f"Found {len(npz_paths)} NPZ files.")
    dfs = []
    for path in npz_paths:
        try:
            df = process_npz(path)
            if df is not None:
                dfs.append(df)
                print(f"[OK] {path} -> rows: {len(df)}")
        except Exception as e:
            print(f"[ERROR] {path} -> {e}")

    if not dfs:
        print("No data collected from NPZ files.")
        return

    combined_df = pd.concat(dfs, ignore_index=True)
    print("Total rows:", len(combined_df), "Example embedding dimension:", len(combined_df.loc[0, "embedding"]))

    # Write to Parquet
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table_dict = {col: pa.array(combined_df[col].tolist()) for col in combined_df.columns}
        table = pa.table(table_dict)
        pq.write_table(table, args.output, compression="snappy")
        print("Parquet written to:", args.output)
    except Exception as e:
        print("failed to write Parquet:", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert NPZ embeddings to a Parquet file with class/target columns expanded."
    )
    parser.add_argument("--root", "-r", default="processed_dataset", help="Root folder for NPZ.")
    parser.add_argument("--output", "-o", default="embeddings.parquet", help="Output Parquet file path.")
    args = parser.parse_args()
    main(args)
