# Image Dataset Processing & Embedding Evaluation
##  Project Goal
The goal of this project is to efficiently process large image datasets, extract meaningful feature embeddings, and evaluate these embeddings to improve product classification and other visual recognition tasks. It aims to provide a modular workflow that allows cropping objects from annotated images, extracting embeddings with any PyTorch-based model, evaluating embedding quality through metrics and greedy combination searches, and preparing the data for visualization platforms.
## Workflow Overview
###  1. Crop images from annotated JSONs
Use [crop.py](crop.py) to extract objects from large images based on bounding box annotations in JSON format. Optional padding can be applied to each crop.
### 2. Extract embeddings
Use [extract_embedding.py](extract_embedding.py) to compute feature embeddings from cropped images using any PyTorch-based model. The embeddings are saved in .npz files along with metadata.
### 3. Evaluate embeddings
• [evaluate_combo.py](evaluate_combo.py): Computes metrics (e.g., k-NN accuracy) for reference and query embeddings. Helps measure embedding quality and compare models.

• [greedy_combo_eval.py](greedy_combo_eval.py): Performs a greedy search to evaluate combinations of embeddings, identifying the best-performing reference/query sets.
### 4. Convert embeddings to Parquet
Use [npz_to_parquet.py](npz_to_parquet.py) to convert .npz embedding files into a single Parquet file, expanding metadata columns (class labels, targets, etc.). The output is compatible with visualization tools like [Apple/Atlas](https://github.com/apple/embedding-atlas?tab=readme-ov-file).
##  Scripts Overview

### 1. crop.py
Purpose: Crop objects from images using JSON annotations.
#### Features:
Supports padding in pixels or as a percentage of object size.\
Resizes crops using resize or resize_with_padding.\
Validates annotations and skips invalid or missing data.\
Organizes crops by split (train, val, test) and object ID.
#### Inputs:
--json_path: Path to JSON annotations.\
--image_folder: Folder containing original images.\
--output_root: Folder to save cropped images.
  
#### Outputs: 
Cropped images organized by split and object ID.

### 2. extract_embeddings.py
Purpose: Extract embeddings from cropped images using a model.
#### Features:
Supports any PyTorch-based feature extraction model.\
Maintains metadata like sample_id, crop_id, class labels, and dataset name.\
Handles multiple splits (train, val, test).
#### Inputs:
--data_root: Root folder containing cropped images.\
--folders: Subfolders corresponding to classes/categories.\
Optional: --company_name or dataset identifier.
  
#### Outputs: 
.npz files containing embeddings and metadata.

### 3. evaluate_combo.py
Evaluate embeddings by computing metrics (like k-NN accuracy) and optionally visualize embedding distributions for deeper insight. This script allows you to compare reference and query embeddings to assess their quality and separability.
#### Features:
Loads embeddings from .npz files and merges them by user-specified IDs.\
Performs k-NN evaluation (cosine similarity) to compute simple accuracy for reference-query combinations.\
Computes cosine similarity matrices among embeddings to measure intra-class and inter-class similarity.\
Supports 2D visualization using PCA, t-SNE, or UMAP. Only the top-N most frequent labels are visualized; all others are grouped as "OTHER".\
Interactive selection: users can incrementally add reference IDs and evaluate queries against the merged references.
#### Inputs:
--train_npz: Reference embedding files.\
--test_npz: Query embedding files.\
--key: Key inside the .npz files where embeddings are stored (e.g., 'train', 'test').
  
#### Outputs: 
Accuracy scores printed for each reference-query pair.
 
Optional visualization of embeddings in 2D using PCA, t-SNE, or UMAP.
  <img width="591" height="450" alt="Screenshot (73)" src="https://github.com/user-attachments/assets/74ef6a5e-5333-4f86-a3eb-fc5af80adb29" />
<img width="606" height="445" alt="Screenshot (72)" src="https://github.com/user-attachments/assets/cc6762ab-f3f8-4864-9aa2-128f21ed3dc2" />
<img width="587" height="464" alt="Screenshot (74)" src="https://github.com/user-attachments/assets/4cabb6c5-92a9-4301-aba4-991119ec93df" />

Cosine similarity metrics (optional) for analyzing embedding relationships.
  
### 4. greedy_combo_eval.py
Purpose: Perform greedy search over embedding combinations to find the best-performing reference/query sets.
#### Features:
Supports all pairwise or n-wise combinations.\
Computes k-NN metrics for each combination.\
Saves results to results.txt.\
Dataset-agnostic.
#### Inputs:
--train_npz / --test_npz: Lists of .npz embeddings.\
--combination_size: Number of embeddings per combination.\
--company_name: Generic dataset name.
  
#### Outputs: 
Evaluation metrics for each combination saved to a file.

### 5. npz_to_parquet.py
Purpose: Convert .npz embedding files into a single Parquet file for visualization.
#### Features:
Expands multi-label fields into separate columns (class_0, class_1, etc.).\
Captures dataset name, sample IDs, and source file name.\
Output is ready for visualization in platforms like Apple/Atlas.
#### Inputs:
--root: Folder containing .npz files.\
--out: Output Parquet file.
  
#### Outputs: 
Single Parquet file with embeddings and metadata.

## Results and Observations
Different combinations of references and queries with varying padding percentages improved product classification accuracy for certain setups.
Query [10] consistently achieved the highest accuracy across experiments.
Excessive padding (>20%) generally reduced model performance.
Combining the best-performing query with the lowest-performing reference did not increase accuracy. The best results occur when each padding is paired with its own reference and query. Accuracy could potentially be improved by combining paddings while considering computational cost.
Individual crops (crop_ids) that were misclassified alone were sometimes correctly predicted in combinations, and vice versa. No clear patterns emerged, highlighting the complex and sometimes unpredictable behavior of ML embeddings.

## Requirements
•Python 3.8+  
•PyTorch  
•NumPy  
•Pandas  
•OpenCV (cv2)  
•scikit-learn, seaborn, matplotlib (optional for visualization)  
•PyArrow (for Parquet conversion)

## Usage Example
```
# Crop images
python crop.py --json_path annotations.json --image_folder images/ --output_root cropped_images/

# Extract embeddings
python extract_embedding.py --data_root cropped_images/ --folders class_folders/ --company_name "datasetX"

# Evaluate embeddings
python evaluate_combo.py --train_npz train_embeddings.npz --test_npz test_embeddings.npz --company_name "datasetX"

# Greedy combination evaluation
python greedy_combo_eval.py --train_npz *.npz --test_npz *.npz --combination_size 2 --company_name "datasetX"

# Convert embeddings to Parquet
python npz_to_parquet.py --root embeddings/ --out embeddings.parquet
```
