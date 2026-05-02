# AgriKD — Baseline Model Selection

> Part of the **AgriKD** paper: *"AgriKD: Cross-Architecture Knowledge Distillation for Efficient Leaf Disease Classification"*

This branch benchmarks candidate **teacher and student architectures** before knowledge distillation. Results from this pipeline directly informed the teacher–student pair chosen for the AgriKD framework.

**The full distillation pipeline (AgriKD) lives on the `main` branch of this repository.**

---

## Role in AgriKD

```
Baseline Model Selection (this branch)
  ↓  benchmark all candidate architectures
  ↓  select best teacher (highest accuracy)
  ↓  select best student (best accuracy/efficiency trade-off)
Knowledge Distillation  →  main branch
  ↓  ViT-B/16 teacher  ×  truncated MobileNetV2 student
  ↓  PCA Projector + GW Linear Projector + L_KL + L_Rel + L_CE
  ↓  5-fold stratified cross-validation
```

In the paper, ViT-B/16 was selected as the teacher and a truncated MobileNetV2 (Bottleneck 1–5) as the student, based on the F1-score / parameter-count trade-off measured by this pipeline.

---

## What this pipeline does

For each model listed in `Config.MODELS`, the pipeline:

1. Splits the dataset (70 / 15 / 15 train/val/test, stratified)
2. Trains the model with cosine-annealing LR + warmup + early stopping
3. Evaluates on the test set using the best validation checkpoint
4. Exports macro and per-class metrics to Excel
5. Generates training curves and confusion matrices

Optionally, **Stratified K-Fold Cross-Validation** can be enabled for more robust estimates.

---

## Repository Structure

```
├── config.py          # All hyperparameters — edit here
├── main.py            # Entry point — runs the full pipeline
├── train.py           # Training loop, early stopping, checkpoint management
├── evaluate.py        # Three evaluation strategies + Excel export
├── models.py          # Pretrained backbones with custom classifier head
├── dataset.py         # Data loading, augmentation, WeightedRandomSampler
├── losses.py          # PolyFocalLoss + class weight utilities
├── visualization.py   # Training curves and dataset statistics
├── save.py            # Result aggregation helpers
├── check_dataset.py   # Dataset sanity-check script
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Key dependencies:** PyTorch ≥ 2.0, torchvision, timm, scikit-learn, openpyxl, pandas, matplotlib, seaborn

---

## Quick Start

### 1. Prepare dataset

Organise images in `ImageFolder` format (one subfolder per class):

```
dataset/
├── class_A/
├── class_B/
└── ...
```

### 2. Configure

Edit `config.py`:

```python
DATASET_PATH = "/path/to/your/dataset"

# Add/remove architectures to benchmark:
MODELS = [
    'vit_base_patch16_224',   # teacher candidate
    'mobilenet_v2',           # student candidate
    # 'resnet101',
    # 'efficientnet_b0',
]
```

### 3. Run

```bash
python main.py
```

Results are saved to `results/<run_number>/`:
- `run_config.xlsx` — full configuration snapshot
- `all_models_results.xlsx` — side-by-side comparison of all models
- `<model_name>/` — per-model Excel, confusion matrix, training curves

---

## Configuration Reference

| Section | Parameter | Default | Description |
|---|---|---|---|
| **Dataset** | `DATASET_PATH` | — | Path to dataset root (`ImageFolder` format) |
| | `TRAIN_RATIO / VAL_RATIO / TEST_RATIO` | 0.7 / 0.15 / 0.15 | Split ratios |
| **Architectures** | `MODELS` | `['vit_base_patch16_224']` | List of backbones to benchmark |
| | `CLASSIFIER_CONFIG` | `[512]` | Hidden layer sizes for the classifier head |
| | `DROPOUT_RATE` | `0.4` | Dropout in classifier head |
| **Training** | `NUM_EPOCHS` | `50` | Maximum training epochs |
| | `LEARNING_RATE` | `9e-7` | Initial LR |
| | `EARLY_STOPPING_PATIENCE` | `10` | Stop if no val_loss improvement |
| | `LOSS_FUNCTION` | `'cross_entropy'` | `'cross_entropy'` or `'poly_focal'` |
| **Cross-Val** | `USE_CROSS_VALIDATION` | `False` | Enable Stratified K-Fold CV |
| | `CV_N_SPLITS` | `5` | Number of folds |
| **Output** | `AUTO_DELETE_CHECKPOINTS` | `False` | Delete checkpoints after evaluation |

---

## Handling Class Imbalance

The pipeline provides two complementary tools (as used in the paper for the imbalanced Potato dataset):

- **WeightedRandomSampler** (`USE_WEIGHTED_SAMPLER = True`) — rebalances at the data level by oversampling minority classes during training.
- **PolyFocalLoss** (`LOSS_FUNCTION = 'poly_focal'`) — focuses the loss on hard/misclassified examples.

> **Note:** enabling both simultaneously may over-correct for imbalance. The paper used `USE_WEIGHTED_SAMPLER = True` with standard cross-entropy for the baseline experiments.

---

## Knowledge Distillation

Once teacher and student candidates are selected from this benchmark, the AgriKD distillation pipeline on the **`main` branch** transfers knowledge from the teacher to the student using:

- PCA Cross-Attention Projector (L_proj1)
- Group-Wise Linear Projector (L_proj2)
- Hinton KD logits distillation (L_KL)
- DIST relational loss (L_Rel)
- Cross-entropy with label smoothing (L_CE)

→ **See `main` branch for the full AgriKD implementation.**

---

## Citation

```bibtex
@article{agrikd2026,
  title   = {AgriKD: Cross-Architecture Knowledge Distillation for Efficient Leaf Disease Classification},
  author  = {Your Name et al.},
  journal = {Your Journal/Conference},
  year    = {2026}
}
```


## Models hỗ trợ

Uncomment trong `Config.MODELS`:

```python
MODELS = [
    'vgg16',
    'resnet18',
    'resnet101',
    'mobilenet_v2',
    'densenet121',
    'efficientnet_b0',
    'vit_base_patch16_224',
]
```

## 3 Chiến thuật Đánh giá

| Strategy | Mô tả |
|----------|-------|
| **Best Checkpoint** | Checkpoint có val_loss thấp nhất |
| **Top-K Average** | Trung bình weights của K checkpoint tốt nhất (K = 2,3,4,5) |
| **Last-N Average** | Trung bình weights của N epoch cuối cùng |

## Cross-Validation

Khi `USE_CROSS_VALIDATION = True`:
- Data `train + val` gộp thành CV pool
- `test` giữ nguyên làm hold-out
- Dùng `StratifiedKFold` (sklearn) chia K fold, giữ tỉ lệ class
- Kết quả cuối: **mean ± std** qua K fold → lưu vào `cv_summary_results.xlsx`

## Metrics

Tất cả metrics dùng **Macro averaging** (trung bình đều giữa các class):

| Metric | Cách tính |
|--------|-----------|
| Accuracy | Overall correct / total |
| Precision | Macro average |
| Recall | Macro average |
| F1-Score | Macro average |
| AUC | Macro average, one-vs-rest |

Kết quả bao gồm cả **per-class breakdown** (Precision, Recall, F1, Specificity, AUC, Support).

## Reproduce kết quả

1. Set `RANDOM_SEED = 42` (mặc định) — đảm bảo cùng data split, cùng weight init
2. Đặt đúng `DATASET_PATH`
3. Chọn model trong `MODELS`
4. Chạy `python main.py`

Seed cố định cho: `random`, `numpy`, `torch`, `CUDA`, `cudnn.deterministic`.

## Ghi chú

- **WRS + Focal Loss đồng thời**: Không lỗi code, nhưng có thể double-correct class imbalance. Cân nhắc chỉ bật 1 trong 2.
- **LR Scheduler**: Linear Warmup → Cosine Annealing
- **Data Augmentation** (chỉ train): RandomFlip, RandomRotation(90°), ColorJitter

## 💾 Checkpoints

Checkpoints được lưu trong `checkpoints/`:

```
checkpoints/
├── vgg16/
│   ├── epoch_001_val_loss_0.xxxx.pth
│   ├── epoch_002_val_loss_0.xxxx.pth
│   ├── best_checkpoint.pth
│   └── checkpoint_info.json
├── resnet101/
│   └── ...
└── ...
```

## 🔧 Tùy chỉnh

### Thay đổi learning rate decay:

Trong `config.py`:

```python
LR_DECAY_PATIENCE = 5  # Giảm LR sau 5 epochs val_loss không cải thiện
LR_DECAY_FACTOR = 0.5  # Nhân LR với 0.5
```

### Thay đổi custom classifier:

Trong `config.py`:

```python
CLASSIFIER_CONFIG = [256, 128, 64]  # 3 hidden layers
DROPOUT_RATE = 0.5
```

### Thay đổi data augmentation:

Trong `dataset.py`, function `get_transforms()`:

```python
transform = transforms.Compose([
    transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    # Thêm augmentation khác...
])
```

### Thêm/bớt models:

Trong `config.py`:

```python
MODELS = [
    'vgg16',
    'resnet101',
    # Thêm/bớt models ở đây
]
```

## 📋 Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA (recommended) hoặc CPU
- RAM: >= 8GB
- GPU: >= 6GB VRAM (recommended)

## 🎓 Sử dụng cho Research

Code này được thiết kế để:
- Dễ dàng thay đổi dataset
- Tự động hóa toàn bộ pipeline
- Export kết quả professional
- Tái sử dụng cho nhiều experiments

Chỉ cần thay đổi `DATASET_PATH` trong `config.py` và chạy `python main.py`!

## 📝 Citation

Nếu sử dụng code này cho research, vui lòng ghi nguồn phù hợp.

## 🐛 Troubleshooting

### Lỗi out of memory:
- Giảm `BATCH_SIZE` trong `config.py`
- Giảm `NUM_WORKERS`

### Lỗi không tìm thấy dataset:
- Kiểm tra đường dẫn `DATASET_PATH` trong `config.py`
- Đảm bảo folder structure đúng format (classes trong subfolder)

### Model không train:
- Kiểm tra GPU/CUDA availability
- Kiểm tra dependencies đã cài đủ chưa

## 📧 Support

Nếu có vấn đề, vui lòng mở issue hoặc liên hệ.
