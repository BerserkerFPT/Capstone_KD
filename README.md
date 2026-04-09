# Baseline Research - Pretrained Models Evaluation

Pipeline tự động train & evaluate pretrained models cho bài toán image classification.

## Cấu trúc Project

```
├── config.py          # Toàn bộ cấu hình (dataset, training, loss, sampler, CV, ...)
├── main.py            # Main pipeline - chạy file này
├── train.py           # Training loop, early stopping, checkpoint management
├── evaluate.py        # 3 chiến thuật đánh giá + export Excel
├── models.py          # Pretrained models với custom classifier head
├── dataset.py         # Data loading, augmentation, WeightedRandomSampler
├── losses.py          # PolyFocalLoss + class weight computation
├── visualization.py   # Training curves, dataset statistics
├── requirements.txt   # Dependencies
└── results/           # Kết quả tự động lưu theo từng lần chạy (results/1/, results/2/, ...)
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Cách chạy

```bash
python main.py
```

Pipeline tự động: Validate config → Load dataset → Train từng model → Evaluate 3 strategies → Export Excel + Charts.

Kết quả mỗi lần chạy lưu riêng tại `results/<run_number>/` gồm:
- `run_config.xlsx` — toàn bộ config của lần chạy
- `all_models_results.xlsx` — bảng so sánh tất cả model
- `<model_name>/` — kết quả chi tiết, confusion matrix, training curves

## Cấu hình (`config.py`)

Mở `config.py`, chỉnh các biến cần thiết:

| Nhóm | Biến quan trọng | Mô tả |
|------|-----------------|-------|
| **Dataset** | `DATASET_PATH` | Đường dẫn tới thư mục dataset (mỗi class = 1 subfolder) |
| | `TRAIN_RATIO / VAL_RATIO / TEST_RATIO` | Tỉ lệ chia data (mặc định 70/15/15) |
| **Model** | `MODELS` | List model cần train (comment/uncomment để chọn) |
| | `CLASSIFIER_CONFIG` | Hidden layers của classifier head, VD: `[512]` |
| | `DROPOUT_RATE` | Dropout rate cho classifier |
| **Training** | `BATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE` | Hyperparameters cơ bản |
| | `WEIGHT_DECAY` | L2 regularization |
| | `EARLY_STOPPING_PATIENCE` | Dừng sớm nếu val_loss không giảm sau N epochs |
| **Loss** | `LOSS_FUNCTION` | `'cross_entropy'` hoặc `'poly_focal'` |
| | `label_smoothing` | Label smoothing (chỉ cho CrossEntropy) |
| | `FOCAL_GAMMA`, `POLY_EPSILON` | Params cho PolyFocalLoss |
| **Sampler** | `USE_WEIGHTED_SAMPLER` | `True/False` — bật WeightedRandomSampler xử lý class imbalance |
| **Cross-Val** | `USE_CROSS_VALIDATION` | `True/False` — bật Stratified K-Fold CV |
| | `CV_N_SPLITS` | Số fold (mặc định 5) |
| **Output** | `AUTO_DELETE_CHECKPOINTS` | Tự xóa checkpoints sau evaluate để tiết kiệm disk |

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
