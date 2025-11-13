# Baseline Research - Pretrained Models Evaluation

Hệ thống đánh giá baseline cho 8 pretrained models với 3 chiến thuật đánh giá khác nhau.

## 📁 Cấu trúc Project

```
Capstone/
├── config.py              # Cấu hình dataset, training, models
├── models.py              # Định nghĩa 8 pretrained models với custom classifier
├── dataset.py             # Data loading, augmentation, và splitting
├── train.py               # Training với early stopping và checkpoint saving
├── evaluate.py            # 3 chiến thuật đánh giá và export kết quả
├── main.py                # Main pipeline để chạy toàn bộ
├── requirements.txt       # Dependencies
└── README.md             # Hướng dẫn sử dụng
```

## 🚀 Cài đặt

```bash
pip install -r requirements.txt
```

## ⚙️ Cấu hình

Mở file `config.py` và thay đổi đường dẫn dataset:

```python
class Config:
    # Thay đổi đường dẫn dataset ở đây
    DATASET_PATH = r"D:\Capstone\MorningaLeaf_Dataset"  
    # hoặc
    DATASET_PATH = r"D:\Capstone\TomatoDataset\TomatoDataset"
```

### Các cấu hình khác:

- **Train/Val/Test split**: Mặc định 70/15/15
- **Batch size**: 32
- **Number of epochs**: 50
- **Learning rate**: 0.001
- **Learning rate decay**: Giảm 0.5x sau 5 epochs val_loss không cải thiện
- **Early stopping patience**: 10 epochs
- **Custom classifier**: [256, 128, 64] với dropout 0.5 (Chỉ cần thêm số vào thì sẽ tự define và chạy)

## 🎯 8 Pretrained Models

1. VGG16
2. ResNet101
3. DenseNet121
4. EfficientNet-B0
5. ConvNeXt-Tiny
6. ViT-Base-Patch16-224
7. Swin-Tiny-Patch4-Window7-224
8. ConViT-Tiny

**Lưu ý**: Tất cả backbone weights đều được đóng băng (frozen), chỉ train custom classifier.

## 📊 3 Chiến thuật Đánh giá

### Strategy 1: Best Checkpoint
- Lưu checkpoint có val_loss thấp nhất
- Đánh giá trên tập test

### Strategy 2: Top-K Average
- Tìm K checkpoints có val_loss thấp nhất
- Trung bình trọng số của K checkpoints
- Đánh giá với K = 2, 3, 4, 5

### Strategy 3: Last-N Average
- Lấy 10 checkpoints của 10 epoch cuối cùng
- Trung bình trọng số
- Đánh giá trên tập test

## 🏃‍♂️ Cách chạy

### Chạy toàn bộ pipeline:

```bash
python main.py
```

Pipeline sẽ tự động:
1. Validate cấu hình
2. Load và split dataset
3. Train 8 models với early stopping
4. Đánh giá với 3 strategies
5. Export kết quả ra Excel
6. Tạo performance charts

### Chạy riêng từng bước:

```bash
# Test dataset loading
python dataset.py

# Test một model
python train.py

# Test models
python models.py
```

## 📈 Kết quả

Sau khi chạy xong, kết quả sẽ được lưu trong thư mục `results/`:

```
results/
├── baseline_results_YYYYMMDD_HHMMSS.xlsx       # Kết quả 48 experiments (8 models × 6 strategies)
├── performance_comparison.png                   # Chart tổng hợp so sánh toàn bộ
└── experiment_info_YYYYMMDD_HHMMSS.json        # Thông tin experiment
```

### Kết quả bao gồm 48 experiments:
- 8 models × 6 strategies:
  - Strategy 1: Best checkpoint
  - Strategy 2: Top-2, Top-3, Top-4, Top-5 average
  - Strategy 3: Last 10 epochs average

### Metrics được đánh giá:

- Accuracy (%)
- Precision (%)
- Recall (%)
- F1-Score (%)
- AUC (%)

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
