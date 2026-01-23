# Hướng dẫn Setup Code từ Private GitHub Repo trên Kaggle

## ⚠️ VẤN ĐỀ: Kaggle không cho nhập password khi clone private repo

## ✅ GIẢI PHÁP: Sử dụng GitHub Personal Access Token

---

## 🔑 Bước 1: Tạo GitHub Personal Access Token

### 1.1. Truy cập GitHub Settings
1. Đăng nhập GitHub
2. Click avatar (góc trên bên phải) → **Settings**
3. Scroll xuống cuối → Click **Developer settings**
4. Click **Personal access tokens** → **Tokens (classic)**

### 1.2. Generate New Token
1. Click **"Generate new token"** → **"Generate new token (classic)"**
2. Điền thông tin:
   - **Note**: `Kaggle Training` (hoặc tên bất kỳ)
   - **Expiration**: Chọn thời gian (khuyến nghị: 90 days)
   - **Select scopes**: ✅ Chọn **`repo`** (full control of private repositories)
3. Scroll xuống → Click **"Generate token"**

### 1.3. Copy Token
- Token sẽ hiển thị dạng: `ghp_xxxxxxxxxxxxxxxxxxxx`
- ⚠️ **QUAN TRỌNG**: Copy và lưu lại ngay! Token chỉ hiển thị 1 lần
- Nếu mất token, phải tạo token mới

---

## 📝 Bước 2: Chuẩn bị Script Clone

### Option A: Sử dụng script có sẵn (Khuyên dùng)

**Trong Kaggle Notebook:**

```python
# Cell 1: Tạo script clone
%%writefile clone_repo.py
import os
import subprocess

# ============ THAY ĐỔI THÔNG TIN Ở ĐÂY ============
GITHUB_USERNAME = "BerserkerFPT"
GITHUB_TOKEN = "ghp_YOUR_TOKEN_HERE"  # ⚠️ Paste token ở đây
REPO_OWNER = "BerserkerFPT"
REPO_NAME = "Capstone_KD"
BRANCH = "TomatoLeaf_VaibhavSolapure"
# ==================================================

# Clone with token
clone_url = f"https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com/{REPO_OWNER}/{REPO_NAME}.git"
subprocess.run(["git", "clone", "-b", BRANCH, clone_url], check=True)
print(f"✅ Cloned successfully to {REPO_NAME}/")

# Cell 2: Chạy script
!python clone_repo.py

# Cell 3: Verify
%cd Capstone_KD
!ls -la
```

### Option B: Clone trực tiếp (Đơn giản hơn)

```python
# Cell 1: Clone với token
import os

# ⚠️ THAY ĐỔI TOKEN VÀ USERNAME
GITHUB_USERNAME = "BerserkerFPT"
GITHUB_TOKEN = "ghp_YOUR_TOKEN_HERE"  # Paste token ở đây

# Clone
!git clone -b TomatoLeaf_VaibhavSolapure \
  https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com/BerserkerFPT/Capstone_KD.git

# Cell 2: Change directory
%cd Capstone_KD
!ls
```

---

## 🚀 Bước 3: Complete Kaggle Notebook Setup

### Full Notebook Code:

```python
# ==================== CELL 1: Clone Repository ====================
import os
import subprocess

# ⚠️ ĐIỀN THÔNG TIN CỦA BẠN
GITHUB_USERNAME = "BerserkerFPT"
GITHUB_TOKEN = "ghp_xxxxxxxxxxxxxxxxxxxx"  # ⚠️ PASTE TOKEN Ở ĐÂY

# Clone repository
clone_url = f"https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com/BerserkerFPT/Capstone_KD.git"
result = subprocess.run(
    ["git", "clone", "-b", "TomatoLeaf_VaibhavSolapure", clone_url],
    capture_output=True, text=True
)

if result.returncode == 0:
    print("✅ Clone thành công!")
    os.chdir("Capstone_KD")
    print(f"📁 Files: {os.listdir('.')}")
else:
    print(f"❌ Clone failed: {result.stderr}")

# ==================== CELL 2: Install Dependencies ====================
!pip install -q timm openpyxl seaborn

# Check GPU
import torch
print(f"\n{'='*70}")
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"{'='*70}\n")

# ==================== CELL 3: Verify Dataset ====================
import os

# List available datasets
print("Available Kaggle datasets:")
datasets = os.listdir('/kaggle/input')
for d in datasets:
    print(f"  📦 /kaggle/input/{d}")

# Update dataset path if needed
from config import Config

# ⚠️ Update this to match your dataset name on Kaggle
DATASET_NAME = "tomato-dataset"  # Thay đổi theo tên dataset trên Kaggle

Config.DATASET_PATH = f"/kaggle/input/{DATASET_NAME}/TomatoDataset"

# Verify dataset exists
if os.path.exists(Config.DATASET_PATH):
    classes = os.listdir(Config.DATASET_PATH)
    print(f"\n✅ Dataset found!")
    print(f"Classes ({len(classes)}): {classes}")
else:
    print(f"\n❌ Dataset NOT found at: {Config.DATASET_PATH}")
    print("Please update DATASET_NAME variable above")

# ==================== CELL 4: Optional - Test with fewer models ====================
# Uncomment để test nhanh với ít models
# from config import Config
# Config.MODELS = ['resnet18', 'mobilenet_v2', 'efficientnet_b0']
# Config.NUM_EPOCHS = 20  # Test với ít epochs

# ==================== CELL 5: Run Training ====================
from main import main

# Run training
main()

# ==================== CELL 6: View Results ====================
import pandas as pd
import os

# Find latest results folder
results_base = "/kaggle/working/results"
runs = sorted([int(d) for d in os.listdir(results_base) if d.isdigit()])
latest_run = runs[-1]

# Load results
results_path = f"{results_base}/{latest_run}/all_models_results.xlsx"
df = pd.read_excel(results_path)

print(f"\n{'='*70}")
print(f"RESULTS - Run #{latest_run}")
print(f"{'='*70}\n")
print(df.to_string(index=False))

# Display chart
from IPython.display import Image, display
chart_path = f"{results_base}/{latest_run}/performance_comparison.png"
display(Image(filename=chart_path))

# ==================== CELL 7: Download Results ====================
# Zip kết quả để download
!zip -r /kaggle/working/results_run_{latest_run}.zip /kaggle/working/results/{latest_run}/

print(f"\n✅ Results saved to: results_run_{latest_run}.zip")
print(f"Download từ Output panel (panel bên phải) →")
```

---

## 🔒 BẢO MẬT: Cách giữ token an toàn

### ⚠️ QUAN TRỌNG: Không commit token lên GitHub!

### Cách 1: Dùng Kaggle Secrets (Khuyến nghị)
```python
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
GITHUB_TOKEN = user_secrets.get_secret("GITHUB_TOKEN")
```

**Setup Kaggle Secret:**
1. Kaggle Notebook → Add-ons (panel phải)
2. Secrets → + Add a new secret
3. Label: `GITHUB_TOKEN`
4. Value: Paste token của bạn
5. Click Add

### Cách 2: Xóa token sau khi clone
```python
# Clone
!git clone https://user:token@github.com/repo.git

# Xóa token khỏi git config
%cd repo
!git config --unset credential.helper
!git remote set-url origin https://github.com/BerserkerFPT/Capstone_KD.git
```

### Cách 3: Dùng biến môi trường tạm
```python
import os
os.environ['GITHUB_TOKEN'] = 'your_token_here'
TOKEN = os.environ.get('GITHUB_TOKEN')
# Delete variable sau khi dùng xong
del os.environ['GITHUB_TOKEN']
```

---

## 🐛 Troubleshooting

### Lỗi: "Authentication failed"
- ✅ Kiểm tra token có đúng không (copy đầy đủ)
- ✅ Kiểm tra token có scope `repo` không
- ✅ Kiểm tra token chưa expire

### Lỗi: "Repository not found"
- ✅ Kiểm tra tên repo và branch đúng không
- ✅ Kiểm tra token có quyền access repo không

### Lỗi: "Dataset not found"
- ✅ Verify dataset đã add vào notebook
- ✅ Check tên dataset: `!ls /kaggle/input`
- ✅ Update `DATASET_PATH` cho đúng

### Token không work?
1. Tạo token mới
2. Đảm bảo chọn scope `repo`
3. Copy ngay sau khi tạo
4. Test bằng cách clone local trước

---

## 📋 Checklist Trước Khi Chạy

- [ ] Tạo GitHub Personal Access Token với scope `repo`
- [ ] Copy token và lưu lại an toàn
- [ ] Enable GPU trong Kaggle (Settings → GPU T4 x2)
- [ ] Add dataset vào Kaggle notebook (Add Data)
- [ ] Update `GITHUB_TOKEN` trong script
- [ ] Update `DATASET_PATH` nếu cần
- [ ] Enable Internet trong Kaggle (để cài packages)
- [ ] Có đủ GPU quota (30h/week)

---

## 💡 Tips

1. **Test trước với 1-2 models:**
   ```python
   Config.MODELS = ['resnet18']
   Config.NUM_EPOCHS = 10
   ```

2. **Monitor GPU:**
   ```python
   !nvidia-smi
   ```

3. **Save progress thường xuyên:**
   - Click "Save Version" trong Kaggle

4. **Estimate time:**
   - 1 model × 150 epochs ≈ 2-3 hours
   - 10 models ≈ 20-30 hours total

---

## 🎯 TL;DR (Quick Start)

```python
# 1. Tạo token: https://github.com/settings/tokens (scope: repo)

# 2. Clone trong Kaggle:
!git clone -b TomatoLeaf_VaibhavSolapure \
  https://USERNAME:TOKEN@github.com/BerserkerFPT/Capstone_KD.git

# 3. Setup:
%cd Capstone_KD
!pip install -q timm openpyxl seaborn

# 4. Update config:
from config import Config
Config.DATASET_PATH = "/kaggle/input/your-dataset/TomatoDataset"

# 5. Run:
from main import main
main()
```

Xong! 🎉
