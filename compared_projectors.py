import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------
# 1. ĐƯỜNG DẪN CHECKPOINT CỦA BẠN (ĐIỀN VÀO ĐÂY)
# ---------------------------------------------------------
# Cần trỏ đến file ảnh test của bạn
SAMPLE_IMAGE = "/workspace/kaggle/working/ProcessedOriginal/Healthy/IMG_20250314_102903.jpg"

# Thư mục Dataset Test (Trỏ vào folder chứa các class thư mục con, ví dụ: /test)
# Phải đảm bảo bên trong TEST_DIR có Folder của các Class (Healthy, Rust,...)
TEST_DIR = "/workspace/kaggle/working/ProcessedOriginal" # <--- HÃY CHỈNH LẠI THÀNH FOLDER CHỨA ẢNH TEST CỦA BẠN

# Checkpoint Teacher
TEACHER_CKPT = "/Strategy_2_K2.pth"

# Checkpoint của nhánh V1 (Linear)
STUDENT_V1_CKPT = "/Compare/V1/checkpoints/run_1/saved_checkpoints/strategy1_best_epoch_18.pth"
PROJ_V1_CKPT = "/Compare/V1/checkpoints/run_1/best_gl_projector.pth"

# Checkpoint của nhánh V2 (Bottleneck ResNet)
STUDENT_V2_CKPT = "/Compare/V2_Bottleneck/checkpoints/run_3/saved_checkpoints/strategy1_best_epoch_24.pth"
PROJ_V2_CKPT = "/Compare/V2_Bottleneck/checkpoints/run_3/best_gl_projector.pth"

NUM_CLASSES = 5
BATCH_SIZE = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------------------------------------------------
# 2. HELPER IMPORT ĐỂ KHÔNG BỊ CONFLICT GIỮA V1 VÀ V2
# ---------------------------------------------------------
def load_v1():
    sys.path.insert(0, '/Compare/V1')
    from Teacher_extraction import TeacherExtractor
    from main import StudentWithHead
    from GWLinear_projector import GWLinearProjector as ProjV1
    
    t = TeacherExtractor(pretrained=False) # Head custom
    s = StudentWithHead(num_classes=NUM_CLASSES)
    p = ProjV1()
    
    sys.path.pop(0)
    for k in list(sys.modules.keys()):
        if k in ['Teacher_extraction', 'Student_extraction', 'main', 'GWLinear_projector', 'loss_functions', 'dataset']:
            del sys.modules[k]
    return t, s, p

def load_v2():
    sys.path.insert(0, '/Compare/V2_Bottleneck')
    from main import StudentWithHead
    from GWLinear_projector import GWLinearProjector as ProjV2
    
    s = StudentWithHead(num_classes=NUM_CLASSES)
    p = ProjV2()
    
    sys.path.pop(0)
    for k in list(sys.modules.keys()):
        if k in ['Teacher_extraction', 'Student_extraction', 'main', 'GWLinear_projector', 'loss_functions', 'dataset']:
            del sys.modules[k]
    return s, p

# ---------------------------------------------------------
# 3. METRICS CƠ BẢN (Tính trên Matrix [196, 768])
# ---------------------------------------------------------
def cka_score(X, Y):
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    hsic = torch.trace(X.T @ Y @ Y.T @ X)
    var1 = torch.trace(X.T @ X @ X.T @ X)
    var2 = torch.trace(Y.T @ Y @ Y.T @ Y)
    return (hsic / (torch.sqrt(var1) * torch.sqrt(var2) + 1e-8)).item()

def evaluate_metrics(feat_t, feat_s):
    with torch.no_grad():
        cos_sim = F.cosine_similarity(feat_t.flatten(), feat_s.flatten(), dim=0).item()
        mse = F.mse_loss(feat_s, feat_t).item()
        l1 = F.l1_loss(feat_s, feat_t).item()
        cka = cka_score(feat_t, feat_s)
    return {"Cosine Sim": cos_sim, "MSE": mse, "L1 Loss": l1, "CKA": cka}

# ---------------------------------------------------------
# 4. CHẠY QUÉT TOÀN BỘ DATASET ĐỂ TÍNH TRUNG BÌNH THEO CLASS
# ---------------------------------------------------------
def evaluate_full_dataset(teacher, student_v1, proj_v1, student_v2, proj_v2):
    if not os.path.exists(TEST_DIR):
        print(f"\n❌ [Cảnh báo] Không tìm thấy thư mục DATASET tại: {TEST_DIR}.")
        print("Vui lòng cập nhật biến TEST_DIR để quét toàn bộ Dataset.")
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    classes = dataset.classes
    
    print(f"\n📂 Đã nạp thành công bộ Test Dataset. Gồm {len(dataset)} ảnh, chia thành {len(classes)} classes.")
    print("="*70)

    # Khởi tạo bộ nhớ cộng dồn cho V1 và V2
    # Cấu trúc: stats[version][class_name] = {'cos': 0, 'mse': 0, 'l1': 0, 'cka': 0, 'count': 0}
    stats = {
        'V1': {c: {'cos': 0.0, 'mse': 0.0, 'l1': 0.0, 'cka': 0.0, 'count': 0} for c in classes},
        'V2': {c: {'cos': 0.0, 'mse': 0.0, 'l1': 0.0, 'cka': 0.0, 'count': 0} for c in classes}
    }

    print("⏳ Đang quét toàn bộ ảnh để tính Score. Vui lòng đợi nhé...")
    teacher.model.eval(); student_v1.eval(); proj_v1.eval(); student_v2.eval(); proj_v2.eval()

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating classes"):
            images = images.to(DEVICE)
            
            # Forward Batch Teacher
            t_out = teacher.extract(images)
            batch_feat_t = t_out["block_mean"]  # [B, 196, 768]
            
            # Forward Batch V1
            feat_s1, _ = student_v1(images)
            batch_feat_p1 = proj_v1(feat_s1)
            
            # Forward Batch V2
            feat_s2, _ = student_v2(images)
            batch_feat_p2 = proj_v2(feat_s2)

            for i in range(len(labels)):
                c_name = classes[labels[i].item()]
                
                # Cắt ra 1 ảnh
                ft = batch_feat_t[i]
                fp1 = batch_feat_p1[i]
                fp2 = batch_feat_p2[i]
                
                m1 = evaluate_metrics(ft, fp1)
                m2 = evaluate_metrics(ft, fp2)
                
                stats['V1'][c_name]['cos'] += m1['Cosine Sim']
                stats['V1'][c_name]['mse'] += m1['MSE']
                stats['V1'][c_name]['l1'] += m1['L1 Loss']
                stats['V1'][c_name]['cka'] += m1['CKA']
                stats['V1'][c_name]['count'] += 1

                stats['V2'][c_name]['cos'] += m2['Cosine Sim']
                stats['V2'][c_name]['mse'] += m2['MSE']
                stats['V2'][c_name]['l1'] += m2['L1 Loss']
                stats['V2'][c_name]['cka'] += m2['CKA']
                stats['V2'][c_name]['count'] += 1

    # Print Bảng Kết Quả ra Terminal & Pandas
    print("\n" + "="*85)
    print(f"BẢNG SO SÁNH METRICS PROJECTOR THEO TỪNG CLASS (DATASET: {len(dataset)} ảnh)")
    print("="*85)
    
    rows = []
    
    for c in classes:
        n = stats['V1'][c]['count']
        if n == 0: continue
        
        # Trung bình cộng V1
        v1_cos = stats['V1'][c]['cos'] / n
        v1_mse = stats['V1'][c]['mse'] / n
        v1_l1 = stats['V1'][c]['l1'] / n
        v1_cka = stats['V1'][c]['cka'] / n
        
        # Trung bình cộng V2
        v2_cos = stats['V2'][c]['cos'] / n
        v2_mse = stats['V2'][c]['mse'] / n
        v2_l1 = stats['V2'][c]['l1'] / n
        v2_cka = stats['V2'][c]['cka'] / n

        # Thêm mũi tên
        s_cos = f"{v1_cos:.3f} 👈 | {v2_cos:.3f}" if v1_cos > v2_cos else f"{v1_cos:.3f} | {v2_cos:.3f} 👈"
        s_cka = f"{v1_cka:.3f} 👈 | {v2_cka:.3f}" if v1_cka > v2_cka else f"{v1_cka:.3f} | {v2_cka:.3f} 👈"
        s_mse = f"{v1_mse:.3f} 👈 | {v2_mse:.3f}" if v1_mse < v2_mse else f"{v1_mse:.3f} | {v2_mse:.3f} 👈"
        s_l1  = f"{v1_l1:.3f} 👈 | {v2_l1:.3f}"  if v1_l1 < v2_l1 else  f"{v1_l1:.3f} | {v2_l1:.3f} 👈"

        print(f"👉 Class: {c.upper():<15} (N={n})")
        print(f"    Cos (Lớn Tốt) : {s_cos}")
        print(f"    CKA (Lớn Tốt) : {s_cka}")
        print(f"    MSE (Nhỏ Tốt) : {s_mse}")
        print(f"    L1  (Nhỏ Tốt) : {s_l1}")
        print("-"*85)
        
        rows.append({
            "Class": c, "Count": n,
            "V1_Cos": v1_cos, "V2_Cos": v2_cos,
            "V1_CKA": v1_cka, "V2_CKA": v2_cka,
            "V1_MSE": v1_mse, "V2_MSE": v2_mse,
            "V1_L1": v1_l1, "V2_L1": v2_l1
        })

    # Summary toàn Dataset
    df = pd.DataFrame(rows)
    print("\n✅ TỔNG QUAN TRUNG BÌNH TOÀN DATASET:")
    print(f"    - Cosine Sim: V1 ({df['V1_Cos'].mean():.4f}) vs V2 ({df['V2_Cos'].mean():.4f}) -> {( 'V1' if df['V1_Cos'].mean() > df['V2_Cos'].mean() else 'V2' )} WIN")
    print(f"    - CKA       : V1 ({df['V1_CKA'].mean():.4f}) vs V2 ({df['V2_CKA'].mean():.4f}) -> {( 'V1' if df['V1_CKA'].mean() > df['V2_CKA'].mean() else 'V2' )} WIN")
    print(f"    - MSE Loss  : V1 ({df['V1_MSE'].mean():.4f}) vs V2 ({df['V2_MSE'].mean():.4f}) -> {( 'V1' if df['V1_MSE'].mean() < df['V2_MSE'].mean() else 'V2' )} WIN")
    print(f"    - L1 Loss   : V1 ({df['V1_L1'].mean():.4f})  vs V2 ({df['V2_L1'].mean():.4f})  -> {( 'V1' if df['V1_L1'].mean() < df['V2_L1'].mean() else 'V2' )} WIN")
    print("="*85)
    
    # Save Excel
    df.to_excel("compare_projectors_full_dataset.xlsx", index=False)
    print("\n📁 Đã xuất dữ liệu chi tiết ra file `compare_projectors_full_dataset.xlsx` để vẽ chart/báo cáo.")

    # -----------------------------------------------
    # QUAN TRỌNG: VẼ BAR CHART CHO TỪNG METRIC
    # -----------------------------------------------
    metrics_info = [
        ("Cos", "Cosine Similarity (Cao hơn là tốt hơn)"),
        ("CKA", "CKA Score (Cao hơn là tốt hơn)"),
        ("MSE", "MSE Loss (Thấp hơn là tốt hơn)"),
        ("L1",  "L1 Loss (Thấp hơn là tốt hơn)")
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    x = np.arange(len(df))  # Vị trí các nhóm bar
    width = 0.35  # Độ rộng
    
    for i, (metric_key, title) in enumerate(metrics_info):
        ax = axes[i]
        
        v1_vals = df[f"V1_{metric_key}"]
        v2_vals = df[f"V2_{metric_key}"]
        
        rects1 = ax.bar(x - width/2, v1_vals, width, label='V1 (Linear)', color='#4c72b0')
        rects2 = ax.bar(x + width/2, v2_vals, width, label='V2 (ResNet)', color='#c44e52')

        ax.set_ylabel('Điểm số / Sai số')
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df["Class"], rotation=20, ha='right', fontsize=9)
        ax.legend()
        
        # Gắn số lên đỉnh cột để dễ đọc
        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), textcoords="offset points", xytext=(0,3), ha='center', va='bottom', fontsize=8)
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), textcoords="offset points", xytext=(0,3), ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    plt.savefig("compare_metrics_barchart.png", dpi=300)
    print("📈 TUYỆT VỜI! Đã tạo xong Biểu đồ Cột SO SÁNH (Bar Chart) 4 hệ số tại file: `compare_metrics_barchart.png`")

# ---------------------------------------------------------
# 5. TRỰC QUAN HOÁ 1 ẢNH (VISUALIZATION MẪU)
# ---------------------------------------------------------
def overlay_heatmap(img_tensor, feat, ax, title):
    img_np = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img_np = np.array([0.229, 0.224, 0.225]) * img_np + np.array([0.485, 0.456, 0.406])
    img_np = np.clip(img_np, 0, 1)

    f = feat.view(14, 14, 768)
    heatmap = f.mean(dim=-1).detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)
    
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = heatmap_colored / 255.0
    
    ax.imshow(0.5 * img_np + 0.5 * heatmap_colored)
    ax.set_title(title, fontsize=11)
    ax.axis('off')

def visualize_single_image(teacher, student_v1, proj_v1, student_v2, proj_v2):
    if not os.path.exists(SAMPLE_IMAGE):
        return

    img = Image.open(SAMPLE_IMAGE).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        t_out = teacher.extract(x)
        feat_t = t_out["block_mean"].squeeze(0)
        
        feat_s1, _ = student_v1(x)
        feat_p1 = proj_v1(feat_s1).squeeze(0)
        
        feat_s2, _ = student_v2(x)
        feat_p2 = proj_v2(feat_s2).squeeze(0)

    m1 = evaluate_metrics(feat_t, feat_p1)
    m2 = evaluate_metrics(feat_t, feat_p2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    overlay_heatmap(x, feat_t, axes[0], "Teacher (Ground Truth)")
    overlay_heatmap(x, feat_p1, axes[1], f"V1 Projection\nCos: {m1['Cosine Sim']:.3f} | L1: {m1['L1 Loss']:.3f}")
    overlay_heatmap(x, feat_p2, axes[2], f"V2 Projection\nCos: {m2['Cosine Sim']:.3f} | L1: {m2['L1 Loss']:.3f}")
    
    plt.tight_layout()
    plt.savefig("compare_visualization_simple.png", dpi=300)
    print("\n✅ Thêm nữa: Đã lưu trực quan hóa heatmap ảnh test tại file: compare_visualization_simple.png")
    # plt.show()

# ---------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------
def main():
    print("🚀 Đang khởi tạo mô hình...")
    teacher, student_v1, proj_v1 = load_v1()
    student_v2, proj_v2 = load_v2()

    teacher.to(DEVICE)
    student_v1.to(DEVICE); proj_v1.to(DEVICE)
    student_v2.to(DEVICE); proj_v2.to(DEVICE)

    if os.path.exists(TEACHER_CKPT):
        teacher.model.load_state_dict(torch.load(TEACHER_CKPT, map_location=DEVICE).get('model_state_dict', torch.load(TEACHER_CKPT, map_location=DEVICE)))
    
    if os.path.exists(STUDENT_V1_CKPT): student_v1.load_state_dict(torch.load(STUDENT_V1_CKPT, map_location=DEVICE).get('student_state_dict', torch.load(STUDENT_V1_CKPT, map_location=DEVICE)))
    if os.path.exists(PROJ_V1_CKPT): proj_v1.load_state_dict(torch.load(PROJ_V1_CKPT, map_location=DEVICE))
    if os.path.exists(STUDENT_V2_CKPT): student_v2.load_state_dict(torch.load(STUDENT_V2_CKPT, map_location=DEVICE).get('student_state_dict', torch.load(STUDENT_V2_CKPT, map_location=DEVICE)))
    if os.path.exists(PROJ_V2_CKPT): proj_v2.load_state_dict(torch.load(PROJ_V2_CKPT, map_location=DEVICE))

    # TÍNH TOÁN TRÊN TỪNG CLASS TOÀN BỘ DATASET
    evaluate_full_dataset(teacher, student_v1, proj_v1, student_v2, proj_v2)

    # VẼ RENDER CỦA ẢNH MẪU (như bạn yêu cầu lúc nãy)
    visualize_single_image(teacher, student_v1, proj_v1, student_v2, proj_v2)

if __name__ == "__main__":
    main()
