import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------
# 1. ĐƯỜNG DẪN CHECKPOINT CỦA BẠN (ĐIỀN VÀO ĐÂY)
# ---------------------------------------------------------
# Cần trỏ đến file ảnh test của bạn
SAMPLE_IMAGE = "c:/Users/Acer/Downloads/sample.jpg" # <--- Thay đổi đường dẫn ảnh

# Checkpoint Teacher
TEACHER_CKPT = "path/to/teacher_model.pth"

# Checkpoint của nhánh V1 (Linear)
STUDENT_V1_CKPT = "Capstone_KD-feature-dist-kd-strategy-/checkpoints/latest.pth"
PROJ_V1_CKPT = "Capstone_KD-feature-dist-kd-strategy-/checkpoints/latest_gl_projector.pth"

# Checkpoint của nhánh V2 (Bottleneck ResNet)
STUDENT_V2_CKPT = "Capstone_KD-feature-v2-dist-kd-strategy-/checkpoints/latest.pth"
PROJ_V2_CKPT = "Capstone_KD-feature-v2-dist-kd-strategy-/checkpoints/latest_gl_projector.pth"

NUM_CLASSES = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------------------------------------------------
# 2. HELPER IMPORT ĐỂ KHÔNG BỊ CONFLICT GIỮA V1 VÀ V2
# ---------------------------------------------------------
def load_v1():
    # Chuyển vào folder V1 để import
    sys.path.insert(0, os.path.abspath('Capstone_KD-feature-dist-kd-strategy-'))
    from Teacher_extraction import TeacherExtractor
    from main import StudentWithHead
    from GWLinear_projector import GWLinearProjector as ProjV1
    
    t = TeacherExtractor(pretrained=False) # Head custom
    s = StudentWithHead(num_classes=NUM_CLASSES)
    p = ProjV1()
    
    # Dọn dẹp path
    sys.path.pop(0)
    for k in list(sys.modules.keys()):
        if k in ['Teacher_extraction', 'Student_extraction', 'main', 'GWLinear_projector', 'loss_functions', 'dataset']:
            del sys.modules[k]
    return t, s, p

def load_v2():
    # Chuyển vào folder V2 để import
    sys.path.insert(0, os.path.abspath('Capstone_KD-feature-v2-dist-kd-strategy-'))
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
# 3. METRICS CƠ BẢN
# ---------------------------------------------------------
def cka_score(X, Y):
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    hsic = torch.trace(X.T @ Y @ Y.T @ X)
    var1 = torch.trace(X.T @ X @ X.T @ X)
    var2 = torch.trace(Y.T @ Y @ Y.T @ Y)
    return (hsic / (torch.sqrt(var1) * torch.sqrt(var2) + 1e-8)).item()

def evaluate_metrics(feat_t, feat_s):
    # feat_t, feat_s: [196, 768]
    with torch.no_grad():
        cos_sim = F.cosine_similarity(feat_t.flatten(), feat_s.flatten(), dim=0).item()
        mse = F.mse_loss(feat_s, feat_t).item()
        l1 = F.l1_loss(feat_s, feat_t).item()
        cka = cka_score(feat_t, feat_s)
    return {"Cosine Sim": cos_sim, "MSE": mse, "L1 Loss": l1, "CKA": cka}

# ---------------------------------------------------------
# 4. TRỰC QUAN HOÁ (VISUALIZATION)
# ---------------------------------------------------------
def overlay_heatmap(img_tensor, feat, ax, title):
    # 1. Chuyển ảnh về dạng numpy RGB
    img_np = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img_np = np.array([0.229, 0.224, 0.225]) * img_np + np.array([0.485, 0.456, 0.406])
    img_np = np.clip(img_np, 0, 1)

    # 2. Xử lý feature map [196, 768] -> [14, 14]
    f = feat.view(14, 14, 768)
    heatmap = f.mean(dim=-1).detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)
    
    # 3. Overlay
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = heatmap_colored / 255.0
    
    ax.imshow(0.5 * img_np + 0.5 * heatmap_colored)
    ax.set_title(title, fontsize=11)
    ax.axis('off')

# ---------------------------------------------------------
# 5. CHẠY CHƯƠNG TRÌNH SO SÁNH
# ---------------------------------------------------------
def main():
    print("🚀 Đang khởi tạo mô hình...")
    teacher, student_v1, proj_v1 = load_v1()
    student_v2, proj_v2 = load_v2()

    # Move to device
    teacher.to(DEVICE)
    student_v1.to(DEVICE); proj_v1.to(DEVICE)
    student_v2.to(DEVICE); proj_v2.to(DEVICE)

    # Load Teacher
    if os.path.exists(TEACHER_CKPT):
        teacher.model.load_state_dict(torch.load(TEACHER_CKPT, map_location=DEVICE).get('model_state_dict', torch.load(TEACHER_CKPT, map_location=DEVICE)))
        print("✅ Đã load Teacher")
    else:
        print("⚠️ Không tìm thấy CHECKPOINT TEACHER, sẽ chạy với mô hình init ngẫu nhiên để test script!")

    # Load V1
    if os.path.exists(STUDENT_V1_CKPT): student_v1.load_state_dict(torch.load(STUDENT_V1_CKPT, map_location=DEVICE).get('student_state_dict', torch.load(STUDENT_V1_CKPT, map_location=DEVICE)))
    if os.path.exists(PROJ_V1_CKPT): proj_v1.load_state_dict(torch.load(PROJ_V1_CKPT, map_location=DEVICE))
    
    # Load V2
    if os.path.exists(STUDENT_V2_CKPT): student_v2.load_state_dict(torch.load(STUDENT_V2_CKPT, map_location=DEVICE).get('student_state_dict', torch.load(STUDENT_V2_CKPT, map_location=DEVICE)))
    if os.path.exists(PROJ_V2_CKPT): proj_v2.load_state_dict(torch.load(PROJ_V2_CKPT, map_location=DEVICE))

    # Đưa về chế độ Eval
    teacher.model.eval(); student_v1.eval(); proj_v1.eval(); student_v2.eval(); proj_v2.eval()

    # --- Đọc ảnh ---
    if not os.path.exists(SAMPLE_IMAGE):
        print(f"❌ Không tìm thấy ảnh tại: {SAMPLE_IMAGE}. Vui lòng sửa lại đường dẫn dòng 14!")
        return

    img = Image.open(SAMPLE_IMAGE).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    x = transform(img).unsqueeze(0).to(DEVICE)

    print("📸 Đang chạy Inference...")
    with torch.no_grad():
        # Teacher output
        t_out = teacher.extract(x)
        feat_t = t_out["block_mean"].squeeze(0)  # [196, 768]
        
        # V1 output
        feat_s1, _ = student_v1(x)               # [96, 14, 14]
        feat_p1 = proj_v1(feat_s1).squeeze(0)     # [196, 768]
        
        # V2 output
        feat_s2, _ = student_v2(x)               # [96, 14, 14]
        feat_p2 = proj_v2(feat_s2).squeeze(0)     # [196, 768]

    # --- Đo đạc metrics ---
    metrics_v1 = evaluate_metrics(feat_t, feat_p1)
    metrics_v2 = evaluate_metrics(feat_t, feat_p2)

    print("\n" + "="*50)
    print(f"{'Metric':<15} | {'V1 (Linear)':<12} | {'V2 (ResNet)':<12}")
    print("-" * 50)
    for k in metrics_v1.keys():
        v1_val, v2_val = metrics_v1[k], metrics_v2[k]
        
        if k in ["Cosine Sim", "CKA"]:
            p1, p2 = ("👈", "") if v1_val > v2_val else ("", "👈")
        else:
            p1, p2 = ("👈", "") if v1_val < v2_val else ("", "👈")
            
        print(f"{k:<15} | {v1_val:<12.4f} {p1:2} | {v2_val:<12.4f} {p2:2}")
    print("="*50)

    # --- Vẽ Heatmap ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    overlay_heatmap(x, feat_t, axes[0], "Teacher (Ground Truth Ground)")
    overlay_heatmap(x, feat_p1, axes[1], f"V1 Projection\nCos: {metrics_v1['Cosine Sim']:.3f} | L1: {metrics_v1['L1 Loss']:.3f}")
    overlay_heatmap(x, feat_p2, axes[2], f"V2 Projection\nCos: {metrics_v2['Cosine Sim']:.3f} | L1: {metrics_v2['L1 Loss']:.3f}")
    
    plt.tight_layout()
    plt.savefig("compare_visualization_simple.png", dpi=300)
    print("\n✅ Đã lưu kết quả trực quan hóa tại file: compare_visualization_simple.png")
    plt.show()

if __name__ == "__main__":
    main()
