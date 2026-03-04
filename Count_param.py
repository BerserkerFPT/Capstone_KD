# Cách 1: Thêm path rồi import bình thường
from main import StudentWithHead

# Đếm số tham số
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# Sử dụng
model = StudentWithHead(num_classes=5)

total, trainable = count_parameters(model)
print(f"Total parameters: {total:,}")
print(f"Trainable parameters: {trainable:,}")
print(f"Size: ~{total * 4 / 1024 / 1024:.2f} MB (float32)")

# import torch

# ckpt_path = "/home/student/dunglmde180498/ViTBase-Resnet50Code_AddLogits_Completed_Ver1_MobileNet85,13/strategy_2_top_k3_avg.pth"
# try:
#     checkpoint = torch.load(ckpt_path, map_location='cpu')
#     print("Checkpoint loaded successfully!")
#     print("Type:", type(checkpoint))
#     if isinstance(checkpoint, dict):
#         print("Keys:", list(checkpoint.keys()))
#     else:
#         print("Checkpoint is not a dict.")
# except Exception as e:
#     print("Checkpoint is NOT valid!")
#     print("Error:", e)