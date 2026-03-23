import torch
import timm
import torch.nn.functional as F
import math
import torch.nn as nn
class CustomHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 5),
        )
    def forward(self, x):
        return self.classifier(x)

# class CustomHead(nn.Module):
#     def __init__(self):
#         super().__init__()
#         prev_dim = 768
#         hidden_dim = 512

#         self.classifier = nn.Sequential(
#             nn.LayerNorm(prev_dim),
#             nn.Linear(prev_dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(0.4),
#             nn.Linear(hidden_dim, 10)
#         )

#     def forward(self, x):
#         return self.classifier(x)
class TeacherExtractor:
    def __init__(self, 
                 model_name="vit_base_patch16_224", 
                 pretrained=False,
                 checkpoint_path=None,
                 block_ids=[11,10,9,8,7],
                 block_qkv_id=[11,10,9,8,7]):
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.head = CustomHead()
        # Load checkpoint nếu có
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            # Xử lý các trường hợp checkpoint khác nhau
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'],strict=True)

        self.model.eval()
        self.cache = {"block_outs": [], "logits": None, "qkv_outs": []}
        self.block_ids = block_ids
        self.block_qkv_id = block_qkv_id
        # ===== ĐÓNG BĂNG ViT =====
        for param in self.model.parameters():
            param.requires_grad = False
    
        self.hooks = []
    def _hook_block_multi(self, module, input, output):
        self.cache["block_outs"].append(output.detach())
    # ---- hook QKV ----
    def _hook_qkv(self, module, input, output):
        # output: [B, 197, 2304]
        self.cache["qkv_outs"].append(output.detach())

    def _hook_logits(self, module, input, output):
        self.cache["logits"] = output.detach()

    def _register_hooks(self):
        for i in self.block_qkv_id:
            h_qkv = self.model.blocks[i].attn.qkv.register_forward_hook(self._hook_qkv) ## Cái này để lấy QKV từ block 11
            self.hooks.append(h_qkv)

        h_logits = self.model.head.register_forward_hook(self._hook_logits) ## Cái này để lấy logits
        self.hooks.append(h_logits)
           # Lấy output từ các block này
        for i in self.block_ids:
            h = self.model.blocks[i].register_forward_hook(self._hook_block_multi)
            self.hooks.append(h)

    # ---- hook Transformer Block output ---- ( đang lấy block 11)
    def _remove_hooks(self):
        # cleanup hooks (rất quan trọng)
        for h in self.hooks:
            h.remove()
        self.hooks = []
    
    def extract(self, x):
        """
        Forward pass và extract Q, K, V, Attntion teacher, block11_out
        
        Args:
            x: input tensor [B, 3, 224, 224]
        
        Returns:
            dict với các key: Q_t, K_t, V_t, Attn_t, block11_out
        """
        self.cache["block_outs"].clear()
        self.cache["qkv_outs"].clear()
        self.cache["logits"] = None
        self._register_hooks()
        
        # ===== forward teacher =====
        with torch.no_grad():
            _ = self.model(x)
        # ==== get logits =====    
        logits = self.cache["logits"]
        # ===== retrieve outputs =====
        qkv_out = self.cache["qkv_outs"]# [B, 197, 2304]
        if len(qkv_out) > 1:
            qkv_out = torch.stack(qkv_out, dim=0)
            qkv_out = qkv_out.mean(dim=0)
        else:
            qkv_out = qkv_out[0]
        block_outs = self.cache["block_outs"]   # list length = 5
        # # ===== split Q K V =====
        Q = qkv_out[:, :, 0:768]
        K = qkv_out[:, :, 768:1536]
        V = qkv_out[:, :, 1536:2304]

        Q_t = Q[:, 1:, :] 
        K_t = K[:, 1:, :] 
        V_t = V[:, 1:, :] 

        # [5, B, 197, 768]
        if len(block_outs) > 1:
            #Trung bình các outputs của nhiều transformer block ( thì chạy code này ) 
            block_stack = torch.stack(block_outs, dim=0)
            block_mean = block_stack.mean(dim=0)    # [B,197,768]
            block_mean = block_mean[:, 1:, :]
        else:
            #Lấy output của 1 transformer block (thì chạy code này)
            block_mean = block_outs[0][:, 1:, :]
        
        
        # Tính attention map để bỏ vào hàm loss
        d = Q_t.shape[-1]  # 768
        attn_logits = torch.matmul(Q_t, K_t.transpose(-2, -1)) / math.sqrt(d)
        attn_weights = F.softmax(attn_logits, dim=-1)
        Attn_t = torch.matmul(attn_weights, V_t)
        
        self._remove_hooks()
        
        return {
            "Q_t": Q_t,
            "K_t": K_t,
            "V_t": V_t,
            "Attn_t": Attn_t,
            "block_mean": block_mean,
            "logits": logits
        }
    
    def to(self, device):
        """Move model to device"""
        self.model = self.model.to(device)
        return self


# ===== Test =====
if __name__ == "__main__":
    teacher = TeacherExtractor(checkpoint_path="path/to/your/checkpoint.pth")
    
    x = torch.randn(2, 3, 224, 224)
    outputs = teacher.extract(x)
    
    print("Q teacher:", outputs["Q_t"].shape)
    print("K teacher:", outputs["K_t"].shape)
    print("V teacher:", outputs["V_t"].shape)
    print("Block11:", outputs["block_mean"].shape)
    print("Attention teacher:", outputs["Attn_t"].shape)  # [B, 196, 768]
    print("Logits:", outputs["logits"].shape)
