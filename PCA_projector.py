import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PCAttentionProjector(nn.Module):
    """
    PCAttention Projector (Section 3.2 paper)
    Maps CNN feature map -> Q, K, V -> Partial Cross Attention
    """

    def __init__(self, in_channels=96, embed_dim=768, p=0.5,dropout=0.5):
        super().__init__()
        self.embed_dim = embed_dim
        self.p = p  # probability for partial cross replacement
        self.dropout = nn.Dropout(dropout)
        # 3 × 3x3 Conv layers
        self.conv_q = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
        self.conv_k = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
        self.conv_v = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
        self.ln_q = nn.LayerNorm(embed_dim)
        self.ln_k = nn.LayerNorm(embed_dim)
        self.ln_v = nn.LayerNorm(embed_dim)
    def _flatten_tokens(self, x):
        """
        [B, C, H, W] -> [B, HW, C]
        """
        return x.flatten(2).transpose(1, 2)

    def _partial_replace(self, Ms, Mt):
        """
        g(Ms): randomly replace student matrix with teacher matrix
        """
        if Mt is None:
            return Ms

        mask = torch.rand_like(Ms) < self.p
        return torch.where(mask, Mt, Ms)

    def forward(
        self,
        feat_map,          # [B,1024,14,14]
        QT=None, KT=None, VT=None  # teacher Q,K,V [B,196,768]
    ):
        """
        Returns:
            PCAttnS: [B,196,768]
            QS, KS, VS: student projected matrices
        """

        # ---- 1. CNN → Q,K,V via Conv ----
        QS_map = self.dropout(self.conv_q(feat_map))  # [B,768,14,14]
        KS_map = self.dropout(self.conv_k(feat_map))  # [B,768,14,14]
        VS_map = self.dropout(self.conv_v(feat_map))  # [B,768,14,14]

        # ---- 2. Flatten to tokens ----
        QS = self._flatten_tokens(QS_map)  # [B,196,768]
        KS = self._flatten_tokens(KS_map)
        VS = self._flatten_tokens(VS_map)

        # ---- 3. Partial Cross Attention ----
        QS_p = self._partial_replace(QS, QT)
        KS_p = self._partial_replace(KS, KT)
        VS_p = self._partial_replace(VS, VT)

        QS_p = self.ln_q(QS_p)
        KS_p = self.ln_k(KS_p)
        VS_p = self.ln_v(VS_p)
        d = self.embed_dim

        attn = torch.softmax(
            (QS_p @ KS_p.transpose(-2, -1)) / math.sqrt(d),
            dim=-1
        )  # [B,196,196]

        PCAttnS = attn @ VS_p  # [B,196,768]

        return {
            "PCAttnS": PCAttnS,
            "QS": QS,
            "KS": KS,
            "VS": VS,
        }
