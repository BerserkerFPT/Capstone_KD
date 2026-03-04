import torch
import torch.nn as nn

class GWLinearProjector(nn.Module):
    def __init__(self, in_dim=96, out_dim=768, drop_p=0.4):
        super().__init__()

        # 4 shared linear layers (one   per group)
        self.group_fc = nn.ModuleList([
            nn.Linear(in_dim, out_dim),
            nn.Linear(in_dim, out_dim),
            nn.Linear(in_dim, out_dim),
            nn.Linear(in_dim, out_dim),
        ])

        self.dropout = nn.Dropout(drop_p)
        self.ln = nn.LayerNorm(768)
    def forward(self, h_s):
        """
        h_s: [B, 1024, 14, 14] or [B, 196, 1024]
        return: h'_s [B, 196, 768]
        """

        if h_s.dim() == 3:
            B, N, C = h_s.shape
            h_s = h_s.view(B, 14, 14, C)
        else:
            B, C, H, W = h_s.shape
            h_s = h_s.permute(0, 2, 3, 1)  # [B,14,14,1024]

        out = torch.zeros(B, 14, 14, 768, device=h_s.device)

        # Group 0
        g0 = h_s[:, 0:7, 0:7, :].reshape(B, -1, 96)
        out[:, 0:7, 0:7, :] = self.group_fc[0](self.dropout(g0)).view(B, 7, 7, 768)

        # Group 1
        g1 = h_s[:, 0:7, 7:14, :].reshape(B, -1, 96)
        out[:, 0:7, 7:14, :] = self.group_fc[1](self.dropout(g1)).view(B, 7, 7, 768)

        # Group 2
        g2 = h_s[:, 7:14, 0:7, :].reshape(B, -1, 96)
        out[:, 7:14, 0:7, :] = self.group_fc[2](self.dropout(g2)).view(B, 7, 7, 768)

        # Group 3
        g3 = h_s[:, 7:14, 7:14, :].reshape(B, -1, 96)
        out[:, 7:14, 7:14, :] = self.group_fc[3](self.dropout(g3)).view(B, 7, 7, 768)
        out = out.view(B, 196, 768)
        out = self.ln(out)
        return out

