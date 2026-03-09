import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PCALoss(nn.Module):
    """
    Partially Cross Attention Loss (L_proj1)
    
    L_proj1 = ||Attn_T - PCAttn_S||^2 + ||V_T * V_T / sqrt(d) - V_S * V_S / sqrt(d)||^2
    
    Note: V_T * V_T is element-wise multiplication
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='mean')
    
    def forward(self, Attn_t, PCAttn_s, V_t, V_s):
        """
        Args:
            Attn_t: Teacher attention output [B, 196, 768]
            PCAttn_s: Student partially cross attention output [B, 196, 768]
            V_t: Teacher Value matrix [B, 196, 768]
            V_s: Student Value matrix [B, 196, 768]
        
        Returns:
            loss: scalar
        """
        d = V_t.shape[-1]  # 768
        sqrt_d = math.sqrt(d)
        
        # Attention loss
        attn_loss = self.mse(Attn_t, PCAttn_s)
        
        # Value correlation loss (element-wise multiplication)
        V_t_corr = (V_t * V_t) / sqrt_d  # [B, 196, 768]
        V_s_corr = (V_s * V_s) / sqrt_d  # [B, 196, 768]
        value_loss = self.mse(V_t_corr, V_s_corr)
        
        return attn_loss + value_loss


class GWLLoss(nn.Module):
    """
    Group-wise Linear Loss (L_proj2)
    
    L_proj2 = ||h_T - h'_S||^2
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='mean')
    
    def forward(self, h_t, h_s_proj):
        """
        Args:
            h_t: Teacher feature (block11 output) [B, 196, 768]
            h_s_proj: Student projected feature [B, 196, 768]
        
        Returns:
            loss: scalar
        """
        return self.mse(h_t, h_s_proj)
class LogitsKDLoss(nn.Module):
    """
    Knowledge Distillation loss on logits (Hinton et al.)
    """
    def __init__(self, temperature=4.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits):
        """
        Args:
            student_logits: [B, num_classes]
            teacher_logits: [B, num_classes]
        """
        T = self.temperature

        soft_student = F.log_softmax(student_logits / T, dim=-1)
        soft_teacher = F.softmax(teacher_logits / T, dim=-1)

        loss = F.kl_div(
            soft_student,
            soft_teacher,
            reduction="batchmean"
        ) * (T**2)

        return loss

class ProjectionLoss(nn.Module):
    """
    Projection Loss (L_proj1 + L_proj2)
    
    Combines PCA attention loss and Group-wise Linear loss.
    Only used when use_projection=True.
    """
    def __init__(self):
        super().__init__()
        self.pca_loss = PCALoss()
        self.gwl_loss = GWLLoss()
    
    def forward(self, Attn_t, PCAttn_s, V_t, V_s, h_t, h_s_proj):
        """
        Args:
            Attn_t: Teacher attention output [B, 196, 768]
            PCAttn_s: Student partially cross attention output [B, 196, 768]
            V_t: Teacher Value matrix [B, 196, 768]
            V_s: Student Value matrix [B, 196, 768]
            h_t: Teacher feature (block11 output) [B, 196, 768]
            h_s_proj: Student projected feature from GL projector [B, 196, 768]
        
        Returns:
            l_proj1: PCA loss (scalar)
            l_proj2: GL loss (scalar)
        """
        l_proj1 = self.pca_loss(Attn_t, PCAttn_s, V_t, V_s)
        l_proj2 = self.gwl_loss(h_t, h_s_proj)
        
        return l_proj1, l_proj2


def cosine_similarity(a, b, eps=1e-8):
    """
    Calculates cosine similarity between two tensors.
    """
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    """
    Calculates Pearson correlation using cosine similarity on centered data.
    """
    return cosine_similarity(a - a.mean(1).unsqueeze(1), b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    """
    Calculates Inter-class relation loss (1 - mean correlation).
    """
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    """
    Calculates Intra-class relation loss (correlation between columns).
    """
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


class DIST(nn.Module):
    """
    Distillation from A Stronger Teacher (DIST) Loss
    """
    def __init__(self, beta=2.0, gamma=2.0, temperature=1.0):
        super(DIST, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature

    def forward(self, z_s, z_t):
        """
        Args:
            z_s: Student logits [B, num_classes]
            z_t: Teacher logits [B, num_classes]
        """
        # Softmax with temperature (usually T=1 for DIST)
        y_s = (z_s / self.temperature).softmax(dim=1)
        y_t = (z_t / self.temperature).softmax(dim=1)

        inter_loss = inter_class_relation(y_s, y_t)
        intra_loss = intra_class_relation(y_s, y_t)

        loss = self.beta * inter_loss + self.gamma * intra_loss
        return loss


# ===== Test =====
if __name__ == "__main__":
    B = 2

    # Fake data
    Attn_t = torch.randn(B, 196, 768)
    PCAttn_s = torch.randn(B, 196, 768)
    V_t = torch.randn(B, 196, 768)
    V_s = torch.randn(B, 196, 768)
    h_t = torch.randn(B, 196, 768)
    h_s_proj = torch.randn(B, 196, 768)

    # Test PCA Loss
    pca_loss_fn = PCALoss()
    pca_loss = pca_loss_fn(Attn_t, PCAttn_s, V_t, V_s)
    print(f"PCA Loss (L_proj1): {pca_loss.item():.4f}")

    # Test GWL Loss
    gwl_loss_fn = GWLLoss()
    gwl_loss = gwl_loss_fn(h_t, h_s_proj)
    print(f"GWL Loss (L_proj2): {gwl_loss.item():.4f}")

    # Test Projection Loss
    proj_loss_fn = ProjectionLoss()
    l_proj1, l_proj2 = proj_loss_fn(Attn_t, PCAttn_s, V_t, V_s, h_t, h_s_proj)
    print(f"\nL_proj1: {l_proj1.item():.4f}, L_proj2: {l_proj2.item():.4f}")

    # Test DIST Loss
    dist_loss_fn = DIST(beta=2.0, gamma=2.0, temperature=1.0)
    z_s = torch.randn(B, 10)
    z_t = torch.randn(B, 10)
    dist_loss = dist_loss_fn(z_s, z_t)
    print(f"DIST Loss: {dist_loss.item():.4f}")
