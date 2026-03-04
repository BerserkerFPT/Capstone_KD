import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models.feature_extraction import create_feature_extractor


class StudentExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Load pretrained MobileNetV2
        if pretrained:
            weights = MobileNet_V2_Weights.DEFAULT
            mobilenet = mobilenet_v2(weights=weights)
        else:
            mobilenet = mobilenet_v2(weights=None)
        
        # ❗ Student KHÔNG đóng băng
        # (không set requires_grad = False)
        
        return_nodes = {
            "features.11": "feat_14x14_96"
        }
        
        self.backbone = create_feature_extractor(
            mobilenet,
            return_nodes=return_nodes
        )
        # self.pool = nn.AvgPool2d(kernel_size=4, stride=4)
    def forward(self, x):
        """
        Forward pass để extract feature map từ layer3
        
        Args:
            x: input tensor [B, 3, 224, 224]
        
        Returns:
            feat_map: [B, 96, 14, 14]
        """
        out = self.backbone(x)
        feat_map = out["feat_14x14_96"]  # [B, 96, 56, 56]
        # feat_map = self.pool(feat_map)
        return feat_map
    
    def train_mode(self):
        """Set model to training mode"""
        self.backbone.train()
        return self
    
    def eval_mode(self):
        """Set model to evaluation mode"""
        self.backbone.eval()
        return self


# ===== Test =====
if __name__ == "__main__":
    student = StudentExtractor()
    student.train_mode()
    
    x = torch.randn(2, 3, 224, 224)
    feat_map = student(x)
    
    print("Student feature map:", feat_map.shape)  # [B, 96, 14, 14]
