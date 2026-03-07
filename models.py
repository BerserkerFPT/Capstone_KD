"""
Pretrained models with custom classifiers
"""
import torch
import torch.nn as nn
import timm
from torchvision import models
from config import Config


class CustomClassifier(nn.Module):
    """Custom fully connected layers for CNN models (no BatchNorm)"""
    def __init__(self, in_features, num_classes):
        super(CustomClassifier, self).__init__()
        
        layers = []
        prev_dim = in_features
        
        # Add hidden layers from config
        for hidden_dim in Config.CLASSIFIER_CONFIG:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=False),  # inplace=False is safer for gradient computation
                nn.Dropout(Config.DROPOUT_RATE)
            ])
            prev_dim = hidden_dim
        
        # Add final classification layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.classifier(x)


class TransformerClassifier(nn.Module):
    """Simple classifier for Vision Transformers (no BatchNorm)"""
    def __init__(self, in_features, num_classes):
        super(TransformerClassifier, self).__init__()
        
        layers = []
        prev_dim = in_features
        
        # Add hidden layers from config
        for hidden_dim in Config.CLASSIFIER_CONFIG:
            layers.extend([
                nn.LayerNorm(prev_dim),
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),  # GELU is better for Transformers
                nn.Dropout(Config.DROPOUT_RATE)
            ])
            prev_dim = hidden_dim
        
        # Add final classification layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.classifier(x)


def get_model(model_name, num_classes, freeze_backbone=False):
    """
    Get pretrained model with custom classifier
    
    Args:
        model_name: Name of the pretrained model
        num_classes: Number of output classes
        freeze_backbone: Whether to freeze backbone weights
    
    Returns:
        model: PyTorch model with custom classifier
    """
    
    if model_name == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        in_features = model.classifier[0].in_features
        # Freeze or unfreeze backbone
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
        else:
            for param in model.features.parameters():
                param.requires_grad = True
        model.classifier = CustomClassifier(in_features, num_classes)

    elif model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        # Freeze or unfreeze backbone
        if freeze_backbone:
            for name, param in model.named_parameters():
                if not name.startswith('fc'):
                    param.requires_grad = False
        else:
            for name, param in model.named_parameters():
                if not name.startswith('fc'):
                    param.requires_grad = True
        # Replace classifier
        model.fc = CustomClassifier(in_features, num_classes)

    elif model_name == 'resnet101':
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        # Freeze or unfreeze backbone
        if freeze_backbone:
            for name, param in model.named_parameters():
                if not name.startswith('fc'):
                    param.requires_grad = False
        else:
            for name, param in model.named_parameters():
                if not name.startswith('fc'):
                    param.requires_grad = True
        # Replace classifier
        model.fc = CustomClassifier(in_features, num_classes)

    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        # Freeze or unfreeze backbone
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
        else:
            for param in model.features.parameters():
                param.requires_grad = True
        # Replace classifier
        model.classifier = CustomClassifier(in_features, num_classes)

    elif model_name == 'densenet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        in_features = model.classifier.in_features
        # Freeze or unfreeze backbone
        if freeze_backbone:
            for name, param in model.named_parameters():
                if not name.startswith('classifier'):
                    param.requires_grad = False
        else:
            for name, param in model.named_parameters():
                if not name.startswith('classifier'):
                    param.requires_grad = True
        # Replace classifier
        model.classifier = CustomClassifier(in_features, num_classes)

    elif model_name == 'efficientnet_b0':
        model = timm.create_model('efficientnet_b0', pretrained=True)
        in_features = model.classifier.in_features
        # Freeze or unfreeze backbone
        if freeze_backbone:
            for name, param in model.named_parameters():
                if not name.startswith('classifier'):
                    param.requires_grad = False
        else:
            for name, param in model.named_parameters():
                if not name.startswith('classifier'):
                    param.requires_grad = True
        # Replace classifier
        model.classifier = CustomClassifier(in_features, num_classes)

    elif model_name == 'convnext_tiny':
        model = timm.create_model('convnext_tiny', pretrained=True)
        in_features = model.head.fc.in_features
        # Freeze or unfreeze backbone
        if freeze_backbone:
            for name, param in model.named_parameters():
                if not name.startswith('head'):
                    param.requires_grad = False
        else:
            for name, param in model.named_parameters():
                if not name.startswith('head'):
                    param.requires_grad = True
        # Replace classifier (no BatchNorm)
        model.head.fc = CustomClassifier(in_features, num_classes)

    elif model_name == 'vit_base_patch16_224':
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        in_features = model.head.in_features
        # Freeze or unfreeze backbone
        if freeze_backbone:
            for name, param in model.named_parameters():
                if not name.startswith('head'):
                    param.requires_grad = False
        else:
            for name, param in model.named_parameters():
                if not name.startswith('head'):
                    param.requires_grad = True
        # Replace classifier (Transformer - NO BatchNorm, use GELU)
        model.head = TransformerClassifier(in_features, num_classes)

    elif model_name == 'swin_tiny_patch4_window7_224':
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
        # Swin has ClassifierHead with nested fc, not direct Linear
        in_features = model.head.fc.in_features
        # Freeze or unfreeze backbone
        if freeze_backbone:
            for name, param in model.named_parameters():
                if not name.startswith('head'):
                    param.requires_grad = False
        else:
            for name, param in model.named_parameters():
                if not name.startswith('head'):
                    param.requires_grad = True
        # Replace only fc inside head to keep global_pool
        model.head.fc = TransformerClassifier(in_features, num_classes)

    elif model_name == 'convit_tiny':
        model = timm.create_model('convit_tiny', pretrained=True)
        in_features = model.head.in_features
        # Freeze or unfreeze backbone
        if freeze_backbone:
            for name, param in model.named_parameters():
                if not name.startswith('head'):
                    param.requires_grad = False
        else:
            for name, param in model.named_parameters():
                if not name.startswith('head'):
                    param.requires_grad = True
        # Replace classifier (Transformer - NO BatchNorm, use GELU)
        model.head = TransformerClassifier(in_features, num_classes)
    
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    return model


def count_parameters(model):
    """Count trainable and total parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


if __name__ == "__main__":
    # Test all models
    num_classes = 4
    print("\n" + "="*70)
    print("Testing all pretrained models")
    print("="*70)
    
    for model_name in Config.MODELS:
        try:
            model = get_model(model_name, num_classes, freeze_backbone=True)
            total, trainable = count_parameters(model)
            print(f"\n{model_name}:")
            print(f"  Total parameters: {total:,}")
            print(f"  Trainable parameters: {trainable:,}")
            print(f"  Frozen parameters: {total - trainable:,}")
            print(f"  ✓ Model loaded successfully")
        except Exception as e:
            print(f"\n{model_name}:")
            print(f"  ✗ Error: {str(e)}")
