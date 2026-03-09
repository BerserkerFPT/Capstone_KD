import torch
import torch.nn as nn
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Import modules
from dataset import DatasetHandler, set_seed
from Student_extraction import StudentExtractor

# ===== SEED CONSISTENCY =====
SEED = 42
set_seed(SEED)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = str(SEED)

class StudentWithHead(nn.Module):
    """Student model với classification head"""
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        self.backbone = StudentExtractor(pretrained=pretrained)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(96,512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        feat_map = self.backbone(x)
        pooled = self.gap(feat_map)
        pooled = pooled.flatten(1)
        logits = self.classifier(pooled)
        return feat_map, logits


@torch.no_grad()
def test_model(model, test_loader, device, class_names=None):
    model.eval()

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(test_loader, desc="Testing")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        _, logits = model(images)
        _, predicted = logits.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'acc': f'{100.*correct/total:.2f}%'})

    accuracy = 100. * correct / total

    result = {"accuracy": accuracy}

    if class_names is not None:
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        report = classification_report(
            all_labels, all_preds,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )

        cm = confusion_matrix(all_labels, all_preds)

        result["classification_report"] = report
        result["confusion_matrix"] = cm

    return result


def export_metrics_to_excel(metrics, class_names, save_path="test_metrics.xlsx"):
    """Export per-class metrics and confusion matrix to Excel"""
    report = metrics["classification_report"]
    cm = metrics["confusion_matrix"]

    with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
        # Sheet 1: Per-class metrics
        rows = []
        for cls_name in class_names:
            m = report[cls_name]
            rows.append({
                "Class": cls_name,
                "Precision": round(m["precision"], 4),
                "Recall": round(m["recall"], 4),
                "F1-Score": round(m["f1-score"], 4),
                "Support": int(m["support"])
            })
        for avg_type in ["macro avg", "weighted avg"]:
            m = report[avg_type]
            rows.append({
                "Class": avg_type.title(),
                "Precision": round(m["precision"], 4),
                "Recall": round(m["recall"], 4),
                "F1-Score": round(m["f1-score"], 4),
                "Support": int(m["support"])
            })
        rows.append({
            "Class": "Overall Accuracy",
            "Precision": "",
            "Recall": "",
            "F1-Score": round(report["accuracy"], 4),
            "Support": int(report["macro avg"]["support"])
        })

        df_metrics = pd.DataFrame(rows)
        df_metrics.to_excel(writer, sheet_name="Per-Class Metrics", index=False)

        # Sheet 2: Confusion Matrix
        df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
        df_cm.index.name = "Actual \\ Predicted"
        df_cm.to_excel(writer, sheet_name="Confusion Matrix")

    print(f"\n📁 Metrics exported to: {save_path}")


def main():
    # Config - PHẢI GIỐNG VỚI main.py
    config = {
        "data_dir": r"/TomatoDataset",
        "num_classes": 10,
        "batch_size": 32,
        "num_workers": 16,
        "checkpoint_path": "/Capstone_KD_Testing_Ver1/checkpoints/run_7/saved_checkpoints/strategy3_last_10_averaged.pth",  # Đường dẫn checkpoint
        "device": "cuda:0"
    }
    
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ===== QUAN TRỌNG: Dùng cùng seed và DatasetHandler =====
    data_handler = DatasetHandler(
        root_dir=config["data_dir"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        random_seed=SEED  # Phải giống với training
    )
    
    # Verify consistency
    print("\n===== Verifying split consistency =====")
    data_handler.verify_split_consistency()
    
    _, _, test_loader = data_handler.get_dataloaders()
    
    print(f"\nTest samples: {len(test_loader.dataset)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Classes: {data_handler.get_class_names()}")
    
    # Load model
    model = StudentWithHead(num_classes=config["num_classes"], pretrained=False)
    
    # Load checkpoint
    checkpoint = torch.load(config["checkpoint_path"], map_location=device)
    if 'student_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['student_state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    print(f"\n✅ Loaded checkpoint from: {config['checkpoint_path']}")
    
    # Test
    class_names = data_handler.get_class_names()
    metrics = test_model(model, test_loader, device, class_names=class_names)
    print(f"\n🏆 Test Accuracy: {metrics['accuracy']:.2f}%")

    # Print per-class metrics
    if "classification_report" in metrics:
        report = metrics["classification_report"]
        print("\n" + "="*60)
        print("📊 Per-Class Metrics")
        print("="*60)
        print(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
        print("-" * 65)
        for cls_name in class_names:
            m = report[cls_name]
            print(f"{cls_name:<25} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1-score']:>10.4f} {m['support']:>10.0f}")
        print("-" * 65)
        m = report["macro avg"]
        print(f"{'Macro Avg':<25} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1-score']:>10.4f} {m['support']:>10.0f}")
        m = report["weighted avg"]
        print(f"{'Weighted Avg':<25} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1-score']:>10.4f} {m['support']:>10.0f}")

        # Export to Excel
        save_dir = os.path.dirname(config["checkpoint_path"])
        excel_path = os.path.join(save_dir, "test_metrics.xlsx")
        export_metrics_to_excel(metrics, class_names, save_path=excel_path)


if __name__ == "__main__":
    main()