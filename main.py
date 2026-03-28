import os
import copy
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, confusion_matrix,
                            accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score)

# Import các module đã tạo
from Teacher_extraction import TeacherExtractor
from Student_extraction import StudentExtractor
from PCA_projector import PCAttentionProjector
from GWLinear_projector import GWLinearProjector
from loss_functions import ProjectionLoss, LogitsKDLoss, DIST, PolyFocalLoss, compute_class_weights
from dataset import DatasetHandler, set_seed
from visualization import plot_training_curves, plot_dwa_curves
torch.use_deterministic_algorithms(True, warn_only=True)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

class DynamicWeightAveraging:
    def __init__(self, num_tasks, temperature=2.0, initial_lambdas=None):
        self.num_tasks = num_tasks
        self.temperature = temperature
        self.loss_history = []
        self.initial_lambdas = initial_lambdas  # Custom weights for first 2 epochs

    def get_lambdas(self):
        t = len(self.loss_history) + 1
        if t <= 2:
            if self.initial_lambdas is not None:
                return self.initial_lambdas.clone()
            return torch.ones(self.num_tasks, dtype=torch.float32)

        L_t_minus_1 = self.loss_history[-1]
        L_t_minus_2 = self.loss_history[-2]
        
        w = L_t_minus_1 / (L_t_minus_2 + 1e-8)
        lambdas = self.num_tasks * torch.nn.functional.softmax(w / self.temperature, dim=0)
        return lambdas

    def update_loss_history(self, epoch_losses):
        if not isinstance(epoch_losses, torch.Tensor):
            epoch_losses = torch.tensor(epoch_losses, dtype=torch.float32)
        assert epoch_losses.size(0) == self.num_tasks
        self.loss_history.append(epoch_losses.clone().detach())
        if len(self.loss_history) > 2:
            self.loss_history.pop(0)


class StudentWithHead(nn.Module):
    """
    Student model với classification head
    """
    def __init__(self, num_classes, pretrained=True, feature_dim=96,
                 fc_hidden=None, fc_dropout=0.7):
        super().__init__()
        if fc_hidden is None:
            fc_hidden = [512, 256]
        self.backbone = StudentExtractor(pretrained=pretrained)
        
        # Classification head: Global Average Pooling + MLP
        self.gap = nn.AdaptiveAvgPool2d(1)
        layers = []
        in_dim = feature_dim
        for h in fc_hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(fc_dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Returns:
            feat_map: [B, 1024, 14, 14] - for distillation
            logits: [B, num_classes] - for classification
        """
        feat_map = self.backbone(x)  # [B, 1024, 14, 14]
        
        # Classification
        pooled = self.gap(feat_map)  # [B, 1024, 1, 1]
        pooled = pooled.flatten(1)   # [B, 1024]
        logits = self.classifier(pooled)  # [B, num_classes]
        
        return feat_map, logits


# =============================================================================
# CheckpointManager: keep last N + top K best checkpoints
# =============================================================================
class CheckpointManager:
    def __init__(self, save_dir, keep_last_n=10, keep_top_k=5):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.checkpoints = []  # List of (epoch, val_loss, path)
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.keep_last_n = keep_last_n
        self.keep_top_k = keep_top_k

    def save(self, student_state_dict, optimizer_state_dict, scheduler_state_dict,
             epoch, val_loss, val_acc, pca_projector_state_dict=None, gl_projector_state_dict=None):
        checkpoint = {
            'epoch': epoch,
            'student_state_dict': student_state_dict,
            'optimizer_student_state_dict': optimizer_state_dict,
            'scheduler_student_state_dict': scheduler_state_dict,
            'val_loss': val_loss,
            'val_acc': val_acc
        }
        path = os.path.join(self.save_dir, f'epoch_{epoch:03d}_val_loss_{val_loss:.4f}.pth')
        torch.save(checkpoint, path)

        self.checkpoints.append({
            'epoch': epoch, 'val_loss': val_loss, 'path': path
        })

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            best_path = os.path.join(self.save_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            if pca_projector_state_dict is not None:
                torch.save(pca_projector_state_dict, os.path.join(self.save_dir, 'best_pca_projector.pth'))
            if gl_projector_state_dict is not None:
                torch.save(gl_projector_state_dict, os.path.join(self.save_dir, 'best_gl_projector.pth'))
            print(f"\U0001f4be Best model saved (epoch {epoch}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.2f}%)")

        self._cleanup()
        return path

    def _cleanup(self):
        if len(self.checkpoints) <= self.keep_last_n + self.keep_top_k:
            return
        sorted_by_epoch = sorted(self.checkpoints, key=lambda x: x['epoch'])
        last_n = set(cp['epoch'] for cp in sorted_by_epoch[-self.keep_last_n:])
        sorted_by_loss = sorted(self.checkpoints, key=lambda x: x['val_loss'])
        top_k = set(cp['epoch'] for cp in sorted_by_loss[:self.keep_top_k])
        keep_epochs = last_n | top_k
        to_keep = []
        for cp in self.checkpoints:
            if cp['epoch'] in keep_epochs:
                to_keep.append(cp)
            else:
                p = cp['path']
                try:
                    if os.path.exists(p): os.remove(p)
                except Exception: pass
        self.checkpoints = to_keep

    def get_best_checkpoint(self):
        if not self.checkpoints:
            return None
        cp = min(self.checkpoints, key=lambda x: x['val_loss'])
        return (cp['epoch'], cp['val_loss'], cp['path'])

    def get_top_k_checkpoints(self, k):
        return [(cp['epoch'], cp['val_loss'], cp['path']) for cp in sorted(self.checkpoints, key=lambda x: x['val_loss'])[:k]]

    def get_last_n_checkpoints(self, n):
        return [(cp['epoch'], cp['val_loss'], cp['path']) for cp in sorted(self.checkpoints, key=lambda x: x['epoch'])[-n:]]

    def save_info(self):
        info = {'checkpoints': self.checkpoints}
        with open(os.path.join(self.save_dir, 'checkpoint_info.json'), 'w') as f:
            json.dump(info, f, indent=4)


# =============================================================================
# Helper functions for checkpoint averaging
# =============================================================================
def average_student_weights(checkpoint_paths, device):
    """Average student model weights, skip BN running stats"""
    if not checkpoint_paths:
        return None
    if len(checkpoint_paths) == 1:
        cp = torch.load(checkpoint_paths[0], map_location=device)
        return cp['student_state_dict']

    first = torch.load(checkpoint_paths[0], map_location=device)
    averaged = copy.deepcopy(first['student_state_dict'])

    keys_to_avg = []
    keys_to_keep = []
    for key in averaged.keys():
        if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
            keys_to_keep.append(key)
        else:
            keys_to_avg.append(key)

    for path in checkpoint_paths[1:]:
        cp = torch.load(path, map_location=device)
        sd = cp['student_state_dict']
        for key in keys_to_avg:
            averaged[key] = averaged[key] + sd[key]

    n = len(checkpoint_paths)
    for key in keys_to_avg:
        averaged[key] = averaged[key] / n

    return averaged


def update_bn_stats(model, train_loader, device, num_batches=100):
    """
    Update BatchNorm running statistics after loading averaged weights.
    
    IMPORTANT: For frozen backbone models, we should NOT update the backbone BN layers
    because they already have good statistics from ImageNet pretraining.
    We only update BN layers that are in trainable (unfrozen) parts.
    """
    # Identify which BN layers are in trainable parts
    trainable_bn_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            has_trainable = False
            for param in module.parameters():
                if param.requires_grad:
                    has_trainable = True
                    break
            if has_trainable:
                trainable_bn_layers.append((name, module))

    if not trainable_bn_layers:
        print("      (No trainable BN layers found, skipping BN update)")
        return

    print(f"      (Found {len(trainable_bn_layers)} trainable BN layers to update)")

    # Set model to eval mode first
    model.eval()

    # Only set trainable BN layers to train mode and reset their statistics
    for name, module in trainable_bn_layers:
        module.train()
        module.momentum = None  # Use cumulative moving average
        module.reset_running_stats()

    # Forward pass to accumulate BN statistics (no gradient computation)
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(train_loader):
            if batch_idx >= num_batches:
                break
            images = images.to(device)
            _ = model(images)

    # Set everything back to eval mode
    model.eval()


class DistillationPipeline:
    def __init__(
        self,
        data_dir,
        num_classes,
        batch_size=32,
        num_workers=16,
        lr_student=1e-4,
        # lr_teacher=1e-4,
        epochs=120,
        warmup_epochs_student=5,
        # warmup_epochs_teacher=5,
        device="cuda",
        save_dir="checkpoints",
        dwa_init_lambdas=None,  # list of 5 initial lambda values for DWA
        patience=15,  # early stopping patience
        start_factor_student=1e-8,
        # start_factor_teacher=1e-8,  # warmup start factor
        eta_min_student=1e-7,
        block_ids=[11,10,9,8,7],
        block_qkv_id=11,
        temperature=4.0,
        dist_beta=2.0,
        dist_gamma=2.0,
        last_n_epochs=10,
        keep_last_n=10,
        keep_top_k=5,
        # eta_min_teacher=1e-7,  # cosine annealing min lr
        teacher_checkpoint=None,
        student_fc_dropout=0.7,
        student_fc_hidden=None,
        pca_dropout=0.5,
        pca_partial_p=0.5,
        gw_drop_p=0.4,
        label_smoothing=0.1,
        use_projection=True,  # ablation: set False to skip PCA/GL projectors
        # --- Ablation: individual loss flags ---
        use_ce=True,      # Cross-Entropy loss
        use_proj1=True,   # L_proj1 PCA projection loss
        use_proj2=True,   # L_proj2 GWLinear projection loss
        use_logits=True,  # L_logits Hinton KD logits loss
        use_dist=True,    # L_dist DIST relational loss
        # --- DWA hyperparameters ---
        dwa_temperature=2.0,   # Temperature T for DWA softmax
        dwa_num_tasks=5,       # Auto-computed from active losses in Config
        # --- Random seed ---
        random_seed=42,
        # --- Weighted sampler & Focal loss ---
        use_weighted_sampler=False,
        use_focal_loss=False,
        focal_gamma=2.0,
        poly_epsilon=1.0,
        class_weight_method='inverse_freq',
    ):
        # ===== SET GLOBAL SEED FIRST (before any model/data init) =====
        self.random_seed = random_seed
        set_seed(self.random_seed)
        print(f"🌱 Global random seed set to: {self.random_seed}")

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.warmup_epochs_student = warmup_epochs_student
        # self.warmup_epochs_teacher = warmup_epochs_teacher
        self.save_dir = save_dir
        self.dwa_init_lambdas = dwa_init_lambdas
        self.temperature = temperature
        self.patience = patience
        self.start_factor_student = start_factor_student
        # self.start_factor_teacher = start_factor_teacher
        self.eta_min_student = eta_min_student
        self.dist_beta = dist_beta
        self.dist_gamma = dist_gamma
        self.use_projection = use_projection
        # --- Ablation flags ---
        self.use_ce     = use_ce
        self.use_proj1  = use_proj1 and use_projection  # proj1 requires use_projection
        self.use_proj2  = use_proj2 and use_projection  # proj2 requires use_projection
        self.use_logits = use_logits
        self.use_dist   = use_dist
        # --- DWA hyperparameters ---
        self.dwa_temperature = dwa_temperature
        self.dwa_num_tasks   = dwa_num_tasks
        # --- Weighted sampler & Focal loss ---
        self.use_weighted_sampler = use_weighted_sampler
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.poly_epsilon = poly_epsilon
        self.class_weight_method = class_weight_method
        if not self.use_ce:
            print("\u26a0\ufe0f  WARNING: USE_CE=False. CE loss is disabled — classification may fail!")
        # self.eta_min_teacher = eta_min_teacher,
        os.makedirs(save_dir, exist_ok=True)
        
        # Auto-detect experiment run number
        run_number = 1
        while os.path.exists(os.path.join(save_dir, f"run_{run_number}")):
            run_number += 1
        self.save_dir = os.path.join(save_dir, f"run_{run_number}")
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"📂 Experiment run #{run_number}, saving to: {self.save_dir}")
        
        # ===== Dataset =====
        print("Loading dataset...")
        self.data_handler = DatasetHandler(
            root_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            use_weighted_sampler=use_weighted_sampler,
            random_seed=self.random_seed
        )
        self.train_loader, self.val_loader, self.test_loader = self.data_handler.get_dataloaders()
        
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
        print(f"Num classes: {num_classes}")
        
        # ===== Models =====
        print("\nInitializing models...")
        
        # Teacher (frozen, inference only)
        self.teacher = TeacherExtractor(pretrained=False,
                                        checkpoint_path=teacher_checkpoint,
                                        block_ids=block_ids,
                                        block_qkv_id=block_qkv_id)
        self.teacher.to(self.device)
        print("✅ Teacher (ViT-B/16) loaded and frozen")
        
        # Student with classification head
        self.student_fc_dropout = student_fc_dropout
        self.student_fc_hidden = student_fc_hidden if student_fc_hidden else [512, 256]
        self.student = StudentWithHead(
            num_classes=num_classes, pretrained=True,
            fc_hidden=self.student_fc_hidden, fc_dropout=self.student_fc_dropout
        )
        self.student = self.student.to(self.device)
        print("✅ Student (ResNet-50) loaded")
        
        # # Teacher Head (trainable)
        # self.teacher_head = TeacherHead(num_classes=num_classes, embed_dim=768)
        # self.teacher_head = self.teacher_head.to(self.device)
        # print("✅ Teacher Head (trainable) loaded")
        
        # Projectors (only created when use_projection=True)
        self.pca_dropout = pca_dropout
        self.pca_partial_p = pca_partial_p
        self.gw_drop_p = gw_drop_p
        if self.use_projection:
            self.pca_projector = PCAttentionProjector(
                in_channels=96, embed_dim=768,
                p=self.pca_partial_p, dropout=self.pca_dropout
            )
            self.pca_projector = self.pca_projector.to(self.device)
            print("✅ PCA Projector loaded")
            
            self.gl_projector = GWLinearProjector(in_dim=96, out_dim=768, drop_p=self.gw_drop_p)
            self.gl_projector = self.gl_projector.to(self.device)
            print("✅ GL Projector loaded")
        else:
            self.pca_projector = None
            self.gl_projector = None
            print("⏭️  Projectors skipped (use_projection=False)")
        
        # ===== Loss functions =====
        self.label_smoothing = label_smoothing
        self.kd_loss_fn = ProjectionLoss() if self.use_projection else None
        if self.use_focal_loss:
            # Compute class weights from training data
            train_labels = self.data_handler.get_train_labels()
            class_weights = compute_class_weights(train_labels, method=self.class_weight_method)
            self.ce_loss_fn = PolyFocalLoss(
                gamma=self.focal_gamma,
                epsilon=self.poly_epsilon,
                alpha=class_weights
            )
            print(f"\u2705 PolyFocalLoss enabled (gamma={self.focal_gamma}, epsilon={self.poly_epsilon}, method={self.class_weight_method})")
        else:
            self.ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.logits_loss = LogitsKDLoss(temperature=temperature)
        self.dist_loss_fn = DIST(beta=dist_beta, gamma=dist_gamma)
        # ===== Optimizer (chỉ train student + projectors nếu có) =====
        trainable_params = list(self.student.parameters())
        if self.use_projection:
            trainable_params += list(self.pca_projector.parameters()) + \
                                list(self.gl_projector.parameters())
        # self.optimizer_teacher = optim.Adam(self.teacher_head.parameters(), lr=lr_teacher)
        self.optimizer_student = optim.Adam(trainable_params, lr=lr_student)
        
        # ===== Scheduler: Linear warmup + Cosine Annealing (epoch-level) =====
        self.scheduler_student = self._get_scheduler()
        
        # Store for evaluation strategies
        self.num_classes = num_classes
        self.last_n_epochs = last_n_epochs
        
        # Checkpoint Manager (keeps last N + top K best checkpoints)
        self.checkpoint_manager = CheckpointManager(
            save_dir=self.save_dir,
            keep_last_n=keep_last_n,
            keep_top_k=keep_top_k
        )
        
        print(f"\n✅ Pipeline initialized on {self.device}")

        # ===== Export config to Excel =====
        self._export_config_to_excel()
    
    def _export_config_to_excel(self):
        """
        Xuất toàn bộ hyperparameters và config của run hiện tại
        ra file Excel (config_run_<N>.xlsx) trong self.save_dir.

        Gồm 2 sheet:
          - "Config"         : tất cả hyperparameters
          - "Ablation Flags" : các flag bật/tắt loss
        """
        import datetime

        run_name = os.path.basename(self.save_dir)   # e.g. "run_3"
        excel_path = os.path.join(self.save_dir, f"config_{run_name}.xlsx")

        # ── Sheet 1: all hyperparameters ─────────────────────────────
        config_rows = [
            # ── Dataset ──
            {"Group": "Dataset",   "Parameter": "data_dir",              "Value": getattr(self.data_handler, 'root_dir', 'N/A')},
            {"Group": "Dataset",   "Parameter": "num_classes",           "Value": self.num_classes},
            {"Group": "Dataset",   "Parameter": "batch_size",            "Value": self.data_handler.batch_size
                                                                                   if hasattr(self.data_handler, 'batch_size') else 'N/A'},
            # ── Training ──
            {"Group": "Training",  "Parameter": "epochs",               "Value": self.epochs},
            {"Group": "Training",  "Parameter": "lr_student",           "Value": self.optimizer_student.param_groups[0]['lr']},
            {"Group": "Training",  "Parameter": "warmup_epochs_student","Value": self.warmup_epochs_student},
            {"Group": "Training",  "Parameter": "start_factor_student", "Value": self.start_factor_student},
            {"Group": "Training",  "Parameter": "eta_min_student",      "Value": self.eta_min_student},
            {"Group": "Training",  "Parameter": "patience",             "Value": self.patience},
            {"Group": "Training",  "Parameter": "label_smoothing",      "Value": self.label_smoothing},
            # ── Loss weights (initial λ) ──
            {"Group": "Loss Weights", "Parameter": "dwa_init_lambda_CE",     "Value": self.dwa_init_lambdas[0] if self.dwa_init_lambdas else 1.0},
            {"Group": "Loss Weights", "Parameter": "dwa_init_lambda_Proj1",  "Value": self.dwa_init_lambdas[1] if self.dwa_init_lambdas else 1.0},
            {"Group": "Loss Weights", "Parameter": "dwa_init_lambda_Proj2",  "Value": self.dwa_init_lambdas[2] if self.dwa_init_lambdas else 1.0},
            {"Group": "Loss Weights", "Parameter": "dwa_init_lambda_Logits", "Value": self.dwa_init_lambdas[3] if self.dwa_init_lambdas else 1.0},
            {"Group": "Loss Weights", "Parameter": "dwa_init_lambda_DIST",   "Value": self.dwa_init_lambdas[4] if self.dwa_init_lambdas else 1.0},
            # ── DWA ──
            {"Group": "DWA",       "Parameter": "dwa_temperature",      "Value": self.dwa_temperature},
            {"Group": "DWA",       "Parameter": "dwa_num_tasks",        "Value": self.dwa_num_tasks},
            # ── Logits KD ──
            {"Group": "KD",        "Parameter": "temperature (Hinton)", "Value": self.temperature},
            # ── DIST ──
            {"Group": "DIST",      "Parameter": "dist_beta",            "Value": self.dist_beta},
            {"Group": "DIST",      "Parameter": "dist_gamma",           "Value": self.dist_gamma},
            # ── Projectors ──
            {"Group": "Projector", "Parameter": "use_projection",       "Value": self.use_projection},
            {"Group": "Projector", "Parameter": "pca_dropout",          "Value": self.pca_dropout},
            {"Group": "Projector", "Parameter": "pca_partial_p",        "Value": self.pca_partial_p},
            {"Group": "Projector", "Parameter": "gw_drop_p",            "Value": self.gw_drop_p},
            # ── Student ──
            {"Group": "Student",   "Parameter": "fc_dropout",           "Value": self.student_fc_dropout},
            {"Group": "Student",   "Parameter": "fc_hidden",            "Value": str(self.student_fc_hidden)},
            # ── Checkpoint ──
            {"Group": "Checkpoint","Parameter": "save_dir",             "Value": self.save_dir},
            {"Group": "Checkpoint","Parameter": "keep_last_n",          "Value": self.checkpoint_manager.keep_last_n},
            {"Group": "Checkpoint","Parameter": "keep_top_k",           "Value": self.checkpoint_manager.keep_top_k},
            {"Group": "Checkpoint","Parameter": "last_n_epochs",        "Value": self.last_n_epochs},
            # ── Meta ──
            {"Group": "Meta",      "Parameter": "device",               "Value": str(self.device)},
            {"Group": "Meta",      "Parameter": "random_seed",          "Value": self.random_seed},
            {"Group": "Meta",      "Parameter": "timestamp",            "Value": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
            {"Group": "Meta",      "Parameter": "run",                  "Value": run_name},
        ]

        # ── Sheet 2: ablation flags ───────────────────────────────────
        ablation_rows = [
            {"Loss": "CE",     "Flag": "use_ce",     "Enabled": self.use_ce},
            {"Loss": "Proj1",  "Flag": "use_proj1",  "Enabled": self.use_proj1},
            {"Loss": "Proj2",  "Flag": "use_proj2",  "Enabled": self.use_proj2},
            {"Loss": "Logits", "Flag": "use_logits", "Enabled": self.use_logits},
            {"Loss": "DIST",   "Flag": "use_dist",   "Enabled": self.use_dist},
        ]

        df_config   = pd.DataFrame(config_rows)
        df_ablation = pd.DataFrame(ablation_rows)

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df_config.to_excel(writer,   sheet_name="Config",         index=False)
            df_ablation.to_excel(writer, sheet_name="Ablation Flags", index=False)

        print(f"📋 Config exported to: {excel_path}")

    def _get_scheduler(self):
        """
        Linear warmup + Cosine annealing scheduler using SequentialLR (epoch-level)
        Tạo scheduler riêng cho teacher và student
        """
        warmup_epochs_student = self.warmup_epochs_student
        cosine_epochs_student = self.epochs - self.warmup_epochs_student
        # warmup_epochs_teacher = self.warmup_epochs_teacher
        # cosine_epochs_teacher = self.epochs - self.warmup_epochs_teacher 
    
        # ===== SCHEDULER CHO STUDENT =====
        warmup_scheduler_student = LinearLR(
            self.optimizer_student,
            start_factor=self.start_factor_student,
            end_factor=1.0,
            total_iters=self.warmup_epochs_student
        )
        
        cosine_scheduler_student = CosineAnnealingLR(
            self.optimizer_student,
            T_max=cosine_epochs_student,
            eta_min=self.eta_min_student
        )
        
        scheduler_student = SequentialLR(
            self.optimizer_student,
            schedulers=[warmup_scheduler_student, cosine_scheduler_student],
            milestones=[warmup_epochs_student]
        )
        
        # # ===== SCHEDULER CHO TEACHER HEAD =====
        # warmup_scheduler_teacher = LinearLR(
        #     self.optimizer_teacher,
        #     start_factor=self.start_factor_teacher,
        #     end_factor=1.0,
        #     total_iters=self.warmup_epochs_teacher
        # )
        
        # cosine_scheduler_teacher = CosineAnnealingLR(
        #     self.optimizer_teacher,
        #     T_max=cosine_epochs_teacher,
        #     eta_min=self.eta_min_teacher
        # )
        
        # scheduler_teacher = SequentialLR(
        #     self.optimizer_teacher,
        #     schedulers=[warmup_scheduler_teacher, cosine_scheduler_teacher],
        #     milestones=[warmup_epochs_teacher]
        # )
        
        return scheduler_student
    
    def train_one_epoch(self, epoch, current_lambdas=None):
        """Train for one epoch."""
        # Build ordered list of active losses (same order DWA tracks them)
        # Order: [CE?, Proj1?, Proj2?, Logits?, DIST?]
        if current_lambdas is None:
            # Fallback: equal weights for all active losses
            current_lambdas = torch.ones(self.dwa_num_tasks)
        current_lambdas = current_lambdas.to(self.device)

        self.student.train()
        if self.use_projection:
            self.pca_projector.train()
            self.gl_projector.train()

        total_loss = 0.0
        total_kd_loss = 0.0
        total_logits_loss = 0.0
        total_ce_loss_s = 0.0
        total_l1 = 0.0
        total_l2 = 0.0
        total_dist_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # ===== Teacher forward (no grad) =====
            with torch.no_grad():
                teacher_out = self.teacher.extract(images)
            logit_t = teacher_out["logits"]
            if self.use_projection:
                Q_t = teacher_out["Q_t"]
                K_t = teacher_out["K_t"]
                V_t = teacher_out["V_t"]
                Attn_t = teacher_out["Attn_t"]
                h_t = teacher_out["block_mean"]  # [B, 196, 768]
            
            # ===== Student forward =====
            feat_map, logit_s = self.student(images)

            # ===== PCA & GL Projectors =====
            if self.use_projection:
                pca_out = self.pca_projector(feat_map, Q_t, K_t, V_t)
                PCAttn_s = pca_out["PCAttnS"]
                V_s = pca_out["VS"]
                h_s_proj = self.gl_projector(feat_map)
                l_proj1, l_proj2 = self.kd_loss_fn(Attn_t, PCAttn_s, V_t, V_s, h_t, h_s_proj)
            else:
                l_proj1 = torch.tensor(0.0, device=self.device)
                l_proj2 = torch.tensor(0.0, device=self.device)

            # ===== Compute individual losses =====
            ce_loss_s      = self.ce_loss_fn(logit_s, labels)
            logits_kd_loss = self.logits_loss(logit_s, logit_t.detach())
            dist_loss      = self.dist_loss_fn(logit_s, logit_t.detach())
            
            # ===== Build total loss dynamically (ablation-aware) =====
            # current_lambdas is indexed in activation order: only active losses
            # are given a DWA slot. We iterate the active flags to assign λ_i.
            loss_student = torch.tensor(0.0, device=self.device)
            lam_idx = 0

            if self.use_ce:
                loss_student = loss_student + current_lambdas[lam_idx] * ce_loss_s
                lam_idx += 1

            if self.use_proj1:
                loss_student = loss_student + current_lambdas[lam_idx] * l_proj1
                lam_idx += 1

            if self.use_proj2:
                loss_student = loss_student + current_lambdas[lam_idx] * l_proj2
                lam_idx += 1

            if self.use_logits:
                loss_student = loss_student + current_lambdas[lam_idx] * logits_kd_loss
                lam_idx += 1

            if self.use_dist:
                loss_student = loss_student + current_lambdas[lam_idx] * dist_loss
                # lam_idx += 1  # no need to increment after last

            # ===== Backward =====
            self.optimizer_student.zero_grad()
            loss_student.backward()
            self.optimizer_student.step()

            # ===== Metrics =====
            total_loss += loss_student.item()
            total_kd_loss += (l_proj1.item() + l_proj2.item())
            total_ce_loss_s += ce_loss_s.item()
            total_logits_loss += logits_kd_loss.item()
            total_l1 += l_proj1.item()
            total_l2 += l_proj2.item()
            total_dist_loss += dist_loss.item()
            _, predicted = logit_s.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Rebuild lam_idx for postfix display
            _idx = 0
            _lce  = (current_lambdas[_idx].item() if self.use_ce     else 0.0); _idx += int(self.use_ce)
            _lp1  = (current_lambdas[_idx].item() if self.use_proj1  else 0.0); _idx += int(self.use_proj1)
            _lp2  = (current_lambdas[_idx].item() if self.use_proj2  else 0.0); _idx += int(self.use_proj2)
            _llg  = (current_lambdas[_idx].item() if self.use_logits else 0.0); _idx += int(self.use_logits)
            _lds  = (current_lambdas[_idx].item() if self.use_dist   else 0.0)
            pbar.set_postfix({
                "Loss": f"{loss_student.item():.3f}",
                "CE":    f"{_lce  * ce_loss_s.item():.3f}" if self.use_ce     else "OFF",
                "Proj1": f"{_lp1  * l_proj1.item():.3f}"  if self.use_proj1  else "OFF",
                "Proj2": f"{_lp2  * l_proj2.item():.3f}"  if self.use_proj2  else "OFF",
                "Logits":f"{_llg  * logits_kd_loss.item():.3f}" if self.use_logits else "OFF",
                "DIST":  f"{_lds  * dist_loss.item():.3f}" if self.use_dist   else "OFF",
                "Acc":   f"{100.*correct/total:.1f}%",
                "LR":    f"{self.scheduler_student.get_last_lr()[0]:.4e}",
            })
        
        n_batches = len(self.train_loader)
        avg_loss      = total_loss      / n_batches
        avg_kd_loss   = total_kd_loss   / n_batches
        avg_ce_loss_s = total_ce_loss_s / n_batches
        accuracy      = 100. * correct / total
        avg_raw_l1    = total_l1           / n_batches
        avg_raw_l2    = total_l2           / n_batches
        avg_raw_l3    = total_logits_loss  / n_batches
        avg_raw_dist  = total_dist_loss    / n_batches

        # Rebuild λ* for weighted metrics
        _idx = 0
        _lce  = (_lam := current_lambdas[_idx].item() if self.use_ce     else 0.0); _idx += int(self.use_ce)
        _lp1  = (current_lambdas[_idx].item() if self.use_proj1  else 0.0); _idx += int(self.use_proj1)
        _lp2  = (current_lambdas[_idx].item() if self.use_proj2  else 0.0); _idx += int(self.use_proj2)
        _llg  = (current_lambdas[_idx].item() if self.use_logits else 0.0); _idx += int(self.use_logits)
        _lds  = (current_lambdas[_idx].item() if self.use_dist   else 0.0)

        return {
            "loss":         avg_loss,
            "kd_loss":      avg_kd_loss,
            "ce_loss_s":    avg_ce_loss_s,
            "raw_l1":       avg_raw_l1,
            "raw_l2":       avg_raw_l2,
            "raw_l3":       avg_raw_l3,
            "raw_dist":     avg_raw_dist,
            "ce_weighted":  avg_ce_loss_s * _lce,
            "l1_weighted":  avg_raw_l1    * _lp1,
            "l2_weighted":  avg_raw_l2    * _lp2,
            "l3_weighted":  avg_raw_l3    * _llg,
            "dist_weighted":avg_raw_dist  * _lds,
            "accuracy":     accuracy,
            "lambda_vals":  current_lambdas.cpu().tolist(),  # raw DWA lambdas
        }
    
    
    @torch.no_grad()
    def validate(self, loader, desc="Val", class_names=None, current_lambdas=None):
        """Validate on given loader using total KD loss (not just CE)"""
        self.student.eval()
        if self.use_projection:
            self.pca_projector.eval()
            self.gl_projector.eval()

        if current_lambdas is None:
            current_lambdas = torch.ones(self.dwa_num_tasks)
        current_lambdas = current_lambdas.to(self.device)

        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(loader, desc=f"[{desc}]")

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Teacher forward
            teacher_out = self.teacher.extract(images)
            logit_t = teacher_out["logits"]

            # Student forward
            feat_map, logit_s = self.student(images)

            # Projector losses
            if self.use_projection:
                Q_t = teacher_out["Q_t"]
                K_t = teacher_out["K_t"]
                V_t = teacher_out["V_t"]
                Attn_t = teacher_out["Attn_t"]
                h_t = teacher_out["block_mean"]
                pca_out = self.pca_projector(feat_map, Q_t, K_t, V_t)
                PCAttn_s = pca_out["PCAttnS"]
                V_s = pca_out["VS"]
                h_s_proj = self.gl_projector(feat_map)
                l_proj1, l_proj2 = self.kd_loss_fn(Attn_t, PCAttn_s, V_t, V_s, h_t, h_s_proj)
            else:
                l_proj1 = torch.tensor(0.0, device=self.device)
                l_proj2 = torch.tensor(0.0, device=self.device)

            # Compute all losses
            ce_loss = self.ce_loss_fn(logit_s, labels)
            logits_kd_loss = self.logits_loss(logit_s, logit_t.detach())
            dist_loss = self.dist_loss_fn(logit_s, logit_t.detach())

            # Build total loss (same weighting as training)
            loss = torch.tensor(0.0, device=self.device)
            lam_idx = 0
            if self.use_ce:
                loss = loss + current_lambdas[lam_idx] * ce_loss
                lam_idx += 1
            if self.use_proj1:
                loss = loss + current_lambdas[lam_idx] * l_proj1
                lam_idx += 1
            if self.use_proj2:
                loss = loss + current_lambdas[lam_idx] * l_proj2
                lam_idx += 1
            if self.use_logits:
                loss = loss + current_lambdas[lam_idx] * logits_kd_loss
                lam_idx += 1
            if self.use_dist:
                loss = loss + current_lambdas[lam_idx] * dist_loss

            total_loss += loss.item()

            # Accuracy
            _, predicted = logit_s.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Acc": f"{100.*correct/total:.2f}%"
            })

        avg_loss = total_loss / len(loader)
        accuracy = 100. * correct / total

        result = {
            "loss": avg_loss,
            "accuracy": accuracy
        }

        # Compute per-class metrics if class_names provided
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
            result["all_preds"] = all_preds
            result["all_labels"] = all_labels

        return result
    
    def save_checkpoint(self, epoch, val_loss, val_acc, is_best=False):
        """Save checkpoint using CheckpointManager + latest.pth for resume"""
        student_sd = self.student.state_dict()
        optimizer_sd = self.optimizer_student.state_dict()
        scheduler_sd = self.scheduler_student.state_dict()
        
        pca_sd = self.pca_projector.state_dict() if self.use_projection else None
        gl_sd = self.gl_projector.state_dict() if self.use_projection else None

        # Save via CheckpointManager (handles best.pth + cleanup internally)
        self.checkpoint_manager.save(
            student_state_dict=student_sd,
            optimizer_state_dict=optimizer_sd,
            scheduler_state_dict=scheduler_sd,
            epoch=epoch + 1,
            val_loss=val_loss,
            val_acc=val_acc,
            pca_projector_state_dict=pca_sd,
            gl_projector_state_dict=gl_sd
        )
        
        # Also save latest.pth for resume training
        latest = {
            "epoch": epoch + 1,
            "student_state_dict": student_sd,
            "optimizer_student_state_dict": optimizer_sd,
            "scheduler_student_state_dict": scheduler_sd,
            "val_loss": val_loss,
            "val_acc": val_acc
        }
        torch.save(latest, os.path.join(self.save_dir, "latest.pth"))
        if pca_sd is not None:
            torch.save(pca_sd, os.path.join(self.save_dir, "latest_pca_projector.pth"))
        if gl_sd is not None:
            torch.save(gl_sd, os.path.join(self.save_dir, "latest_gl_projector.pth"))
            
    def load_checkpoint(self, path):
        """Load checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.student.load_state_dict(checkpoint["student_state_dict"])
        
        if self.use_projection:
            dir_name = os.path.dirname(path)
            base_name = os.path.basename(path)
            
            if "latest" in base_name or "best" in base_name:
                prefix = base_name.split("_")[0] # latest or best
                pca_path = os.path.join(dir_name, f"{prefix}_pca_projector.pth")
                gl_path = os.path.join(dir_name, f"{prefix}_gl_projector.pth")
            else:
                prefix = base_name.split("_val_loss")[0] # epoch_010
                pca_path = os.path.join(dir_name, f"{prefix}_pca.pth")
                gl_path = os.path.join(dir_name, f"{prefix}_gl.pth")
                
            if os.path.exists(pca_path):
                self.pca_projector.load_state_dict(torch.load(pca_path, map_location=self.device))
            if os.path.exists(gl_path):
                self.gl_projector.load_state_dict(torch.load(gl_path, map_location=self.device))

        self.optimizer_student.load_state_dict(checkpoint["optimizer_student_state_dict"])
        self.scheduler_student.load_state_dict(checkpoint["scheduler_student_state_dict"])
        
        val_loss = checkpoint.get('val_loss', float('inf'))
        val_acc = checkpoint.get('val_acc', 0.0)
        print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']} with val_loss: {val_loss:.4f}, val_acc: {val_acc:.2f}%")
        
        return checkpoint["epoch"], val_loss
    
    def train(self, resume_path=None):
        """Full training loop"""
        start_epoch = 0
        best_val_loss = float('inf')  # Lower is better
        epochs_no_improve = 0  # Early stopping counter

        # Khởi tạo DWA với num_tasks và temperature từ config
        # (dwa_num_tasks được auto-tính từ các USE_* flags trong Config)
        # Build initial_lambdas from config (only for active losses)
        init_lambdas = None
        if self.dwa_init_lambdas is not None:
            all_lambdas = self.dwa_init_lambdas  # [CE, Proj1, Proj2, Logits, DIST]
            active = []
            flags = [self.use_ce, self.use_proj1, self.use_proj2, self.use_logits, self.use_dist]
            for flag, lam in zip(flags, all_lambdas):
                if flag:
                    active.append(lam)
            init_lambdas = torch.tensor(active, dtype=torch.float32)
        dwa = DynamicWeightAveraging(
            num_tasks=self.dwa_num_tasks,
            temperature=self.dwa_temperature,
            initial_lambdas=init_lambdas
        )

        history = {
            "train_loss": [],
            "val_loss":   [],
            "train_acc":  [],
            "val_acc":    [],
            "lr":         [],
            "raw_ce":     [],
            "raw_l1":     [],
            "raw_l2":     [],
            "raw_l3":     [],
            "raw_dist":   [],
            "lambda_ce":  [],
            "lambda_l1":  [],
            "lambda_l2":  [],
            "lambda_l3":  [],
            "lambda_dist":[],
            "epoch_time": []
        }
        training_start_time = time.time()

        if resume_path and os.path.exists(resume_path):
            start_epoch, best_val_loss = self.load_checkpoint(resume_path)
            # start_epoch += 1

        print("\n" + "="*60)
        print("🚀 Starting Training")
        print(f"   Early Stopping: patience = {self.patience}")
        print("="*60)

        for epoch in range(start_epoch, self.epochs):
            # Lấy lambda động cho epoch hiện tại
            current_lambdas = dwa.get_lambdas()

            # Log DWA lambdas theo active losses
            active_names  = (["CE"]     if self.use_ce     else []) + \
                            (["Proj1"]  if self.use_proj1  else []) + \
                            (["Proj2"]  if self.use_proj2  else []) + \
                            (["Logits"] if self.use_logits else []) + \
                            (["DIST"]   if self.use_dist   else [])
            lam_str = ", ".join(f"{n}={current_lambdas[i].item():.4f}"
                                for i, n in enumerate(active_names))
            print(f"\n[DWA] Epoch {epoch+1} lambdas: {lam_str}")

            # Train
            epoch_start_time = time.time()
            train_metrics = self.train_one_epoch(epoch, current_lambdas)

            # Cập nhật DWA: chỉ đưa losses của các tasks đang active
            active_losses = []
            if self.use_ce:     active_losses.append(train_metrics["ce_loss_s"])
            if self.use_proj1:  active_losses.append(train_metrics["raw_l1"])
            if self.use_proj2:  active_losses.append(train_metrics["raw_l2"])
            if self.use_logits: active_losses.append(train_metrics["raw_l3"])
            if self.use_dist:   active_losses.append(train_metrics["raw_dist"])
            dwa.update_loss_history(torch.tensor(active_losses))
            
            # Validate
            val_metrics = self.validate(self.val_loader, desc="Val", current_lambdas=current_lambdas)
            epoch_elapsed = time.time() - epoch_start_time

            # Get current LR (before step)
            current_lr_student = self.scheduler_student.get_last_lr()[0]
            # current_lr_teacher = self.scheduler_teacher.get_last_lr()[0]

            # Record history
            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["train_acc"].append(train_metrics["accuracy"])
            history["val_acc"].append(val_metrics["accuracy"])
            history["lr"].append(current_lr_student)

            history["raw_ce"].append(train_metrics["ce_loss_s"])
            history["raw_l1"].append(train_metrics["raw_l1"])
            history["raw_l2"].append(train_metrics["raw_l2"])
            history["raw_l3"].append(train_metrics["raw_l3"])
            history["raw_dist"].append(train_metrics["raw_dist"])
            history["epoch_time"].append(round(epoch_elapsed, 2))

            # Lambda history: map DWA slots → fixed 5-key history
            lam_idx = 0
            history["lambda_ce"].append(current_lambdas[lam_idx].item() if self.use_ce else 0.0)
            lam_idx += int(self.use_ce)
            history["lambda_l1"].append(current_lambdas[lam_idx].item() if self.use_proj1 else 0.0)
            lam_idx += int(self.use_proj1)
            history["lambda_l2"].append(current_lambdas[lam_idx].item() if self.use_proj2 else 0.0)
            lam_idx += int(self.use_proj2)
            history["lambda_l3"].append(current_lambdas[lam_idx].item() if self.use_logits else 0.0)
            lam_idx += int(self.use_logits)
            history["lambda_dist"].append(current_lambdas[lam_idx].item() if self.use_dist else 0.0)

            # Step scheduler (epoch-level)
            self.scheduler_student.step()
            # self.scheduler_teacher.step()

            # Print epoch summary
            print(f"\n📊 Epoch {epoch+1}/{self.epochs} Summary (LR_S: {current_lr_student:.6f})")
            print(f"   Train - Loss: {train_metrics['loss']:.4f}, "
                  f"CE: {train_metrics['ce_weighted']:.4f}, "
                  f"Proj1: {train_metrics['l1_weighted']:.4f}, "
                  f"Proj2: {train_metrics['l2_weighted']:.4f}, "
                  f"Logits: {train_metrics['l3_weighted']:.4f}, "
                  f"DIST: {train_metrics['dist_weighted']:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.2f}%")
            print(f"   Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.2f}%")
            # Save checkpoint (based on lowest val_loss)
            is_best = val_metrics["loss"] < best_val_loss
            if is_best:
                best_val_loss = val_metrics["loss"]
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            self.save_checkpoint(epoch, val_metrics["loss"], val_metrics["accuracy"], is_best)
            
            # Early stopping check
            if epochs_no_improve >= self.patience:
                print(f"\n⚠️ Early stopping triggered! No improvement for {self.patience} epochs.")
                print(f"   Best val_loss: {best_val_loss:.4f}")
                break
            
            print(f"   Early stopping: {epochs_no_improve}/{self.patience}")
            print()
        
        # Save checkpoint manager info
        self.checkpoint_manager.save_info()

        history["total_training_time"] = round(time.time() - training_start_time, 2)

        # ===== Plot learning curves =====
        plot_training_curves(history, self.save_dir)
        plot_dwa_curves(history, self.save_dir)

        # ===== Evaluate all 3 strategies =====
        all_results = self.evaluate_all_strategies(history)
        
        # ===== Cleanup training checkpoints, keep only strategy files =====
        self._cleanup_training_checkpoints()
        
        return all_results

    def _export_metrics_to_excel(self, metrics, class_names):
        """Export per-class metrics and confusion matrix to Excel"""
        report = metrics["classification_report"]
        cm = metrics["confusion_matrix"]

        excel_path = os.path.join(self.save_dir, "test_metrics.xlsx")

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
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
            # Add overall metrics
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

        print(f"\n📁 Metrics exported to: {excel_path}")

    # =================================================================
    # Evaluation Strategy Methods
    # =================================================================
    @torch.no_grad()
    def evaluate_model_full(self, model, loader, class_names):
        """Full evaluation: per-class precision/recall/F1, AUC, confusion matrix"""
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        running_loss = 0.0
        total = 0
        criterion = nn.CrossEntropyLoss()

        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)
            _, logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            _, preds = logits.max(1)
            running_loss += loss.item() * images.size(0)
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        test_loss = running_loss / total
        accuracy = accuracy_score(all_labels, all_preds) * 100
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0) * 100
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0) * 100
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100
        try:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro') * 100
        except Exception:
            auc = 0.0

        report = classification_report(
            all_labels, all_preds,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        cm = confusion_matrix(all_labels, all_preds)

        return {
            'Test Loss': test_loss,
            'Accuracy (%)': accuracy,
            'Precision (%)': precision,
            'Recall (%)': recall,
            'F1-Score (%)': f1,
            'AUC (%)': auc,
            'classification_report': report,
            'confusion_matrix': cm,
        }

    def _create_student_model(self):
        """Create a fresh StudentWithHead for loading averaged weights"""
        model = StudentWithHead(
            num_classes=self.num_classes, pretrained=False,
            fc_hidden=self.student_fc_hidden, fc_dropout=self.student_fc_dropout
        )
        return model.to(self.device)

    def _print_strategy_results(self, metrics, strategy_name, class_names):
        """Print evaluation results for one strategy"""
        print(f"    {'='*60}")
        print(f"    📊 TEST RESULTS - {strategy_name}:")
        print(f"    {'='*60}")
        print(f"    Test Loss : {metrics['Test Loss']:>8.4f}")
        print(f"    Accuracy  : {metrics['Accuracy (%)']:>8.2f}%")
        print(f"    Precision : {metrics['Precision (%)']:>8.2f}%")
        print(f"    Recall    : {metrics['Recall (%)']:>8.2f}%")
        print(f"    F1-Score  : {metrics['F1-Score (%)']:>8.2f}%")
        print(f"    AUC       : {metrics['AUC (%)']:>8.2f}%")
        print(f"    {'='*60}")
        if 'classification_report' in metrics:
            report = metrics['classification_report']
            print(f"    {'Class':<25} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
            print(f"    {'-'*65}")
            for cls_name in class_names:
                m = report[cls_name]
                print(f"    {cls_name:<25} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1-score']:>10.4f} {m['support']:>10.0f}")
            print(f"    {'-'*65}")
            m = report['macro avg']
            print(f"    {'Macro Avg':<25} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1-score']:>10.4f} {m['support']:>10.0f}")
            m = report['weighted avg']
            print(f"    {'Weighted Avg':<25} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1-score']:>10.4f} {m['support']:>10.0f}")

    def strategy_1_best_checkpoint(self, class_names):
        """Strategy 1: Evaluate best checkpoint (lowest val_loss)"""
        print(f"\n  Strategy 1: Best checkpoint (lowest val_loss)")
        best = self.checkpoint_manager.get_best_checkpoint()
        if best is None:
            print("    No checkpoints available!")
            return None

        epoch, val_loss, path = best
        print(f"    Best checkpoint: Epoch {epoch}, Val Loss: {val_loss:.4f}")

        model = self._create_student_model()
        cp = torch.load(path, map_location=self.device)
        model.load_state_dict(cp['student_state_dict'])

        # Save strategy checkpoint
        save_dir = os.path.join(self.save_dir, 'saved_checkpoints')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'strategy1_best_epoch_{epoch}.pth')
        torch.save({'student_state_dict': model.state_dict(), 'epoch': epoch, 'val_loss': val_loss}, save_path)
        print(f"    ✓ Saved to: {save_path}")

        metrics = self.evaluate_model_full(model, self.test_loader, class_names)
        self._print_strategy_results(metrics, "Strategy 1 (Best Checkpoint)", class_names)
        return metrics

    def strategy_2_top_k_average(self, class_names):
        """Strategy 2: Average top-K checkpoints (K=2,3,4,5) and evaluate"""
        print(f"\n  Strategy 2: Top-K checkpoint averaging")
        results = {}

        for k in [2, 3, 4, 5]:
            print(f"    K={k}:")
            top_k = self.checkpoint_manager.get_top_k_checkpoints(k)

            if len(top_k) < k:
                print(f"      Warning: Only {len(top_k)} checkpoints available")
            if not top_k:
                continue

            paths = [p for _, _, p in top_k]
            avg_weights = average_student_weights(paths, self.device)

            model = self._create_student_model()
            model.load_state_dict(avg_weights, strict=True)

            print(f"      Updating BatchNorm statistics...")
            update_bn_stats(model, self.train_loader, self.device, num_batches=100)

            # Save
            save_dir = os.path.join(self.save_dir, 'saved_checkpoints')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'strategy2_top_{k}_averaged.pth')
            torch.save({'student_state_dict': model.state_dict(), 'k': k}, save_path)
            print(f"      ✓ Saved to: {save_path}")

            metrics = self.evaluate_model_full(model, self.test_loader, class_names)
            self._print_strategy_results(metrics, f"Strategy 2 (Top-{k} Avg)", class_names)
            results[k] = metrics

        return results

    def strategy_3_last_n_average(self, class_names):
        """Strategy 3: Average last N epoch checkpoints"""
        print(f"\n  Strategy 3: Last {self.last_n_epochs} epochs averaging")
        last_n = self.checkpoint_manager.get_last_n_checkpoints(self.last_n_epochs)

        if not last_n:
            print("    No checkpoints available!")
            return None
        if len(last_n) < self.last_n_epochs:
            print(f"    Warning: Only {len(last_n)} checkpoints available")

        epochs = [e for e, _, _ in last_n]
        paths = [p for _, _, p in last_n]
        print(f"    Averaging epochs: {epochs}")

        avg_weights = average_student_weights(paths, self.device)

        model = self._create_student_model()
        model.load_state_dict(avg_weights, strict=True)

        print(f"    Updating BatchNorm statistics...")
        update_bn_stats(model, self.train_loader, self.device, num_batches=100)

        # Save
        save_dir = os.path.join(self.save_dir, 'saved_checkpoints')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'strategy3_last_{self.last_n_epochs}_averaged.pth')
        torch.save({'student_state_dict': model.state_dict(), 'epochs': epochs}, save_path)
        print(f"    ✓ Saved to: {save_path}")

        metrics = self.evaluate_model_full(model, self.test_loader, class_names)
        self._print_strategy_results(metrics, f"Strategy 3 (Last {self.last_n_epochs} Avg)", class_names)
        return metrics

    def evaluate_all_strategies(self, history=None):
        """Run all 3 evaluation strategies and export results to Excel"""
        print("\n" + "="*70)
        print("🧪 Evaluating All Strategies")
        print("="*70)

        class_names = self.data_handler.get_class_names()
        all_results = {}

        # Strategy 1: Best single checkpoint
        metrics_1 = self.strategy_1_best_checkpoint(class_names)
        if metrics_1:
            all_results['Strategy 1 (Best)'] = metrics_1

        # Strategy 2: Top-K averaging
        strategy_2 = self.strategy_2_top_k_average(class_names)
        for k, metrics in strategy_2.items():
            all_results[f'Strategy 2 (Top-{k} Avg)'] = metrics

        # Strategy 3: Last N epochs averaging
        metrics_3 = self.strategy_3_last_n_average(class_names)
        if metrics_3:
            all_results[f'Strategy 3 (Last {self.last_n_epochs} Avg)'] = metrics_3

        # Export all results to Excel
        self._export_all_strategies_to_excel(all_results, class_names, history)

        # Print summary table
        print("\n" + "="*70)
        print("📊 SUMMARY OF ALL STRATEGIES")
        print("="*70)
        print(f"{'Strategy':<35} {'Accuracy':>10} {'F1-Score':>10} {'AUC':>10}")
        print("-" * 70)
        for name, m in all_results.items():
            print(f"{name:<35} {m['Accuracy (%)']:>9.2f}% {m['F1-Score (%)']:>9.2f}% {m['AUC (%)']:>9.2f}%")
        print("=" * 70)

        return all_results

    def _export_all_strategies_to_excel(self, all_results, class_names, history=None):
        """Export all strategy results to Excel with 3 sheets: Macro Results + Per-Class Metrics + Lambda weight"""
        excel_path = os.path.join(self.save_dir, "all_strategies_results.xlsx")

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            # ===== Sheet 1: Macro Results =====
            summary_rows = []
            for strategy_name, metrics in all_results.items():
                summary_rows.append({
                    "Strategy": strategy_name,
                    "Test Loss": round(metrics['Test Loss'], 4),
                    "Accuracy (%)": round(metrics['Accuracy (%)'], 2),
                    "Precision (%)": round(metrics['Precision (%)'], 2),
                    "Recall (%)": round(metrics['Recall (%)'], 2),
                    "F1-Score (%)": round(metrics['F1-Score (%)'], 2),
                    "AUC (%)": round(metrics['AUC (%)'], 2),
                })
            df_summary = pd.DataFrame(summary_rows)
            df_summary.to_excel(writer, sheet_name="Macro Results", index=False)

            # ===== Sheet 2: Per-Class Metrics for all strategies =====
            per_class_rows = []
            for strategy_name, metrics in all_results.items():
                if 'classification_report' not in metrics or 'confusion_matrix' not in metrics:
                    continue
                report = metrics['classification_report']
                cm = metrics['confusion_matrix']

                for idx, cls_name in enumerate(class_names):
                    m = report[cls_name]
                    # Per-class accuracy = correctly classified / total samples of this class
                    cls_total = cm[idx].sum()
                    cls_accuracy = (cm[idx][idx] / cls_total * 100) if cls_total > 0 else 0.0
                    per_class_rows.append({
                        "Strategy": strategy_name,
                        "Class": cls_name,
                        "Accuracy (%)": round(cls_accuracy, 2),
                        "Precision (%)": round(m["precision"] * 100, 2),
                        "Recall (%)": round(m["recall"] * 100, 2),
                        "F1-Score (%)": round(m["f1-score"] * 100, 2),
                        "Support": int(m["support"])
                    })

            if per_class_rows:
                df_per_class = pd.DataFrame(per_class_rows)
                df_per_class.to_excel(writer, sheet_name="Per-Class Metrics", index=False)
                
            # ===== Sheet 3: Lambda weight at lowest val_loss (Strategy 1) =====
            if history:
                best = self.checkpoint_manager.get_best_checkpoint()
                if best is not None:
                    best_epoch = best[0]
                    idx = best_epoch - 1
                    
                    if idx >= 0 and idx < len(history.get("lambda_ce", [])):
                        lambda_dict = {
                            "Strategy": "Strategy 1 (Best)",
                            "Best Epoch": best_epoch,
                        }
                        if self.use_ce:
                            lambda_dict["Lambda_CE"] = round(history["lambda_ce"][idx], 4)
                        if self.use_proj1:
                            lambda_dict["Lambda_Proj1"] = round(history["lambda_l1"][idx], 4)
                        if self.use_proj2:
                            lambda_dict["Lambda_Proj2"] = round(history["lambda_l2"][idx], 4)
                        if self.use_logits:
                            lambda_dict["Lambda_Logits"] = round(history["lambda_l3"][idx], 4)
                        if self.use_dist:
                            lambda_dict["Lambda_DIST"] = round(history["lambda_dist"][idx], 4)

                        df_lambdas = pd.DataFrame([lambda_dict])
                        df_lambdas.to_excel(writer, sheet_name="Lambda weight", index=False)

            # ===== Sheet 4: Training Time =====
            if history and "epoch_time" in history:
                epoch_times = history["epoch_time"]
                total_time = history.get("total_training_time", sum(epoch_times))
                time_rows = []
                for i, t in enumerate(epoch_times):
                    m, s = divmod(int(t), 60)
                    time_rows.append({
                        "Epoch": i + 1,
                        "Time (s)": t,
                        "Time (mm:ss)": f"{m:02d}:{s:02d}"
                    })
                total_m, total_s = divmod(int(total_time), 60)
                time_rows.append({
                    "Epoch": "Total",
                    "Time (s)": total_time,
                    "Time (mm:ss)": f"{total_m:02d}:{total_s:02d}"
                })
                df_time = pd.DataFrame(time_rows)
                df_time.to_excel(writer, sheet_name="Training Time", index=False)

        print(f"\n📁 All strategies results exported to: {excel_path}")

    def _cleanup_training_checkpoints(self):
        """
        Xóa tất cả checkpoint training (epoch_*.pth, best.pth, latest.pth, checkpoint_info.json)
        sau khi đã evaluate xong. Chỉ giữ lại folder saved_checkpoints/ chứa strategy files.
        """
        print("\n🧹 Cleaning up training checkpoints...")
        kept = 0
        removed = 0
        saved_cp_dir = os.path.join(self.save_dir, 'saved_checkpoints')

        for fname in os.listdir(self.save_dir):
            fpath = os.path.join(self.save_dir, fname)
            # Skip the saved_checkpoints directory and the Excel results
            if os.path.isdir(fpath):
                continue
            if fname.endswith('.xlsx') or fname.endswith('.csv'):
                kept += 1
                continue
            if (fname.startswith('epoch_') and fname.endswith('.pth')) or fname == 'checkpoint_info.json':
                try:
                    os.remove(fpath)
                    removed += 1
                except Exception as e:
                    print(f"   Warning: Could not delete {fpath}: {e}")
            else:
                kept += 1

        # Count strategy files kept
        strategy_files = 0
        if os.path.isdir(saved_cp_dir):
            strategy_files = len([f for f in os.listdir(saved_cp_dir) if f.endswith('.pth')])

        print(f"   Removed {removed} training checkpoint files")
        print(f"   Kept {strategy_files} strategy checkpoint files in saved_checkpoints/")
        print(f"   Kept {kept} other files (Excel, etc.)")

    def get_student_model(self):
        """
        Trả về student model (không có projectors) để inference
        """
        return self.student


# ===== Main =====
if __name__ == "__main__":
    from config import Config

    os.environ["CUDA_VISIBLE_DEVICES"] = Config.CUDA_VISIBLE_DEVICES

    Config.print_config()
    config = Config.to_pipeline_dict()
    print(f"[KD] block_ids = {config['block_ids']}")
    print(f"[KD] block_qkv_id = {config['block_qkv_id']}")
    pipeline = DistillationPipeline(**config)
    pipeline.train()






























# import os
# import copy
# import json
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
# from tqdm import tqdm
# import numpy as np
# import pandas as pd
# from sklearn.metrics import (classification_report, confusion_matrix,
#                             accuracy_score, precision_score, recall_score,
#                             f1_score, roc_auc_score)

# # Import các module đã tạo
# from Teacher_extraction import TeacherExtractor
# from Student_extraction import StudentExtractor
# from PCA_projector import PCAttentionProjector
# from GWLinear_projector import GWLinearProjector
# from loss_functions import ProjectionLoss, LogitsKDLoss, DIST
# from dataset import DatasetHandler
# from visualization import plot_training_curves
# torch.use_deterministic_algorithms(True, warn_only=True)

# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# class StudentWithHead(nn.Module):
#     """
#     Student model với classification head
#     """
#     def __init__(self, num_classes, pretrained=True, feature_dim=96,
#                  fc_hidden=None, fc_dropout=0.7):
#         super().__init__()
#         if fc_hidden is None:
#             fc_hidden = [512, 256]
#         self.backbone = StudentExtractor(pretrained=pretrained)
        
#         # Classification head: Global Average Pooling + MLP
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         layers = []
#         in_dim = feature_dim
#         for h in fc_hidden:
#             layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(fc_dropout)]
#             in_dim = h
#         layers.append(nn.Linear(in_dim, num_classes))
#         self.classifier = nn.Sequential(*layers)
    
#     def forward(self, x):
#         """
#         Returns:
#             feat_map: [B, 1024, 14, 14] - for distillation
#             logits: [B, num_classes] - for classification
#         """
#         feat_map = self.backbone(x)  # [B, 1024, 14, 14]
        
#         # Classification
#         pooled = self.gap(feat_map)  # [B, 1024, 1, 1]
#         pooled = pooled.flatten(1)   # [B, 1024]
#         logits = self.classifier(pooled)  # [B, num_classes]
        
#         return feat_map, logits


# # =============================================================================
# # CheckpointManager: keep last N + top K best checkpoints
# # =============================================================================
# class CheckpointManager:
#     def __init__(self, save_dir, keep_last_n=10, keep_top_k=5):
#         self.save_dir = save_dir
#         os.makedirs(self.save_dir, exist_ok=True)
#         self.checkpoints = []  # List of (epoch, val_loss, path)
#         self.best_val_loss = float('inf')
#         self.best_epoch = 0
#         self.keep_last_n = keep_last_n
#         self.keep_top_k = keep_top_k

#     def save(self, student_state_dict, optimizer_state_dict, scheduler_state_dict,
#              epoch, val_loss, val_acc):
#         checkpoint = {
#             'epoch': epoch,
#             'student_state_dict': student_state_dict,
#             'optimizer_student_state_dict': optimizer_state_dict,
#             'scheduler_student_state_dict': scheduler_state_dict,
#             'val_loss': val_loss,
#             'val_acc': val_acc
#         }
#         path = os.path.join(self.save_dir, f'epoch_{epoch:03d}_val_loss_{val_loss:.4f}.pth')
#         torch.save(checkpoint, path)
#         self.checkpoints.append((epoch, val_loss, path))

#         if val_loss < self.best_val_loss:
#             self.best_val_loss = val_loss
#             self.best_epoch = epoch
#             best_path = os.path.join(self.save_dir, 'best.pth')
#             torch.save(checkpoint, best_path)
#             print(f"\U0001f4be Best model saved (epoch {epoch}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.2f}%)")

#         self._cleanup()
#         return path

#     def _cleanup(self):
#         if len(self.checkpoints) <= self.keep_last_n + self.keep_top_k:
#             return
#         sorted_by_epoch = sorted(self.checkpoints, key=lambda x: x[0])
#         last_n = set(cp[0] for cp in sorted_by_epoch[-self.keep_last_n:])
#         sorted_by_loss = sorted(self.checkpoints, key=lambda x: x[1])
#         top_k = set(cp[0] for cp in sorted_by_loss[:self.keep_top_k])
#         keep_epochs = last_n | top_k
#         to_keep = []
#         for epoch, val_loss, path in self.checkpoints:
#             if epoch in keep_epochs:
#                 to_keep.append((epoch, val_loss, path))
#             else:
#                 try:
#                     if os.path.exists(path):
#                         os.remove(path)
#                 except Exception:
#                     pass
#         self.checkpoints = to_keep

#     def get_best_checkpoint(self):
#         if not self.checkpoints:
#             return None
#         return min(self.checkpoints, key=lambda x: x[1])

#     def get_top_k_checkpoints(self, k):
#         return sorted(self.checkpoints, key=lambda x: x[1])[:k]

#     def get_last_n_checkpoints(self, n):
#         return sorted(self.checkpoints, key=lambda x: x[0])[-n:]

#     def save_info(self):
#         info = {'checkpoints': [(e, v, p) for e, v, p in self.checkpoints]}
#         with open(os.path.join(self.save_dir, 'checkpoint_info.json'), 'w') as f:
#             json.dump(info, f, indent=4)


# # =============================================================================
# # Helper functions for checkpoint averaging
# # =============================================================================
# def average_student_weights(checkpoint_paths, device):
#     """Average student model weights, skip BN running stats"""
#     if not checkpoint_paths:
#         return None
#     if len(checkpoint_paths) == 1:
#         cp = torch.load(checkpoint_paths[0], map_location=device)
#         return cp['student_state_dict']

#     first = torch.load(checkpoint_paths[0], map_location=device)
#     averaged = copy.deepcopy(first['student_state_dict'])

#     keys_to_avg = []
#     keys_to_keep = []
#     for key in averaged.keys():
#         if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
#             keys_to_keep.append(key)
#         else:
#             keys_to_avg.append(key)

#     for path in checkpoint_paths[1:]:
#         cp = torch.load(path, map_location=device)
#         sd = cp['student_state_dict']
#         for key in keys_to_avg:
#             averaged[key] = averaged[key] + sd[key]

#     n = len(checkpoint_paths)
#     for key in keys_to_avg:
#         averaged[key] = averaged[key] / n

#     return averaged


# def update_bn_stats(model, train_loader, device, num_batches=100):
#     """
#     Update BatchNorm running statistics after loading averaged weights.
    
#     IMPORTANT: For frozen backbone models, we should NOT update the backbone BN layers
#     because they already have good statistics from ImageNet pretraining.
#     We only update BN layers that are in trainable (unfrozen) parts.
#     """
#     # Identify which BN layers are in trainable parts
#     trainable_bn_layers = []
#     for name, module in model.named_modules():
#         if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
#             has_trainable = False
#             for param in module.parameters():
#                 if param.requires_grad:
#                     has_trainable = True
#                     break
#             if has_trainable:
#                 trainable_bn_layers.append((name, module))

#     if not trainable_bn_layers:
#         print("      (No trainable BN layers found, skipping BN update)")
#         return

#     print(f"      (Found {len(trainable_bn_layers)} trainable BN layers to update)")

#     # Set model to eval mode first
#     model.eval()

#     # Only set trainable BN layers to train mode and reset their statistics
#     for name, module in trainable_bn_layers:
#         module.train()
#         module.momentum = None  # Use cumulative moving average
#         module.reset_running_stats()

#     # Forward pass to accumulate BN statistics (no gradient computation)
#     with torch.no_grad():
#         for batch_idx, (images, _) in enumerate(train_loader):
#             if batch_idx >= num_batches:
#                 break
#             images = images.to(device)
#             _ = model(images)

#     # Set everything back to eval mode
#     model.eval()


# class DistillationPipeline:
#     def __init__(
#         self,
#         data_dir,
#         num_classes,
#         batch_size=32,
#         num_workers=16,
#         lr_student=1e-4,
#         # lr_teacher=1e-4,
#         epochs=120,
#         warmup_epochs_student=5,
#         # warmup_epochs_teacher=5,
#         device="cuda",
#         save_dir="checkpoints",
#         lambda1=1.0,  # weight for L_proj1 (PCA loss)
#         lambda2=1.0,  # weight for L_proj2 (GL loss)
#         lambda3=1.0,  # weight for L_logits (Hinton loss)
#         lambda4=1.0,  # weight for DIST loss
#         patience=15,  # early stopping patience
#         start_factor_student=1e-8,
#         # start_factor_teacher=1e-8,  # warmup start factor
#         eta_min_student=1e-7,
#         block_ids=[11,10,9,8,7],
#         block_qkv_id=11,
#         temperature=4.0,
#         dist_beta=2.0,
#         dist_gamma=2.0,
#         last_n_epochs=10,
#         keep_last_n=10,
#         keep_top_k=5,
#         # eta_min_teacher=1e-7,  # cosine annealing min lr
#         teacher_checkpoint=None,
#         student_fc_dropout=0.7,
#         student_fc_hidden=None,
#         pca_dropout=0.5,
#         pca_partial_p=0.5,
#         gw_drop_p=0.4,
#         label_smoothing=0.1,
#         use_projection=True,  # ablation: set False to skip PCA/GL projectors
#     ):
#         self.device = torch.device(device if torch.cuda.is_available() else "cpu")
#         self.epochs = epochs
#         self.warmup_epochs_student = warmup_epochs_student
#         # self.warmup_epochs_teacher = warmup_epochs_teacher
#         self.save_dir = save_dir
#         self.lambda1 = lambda1
#         self.lambda2 = lambda2
#         self.lambda3 = lambda3
#         self.lambda4 = lambda4
#         self.temperature = temperature
#         self.patience = patience
#         self.start_factor_student = start_factor_student
#         # self.start_factor_teacher = start_factor_teacher
#         self.eta_min_student = eta_min_student
#         self.dist_beta = dist_beta
#         self.dist_gamma = dist_gamma
#         self.use_projection = use_projection
#         # self.eta_min_teacher = eta_min_teacher,
#         os.makedirs(save_dir, exist_ok=True)
        
#         # Auto-detect experiment run number
#         run_number = 1
#         while os.path.exists(os.path.join(save_dir, f"run_{run_number}")):
#             run_number += 1
#         self.save_dir = os.path.join(save_dir, f"run_{run_number}")
#         os.makedirs(self.save_dir, exist_ok=True)
#         print(f"📂 Experiment run #{run_number}, saving to: {self.save_dir}")
        
#         # ===== Dataset =====
#         print("Loading dataset...")
#         self.data_handler = DatasetHandler(
#             root_dir=data_dir,
#             batch_size=batch_size,
#             num_workers=num_workers
#         )
#         self.train_loader, self.val_loader, self.test_loader = self.data_handler.get_dataloaders()
        
#         print(f"Train samples: {len(self.train_loader.dataset)}")
#         print(f"Val samples: {len(self.val_loader.dataset)}")
#         print(f"Test samples: {len(self.test_loader.dataset)}")
#         print(f"Num classes: {num_classes}")
        
#         # ===== Models =====
#         print("\nInitializing models...")
        
#         # Teacher (frozen, inference only)
#         self.teacher = TeacherExtractor(pretrained=False,
#                                         checkpoint_path=teacher_checkpoint,
#                                         block_ids=block_ids,
#                                         block_qkv_id=block_qkv_id)
#         self.teacher.to(self.device)
#         print("✅ Teacher (ViT-B/16) loaded and frozen")
        
#         # Student with classification head
#         self.student_fc_dropout = student_fc_dropout
#         self.student_fc_hidden = student_fc_hidden if student_fc_hidden else [512, 256]
#         self.student = StudentWithHead(
#             num_classes=num_classes, pretrained=True,
#             fc_hidden=self.student_fc_hidden, fc_dropout=self.student_fc_dropout
#         )
#         self.student = self.student.to(self.device)
#         print("✅ Student (ResNet-50) loaded")
        
#         # # Teacher Head (trainable)
#         # self.teacher_head = TeacherHead(num_classes=num_classes, embed_dim=768)
#         # self.teacher_head = self.teacher_head.to(self.device)
#         # print("✅ Teacher Head (trainable) loaded")
        
#         # Projectors (only created when use_projection=True)
#         self.pca_dropout = pca_dropout
#         self.pca_partial_p = pca_partial_p
#         self.gw_drop_p = gw_drop_p
#         if self.use_projection:
#             self.pca_projector = PCAttentionProjector(
#                 in_channels=96, embed_dim=768,
#                 p=self.pca_partial_p, dropout=self.pca_dropout
#             )
#             self.pca_projector = self.pca_projector.to(self.device)
#             print("✅ PCA Projector loaded")
            
#             self.gl_projector = GWLinearProjector(in_dim=96, out_dim=768, drop_p=self.gw_drop_p)
#             self.gl_projector = self.gl_projector.to(self.device)
#             print("✅ GL Projector loaded")
#         else:
#             self.pca_projector = None
#             self.gl_projector = None
#             print("⏭️  Projectors skipped (use_projection=False)")
        
#         # ===== Loss functions =====
#         self.label_smoothing = label_smoothing
#         self.kd_loss_fn = ProjectionLoss() if self.use_projection else None
#         self.ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
#         self.logits_loss = LogitsKDLoss(temperature=temperature)
#         self.dist_loss_fn = DIST(beta=dist_beta, gamma=dist_gamma)
#         # ===== Optimizer (chỉ train student + projectors nếu có) =====
#         trainable_params = list(self.student.parameters())
#         if self.use_projection:
#             trainable_params += list(self.pca_projector.parameters()) + \
#                                 list(self.gl_projector.parameters())
#         # self.optimizer_teacher = optim.Adam(self.teacher_head.parameters(), lr=lr_teacher)
#         self.optimizer_student = optim.Adam(trainable_params, lr=lr_student)
        
#         # ===== Scheduler: Linear warmup + Cosine Annealing (epoch-level) =====
#         self.scheduler_student = self._get_scheduler()
        
#         # Store for evaluation strategies
#         self.num_classes = num_classes
#         self.last_n_epochs = last_n_epochs
        
#         # Checkpoint Manager (keeps last N + top K best checkpoints)
#         self.checkpoint_manager = CheckpointManager(
#             save_dir=self.save_dir,
#             keep_last_n=keep_last_n,
#             keep_top_k=keep_top_k
#         )
        
#         print(f"\n✅ Pipeline initialized on {self.device}")
    
#     def _get_scheduler(self):
#         """
#         Linear warmup + Cosine annealing scheduler using SequentialLR (epoch-level)
#         Tạo scheduler riêng cho teacher và student
#         """
#         warmup_epochs_student = self.warmup_epochs_student
#         cosine_epochs_student = self.epochs - self.warmup_epochs_student
#         # warmup_epochs_teacher = self.warmup_epochs_teacher
#         # cosine_epochs_teacher = self.epochs - self.warmup_epochs_teacher 
    
#         # ===== SCHEDULER CHO STUDENT =====
#         warmup_scheduler_student = LinearLR(
#             self.optimizer_student,
#             start_factor=self.start_factor_student,
#             end_factor=1.0,
#             total_iters=self.warmup_epochs_student
#         )
        
#         cosine_scheduler_student = CosineAnnealingLR(
#             self.optimizer_student,
#             T_max=cosine_epochs_student,
#             eta_min=self.eta_min_student
#         )
        
#         scheduler_student = SequentialLR(
#             self.optimizer_student,
#             schedulers=[warmup_scheduler_student, cosine_scheduler_student],
#             milestones=[warmup_epochs_student]
#         )
        
#         # # ===== SCHEDULER CHO TEACHER HEAD =====
#         # warmup_scheduler_teacher = LinearLR(
#         #     self.optimizer_teacher,
#         #     start_factor=self.start_factor_teacher,
#         #     end_factor=1.0,
#         #     total_iters=self.warmup_epochs_teacher
#         # )
        
#         # cosine_scheduler_teacher = CosineAnnealingLR(
#         #     self.optimizer_teacher,
#         #     T_max=cosine_epochs_teacher,
#         #     eta_min=self.eta_min_teacher
#         # )
        
#         # scheduler_teacher = SequentialLR(
#         #     self.optimizer_teacher,
#         #     schedulers=[warmup_scheduler_teacher, cosine_scheduler_teacher],
#         #     milestones=[warmup_epochs_teacher]
#         # )
        
#         return scheduler_student
    
#     def train_one_epoch(self, epoch):
#         """Train for one epoch"""
#         self.student.train()
#         if self.use_projection:
#             self.pca_projector.train()
#             self.gl_projector.train()
#         # self.teacher_head.train()  # ← Teacher head cũng train!
#         total_loss = 0.0
#         total_kd_loss = 0.0
#         total_logits_loss = 0.0
#         total_ce_loss_s = 0.0
#         total_l1 = 0.0
#         total_l2 = 0.0
#         total_dist_loss = 0.0
#         # total_ce_loss_t = 0.0
#         correct = 0
#         total = 0
        
#         pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
        
#         for images, labels in pbar:
#             images = images.to(self.device)
#             labels = labels.to(self.device)
            
#             # ===== Teacher forward (no grad) =====
#             with torch.no_grad():
#                 teacher_out = self.teacher.extract(images)
#             logit_t = teacher_out["logits"]
#             if self.use_projection:
#                 Q_t = teacher_out["Q_t"]
#                 K_t = teacher_out["K_t"]
#                 V_t = teacher_out["V_t"]
#                 Attn_t = teacher_out["Attn_t"]
#                 h_t = teacher_out["block_mean"]  # [B, 196, 768]
            
#             # ===== Student forward =====
#             feat_map, logit_s = self.student(images)

#             # ===== PCA & GL Projectors (only when use_projection=True) =====
#             if self.use_projection:
#                 pca_out = self.pca_projector(feat_map, Q_t, K_t, V_t)
#                 PCAttn_s = pca_out["PCAttnS"]
#                 V_s = pca_out["VS"]
#                 h_s_proj = self.gl_projector(feat_map)  # [B, 196, 768]
#                 l_proj1, l_proj2 = self.kd_loss_fn(Attn_t, PCAttn_s, V_t, V_s, h_t, h_s_proj)
#             else:
#                 l_proj1 = torch.tensor(0.0, device=self.device)
#                 l_proj2 = torch.tensor(0.0, device=self.device)

#             # ===== Calculate losses =====
#             ce_loss_s = self.ce_loss_fn(logit_s, labels)
#             logits_kd_loss = self.logits_loss(logit_s, logit_t.detach())
#             dist_loss = self.dist_loss_fn(logit_s, logit_t.detach())
            
#             # ===== TÍNH LOSS RIÊNG =====
#             # Loss cho STUDENT (KHÔNG có ce_loss_teacher!)(Offline learning)
#             loss_student = ce_loss_s + self.lambda1 * l_proj1 + self.lambda2 * l_proj2 + self.lambda3 * logits_kd_loss + self.lambda4 * dist_loss
#             # Loss cho TEACHER HEAD (chỉ CE)
#             # loss_teacher = ce_loss_t

#             # ===== BACKWARD RIÊNG CHO STUDENT TRƯỚC =====
#             self.optimizer_student.zero_grad()
#             loss_student.backward()  # ← STUDENT TRƯỚC (thêm retain_graph=True)
#             self.optimizer_student.step()

#             # # ===== BACKWARD RIÊNG CHO TEACHER SAU =====
#             # self.optimizer_teacher.zero_grad()
#             # loss_teacher.backward()  # ← TEACHER SAU (bỏ retain_graph=True)
#             # self.optimizer_teacher.step()

#             # ===== Metrics =====
#             total_loss += loss_student.item()
#             total_kd_loss += (l_proj1.item() + l_proj2.item())
#             total_ce_loss_s += ce_loss_s.item()
#             total_logits_loss += logits_kd_loss.item()
#             total_l1 += l_proj1.item()
#             total_l2 += l_proj2.item()
#             total_dist_loss += dist_loss.item()
#             _, predicted = logit_s.max(1)
#             total += labels.size(0)
#             correct += predicted.eq(labels).sum().item()
            
#             # Update progress bar
#             pbar.set_postfix({
#             "Loss_S": f"{loss_student.item():.3f}",
#             "KD": f"{(self.lambda1*l_proj1.item()+self.lambda2*l_proj2.item()+self.lambda3*logits_kd_loss.item()):.3f}",
#             "DIST": f"{(dist_loss.item()):.3f}",
#             "CE": f"{ce_loss_s.item():.3f}",
#             "Acc": f"{100.*correct/total:.1f}%",
#             "LR": f"{self.scheduler_student.get_last_lr()[0]:.4e}",
#         })
        
#         avg_loss = total_loss / len(self.train_loader)
#         avg_kd_loss = total_kd_loss / len(self.train_loader)
#         avg_ce_loss_s = total_ce_loss_s / len(self.train_loader)
#         accuracy = 100. * correct / total
        
#         return {
#             "loss": avg_loss,
#             "kd_loss": avg_kd_loss,
#             "ce_loss_s": avg_ce_loss_s,
#             "l1_weighted": (total_l1 / len(self.train_loader)) * self.lambda1,
#             "l2_weighted": (total_l2 / len(self.train_loader)) * self.lambda2,
#             "l3_weighted": (total_logits_loss / len(self.train_loader)) * self.lambda3,
#             "dist_weighted": (total_dist_loss / len(self.train_loader)) * self.lambda4,
#             "accuracy": accuracy
#         }
    
#     @torch.no_grad()
#     def validate(self, loader, desc="Val", class_names=None):
#         """Validate on given loader, optionally compute per-class metrics"""
#         self.student.eval()

#         total_loss = 0.0
#         correct = 0
#         total = 0
#         all_preds = []
#         all_labels = []

#         pbar = tqdm(loader, desc=f"[{desc}]")

#         for images, labels in pbar:
#             images = images.to(self.device)
#             labels = labels.to(self.device)

#             # Student forward
#             feat_map, logit_s = self.student(images)

#             # CE Loss
#             ce_loss = self.ce_loss_fn(logit_s, labels)
#             total_loss += ce_loss.item()

#             # Accuracy
#             _, predicted = logit_s.max(1)
#             total += labels.size(0)
#             correct += predicted.eq(labels).sum().item()

#             all_preds.extend(predicted.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())

#             pbar.set_postfix({
#                 "Loss": f"{ce_loss.item():.4f}",
#                 "Acc": f"{100.*correct/total:.2f}%"
#             })

#         avg_loss = total_loss / len(loader)
#         accuracy = 100. * correct / total

#         result = {
#             "loss": avg_loss,
#             "accuracy": accuracy
#         }

#         # Compute per-class metrics if class_names provided
#         if class_names is not None:
#             import numpy as np
#             all_preds = np.array(all_preds)
#             all_labels = np.array(all_labels)

#             report = classification_report(
#                 all_labels, all_preds,
#                 target_names=class_names,
#                 output_dict=True,
#                 zero_division=0
#             )

#             cm = confusion_matrix(all_labels, all_preds)

#             result["classification_report"] = report
#             result["confusion_matrix"] = cm
#             result["all_preds"] = all_preds
#             result["all_labels"] = all_labels

#         return result
    
#     def save_checkpoint(self, epoch, val_loss, val_acc, is_best=False):
#         """Save checkpoint using CheckpointManager + latest.pth for resume"""
#         student_sd = self.student.state_dict()
#         optimizer_sd = self.optimizer_student.state_dict()
#         scheduler_sd = self.scheduler_student.state_dict()
        
#         # Save via CheckpointManager (handles best.pth + cleanup internally)
#         self.checkpoint_manager.save(
#             student_state_dict=student_sd,
#             optimizer_state_dict=optimizer_sd,
#             scheduler_state_dict=scheduler_sd,
#             epoch=epoch + 1,
#             val_loss=val_loss,
#             val_acc=val_acc
#         )
        
#         # Also save latest.pth for resume training
#         latest = {
#             "epoch": epoch + 1,
#             "student_state_dict": student_sd,
#             "optimizer_student_state_dict": optimizer_sd,
#             "scheduler_student_state_dict": scheduler_sd,
#             "val_loss": val_loss,
#             "val_acc": val_acc
#         }
#         torch.save(latest, os.path.join(self.save_dir, "latest.pth"))
    
#     def load_checkpoint(self, path):
#         """Load checkpoint"""
#         checkpoint = torch.load(path, map_location=self.device)
        
#         self.student.load_state_dict(checkpoint["student_state_dict"])
#         # self.pca_projector.load_state_dict(checkpoint["pca_projector_state_dict"])
#         # self.gl_projector.load_state_dict(checkpoint["gl_projector_state_dict"])
#         self.optimizer_student.load_state_dict(checkpoint["optimizer_student_state_dict"])
#         self.scheduler_student.load_state_dict(checkpoint["scheduler_student_state_dict"])
        
#         val_loss = checkpoint.get('val_loss', float('inf'))
#         val_acc = checkpoint.get('val_acc', 0.0)
#         print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']} with val_loss: {val_loss:.4f}, val_acc: {val_acc:.2f}%")
        
#         return checkpoint["epoch"], val_loss
    
#     def train(self, resume_path=None):
#         """Full training loop"""
#         start_epoch = 0
#         best_val_loss = float('inf')  # Lower is better
#         epochs_no_improve = 0  # Early stopping counter

#         history = {
#             "train_loss": [],
#             "val_loss":   [],
#             "train_acc":  [],
#             "val_acc":    [],
#             "lr":         [],
#         }

#         if resume_path and os.path.exists(resume_path):
#             start_epoch, best_val_loss = self.load_checkpoint(resume_path)
#             # start_epoch += 1

#         print("\n" + "="*60)
#         print("🚀 Starting Training")
#         print(f"   Early Stopping: patience = {self.patience}")
#         print("="*60)

#         for epoch in range(start_epoch, self.epochs):
#             # Train
#             train_metrics = self.train_one_epoch(epoch)
            
#             # Validate
#             val_metrics = self.validate(self.val_loader, desc="Val")
            
#             # Get current LR (before step)
#             current_lr_student = self.scheduler_student.get_last_lr()[0]
#             # current_lr_teacher = self.scheduler_teacher.get_last_lr()[0]

#             # Record history
#             history["train_loss"].append(train_metrics["loss"])
#             history["val_loss"].append(val_metrics["loss"])
#             history["train_acc"].append(train_metrics["accuracy"])
#             history["val_acc"].append(val_metrics["accuracy"])
#             history["lr"].append(current_lr_student)

#             # Step scheduler (epoch-level)
#             self.scheduler_student.step()
#             # self.scheduler_teacher.step()

#             # Print epoch summary
#             print(f"\n📊 Epoch {epoch+1}/{self.epochs} Summary (LR_S: {current_lr_student:.6f}")
#             print(f"   Train - Loss: {train_metrics['loss']:.4f}, "
#                   f"KD: {train_metrics['l1_weighted']+train_metrics['l2_weighted']+train_metrics['l3_weighted']:.4f}, "
#                   f"DIST: {train_metrics['dist_weighted']:.4f}, " 
#                   f"CE: {train_metrics['ce_loss_s']:.4f}, "
#                   f"Acc: {train_metrics['accuracy']:.2f}%")
#             print(f"   Val   - Loss: {val_metrics['loss']:.4f}, "
#                   f"Acc: {val_metrics['accuracy']:.2f}%")
#             print(f" L1 projection: {train_metrics['l1_weighted']:.4f}")
#             print(f" L2 projection: {train_metrics['l2_weighted']:.4f}")
#             print(f" Logits projection: {train_metrics['l3_weighted']:.4f}")
#             # Save checkpoint (based on lowest val_loss)
#             is_best = val_metrics["loss"] < best_val_loss
#             if is_best:
#                 best_val_loss = val_metrics["loss"]
#                 epochs_no_improve = 0
#             else:
#                 epochs_no_improve += 1
            
#             self.save_checkpoint(epoch, val_metrics["loss"], val_metrics["accuracy"], is_best)
            
#             # Early stopping check
#             if epochs_no_improve >= self.patience:
#                 print(f"\n⚠️ Early stopping triggered! No improvement for {self.patience} epochs.")
#                 print(f"   Best val_loss: {best_val_loss:.4f}")
#                 break
            
#             print(f"   Early stopping: {epochs_no_improve}/{self.patience}")
#             print()
        
#         # Save checkpoint manager info
#         self.checkpoint_manager.save_info()

#         # ===== Plot learning curves =====
#         plot_training_curves(history, self.save_dir)

#         # ===== Evaluate all 3 strategies =====
#         all_results = self.evaluate_all_strategies()
        
#         # ===== Cleanup training checkpoints, keep only strategy files =====
#         self._cleanup_training_checkpoints()
        
#         return all_results

#     def _export_metrics_to_excel(self, metrics, class_names):
#         """Export per-class metrics and confusion matrix to Excel"""
#         report = metrics["classification_report"]
#         cm = metrics["confusion_matrix"]

#         excel_path = os.path.join(self.save_dir, "test_metrics.xlsx")

#         with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
#             # Sheet 1: Per-class metrics
#             rows = []
#             for cls_name in class_names:
#                 m = report[cls_name]
#                 rows.append({
#                     "Class": cls_name,
#                     "Precision": round(m["precision"], 4),
#                     "Recall": round(m["recall"], 4),
#                     "F1-Score": round(m["f1-score"], 4),
#                     "Support": int(m["support"])
#                 })
#             # Add overall metrics
#             for avg_type in ["macro avg", "weighted avg"]:
#                 m = report[avg_type]
#                 rows.append({
#                     "Class": avg_type.title(),
#                     "Precision": round(m["precision"], 4),
#                     "Recall": round(m["recall"], 4),
#                     "F1-Score": round(m["f1-score"], 4),
#                     "Support": int(m["support"])
#                 })
#             rows.append({
#                 "Class": "Overall Accuracy",
#                 "Precision": "",
#                 "Recall": "",
#                 "F1-Score": round(report["accuracy"], 4),
#                 "Support": int(report["macro avg"]["support"])
#             })

#             df_metrics = pd.DataFrame(rows)
#             df_metrics.to_excel(writer, sheet_name="Per-Class Metrics", index=False)

#             # Sheet 2: Confusion Matrix
#             df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
#             df_cm.index.name = "Actual \\ Predicted"
#             df_cm.to_excel(writer, sheet_name="Confusion Matrix")

#         print(f"\n📁 Metrics exported to: {excel_path}")

#     # =================================================================
#     # Evaluation Strategy Methods
#     # =================================================================
#     @torch.no_grad()
#     def evaluate_model_full(self, model, loader, class_names):
#         """Full evaluation: per-class precision/recall/F1, AUC, confusion matrix"""
#         model.eval()
#         all_preds = []
#         all_labels = []
#         all_probs = []
#         running_loss = 0.0
#         total = 0
#         criterion = nn.CrossEntropyLoss()

#         for images, labels in tqdm(loader, desc="Evaluating", leave=False):
#             images = images.to(self.device)
#             labels = labels.to(self.device)
#             _, logits = model(images)
#             loss = criterion(logits, labels)
#             probs = torch.softmax(logits, dim=1)
#             _, preds = logits.max(1)
#             running_loss += loss.item() * images.size(0)
#             total += labels.size(0)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#             all_probs.extend(probs.cpu().numpy())

#         all_preds = np.array(all_preds)
#         all_labels = np.array(all_labels)
#         all_probs = np.array(all_probs)

#         test_loss = running_loss / total
#         accuracy = accuracy_score(all_labels, all_preds) * 100
#         precision = precision_score(all_labels, all_preds, average='macro', zero_division=0) * 100
#         recall = recall_score(all_labels, all_preds, average='macro', zero_division=0) * 100
#         f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100
#         try:
#             auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro') * 100
#         except Exception:
#             auc = 0.0

#         report = classification_report(
#             all_labels, all_preds,
#             target_names=class_names,
#             output_dict=True,
#             zero_division=0
#         )
#         cm = confusion_matrix(all_labels, all_preds)

#         return {
#             'Test Loss': test_loss,
#             'Accuracy (%)': accuracy,
#             'Precision (%)': precision,
#             'Recall (%)': recall,
#             'F1-Score (%)': f1,
#             'AUC (%)': auc,
#             'classification_report': report,
#             'confusion_matrix': cm,
#         }

#     def _create_student_model(self):
#         """Create a fresh StudentWithHead for loading averaged weights"""
#         model = StudentWithHead(
#             num_classes=self.num_classes, pretrained=False,
#             fc_hidden=self.student_fc_hidden, fc_dropout=self.student_fc_dropout
#         )
#         return model.to(self.device)

#     def _print_strategy_results(self, metrics, strategy_name, class_names):
#         """Print evaluation results for one strategy"""
#         print(f"    {'='*60}")
#         print(f"    📊 TEST RESULTS - {strategy_name}:")
#         print(f"    {'='*60}")
#         print(f"    Test Loss : {metrics['Test Loss']:>8.4f}")
#         print(f"    Accuracy  : {metrics['Accuracy (%)']:>8.2f}%")
#         print(f"    Precision : {metrics['Precision (%)']:>8.2f}%")
#         print(f"    Recall    : {metrics['Recall (%)']:>8.2f}%")
#         print(f"    F1-Score  : {metrics['F1-Score (%)']:>8.2f}%")
#         print(f"    AUC       : {metrics['AUC (%)']:>8.2f}%")
#         print(f"    {'='*60}")
#         if 'classification_report' in metrics:
#             report = metrics['classification_report']
#             print(f"    {'Class':<25} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
#             print(f"    {'-'*65}")
#             for cls_name in class_names:
#                 m = report[cls_name]
#                 print(f"    {cls_name:<25} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1-score']:>10.4f} {m['support']:>10.0f}")
#             print(f"    {'-'*65}")
#             m = report['macro avg']
#             print(f"    {'Macro Avg':<25} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1-score']:>10.4f} {m['support']:>10.0f}")
#             m = report['weighted avg']
#             print(f"    {'Weighted Avg':<25} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1-score']:>10.4f} {m['support']:>10.0f}")

#     def strategy_1_best_checkpoint(self, class_names):
#         """Strategy 1: Evaluate best checkpoint (lowest val_loss)"""
#         print(f"\n  Strategy 1: Best checkpoint (lowest val_loss)")
#         best = self.checkpoint_manager.get_best_checkpoint()
#         if best is None:
#             print("    No checkpoints available!")
#             return None

#         epoch, val_loss, path = best
#         print(f"    Best checkpoint: Epoch {epoch}, Val Loss: {val_loss:.4f}")

#         model = self._create_student_model()
#         cp = torch.load(path, map_location=self.device)
#         model.load_state_dict(cp['student_state_dict'])

#         # Save strategy checkpoint
#         save_dir = os.path.join(self.save_dir, 'saved_checkpoints')
#         os.makedirs(save_dir, exist_ok=True)
#         save_path = os.path.join(save_dir, f'strategy1_best_epoch_{epoch}.pth')
#         torch.save({'student_state_dict': model.state_dict(), 'epoch': epoch, 'val_loss': val_loss}, save_path)
#         print(f"    ✓ Saved to: {save_path}")

#         metrics = self.evaluate_model_full(model, self.test_loader, class_names)
#         self._print_strategy_results(metrics, "Strategy 1 (Best Checkpoint)", class_names)
#         return metrics

#     def strategy_2_top_k_average(self, class_names):
#         """Strategy 2: Average top-K checkpoints (K=2,3,4,5) and evaluate"""
#         print(f"\n  Strategy 2: Top-K checkpoint averaging")
#         results = {}

#         for k in [2, 3, 4, 5]:
#             print(f"    K={k}:")
#             top_k = self.checkpoint_manager.get_top_k_checkpoints(k)

#             if len(top_k) < k:
#                 print(f"      Warning: Only {len(top_k)} checkpoints available")
#             if not top_k:
#                 continue

#             paths = [p for _, _, p in top_k]
#             avg_weights = average_student_weights(paths, self.device)

#             model = self._create_student_model()
#             model.load_state_dict(avg_weights, strict=True)

#             print(f"      Updating BatchNorm statistics...")
#             update_bn_stats(model, self.train_loader, self.device, num_batches=100)

#             # Save
#             save_dir = os.path.join(self.save_dir, 'saved_checkpoints')
#             os.makedirs(save_dir, exist_ok=True)
#             save_path = os.path.join(save_dir, f'strategy2_top_{k}_averaged.pth')
#             torch.save({'student_state_dict': model.state_dict(), 'k': k}, save_path)
#             print(f"      ✓ Saved to: {save_path}")

#             metrics = self.evaluate_model_full(model, self.test_loader, class_names)
#             self._print_strategy_results(metrics, f"Strategy 2 (Top-{k} Avg)", class_names)
#             results[k] = metrics

#         return results

#     def strategy_3_last_n_average(self, class_names):
#         """Strategy 3: Average last N epoch checkpoints"""
#         print(f"\n  Strategy 3: Last {self.last_n_epochs} epochs averaging")
#         last_n = self.checkpoint_manager.get_last_n_checkpoints(self.last_n_epochs)

#         if not last_n:
#             print("    No checkpoints available!")
#             return None
#         if len(last_n) < self.last_n_epochs:
#             print(f"    Warning: Only {len(last_n)} checkpoints available")

#         epochs = [e for e, _, _ in last_n]
#         paths = [p for _, _, p in last_n]
#         print(f"    Averaging epochs: {epochs}")

#         avg_weights = average_student_weights(paths, self.device)

#         model = self._create_student_model()
#         model.load_state_dict(avg_weights, strict=True)

#         print(f"    Updating BatchNorm statistics...")
#         update_bn_stats(model, self.train_loader, self.device, num_batches=100)

#         # Save
#         save_dir = os.path.join(self.save_dir, 'saved_checkpoints')
#         os.makedirs(save_dir, exist_ok=True)
#         save_path = os.path.join(save_dir, f'strategy3_last_{self.last_n_epochs}_averaged.pth')
#         torch.save({'student_state_dict': model.state_dict(), 'epochs': epochs}, save_path)
#         print(f"    ✓ Saved to: {save_path}")

#         metrics = self.evaluate_model_full(model, self.test_loader, class_names)
#         self._print_strategy_results(metrics, f"Strategy 3 (Last {self.last_n_epochs} Avg)", class_names)
#         return metrics

#     def evaluate_all_strategies(self):
#         """Run all 3 evaluation strategies and export results to Excel"""
#         print("\n" + "="*70)
#         print("🧪 Evaluating All Strategies")
#         print("="*70)

#         class_names = self.data_handler.get_class_names()
#         all_results = {}

#         # Strategy 1: Best single checkpoint
#         metrics_1 = self.strategy_1_best_checkpoint(class_names)
#         if metrics_1:
#             all_results['Strategy 1 (Best)'] = metrics_1

#         # Strategy 2: Top-K averaging
#         strategy_2 = self.strategy_2_top_k_average(class_names)
#         for k, metrics in strategy_2.items():
#             all_results[f'Strategy 2 (Top-{k} Avg)'] = metrics

#         # Strategy 3: Last N epochs averaging
#         metrics_3 = self.strategy_3_last_n_average(class_names)
#         if metrics_3:
#             all_results[f'Strategy 3 (Last {self.last_n_epochs} Avg)'] = metrics_3

#         # Export all results to Excel
#         self._export_all_strategies_to_excel(all_results, class_names)

#         # Print summary table
#         print("\n" + "="*70)
#         print("📊 SUMMARY OF ALL STRATEGIES")
#         print("="*70)
#         print(f"{'Strategy':<35} {'Accuracy':>10} {'F1-Score':>10} {'AUC':>10}")
#         print("-" * 70)
#         for name, m in all_results.items():
#             print(f"{name:<35} {m['Accuracy (%)']:>9.2f}% {m['F1-Score (%)']:>9.2f}% {m['AUC (%)']:>9.2f}%")
#         print("=" * 70)

#         return all_results

#     def _export_all_strategies_to_excel(self, all_results, class_names):
#         """Export all strategy results to Excel with 2 sheets: Macro Results + Per-Class Metrics"""
#         excel_path = os.path.join(self.save_dir, "all_strategies_results.xlsx")

#         with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
#             # ===== Sheet 1: Macro Results =====
#             summary_rows = []
#             for strategy_name, metrics in all_results.items():
#                 summary_rows.append({
#                     "Strategy": strategy_name,
#                     "Test Loss": round(metrics['Test Loss'], 4),
#                     "Accuracy (%)": round(metrics['Accuracy (%)'], 2),
#                     "Precision (%)": round(metrics['Precision (%)'], 2),
#                     "Recall (%)": round(metrics['Recall (%)'], 2),
#                     "F1-Score (%)": round(metrics['F1-Score (%)'], 2),
#                     "AUC (%)": round(metrics['AUC (%)'], 2),
#                 })
#             df_summary = pd.DataFrame(summary_rows)
#             df_summary.to_excel(writer, sheet_name="Macro Results", index=False)

#             # ===== Sheet 2: Per-Class Metrics for all strategies =====
#             per_class_rows = []
#             for strategy_name, metrics in all_results.items():
#                 if 'classification_report' not in metrics or 'confusion_matrix' not in metrics:
#                     continue
#                 report = metrics['classification_report']
#                 cm = metrics['confusion_matrix']

#                 for idx, cls_name in enumerate(class_names):
#                     m = report[cls_name]
#                     # Per-class accuracy = correctly classified / total samples of this class
#                     cls_total = cm[idx].sum()
#                     cls_accuracy = (cm[idx][idx] / cls_total * 100) if cls_total > 0 else 0.0
#                     per_class_rows.append({
#                         "Strategy": strategy_name,
#                         "Class": cls_name,
#                         "Accuracy (%)": round(cls_accuracy, 2),
#                         "Precision (%)": round(m["precision"] * 100, 2),
#                         "Recall (%)": round(m["recall"] * 100, 2),
#                         "F1-Score (%)": round(m["f1-score"] * 100, 2),
#                         "Support": int(m["support"])
#                     })

#             if per_class_rows:
#                 df_per_class = pd.DataFrame(per_class_rows)
#                 df_per_class.to_excel(writer, sheet_name="Per-Class Metrics", index=False)

#         print(f"\n📁 All strategies results exported to: {excel_path}")

#     def _cleanup_training_checkpoints(self):
#         """
#         Xóa tất cả checkpoint training (epoch_*.pth, best.pth, latest.pth, checkpoint_info.json)
#         sau khi đã evaluate xong. Chỉ giữ lại folder saved_checkpoints/ chứa strategy files.
#         """
#         print("\n🧹 Cleaning up training checkpoints...")
#         kept = 0
#         removed = 0
#         saved_cp_dir = os.path.join(self.save_dir, 'saved_checkpoints')

#         for fname in os.listdir(self.save_dir):
#             fpath = os.path.join(self.save_dir, fname)
#             # Skip the saved_checkpoints directory and the Excel results
#             if os.path.isdir(fpath):
#                 continue
#             if fname.endswith('.xlsx') or fname.endswith('.csv'):
#                 kept += 1
#                 continue
#             # Remove training checkpoint files
#             if fname.endswith('.pth') or fname == 'checkpoint_info.json':
#                 try:
#                     os.remove(fpath)
#                     removed += 1
#                 except Exception as e:
#                     print(f"   Warning: Could not delete {fpath}: {e}")
#             else:
#                 kept += 1

#         # Count strategy files kept
#         strategy_files = 0
#         if os.path.isdir(saved_cp_dir):
#             strategy_files = len([f for f in os.listdir(saved_cp_dir) if f.endswith('.pth')])

#         print(f"   Removed {removed} training checkpoint files")
#         print(f"   Kept {strategy_files} strategy checkpoint files in saved_checkpoints/")
#         print(f"   Kept {kept} other files (Excel, etc.)")

#     def get_student_model(self):
#         """
#         Trả về student model (không có projectors) để inference
#         """
#         return self.student


# # ===== Main =====
# if __name__ == "__main__":
#     from config import Config

#     os.environ["CUDA_VISIBLE_DEVICES"] = Config.CUDA_VISIBLE_DEVICES

#     Config.print_config()
#     config = Config.to_pipeline_dict()
#     print(f"[KD] block_ids = {config['block_ids']}")
#     print(f"[KD] block_qkv_id = {config['block_qkv_id']}")
#     pipeline = DistillationPipeline(**config)
#     pipeline.train()

