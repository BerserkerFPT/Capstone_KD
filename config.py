"""
Centralized configuration for Knowledge Distillation pipeline.
Modify all hyperparameters here instead of editing main.py directly.
"""


class Config:
    # ===================== Dataset =====================
    DATA_DIR = r"/workspace/kaggle/working/ProcessedOriginal"
    NUM_CLASSES = 5
    IMAGE_SIZE = 224
    BATCH_SIZE = 256
    NUM_WORKERS = 16
    RANDOM_SEED = 42

    # Train / Val / Test split
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    # ===================== Device =====================
    DEVICE = "cuda"
    CUDA_VISIBLE_DEVICES = "0"

    # ===================== Teacher =====================
    TEACHER_CHECKPOINT = (
        r"/Strategy_2_K2.pth"
    )
    # Transformer block indices for extracting features / QKV
    BLOCK_IDS = [0]
    BLOCK_QKV_ID = [11]

    # ===================== Student (MobileNetV2) =====================
    STUDENT_PRETRAINED = True
    STUDENT_FEATURE_DIM = 96          # output channels of MobileNetV2 features.11
    STUDENT_FC_DROPOUT = 0.3          # dropout in StudentWithHead classifier
    STUDENT_FC_HIDDEN = [512, 256]    # hidden dims of classifier MLP

    # ===================== Projectors =====================
    EMBED_DIM = 768                   # teacher ViT embed dim (also projector output dim)

    # -- PCAttentionProjector --
    PCA_DROPOUT = 0.2             # dropout for Conv2d layers in PCA projector
    PCA_PARTIAL_P = 0.5              # probability of replacing student Q/K/V with teacher's

    # -- GWLinearProjector --
    GW_DROP_P = 0.4                   # dropout in group-wise linear projector

    # ===================== Training =====================
    EPOCHS = 150
    LR_STUDENT = 5e-3

    # Warmup + Cosine Annealing scheduler
    WARMUP_EPOCHS_STUDENT = int(0.1 * EPOCHS)   # 15% of total epochs
    START_FACTOR_STUDENT = 0.1
    ETA_MIN_STUDENT = 1e-8

    # Early stopping
    PATIENCE = 30

    # ===================== DWA Initial Lambda Values =====================
    # Initial weights for first 2 epochs before DWA computes dynamic weights
    DWA_INIT_LAMBDA_CE     = 1.0    # Cross-Entropy loss
    DWA_INIT_LAMBDA_PROJ1  = 1.0    # L_proj1 (PCA attention loss)
    DWA_INIT_LAMBDA_PROJ2  = 1.0    # L_proj2 (GW linear loss)
    DWA_INIT_LAMBDA_LOGITS = 1.0    # L_logits (Hinton KD loss)
    DWA_INIT_LAMBDA_DIST   = 1.0    # L_dist (DIST loss)

    LABEL_SMOOTHING = 0.1   # CrossEntropyLoss label smoothing

    # Ablation study: set False to disable PCA/GL projectors (CE + Logits / CE + Logits + Relation)
    USE_PROJECTION = True

    # ===================== Ablation: Enable/Disable Individual Losses =====================
    # Set any flag to False to remove that loss from training.
    # DWA will automatically adjust to the active losses.
    USE_CE     = True   # Cross-Entropy loss (classification, should almost always be True)
    USE_PROJ1  = True   # L_proj1 — PCA Attention projection loss
    USE_PROJ2  = True   # L_proj2 — GWLinear projection loss
    USE_LOGITS = True   # L_logits — Hinton KD logits loss
    USE_DIST   = True   # L_dist  — DIST relational loss

    # ===================== DWA Hyperparameters =====================
    # Temperature T for DWA softmax (higher T → more uniform weights)
    DWA_TEMPERATURE = 2.5

    # Hinton KD temperature (for logits distillation)
    TEMPERATURE = 4.0

    # ===================== Weighted Random Sampler =====================
    USE_WEIGHTED_SAMPLER = False   # Use inverse-frequency weighted sampling for imbalanced data

    # ===================== Focal Loss =====================
    USE_FOCAL_LOSS = False         # Use PolyFocalLoss instead of CrossEntropyLoss
    FOCAL_GAMMA = 2.0              # Focusing parameter: higher = more focus on hard examples
    POLY_EPSILON = 1.0             # Poly coefficient: boosts gradient for ambiguous samples
    CLASS_WEIGHT_METHOD = 'inverse_freq'  # 'inverse_freq' or 'effective_num'

    # DIST loss hyper-params
    DIST_BETA = 1.0
    DIST_GAMMA = 1.0

    # ===================== Checkpoints & Evaluation =====================
    SAVE_DIR = "checkpoints"

    # Checkpoint manager: keep last N + top K best during training
    KEEP_LAST_N = 10
    KEEP_TOP_K = 5

    # Strategy 3: average last N epochs
    LAST_N_EPOCHS = 10

    # Strategy 2: top-K values to try
    TOP_K_VALUES = [2, 3, 4, 5]

    # ===================== Helper =====================
    @classmethod
    def to_pipeline_dict(cls):
        """Return a dict that can be unpacked into DistillationPipeline(**config)."""
        # Auto-compute DWA num_tasks from active losses
        dwa_num_tasks = sum([
            cls.USE_CE,
            cls.USE_PROJ1 and cls.USE_PROJECTION,
            cls.USE_PROJ2 and cls.USE_PROJECTION,
            cls.USE_LOGITS,
            cls.USE_DIST,
        ])
        return {
            "data_dir": cls.DATA_DIR,
            "num_classes": cls.NUM_CLASSES,
            "batch_size": cls.BATCH_SIZE,
            "num_workers": cls.NUM_WORKERS,
            "epochs": cls.EPOCHS,
            "lr_student": cls.LR_STUDENT,
            "warmup_epochs_student": cls.WARMUP_EPOCHS_STUDENT,
            "start_factor_student": cls.START_FACTOR_STUDENT,
            "eta_min_student": cls.ETA_MIN_STUDENT,
            "block_ids": cls.BLOCK_IDS,
            "block_qkv_id": cls.BLOCK_QKV_ID,
            "device": cls.DEVICE,
            "save_dir": cls.SAVE_DIR,
            "dwa_init_lambdas": [
                cls.DWA_INIT_LAMBDA_CE,
                cls.DWA_INIT_LAMBDA_PROJ1,
                cls.DWA_INIT_LAMBDA_PROJ2,
                cls.DWA_INIT_LAMBDA_LOGITS,
                cls.DWA_INIT_LAMBDA_DIST,
            ],
            "temperature": cls.TEMPERATURE,
            "patience": cls.PATIENCE,
            "dist_beta": cls.DIST_BETA,
            "dist_gamma": cls.DIST_GAMMA,
            "last_n_epochs": cls.LAST_N_EPOCHS,
            "keep_last_n": cls.KEEP_LAST_N,
            "keep_top_k": cls.KEEP_TOP_K,
            # New config params forwarded to sub-modules
            "teacher_checkpoint": cls.TEACHER_CHECKPOINT,
            "student_fc_dropout": cls.STUDENT_FC_DROPOUT,
            "student_fc_hidden": cls.STUDENT_FC_HIDDEN,
            "pca_dropout": cls.PCA_DROPOUT,
            "pca_partial_p": cls.PCA_PARTIAL_P,
            "gw_drop_p": cls.GW_DROP_P,
            "label_smoothing": cls.LABEL_SMOOTHING,
            "use_projection": cls.USE_PROJECTION,
            # Ablation: individual loss flags
            "use_ce":     cls.USE_CE,
            "use_proj1":  cls.USE_PROJ1,
            "use_proj2":  cls.USE_PROJ2,
            "use_logits": cls.USE_LOGITS,
            "use_dist":   cls.USE_DIST,
            # DWA hyperparams
            "dwa_temperature": cls.DWA_TEMPERATURE,
            "dwa_num_tasks":   dwa_num_tasks,
            # Weighted sampler & Focal loss
            "use_weighted_sampler": cls.USE_WEIGHTED_SAMPLER,
            "use_focal_loss": cls.USE_FOCAL_LOSS,
            "focal_gamma": cls.FOCAL_GAMMA,
            "poly_epsilon": cls.POLY_EPSILON,
            "class_weight_method": cls.CLASS_WEIGHT_METHOD,
        }

    @classmethod
    def print_config(cls):
        """Print all config values."""
        print("\n" + "=" * 60)
        print("📋 Configuration")
        print("=" * 60)
        for key, value in cls.to_pipeline_dict().items():
            print(f"  {key}: {value}")
        print("=" * 60 + "\n")
