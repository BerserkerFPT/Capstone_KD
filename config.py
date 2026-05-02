"""
Centralized configuration for AgriKD pipeline.
Edit hyperparameters here; do not modify main.py directly.
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
    VAL_RATIO   = 0.15
    TEST_RATIO  = 0.15

    # ===================== Device =====================
    DEVICE = "cuda"
    CUDA_VISIBLE_DEVICES = "0"

    # ===================== Teacher (ViT-B/16) =====================
    TEACHER_CHECKPOINT = (
        r"/Strategy_2_K2.pth"
    )
    BLOCK_IDS    = [0]   # feature extraction block indices
    BLOCK_QKV_ID = [11]  # QKV extraction block index

    # ===================== Student (Truncated MobileNetV2) =====================
    STUDENT_PRETRAINED  = True
    STUDENT_FEATURE_DIM = 96           # output channels of Bottleneck 5 (14×14×96)
    STUDENT_FC_DROPOUT  = 0.3
    STUDENT_FC_HIDDEN   = [512, 256]   # MLP classifier hidden dims

    # ===================== Projectors =====================
    EMBED_DIM = 768          # ViT embed dim; also projector output dim

    PCA_DROPOUT   = 0.2      # PCAttentionProjector Conv2d dropout
    PCA_PARTIAL_P = 0.5      # probability of replacing student Q/K/V with teacher's

    GW_DROP_P = 0.4          # GWLinearProjector dropout

    # ===================== Training =====================
    EPOCHS     = 150
    LR_STUDENT = 5e-3

    WARMUP_EPOCHS_STUDENT = int(0.1 * EPOCHS)
    START_FACTOR_STUDENT  = 0.1
    ETA_MIN_STUDENT       = 1e-8

    PATIENCE = 30

    # ===================== Loss Weights (λ₁ … λ₅) =====================
    # Dataset-specific values; use HEURISTIC_WEIGHT_INIT_MODE to auto-compute.
    LAMBDA_CE     = 1.0   # λ_CE
    LAMBDA_PROJ1  = 1.0   # λ_proj1  (L_proj1 — PCA attention)
    LAMBDA_PROJ2  = 1.0   # λ_proj2  (L_proj2 — GW linear)
    LAMBDA_LOGITS = 1.0   # λ_logits (L_KL   — Hinton KD)
    LAMBDA_DIST   = 1.0   # λ_dist   (L_Rel  — DIST relational)

    LABEL_SMOOTHING = 0.1

    # ===================== Loss Ablation Flags =====================
    USE_PROJECTION = True    # False → disable both projectors (CE + KL only)

    USE_CE     = True   # cross-entropy
    USE_PROJ1  = True   # L_proj1
    USE_PROJ2  = True   # L_proj2
    USE_LOGITS = True   # L_KL (Hinton)
    USE_DIST   = True   # L_Rel (DIST)

    TEMPERATURE = 4.0   # Hinton KD temperature

    # ===================== Cross-Validation =====================
    USE_CROSS_VALIDATION = False   # StratifiedKFold CV
    CV_N_SPLITS          = 5

    # Per-fold teacher checkpoints (for imbalanced datasets, e.g. Potato 1:10).
    # Set None to use a single TEACHER_CHECKPOINT for all folds.
    CV_TEACHER_CHECKPOINTS = None  # List[str] | None

    # ===================== Imbalanced Data Handling =====================
    USE_WEIGHTED_SAMPLER = False
    USE_FOCAL_LOSS       = False
    FOCAL_GAMMA          = 2.0
    POLY_EPSILON         = 1.0
    CLASS_WEIGHT_METHOD  = 'inverse_freq'   # 'inverse_freq' | 'effective_num'

    # ===================== DIST Loss =====================
    DIST_BETA  = 1.0
    DIST_GAMMA = 1.0

    # ===================== Heuristic Loss Weight Initialisation =====================
    # (§ Loss Weight Initialisation in the paper)
    # Runs 5 single-loss experiments in isolation on 70% train / 15% val.
    # Test set (15%) is held out and never touched.
    # Contribution F1-scores are normalised to produce dataset-specific λ values.
    # Results exported to ablation_weight_summary.xlsx.
    HEURISTIC_WEIGHT_INIT_MODE = False

    # ===================== Checkpoints & Evaluation =====================
    SAVE_DIR = "checkpoints"

    KEEP_LAST_N = 10
    KEEP_TOP_K  = 5

    LAST_N_EPOCHS = 10
    TOP_K_VALUES  = [2, 3, 4, 5]

    KEEP_BEST_F1_CHECKPOINT_ONLY = True

    # ===================== Helper =====================
    @classmethod
    def to_pipeline_dict(cls):
        """Return a dict that can be unpacked into DistillationPipeline(**config)."""
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
            "loss_lambdas": [
                cls.LAMBDA_CE,
                cls.LAMBDA_PROJ1,
                cls.LAMBDA_PROJ2,
                cls.LAMBDA_LOGITS,
                cls.LAMBDA_DIST,
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
            # Weighted sampler & Focal loss
            "use_weighted_sampler": cls.USE_WEIGHTED_SAMPLER,
            "use_focal_loss": cls.USE_FOCAL_LOSS,
            "focal_gamma": cls.FOCAL_GAMMA,
            "poly_epsilon": cls.POLY_EPSILON,
            "class_weight_method": cls.CLASS_WEIGHT_METHOD,
            "random_seed": cls.RANDOM_SEED,
            # Cross-Validation
            "use_cross_validation": cls.USE_CROSS_VALIDATION,
            "cv_n_splits": cls.CV_N_SPLITS,
            "heuristic_weight_init_mode": cls.HEURISTIC_WEIGHT_INIT_MODE,
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
