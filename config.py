"""
Centralized configuration for Knowledge Distillation pipeline.
Modify all hyperparameters here instead of editing main.py directly.
"""


class Config:
    # ===================== Dataset =====================
    DATA_DIR = r"/Capstone_KD_Testing_Ver1/kaggle/working/ProcessedOriginal"
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
        r"/strategy2_top_4_averaged.pth"
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
    PCA_DROPOUT = 0.5                 # dropout for Conv2d layers in PCA projector
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

    # ===================== Loss Weights =====================
    LAMBDA1 = 0.05          # L_proj1  (PCA attention loss)
    LAMBDA2 = 0.05          # L_proj2  (GW linear loss)
    LAMBDA3 = 0.5           # L_logits (Hinton KD loss)
    LAMBDA4 = 0.75         # L_dist   (DIST loss)

    LABEL_SMOOTHING = 0.1   # CrossEntropyLoss label smoothing

    # Ablation study: set False to disable PCA/GL projectors (CE + Logits / CE + Logits + Relation)
    USE_PROJECTION = True

    # Hinton KD temperature
    TEMPERATURE = 4.0

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
            "lambda1": cls.LAMBDA1,
            "lambda2": cls.LAMBDA2,
            "lambda3": cls.LAMBDA3,
            "lambda4": cls.LAMBDA4,
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
