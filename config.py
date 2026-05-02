"""
AgriKD — Baseline Model Selection Pipeline
Configuration for benchmarking candidate teacher and student architectures.
"""
import os

class Config:

    # ===================== Dataset =====================
    DATASET_PATH = r"/home/student/TomatoDataset"
    TRAIN_RATIO  = 0.7
    VAL_RATIO    = 0.15
    TEST_RATIO   = 0.15

    # ===================== Training =====================
    BATCH_SIZE   = 32
    NUM_EPOCHS   = 50
    LEARNING_RATE = 9e-7
    WEIGHT_DECAY  = 0.1          # L2 regularisation
    WARMUP_EPOCHS = int(NUM_EPOCHS * 0.1)
    ETA_MIN       = 1e-5         # CosineAnnealingLR minimum LR
    NUM_WORKERS   = 16

    # Early stopping
    EARLY_STOPPING_PATIENCE = 10  # epochs without val_loss improvement

    # LR decay (ReduceLROnPlateau fallback)
    LR_DECAY_PATIENCE = 5
    LR_DECAY_FACTOR   = 0.5

    # ===================== Class-Imbalance Handling =====================
    USE_WEIGHTED_SAMPLER = True   # WeightedRandomSampler (data-level rebalancing)

    # ===================== Cross-Validation =====================
    USE_CROSS_VALIDATION = False  # Stratified K-Fold CV
    CV_N_SPLITS          = 5

    # ===================== Loss Function =====================
    # 'cross_entropy'  — standard CE with optional label smoothing
    # 'poly_focal'     — PolyFocalLoss (recommended for imbalanced datasets)
    LOSS_FUNCTION    = 'cross_entropy'
    LABEL_SMOOTHING  = 0.15       # only for cross_entropy
    FOCAL_GAMMA      = 2.0        # only for poly_focal — focusing parameter
    POLY_EPSILON     = 1.0        # only for poly_focal — poly modulation coefficient
    CLASS_WEIGHT_METHOD = 'inverse_freq'  # 'inverse_freq' or 'effective_num'

    # ===================== Candidate Architectures =====================
    # Add/remove models to benchmark. Uncomment to include.
    # Teacher candidates: large, high-accuracy models
    # Student candidates: lightweight, efficient models
    MODELS = [
        # 'vgg16',  
        # 'resnet18',
        # 'resnet101',
        # 'mobilenet_v2'
        # 'densenet121'
        # 'efficientnet_b0',
        'vit_base_patch16_224'
    ]
    
    # Custom classifier configuration

    CLASSIFIER_CONFIG = [512]  
    DROPOUT_RATE = 0.4 
    
    # ===================== Image Configuration =====================
    IMAGE_SIZE = 224

    # ===================== Evaluation Strategies =====================
    # Strategy 1: best checkpoint (lowest val_loss)
    # Strategy 2: weight-average of Top-K best checkpoints
    TOP_K_VALUES  = [2, 3, 4, 5]
    # Strategy 3: weight-average of last N epoch checkpoints
    LAST_N_EPOCHS = 10

    # Checkpoint storage limits (disk optimisation)
    KEEP_LAST_N_CHECKPOINTS = 10
    KEEP_TOP_K_CHECKPOINTS  = 5

    # ===================== Output =====================
    if os.path.exists('/kaggle'):
        CHECKPOINTS_DIR = "/kaggle/working/checkpoints"
        RESULTS_DIR     = "/kaggle/working/results"
    else:
        CHECKPOINTS_DIR = "checkpoints"
        RESULTS_DIR     = "results"

    AUTO_DELETE_CHECKPOINTS = False  # delete checkpoints after evaluation
    KEEP_RESULTS            = True   # always keep Excel/chart outputs

    RANDOM_SEED = 42

    # ===================== W&B (optional) =====================
    USE_WANDB       = False
    WANDB_API_KEY   = ""           # set your key here
    WANDB_PROJECT   = "AgriKD-Baseline"
    WANDB_ENTITY    = None
    EXPERIMENT_NAME = "baseline_exp1"  # change for each experiment run
    
    @classmethod
    def get_num_classes(cls):
        """Automatically detect number of classes from dataset path"""
        if os.path.exists(cls.DATASET_PATH):
            classes = [d for d in os.listdir(cls.DATASET_PATH) 
                      if os.path.isdir(os.path.join(cls.DATASET_PATH, d))]
            return len(classes)
        return 0
    
    @classmethod
    def validate_config(cls):
        """Validate configuration"""
        if not os.path.exists(cls.DATASET_PATH):
            raise ValueError(f"Dataset path does not exist: {cls.DATASET_PATH}")
        
        if cls.TRAIN_RATIO + cls.VAL_RATIO + cls.TEST_RATIO != 1.0:
            raise ValueError("Train/Val/Test ratios must sum to 1.0")
        
        if cls.EARLY_STOPPING_PATIENCE >= cls.NUM_EPOCHS:
            raise ValueError("Early stopping patience should be less than num_epochs")
        
        print(f"✓ Config validated successfully")
        print(f"  Dataset: {cls.DATASET_PATH}")
        print(f"  Number of classes: {cls.get_num_classes()}")
        print(f"  Models to train: {len(cls.MODELS)}")
