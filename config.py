"""
Configuration file for baseline research
"""
import os

class Config:
    # ===================== Dataset Configuration =====================
    # Tự động detect môi trường (Kaggle hoặc local)
    if os.path.exists('/kaggle/input'):
        # Kaggle environment - cập nhật path này theo dataset của bạn trên Kaggle
        DATASET_PATH = "/kaggle/input/tomato-dataset/TomatoDataset"  # Thay đổi theo tên dataset trên Kaggle
    elif os.path.exists("/home/student/vuthde181070/dataset/TomatoDataset"):
        # Server environment
        DATASET_PATH = "/home/student/vuthde181070/dataset/TomatoDataset"
    else:
        # Local environment
        DATASET_PATH = r"/home/student/kaggle/working/ProcessedOriginal"
    # Train/Val/Test split ratios
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
        
    # ===================== Training Configuration =====================
    BATCH_SIZE = 64
    NUM_EPOCHS = 200
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 1e-7  # L2 regularization để chống overfitting
    WARMUP_EPOCHS = int(NUM_EPOCHS * 0.06)
    ETA_MIN = 1e-6 #For CosineAnnealing LR
  # Convert to integer for scheduler
    # Kaggle có 2 CPU cores, nên dùng NUM_WORKERS = 2
    # Set to 0 to avoid multiprocessing issues with limited memory
    NUM_WORKERS = 16
    
    # Early Stopping
    EARLY_STOPPING_PATIENCE = 15  # Stop if val_loss doesn't improve for 20 epochs
    
    # Learning Rate Decay
    LR_DECAY_PATIENCE = 5  # Reduce LR if val_loss doesn't improve for 10 epochs
    LR_DECAY_FACTOR = 0.5  # Multiply LR by this factor when decaying
    
    # ===================== Model Configuration =====================
    MODELS = [
        # 'vgg16',  
        # 'resnet18',
        # 'resnet101',
        # 'mobilenet_v2',
        'densenet121'
        # 'efficientnet_b0',
        # 'vit_base_patch16_224'
    ]
    
    # Custom classifier configuration
    # Định nghĩa các lớp fully connected tùy chỉnh
    # Format: [hidden_dim1, hidden_dim2, ..., num_classes]
    # Đơn giản hóa cho dataset nhỏ (~10k ảnh) để tránh overfitting
    CLASSIFIER_CONFIG = [256, 128]  # Giảm từ 3 xuống 2 hidden layers
    DROPOUT_RATE = 0.5  # Tăng dropdown để chống overfitting mạnh hơn
    
    # ===================== Image Configuration =====================
    IMAGE_SIZE = 224
    
    # ===================== Evaluation Configuration =====================
    # Strategy 2: Top-K checkpoints to average
    TOP_K_VALUES = [2, 3, 4, 5]
    
    # Strategy 3: Number of last epochs to average
    LAST_N_EPOCHS = 10
    
    # Checkpoint management - Memory optimization
    KEEP_LAST_N_CHECKPOINTS = 10  # Keep last N epoch checkpoints
    KEEP_TOP_K_CHECKPOINTS = 5    # Keep top K best val_loss checkpoints
    
    # ===================== Output Configuration =====================
    # Kaggle output được lưu tại /kaggle/working
    if os.path.exists('/kaggle'):
        CHECKPOINTS_DIR = "/kaggle/working/checkpoints"
        RESULTS_DIR = "/kaggle/working/results"
    else:
        CHECKPOINTS_DIR = "checkpoints"
        RESULTS_DIR = "results"
    
    # Checkpoint management - TỰ ĐỘNG XÓA SAU KHI EVALUATE
    AUTO_DELETE_CHECKPOINTS = True  # Xóa checkpoints sau khi evaluate xong mỗi model
    KEEP_RESULTS = True             # Luôn giữ results (Excel, charts)
    
    # Random seed for reproducibility
    RANDOM_SEED = 1
        
    # ===================== W&B Configuration =====================
    # W&B tracking
    USE_WANDB = False  # Set to False to disable wandb
    WANDB_API_KEY = "8ad789629890d812ecffc9f0fce138a75f63f992"  # Your wandb API key
    WANDB_PROJECT = "BurmeseGrape-Capstone"  # Tên project trên wandb
    WANDB_ENTITY = None  # Tên team/user wandb (None = default user)
    # EXPERIMENT_NAME sẽ được set động khi chạy (ví dụ: "experiment_1", "experiment_2")
    EXPERIMENT_NAME = "baseline_exp1"  # ⚠️ THAY ĐỔI CHO MỖI EXPERIMENT
    
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
