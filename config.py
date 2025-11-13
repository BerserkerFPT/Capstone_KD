"""
Configuration file for baseline research
"""
import os

class Config:
    # ===================== Dataset Configuration =====================
    # Thay đổi đường dẫn dataset ở đây
    DATASET_PATH = r"/root/Capstone/MorningaLeaf_Dataset"  # Hoặc "D:\Capstone\TomatoDataset\TomatoDataset"
    
    # Train/Val/Test split ratios
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # ===================== Training Configuration =====================
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    NUM_WORKERS = 4
    
    # Early Stopping
    EARLY_STOPPING_PATIENCE = 20  # Stop if val_loss doesn't improve for 10 epochs
    
    # Learning Rate Decay
    LR_DECAY_PATIENCE = 5  # Reduce LR if val_loss doesn't improve for 5 epochs
    LR_DECAY_FACTOR = 0.5  # Multiply LR by this factor when decaying
    
    # ===================== Model Configuration =====================
    MODELS = [
        'vgg16',
        'resnet101',
        'densenet121',
        'efficientnet_b0',
        'convnext_tiny',
        'vit_base_patch16_224',
        'swin_tiny_patch4_window7_224',
        'convit_tiny'
    ]
    
    # Custom classifier configuration
    # Định nghĩa các lớp fully connected tùy chỉnh
    # Format: [hidden_dim1, hidden_dim2, ..., num_classes]
    CLASSIFIER_CONFIG = [256,128,64]  # 2 hidden layers, num_classes sẽ được tự động thêm
    DROPOUT_RATE = 0.0
    
    # ===================== Image Configuration =====================
    IMAGE_SIZE = 224
    
    # ===================== Evaluation Configuration =====================
    # Strategy 2: Top-K checkpoints to average
    TOP_K_VALUES = [2, 3, 4, 5]
    
    # Strategy 3: Number of last epochs to average
    LAST_N_EPOCHS = 10
    
    # ===================== Output Configuration =====================
    CHECKPOINTS_DIR = "checkpoints"
    RESULTS_DIR = "results"
    
    # Random seed for reproducibility
    RANDOM_SEED = 42
    
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
