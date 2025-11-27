"""
Training script with early stopping and checkpoint saving
"""
import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import Config
from models import get_model
from dataset import load_dataset, create_dataloaders


class EarlyStopping:
    """Early stopping based on validation loss"""
    
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class CheckpointManager:
    """Manage model checkpoints - Optimized to keep only last N + top K best checkpoints"""
    
    def __init__(self, save_dir, model_name, keep_last_n=10, keep_top_k=5):
        self.save_dir = os.path.join(save_dir, model_name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.checkpoints = []  # List of (epoch, val_loss, checkpoint_path)
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.keep_last_n = keep_last_n  # Keep last N epochs
        self.keep_top_k = keep_top_k    # Keep top K best checkpoints
    
    def save_checkpoint(self, model, optimizer, epoch, val_loss, is_best=False):
        """Save checkpoint and manage storage efficiently"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
        }
        
        # Save checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'epoch_{epoch:03d}_val_loss_{val_loss:.4f}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Add to checkpoint list
        self.checkpoints.append((epoch, val_loss, checkpoint_path))
        
        # Track best
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
        
        # Save best checkpoint separately
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
        
        # MEMORY OPTIMIZATION: Clean up old checkpoints
        # Keep only: last N epochs + top K best val_loss
        self._cleanup_checkpoints()
        
        return checkpoint_path
    
    def _cleanup_checkpoints(self):
        """Remove checkpoints that are not in last N epochs or top K best val_loss"""
        if len(self.checkpoints) <= self.keep_last_n + self.keep_top_k:
            return  # Not enough checkpoints to cleanup
        
        # Get last N checkpoints by epoch
        sorted_by_epoch = sorted(self.checkpoints, key=lambda x: x[0])
        last_n_epochs = set(cp[0] for cp in sorted_by_epoch[-self.keep_last_n:])
        
        # Get top K checkpoints by val_loss
        sorted_by_loss = sorted(self.checkpoints, key=lambda x: x[1])
        top_k_epochs = set(cp[0] for cp in sorted_by_loss[:self.keep_top_k])
        
        # Combined set of epochs to keep
        epochs_to_keep = last_n_epochs | top_k_epochs
        
        # Find checkpoints to delete
        checkpoints_to_remove = []
        checkpoints_to_keep = []
        
        for epoch, val_loss, path in self.checkpoints:
            if epoch in epochs_to_keep:
                checkpoints_to_keep.append((epoch, val_loss, path))
            else:
                checkpoints_to_remove.append((epoch, val_loss, path))
        
        # Delete old checkpoints
        for epoch, val_loss, path in checkpoints_to_remove:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                print(f"  Warning: Could not delete checkpoint {path}: {e}")
        
        # Update checkpoints list
        self.checkpoints = checkpoints_to_keep
    
    def get_best_checkpoint(self):
        """Get checkpoint with lowest val_loss"""
        if not self.checkpoints:
            return None
        return min(self.checkpoints, key=lambda x: x[1])
    
    def get_top_k_checkpoints(self, k):
        """Get top K checkpoints with lowest val_loss"""
        if not self.checkpoints:
            return []
        sorted_checkpoints = sorted(self.checkpoints, key=lambda x: x[1])
        return sorted_checkpoints[:k]
    
    def get_last_n_checkpoints(self, n):
        """Get last N epoch checkpoints"""
        if not self.checkpoints:
            return []
        sorted_by_epoch = sorted(self.checkpoints, key=lambda x: x[0])
        return sorted_by_epoch[-n:]
    
    def save_checkpoint_info(self):
        """Save checkpoint information to JSON"""
        info = {
            'checkpoints': [(epoch, val_loss, path) for epoch, val_loss, path in self.checkpoints]
        }
        info_path = os.path.join(self.save_dir, 'checkpoint_info.json')
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)


def train_one_epoch(model, train_loader, criterion, optimizer, device, freeze_backbone=True):
    """Train for one epoch"""
    model.train()
    
    # CRITICAL FIX: Set frozen backbone modules to eval mode to prevent BatchNorm stats update
    if freeze_backbone:
        for name, module in model.named_modules():
            # Identify backbone modules (không phải classifier/head/fc)
            if any(backbone_name in name for backbone_name in 
                   ['features', 'layer1', 'layer2', 'layer3', 'layer4',  # VGG, ResNet
                    'blocks', 'stages',  # EfficientNet, ConvNeXt
                    'patch_embed', 'layers', 'pos_drop',  # ViT, Swin
                    'conv_stem', 'bn1']):
                # Check if module is frozen
                if hasattr(module, 'parameters'):
                    params = list(module.parameters())
                    if params and all(not p.requires_grad for p in params):
                        module.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation', leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def train_model(model_name, train_loader, val_loader, num_classes, device):
    """
    Train a single model
    
    Args:
        model_name: Name of the model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        num_classes: Number of classes
        device: Device to train on
    
    Returns:
        checkpoint_manager: CheckpointManager object
    """
    
    print(f"\n{'='*70}")
    print(f"Training {model_name}")
    print(f"{'='*70}")
    
    # Create model
    model = get_model(model_name, num_classes, freeze_backbone=True)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=Config.LEARNING_RATE,
                          weight_decay=Config.WEIGHT_DECAY)  # L2 regularization
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=Config.LR_DECAY_FACTOR, 
        patience=Config.LR_DECAY_PATIENCE,
        # verbose=True  # IMPORTANT: Show LR changes for monitoring
    )
    
    # Early stopping and checkpoint manager
    early_stopping = EarlyStopping(patience=Config.EARLY_STOPPING_PATIENCE)
    checkpoint_manager = CheckpointManager(
        Config.CHECKPOINTS_DIR, 
        model_name,
        keep_last_n=Config.KEEP_LAST_N_CHECKPOINTS,
        keep_top_k=Config.KEEP_TOP_K_CHECKPOINTS
    )
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, freeze_backbone=True)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch results
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch}/{Config.NUM_EPOCHS}] ({epoch_time:.2f}s) - LR: {current_lr:.6f}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Learning rate scheduler step
        scheduler.step(val_loss)
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"  ✓ New best validation loss!")
        
        checkpoint_manager.save_checkpoint(model, optimizer, epoch, val_loss, is_best)
        
        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\n✓ Early stopping triggered at epoch {epoch}")
            break
    
    # Save checkpoint info
    checkpoint_manager.save_checkpoint_info()
    
    print(f"\n✓ Training completed for {model_name}")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Total checkpoints saved: {len(checkpoint_manager.checkpoints)}")
    
    return checkpoint_manager


if __name__ == "__main__":
    # Test training
    from config import Config
    
    Config.validate_config()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("\nLoading dataset...")
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, class_names = load_dataset(
        Config.DATASET_PATH, 
        Config.TRAIN_RATIO, 
        Config.VAL_RATIO, 
        Config.TEST_RATIO,
        Config.RANDOM_SEED
    )
    
    train_loader, val_loader, test_loader = create_dataloaders(
        train_paths, train_labels, 
        val_paths, val_labels, 
        test_paths, test_labels,
        Config.BATCH_SIZE, 
        Config.NUM_WORKERS
    )
    
    num_classes = len(class_names)
    
    # Train first model as test
    checkpoint_manager = train_model(
        Config.MODELS[0], 
        train_loader, 
        val_loader, 
        num_classes, 
        device
    )
