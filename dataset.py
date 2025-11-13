"""
Dataset loading and preprocessing with data augmentation
"""
import os
import random
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from config import Config


class ImageDataset(Dataset):
    """Custom dataset for image classification"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(split='train'):
    """
    Get data transforms with augmentation
    
    Args:
        split: 'train', 'val', or 'test'
    
    Returns:
        transforms: torchvision transforms
    """
    
    if split == 'train':
        # Training with data augmentation (based on paper)
        # Rotation: -30° to +30°, Flipping, Brightness/Contrast: 0.8-1.2
        transform = transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),  # -30° to +30°
            transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2)),  # 0.8-1.2
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation and test without augmentation
        transform = transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def load_dataset(dataset_path, train_ratio, val_ratio, test_ratio, random_seed=42):
    """
    Load and split dataset into train/val/test sets
    
    Args:
        dataset_path: Path to dataset directory
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_seed: Random seed for reproducibility
    
    Returns:
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, class_names
    """
    
    random.seed(random_seed)
    
    # Get all class directories
    class_dirs = sorted([d for d in os.listdir(dataset_path) 
                        if os.path.isdir(os.path.join(dataset_path, d))])
    
    print(f"\nFound {len(class_dirs)} classes: {class_dirs}")
    
    # Create class to index mapping
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_dirs)}
    
    # Collect all images per class
    class_images = defaultdict(list)
    
    for class_name in class_dirs:
        class_path = os.path.join(dataset_path, class_name)
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                img_path = os.path.join(class_path, img_name)
                class_images[class_name].append(img_path)
    
    # Print dataset statistics
    print("\nDataset statistics:")
    total_images = 0
    for class_name in class_dirs:
        count = len(class_images[class_name])
        total_images += count
        print(f"  {class_name}: {count} images")
    print(f"  Total: {total_images} images")
    
    # Split data for each class
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []
    
    for class_name in class_dirs:
        images = class_images[class_name]
        random.shuffle(images)
        
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Split
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]
        
        # Get label index
        label = class_to_idx[class_name]
        
        # Add to lists
        train_paths.extend(train_imgs)
        train_labels.extend([label] * len(train_imgs))
        
        val_paths.extend(val_imgs)
        val_labels.extend([label] * len(val_imgs))
        
        test_paths.extend(test_imgs)
        test_labels.extend([label] * len(test_imgs))
    
    print(f"\nData split:")
    print(f"  Train: {len(train_paths)} images")
    print(f"  Val: {len(val_paths)} images")
    print(f"  Test: {len(test_paths)} images")
    
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, class_dirs


def create_dataloaders(train_paths, train_labels, val_paths, val_labels, 
                       test_paths, test_labels, batch_size, num_workers=4):
    """
    Create PyTorch dataloaders
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Create datasets
    train_dataset = ImageDataset(train_paths, train_labels, transform=get_transforms('train'))
    val_dataset = ImageDataset(val_paths, val_labels, transform=get_transforms('val'))
    test_dataset = ImageDataset(test_paths, test_labels, transform=get_transforms('test'))
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    from config import Config
    
    Config.validate_config()
    
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
    
    print("\n✓ Dataset loaded successfully!")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
