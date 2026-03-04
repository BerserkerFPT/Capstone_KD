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


def worker_init_fn_seed(worker_id):
    """Worker init function for DataLoader reproducibility (must be at module level for pickle)"""
    random.seed(Config.RANDOM_SEED + worker_id)


class ImageDataset(Dataset):
    """Custom dataset for image classification with error handling for corrupted images"""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self._corrupted_cache = set()  # Cache corrupted image indices

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            # Load image with error handling
            image = Image.open(img_path).convert('RGB')

            # Apply transforms
            if self.transform:
                image = self.transform(image)

            return image, label

        except (OSError, IOError) as e:
            # Handle corrupted/broken images
            if idx not in self._corrupted_cache:
                self._corrupted_cache.add(idx)
                print(f"\n  Warning: Corrupted image at {img_path}: {str(e)[:50]}")

            # Return a black image with correct dimensions as fallback
            if self.transform:
                # Create a dummy black image
                dummy_image = Image.new('RGB', (Config.IMAGE_SIZE, Config.IMAGE_SIZE), (0, 0, 0))
                dummy_image = self.transform(dummy_image)
                return dummy_image, label
            else:
                # Without transform, return a tensor of zeros
                return torch.zeros(3, Config.IMAGE_SIZE, Config.IMAGE_SIZE), label


def get_transforms(split='train'):
    """
    Get data transforms with augmentation

    Args:
        split: 'train', 'val', or 'test'

    Returns:
        transforms: torchvision transforms
    """

    if split == 'train':
        # Training WITH data augmentation for better generalization
        # Enable augmentation: Flipping, Rotation, Brightness/Contrast adjustments
        transform = transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.3),  # 30% chance horizontal flip
            transforms.RandomVerticalFlip(p=0.3),     # 30% chance vertical flip
            transforms.RandomRotation(degrees=90),    # Random rotation up to 90 degrees
            transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2)),  # Color variation
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
    IMPORTANT: Uses fixed random seed to ensure identical train/val/test splits across all models

    Args:
        dataset_path: Path to dataset directory
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_seed: Random seed for reproducibility

    Returns:
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, class_names
    """

    # CRITICAL: Set random seed for reproducible train/val/test splits
    random.seed(random_seed)
    torch.manual_seed(random_seed)

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
        # Sort first for deterministic order, then shuffle with fixed seed
        images = sorted(images)  # Deterministic base order
        random.shuffle(images)   # Shuffle with seeded random

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

    # VERIFICATION: Print first few samples for reproducibility check
    print(f"\n✓ Reproducibility check (random_seed={random_seed}):")
    print(f"  First train sample: {Path(train_paths[0]).name if train_paths else 'N/A'}")
    print(f"  First val sample: {Path(val_paths[0]).name if val_paths else 'N/A'}")
    print(f"  First test sample: {Path(test_paths[0]).name if test_paths else 'N/A'}")

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

    # Determine if CUDA is available for pin_memory
    use_cuda = torch.cuda.is_available()

    # Create dataloaders with proper settings for reproducibility
    # OPTIMIZATION: persistent_workers=True keeps workers alive between epochs (faster training)
    # Only use with num_workers > 0
    use_persistent = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,  # Only use pin_memory with CUDA
        persistent_workers=use_persistent,  # Keep workers alive between epochs
        worker_init_fn=worker_init_fn_seed,  # Use module-level function (not lambda) for Windows pickle
        generator=torch.Generator().manual_seed(Config.RANDOM_SEED),  # Shuffle reproducibility
        prefetch_factor=2 if num_workers > 0 else None  # Prefetch batches for faster loading
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,  # Only use pin_memory with CUDA
        persistent_workers=use_persistent,  # Keep workers alive
        prefetch_factor=2 if num_workers > 0 else None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,  # Only use pin_memory with CUDA
        persistent_workers=use_persistent,  # Keep workers alive
        prefetch_factor=2 if num_workers > 0 else None
    )

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
