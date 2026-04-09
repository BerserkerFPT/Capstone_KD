import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
import numpy as np
import random
import os

# =========================
# GLOBAL SEED FUNCTION
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Chỉ bật nếu CUDA hỗ trợ
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

# =========================
# WORKER SEED FUNCTION
# =========================
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class DatasetHandler:
    """
    Handle dataset loading, augmentation, and splitting (70/15/15)
    """
    
    def __init__(
        self,
        root_dir,
        image_size=224,
        batch_size=32,
        num_workers=4,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
        use_weighted_sampler=False,
        fold_indices=None
    ):
        self.root_dir = root_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.use_weighted_sampler = use_weighted_sampler
        self.fold_indices = fold_indices  # (train_idx, val_idx, test_idx) for CV
        
        # ===== SET GLOBAL SEED =====
        set_seed(self.random_seed)
        
        # Cache split indices để đảm bảo consistency
        self._cached_indices = None
        
        # Augmentation transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.val_test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _split_indices(self, dataset):
        """
        Split dataset indices into train/val/test (70/15/15)
        Stratified split to maintain class balance.
        If fold_indices is provided (CV mode), use those directly.
        """
        # Sử dụng fold_indices nếu có (Cross-Validation mode)
        if self.fold_indices is not None:
            return self.fold_indices

        # Sử dụng cache để đảm bảo consistency
        if self._cached_indices is not None:
            return self._cached_indices
        
        # Reset seed trước khi split
        set_seed(self.random_seed)
        
        targets = np.array(dataset.targets)
        indices = np.arange(len(dataset))
        
        # First split: train (70%) vs temp (30%)
        train_indices, temp_indices = train_test_split(
            indices,
            test_size=(self.val_ratio + self.test_ratio),
            stratify=targets,
            random_state=self.random_seed
        )
        
        # Second split: val (15%) vs test (15%) from temp (30%)
        temp_targets = targets[temp_indices]
        val_ratio_adjusted = self.val_ratio / (self.val_ratio + self.test_ratio)
        
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=(1 - val_ratio_adjusted),
            stratify=temp_targets,
            random_state=self.random_seed
        )
        
        # Cache indices
        self._cached_indices = (train_indices, val_indices, test_indices)
        
        return train_indices, val_indices, test_indices
    
    def get_datasets(self):
        """
        Returns train, val, test datasets
        """
        # Load full dataset without transform first (for splitting)
        full_dataset = ImageFolder(root=self.root_dir)
        
        # Get split indices
        train_indices, val_indices, test_indices = self._split_indices(full_dataset)
        
        # Create datasets with appropriate transforms
        train_dataset = ImageFolder(root=self.root_dir, transform=self.train_transform)
        val_dataset = ImageFolder(root=self.root_dir, transform=self.val_test_transform)
        test_dataset = ImageFolder(root=self.root_dir, transform=self.val_test_transform)
        
        # Create subsets
        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(val_dataset, val_indices)
        test_subset = Subset(test_dataset, test_indices)
        
        return train_subset, val_subset, test_subset
    
    def get_train_labels(self):
        """
        Returns labels for the training split.
        """
        full_dataset = ImageFolder(root=self.root_dir)
        train_indices, _, _ = self._split_indices(full_dataset)
        return [full_dataset.targets[i] for i in train_indices]
    
    def get_dataloaders(self):
        """
        Returns train, val, test dataloaders
        """
        train_subset, val_subset, test_subset = self.get_datasets()
        
        # Generator cho reproducibility
        g = torch.Generator()
        g.manual_seed(self.random_seed)

        sampler = None
        shuffle = True

        if self.use_weighted_sampler:
            # Compute class weights from training labels
            train_labels = self.get_train_labels()
            class_sample_counts = torch.bincount(torch.tensor(train_labels))
            weights = 1.0 / class_sample_counts.float()
            samples_weights = weights[torch.tensor(train_labels)]
            sampler = WeightedRandomSampler(
                weights=samples_weights.double(),
                num_samples=len(samples_weights),
                replacement=True
            )
            shuffle = False
            print(f"\u2705 WeightedRandomSampler enabled (class counts: {class_sample_counts.tolist()})")

        train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            generator=g if sampler is None else None,
            drop_last=False,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            drop_last=False,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
        test_loader = DataLoader(
            test_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            drop_last=False,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
        return train_loader, val_loader, test_loader
    
    def get_class_names(self):
        """
        Returns list of class names
        """
        dataset = ImageFolder(root=self.root_dir)
        return dataset.classes
    
    def get_num_classes(self):
        """
        Returns number of classes
        """
        return len(self.get_class_names())
    
    def verify_split_consistency(self):
        """
        Verify that split is consistent across multiple calls
        """
        # Get indices twice
        full_dataset = ImageFolder(root=self.root_dir)
        
        self._cached_indices = None
        train1, val1, test1 = self._split_indices(full_dataset)
        
        self._cached_indices = None
        train2, val2, test2 = self._split_indices(full_dataset)
        
        print(f"Train consistent: {np.array_equal(train1, train2)}")
        print(f"Val consistent: {np.array_equal(val1, val2)}")
        print(f"Test consistent: {np.array_equal(test1, test2)}")
        print(f"Test indices (first 10): {test1[:10]}")
        
        return np.array_equal(test1, test2)
    
# ===== Test =====
if __name__ == "__main__":
    # Example usage
    root_dir = r"/home/student/kaggle/working/ProcessedOriginal"  # Change to your data path
    
    handler = DatasetHandler(
        root_dir=root_dir,
        image_size=224,
        batch_size=32,
        num_workers=0  # Set to 0 for Windows debugging
    )
    
    # Get dataloaders
    train_loader, val_loader, test_loader = handler.get_dataloaders()
    
    print(f"Number of classes: {handler.get_num_classes()}")
    print(f"Class names: {handler.get_class_names()}")
    print(f"\nTrain samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test one batch
    for images, labels in train_loader:
        print(f"\nBatch shape: {images.shape}")  # [B, 3, 224, 224]
        print(f"Labels shape: {labels.shape}")
        break

# for nw in [4, 8, 4]:
#     handler = DatasetHandler(root_dir, num_workers=nw)
#     train_loader, _, _ = handler.get_dataloaders()
#     images, labels = next(iter(train_loader))
#     print(nw, images.mean().item())