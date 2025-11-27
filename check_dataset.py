"""
Script to check dataset for corrupted images
Run this to find and optionally remove corrupted images before training
"""
import os
from PIL import Image
from tqdm import tqdm
from config import Config


def check_corrupted_images(dataset_path, remove_corrupted=False):
    """
    Check all images in dataset for corruption
    
    Args:
        dataset_path: Path to dataset directory
        remove_corrupted: If True, delete corrupted images
    
    Returns:
        corrupted_images: List of corrupted image paths
    """
    print("="*70)
    print(" CHECKING DATASET FOR CORRUPTED IMAGES")
    print("="*70)
    print(f"Dataset path: {dataset_path}")
    
    corrupted_images = []
    total_images = 0
    
    # Get all class directories
    class_dirs = sorted([d for d in os.listdir(dataset_path) 
                        if os.path.isdir(os.path.join(dataset_path, d))])
    
    for class_name in class_dirs:
        class_path = os.path.join(dataset_path, class_name)
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        
        print(f"\nChecking class: {class_name} ({len(image_files)} images)")
        
        for img_name in tqdm(image_files, desc=f"  {class_name}", leave=False):
            img_path = os.path.join(class_path, img_name)
            total_images += 1
            
            try:
                # Try to open and fully load the image
                with Image.open(img_path) as img:
                    img.load()  # Force load the entire image
                    img.convert('RGB')  # Try converting
            except Exception as e:
                corrupted_images.append((img_path, str(e)[:100]))
                print(f"\n  ✗ CORRUPTED: {img_name}")
                print(f"    Error: {str(e)[:100]}")
                
                if remove_corrupted:
                    try:
                        os.remove(img_path)
                        print(f"    → Removed!")
                    except Exception as del_e:
                        print(f"    → Failed to remove: {del_e}")
    
    # Summary
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    print(f"Total images checked: {total_images}")
    print(f"Corrupted images found: {len(corrupted_images)}")
    
    if corrupted_images:
        print("\nCorrupted images:")
        for img_path, error in corrupted_images:
            print(f"  - {img_path}")
            print(f"    Error: {error}")
        
        if not remove_corrupted:
            print("\nTo remove corrupted images, run:")
            print("  python check_dataset.py --remove")
    else:
        print("\n✓ No corrupted images found! Dataset is clean.")
    
    return corrupted_images


if __name__ == "__main__":
    import sys
    
    remove = '--remove' in sys.argv
    
    if remove:
        print("\n⚠ WARNING: This will DELETE corrupted images!")
        confirm = input("Are you sure? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Cancelled.")
            sys.exit(0)
    
    Config.validate_config()
    corrupted = check_corrupted_images(Config.DATASET_PATH, remove_corrupted=remove)
    
    if corrupted and not remove:
        print("\n💡 Tip: The training code now handles corrupted images automatically,")
        print("   but for best results, you should remove or replace them.")
