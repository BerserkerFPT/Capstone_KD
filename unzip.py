"""
Unzip dataset
"""
import zipfile
import os
import sys

zip_file = sys.argv[1] if len(sys.argv) > 1 else "moringa_dataset.zip"
extract_to = "MorningaLeaf_Dataset"  # Giải nén vào folder này

if not os.path.exists(zip_file):
    print(f"✗ File not found: {zip_file}")
    exit(1)

# Create extraction folder
os.makedirs(extract_to, exist_ok=True)

print(f"Extracting {zip_file} to {extract_to}...")
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("✓ Done!")
dataset_path = os.path.abspath(extract_to)
print(f"Dataset location: {dataset_path}")
print(f'\nUpdate config.py:')
print(f'DATASET_PATH = r"{dataset_path}"')


