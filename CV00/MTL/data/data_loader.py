import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image, ImageFile
import pandas as pd
import numpy as np
import os
import time

# To handle truncated images gracefully
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Important Prerequisite ---
# This library is used for high-quality, on-the-fly face detection and alignment.
from facenet_pytorch import MTCNN
# --- Configuration ---
TRAIN_MANIFEST_PATH = 'manifests/train_manifest.csv'
VAL_MANIFEST_PATH = 'manifests/val_manifest.csv'

# Threshold for downsampling, based on our analysis
TARGET_COUNT_PER_CLASS = 90

# Model-specific parameters
IMAGE_SIZE = 112
BATCH_SIZE = 32 # A smaller batch size for demonstration

# --- 1. Custom Dataset Class with Resampling ---
class FaceDataset(Dataset):
    """
    Custom Dataset for Age-Invariant Face Recognition.
    Handles dynamic resampling (downsampling/oversampling), MTCNN alignment, and augmentations.
    """
    def __init__(self, manifest_path, target_count_per_class=None, mtcnn_preprocessor=None, augmentations=None, is_train=True):
        self.manifest_path = manifest_path
        self.target_count_per_class = target_count_per_class
        self.mtcnn = mtcnn_preprocessor
        self.augmentations = augmentations
        self.is_train = is_train

        # Load the original manifest
        df = pd.read_csv(self.manifest_path)

        # --- Filtering Corrupted or Missing Images ---
        print("Filtering out corrupted or missing image files...")
        valid_paths = []
        for idx, row in df.iterrows():
            image_path = row['image_path']
            try:
                # Attempt to open the image. This will raise an error if the file is corrupt or missing.
                with Image.open(image_path) as img:
                    # Check the size, a simple way to verify it's a valid image
                    _ = img.size
                valid_paths.append(row)
            except (IOError, FileNotFoundError) as e:
                print(f"Skipping corrupted or missing image: {image_path} due to error: {e}")

        # Recreate the manifest with only the valid images
        self.manifest = pd.DataFrame(valid_paths)
        print(f"Filtering complete. Removed {len(df) - len(self.manifest)} invalid files.")


        # --- Dynamic Resampling Logic for Training Set ---
        if self.is_train and self.target_count_per_class is not None:
            print(f"Balancing all classes to have exactly {self.target_count_per_class} images...")
            num_images_before = len(df)

            balanced_manifest_list = []
            grouped = df.groupby('identity_id')

            for identity_id, group in grouped:
                num_samples = len(group)

                if num_samples > self.target_count_per_class:
                    # Downsample (cap) the majority classes
                    balanced_group = group.sample(n=self.target_count_per_class, random_state=42)
                elif num_samples < self.target_count_per_class:
                    # Oversample the minority classes by sampling with replacement
                    num_to_add = self.target_count_per_class - num_samples
                    additional_samples = group.sample(n=num_to_add, replace=True, random_state=42)
                    balanced_group = pd.concat([group, additional_samples])
                else:
                    # Class is already perfectly sized
                    balanced_group = group

                balanced_manifest_list.append(balanced_group)

            # Combine all balanced groups into a single DataFrame
            self.manifest = pd.concat(balanced_manifest_list)

            # Important: Shuffle the final manifest so batches contain diverse identities
            self.manifest = self.manifest.sample(frac=1, random_state=42).reset_index(drop=True)

            num_images_after = len(self.manifest)
            print(f"Resampling complete. Dataset size changed from {num_images_before} to {num_images_after}.")

            # Create identity mapping for training set
            unique_identities = sorted(self.manifest['identity_id'].unique())
            self.identity_mapping = {orig_id: new_id for new_id, orig_id in enumerate(unique_identities)}

            # Apply mapping to training data
            self.manifest['identity_id'] = self.manifest['identity_id'].map(self.identity_mapping)

        else:
            # For validation/test set, use the original manifest
            self.manifest = df

        # Factorize age groups to get integer labels
        # FIX: Apply factorize to the entire 'age_group' Series
        self.manifest['age_group_id'] = pd.factorize(self.manifest['age_group'])[0]


    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        # The rest of this function remains the same as before
        sample_row = self.manifest.iloc[idx]
        image_path = sample_row['image_path']
        identity_label = sample_row['identity_id']
        age_label = float(sample_row['age'])
        age_group_label = sample_row['age_group_id']

        try:
            image = Image.open(image_path).convert('RGB')
        except (IOError, FileNotFoundError):
            print(f"Warning: Could not open image {image_path}. Skipping.")

        if self.mtcnn:
            aligned_face = self.mtcnn(image)
            if aligned_face is None:
                center_crop_transform = transforms.Compose([
                    transforms.CenterCrop(min(image.size)),
                    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)) # Ensure size is correct on fallback
                ])
                image = center_crop_transform(image)
            else:
                image = transforms.ToPILImage()(aligned_face)

        if self.augmentations:
            image = self.augmentations(image)

        image_tensor = transforms.ToTensor()(image)
        image_tensor = (image_tensor * 2.0) - 1.0

        return (
            image_tensor.float(),
            torch.tensor(age_label).float(),
            torch.tensor(age_group_label).float(),
            torch.tensor(identity_label).float()
        )




# --- 2. Define Augmentations and MTCNN Preprocessor ---
print("\n--- Initializing Preprocessors and Augmentations ---")

train_augmentations = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
])

val_augmentations = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
])

mtcnn_processor = MTCNN(image_size=160, margin=0, keep_all=False, select_largest=True, post_process=False, device='cpu')




# --- 3. Instantiate Datasets and DataLoaders ---
print("\n--- Creating Datasets and DataLoaders ---")

train_dataset = FaceDataset(
    manifest_path=TRAIN_MANIFEST_PATH,
    target_count_per_class=TARGET_COUNT_PER_CLASS,
    mtcnn_preprocessor=mtcnn_processor,
    augmentations=train_augmentations,
    is_train=True
)

val_dataset = FaceDataset(
    manifest_path=VAL_MANIFEST_PATH,
    target_count_per_class=None, # No balancing for validation
    mtcnn_preprocessor=mtcnn_processor,
    augmentations=val_augmentations,
    is_train=False
)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)


