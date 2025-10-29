"""
Image dataset builder for TRM vision models.
Supports CIFAR-10, CIFAR-100, and ImageNet subsets.
"""

from typing import List
import os
import json
import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("WARNING: PIL not installed. Install with: pip install Pillow")

from common import PuzzleDatasetMetadata


cli = ArgParser()


class ImageDatasetConfig(BaseModel):
    dataset_name: str  # cifar10, cifar100, imagenet-1k-subset
    output_dir: str
    patch_size: int = 16  # ViT patch size
    image_size: int = 224  # Resize images to this size
    train_split: float = 0.9
    seed: int = 42


def download_cifar10():
    """Download CIFAR-10 dataset."""
    try:
        from datasets import load_dataset
        print("Downloading CIFAR-10 from HuggingFace...")
        dataset = load_dataset("cifar10")
        return dataset
    except ImportError:
        print("❌ datasets library not found. Install with: pip install datasets")
        return None
    except Exception as e:
        print(f"❌ Failed to download CIFAR-10: {e}")
        return None


def download_cifar100():
    """Download CIFAR-100 dataset."""
    try:
        from datasets import load_dataset
        print("Downloading CIFAR-100 from HuggingFace...")
        dataset = load_dataset("cifar100")
        return dataset
    except ImportError:
        print("❌ datasets library not found. Install with: pip install datasets")
        return None
    except Exception as e:
        print(f"❌ Failed to download CIFAR-100: {e}")
        return None


def image_to_patches(image_array: np.ndarray, patch_size: int, image_size: int):
    """
    Convert image to patch sequence (ViT-style).
    
    Args:
        image_array: [H, W, 3] RGB image
        patch_size: Size of each patch
        image_size: Target image size (resize if needed)
    
    Returns:
        patches: [num_patches, patch_size*patch_size*3] flattened patches
    """
    if not PIL_AVAILABLE:
        raise ImportError("PIL required for image processing")
    
    # Convert to PIL and resize
    img = Image.fromarray(image_array)
    img = img.resize((image_size, image_size), Image.BICUBIC)
    img_array = np.array(img)  # [image_size, image_size, 3]
    
    # Extract patches
    num_patches_per_dim = image_size // patch_size
    patches = []
    
    for i in range(num_patches_per_dim):
        for j in range(num_patches_per_dim):
            patch = img_array[
                i*patch_size:(i+1)*patch_size,
                j*patch_size:(j+1)*patch_size,
                :
            ]
            # Flatten patch: [patch_size, patch_size, 3] -> [patch_size^2 * 3]
            patches.append(patch.flatten())
    
    return np.array(patches)  # [num_patches, patch_dim]


def create_image_dataset(config: ImageDatasetConfig):
    """Create image dataset for TRM training."""
    
    if not PIL_AVAILABLE:
        raise ImportError("Pillow required. Install with: pip install Pillow")
    
    np.random.seed(config.seed)
    
    # Load dataset
    if config.dataset_name == "cifar10":
        dataset = download_cifar10()
        if dataset is None:
            raise RuntimeError("Failed to download CIFAR-10")
        num_classes = 10
    elif config.dataset_name == "cifar100":
        dataset = download_cifar100()
        if dataset is None:
            raise RuntimeError("Failed to download CIFAR-100")
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {config.dataset_name}")
    
    print(f"Processing {config.dataset_name} dataset...")
    print(f"Image size: {config.image_size}×{config.image_size}")
    print(f"Patch size: {config.patch_size}×{config.patch_size}")
    
    num_patches = (config.image_size // config.patch_size) ** 2
    patch_dim = config.patch_size * config.patch_size * 3
    
    print(f"Patches per image: {num_patches}")
    print(f"Patch dimension: {patch_dim}")
    
    # Process splits
    os.makedirs(config.output_dir, exist_ok=True)
    
    for split_name in ["train", "test"]:
        split_data = dataset[split_name]
        os.makedirs(os.path.join(config.output_dir, split_name), exist_ok=True)
        
        print(f"\nProcessing {split_name} split ({len(split_data)} samples)...")
        
        inputs = []
        labels = []
        puzzle_identifiers = []
        puzzle_indices = [0]
        group_indices = [0]
        
        for idx, sample in enumerate(tqdm(split_data)):
            # Extract image and label
            img = np.array(sample['img'])  # PIL Image -> numpy
            label = sample['label']
            
            # Convert to patches
            patches = image_to_patches(img, config.patch_size, config.image_size)
            
            # Store as "input tokens" (patch indices will be learned embeddings)
            # For now, use raw patch values normalized to [0, 255] -> [0, vocab_size-1]
            # In practice, you'd use a learned patch embedding layer
            
            # Flatten patches to sequence
            inputs.append(patches.flatten().astype(np.int32) % 256)  # Normalize to byte range
            
            # Label is the class
            label_seq = np.full(num_patches, -100, dtype=np.int32)  # Ignore tokens
            label_seq[0] = label  # Only predict at CLS token position
            labels.append(label_seq)
            
            puzzle_identifiers.append(0)  # No puzzle-specific embedding
            puzzle_indices.append(len(inputs))
            
            if (idx + 1) % 10000 == 0:
                print(f"  Processed {idx + 1} samples")
        
        group_indices.append(len(inputs))
        
        # Save as numpy arrays
        results = {
            "inputs": np.array(inputs, dtype=np.int32),
            "labels": np.array(labels, dtype=np.int32),
            "puzzle_identifiers": np.array(puzzle_identifiers, dtype=np.int32),
            "puzzle_indices": np.array(puzzle_indices, dtype=np.int32),
            "group_indices": np.array(group_indices, dtype=np.int32)
        }
        
        for k, v in results.items():
            np.save(os.path.join(config.output_dir, split_name, f"all__{k}.npy"), v)
        
        # Save metadata
        metadata = PuzzleDatasetMetadata(
            seq_len=num_patches,
            vocab_size=256,  # Byte-level representation (will use learned embeddings)
            pad_id=0,
            ignore_label_id=-100,
            blank_identifier_id=0,
            num_puzzle_identifiers=1,
            total_groups=1,
            mean_puzzle_examples=1.0,
            total_puzzles=len(inputs),
            sets=["all"]
        )
        
        with open(os.path.join(config.output_dir, split_name, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f, indent=2)
    
    # Save dataset info
    dataset_info = {
        "dataset_name": config.dataset_name,
        "num_classes": num_classes,
        "image_size": config.image_size,
        "patch_size": config.patch_size,
        "num_patches": num_patches,
        "patch_dim": patch_dim
    }
    
    with open(os.path.join(config.output_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"\n✅ Dataset created successfully!")
    print(f"   Output: {config.output_dir}")
    print(f"   Num classes: {num_classes}")
    print(f"   Patches per image: {num_patches}")
    print(f"   Train samples: {len(dataset['train'])}")
    print(f"   Test samples: {len(dataset['test'])}")


@cli.command(singleton=True)
def main(config: ImageDatasetConfig):
    create_image_dataset(config)


if __name__ == "__main__":
    cli()
