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

try:
    from sklearn.cluster import MiniBatchKMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: sklearn not installed. Install with: pip install scikit-learn")

from common import PuzzleDatasetMetadata


cli = ArgParser()


class ImageDatasetConfig(BaseModel):
    dataset_name: str  # cifar10, cifar100, imagenet-1k-subset
    output_dir: str
    patch_size: int = 8  # Smaller patches for CIFAR (8×8 = 16 patches per 32×32 image)
    vocab_size: int = 2048  # Codebook size (number of unique patch tokens)
    image_size: int = 32  # CIFAR native size
    train_split: float = 0.9
    seed: int = 42
    use_cnn_tokenizer: bool = True  # If True, save raw images; if False, use K-Means patches


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


def image_to_patches(image_array: np.ndarray, patch_size: int):
    """
    Convert image to patch sequence.
    
    Args:
        image_array: [H, W, 3] RGB image
        patch_size: Size of each patch
    
    Returns:
        patches: [num_patches, patch_size*patch_size*3] flattened patches
    """
    H, W, C = image_array.shape
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    patches = []
    
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            patch = image_array[
                i*patch_size:(i+1)*patch_size,
                j*patch_size:(j+1)*patch_size,
                :
            ]
            # Flatten patch: [patch_size, patch_size, 3] -> [patch_size^2 * 3]
            patches.append(patch.flatten())
    
    return np.array(patches, dtype=np.float32)  # [num_patches, patch_dim]


def build_patch_vocabulary(dataset, patch_size: int, vocab_size: int, seed: int):
    """
    Build patch vocabulary using K-Means clustering (like VQ-VAE codebook).
    Each cluster center becomes a 'patch token'.
    
    Args:
        dataset: Train dataset
        patch_size: Size of patches
        vocab_size: Number of clusters (vocabulary size)
        seed: Random seed
    
    Returns:
        kmeans: Fitted KMeans model (codebook)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn required. Install with: pip install scikit-learn")
    
    print(f"\nBuilding patch vocabulary (codebook)...")
    print(f"Sampling patches from training data...")
    
    # Sample patches from subset of training data
    all_patches = []
    sample_size = min(5000, len(dataset))  # Use 5k images max for clustering
    
    for idx in tqdm(range(sample_size)):
        img = np.array(dataset[idx]['img'])
        patches = image_to_patches(img, patch_size)
        all_patches.append(patches)
    
    # Flatten all patches
    all_patches = np.vstack(all_patches)  # [N_total_patches, patch_dim]
    print(f"Collected {len(all_patches):,} patches")
    
    # Normalize to [0, 1]
    all_patches = all_patches / 255.0
    
    # K-Means clustering to create codebook
    print(f"Clustering into {vocab_size} patch tokens...")
    kmeans = MiniBatchKMeans(
        n_clusters=vocab_size,
        random_state=seed,
        batch_size=1024,
        max_iter=100,
        verbose=1
    )
    kmeans.fit(all_patches)
    
    print(f" Codebook created: {vocab_size} unique patch tokens")
    return kmeans


def patches_to_tokens(patches: np.ndarray, kmeans):
    """
    Convert patches to discrete tokens using K-Means.
    
    Args:
        patches: [num_patches, patch_dim] patch features
        kmeans: Fitted KMeans model
    
    Returns:
        tokens: [num_patches] discrete token IDs
    """
    # Find nearest cluster center for each patch
    tokens = kmeans.predict(patches)
    return tokens.astype(np.int32)


def create_cnn_dataset(config: ImageDatasetConfig, dataset, num_classes: int):
    """
    Create dataset for CNN tokenizer (raw images).
    No K-Means clustering needed - images passed directly to CNN.
    """
    print(f"\nProcessing {config.dataset_name} for CNN tokenizer...")
    print(f"Image size: {config.image_size}×{config.image_size}×3")
    print(f"Output classes: {num_classes}")
    
    for split_name, split_data in [("train", dataset['train']), ("test", dataset['test'])]:
        print(f"\nProcessing {split_name} split...")
        os.makedirs(os.path.join(config.output_dir, split_name), exist_ok=True)
        
        inputs = []  # Raw images [N, H, W, C]
        labels = []  # Class labels [N, 1]
        puzzle_identifiers = []
        puzzle_indices = []
        group_indices = [0]
        
        for idx, sample in enumerate(tqdm(split_data)):
            # Get image and label
            image = sample['img']  # PIL Image
            label = sample['label']  # int
            
            # Convert to numpy array
            image_array = np.array(image, dtype=np.float32)  # [H, W, 3]
            
            # Normalize to [-1, 1] or [0, 1] based on preference
            # Using [0, 1] normalization for now
            image_array = image_array / 255.0
            
            # CNN expects [C, H, W] format (PyTorch convention)
            # But we save as [H, W, C] and transpose during loading for efficiency
            inputs.append(image_array)  # [H, W, 3]
            
            # Label: single class prediction
            # For CNN mode, we predict class from global pooling, not sequence
            labels.append([label])  # [1]
            
            puzzle_identifiers.append(0)
            puzzle_indices.append(len(inputs))
        
        group_indices.append(len(inputs))
        
        # Save as numpy arrays
        results = {
            "inputs": np.array(inputs, dtype=np.float32),  # [N, H, W, 3]
            "labels": np.array(labels, dtype=np.int32),    # [N, 1]
            "puzzle_identifiers": np.array(puzzle_identifiers, dtype=np.int32),
            "puzzle_indices": np.array(puzzle_indices, dtype=np.int32),
            "group_indices": np.array(group_indices, dtype=np.int32)
        }
        
        for k, v in results.items():
            np.save(os.path.join(config.output_dir, split_name, f"all__{k}.npy"), v)
        
        # Save metadata
        # CNN produces 8×8=64 spatial positions after downsampling
        spatial_size = config.image_size // 4  # Two 2× pooling layers
        seq_len = spatial_size * spatial_size  # 8×8 = 64
        
        metadata = PuzzleDatasetMetadata(
            seq_len=seq_len,  # 64 spatial positions
            vocab_size=num_classes,  # Output vocab (10 for CIFAR-10)
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
        
        print(f"  Saved {len(inputs)} {split_name} samples")
    
    # Save dataset info
    dataset_info = {
        "dataset_name": config.dataset_name,
        "num_classes": num_classes,
        "image_size": config.image_size,
        "use_cnn_tokenizer": True,
        "spatial_grid": f"{config.image_size // 4}×{config.image_size // 4}",
        "seq_len": (config.image_size // 4) ** 2,
        "output_vocab_size": num_classes,
        "input_format": "raw_images_HWC_float32"
    }
    
    with open(os.path.join(config.output_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"\n CNN Dataset created successfully!")
    print(f"   Output: {config.output_dir}")
    print(f"   Num classes: {num_classes}")
    print(f"   Image format: [H={config.image_size}, W={config.image_size}, C=3]")
    print(f"   Spatial tokens: {(config.image_size // 4) ** 2} ({config.image_size // 4}×{config.image_size // 4} grid)")
    print(f"   Train samples: {len(dataset['train'])}")
    print(f"   Test samples: {len(dataset['test'])}")


def create_image_dataset(config: ImageDatasetConfig):
    """Create image dataset for TRM training."""
    
    if not PIL_AVAILABLE:
        raise ImportError("Pillow required. Install with: pip install Pillow")
    
    np.random.seed(config.seed)
    
    # Load dataset
    if config.dataset_name == "cifar10":
        dataset = download_cifar10()
        num_classes = 10
    elif config.dataset_name == "cifar100":
        dataset = download_cifar100()
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {config.dataset_name}")
    
    if dataset is None:
        return
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    # CNN mode: save raw images directly
    if config.use_cnn_tokenizer:
        print("\n🔧 Using CNN tokenizer mode (saving raw images)...")
        create_cnn_dataset(config, dataset, num_classes)
        return
    
    print(f"Processing {config.dataset_name} dataset...")
    print(f"Patch size: {config.patch_size}×{config.patch_size}")
    print(f"Image size: {config.image_size}×{config.image_size}")
    
    num_patches_per_dim = config.image_size // config.patch_size
    num_patches = num_patches_per_dim ** 2
    patch_dim = config.patch_size * config.patch_size * 3
    
    print(f"Patches per image: {num_patches} ({num_patches_per_dim}×{num_patches_per_dim})")
    print(f"Patch vocabulary size: {config.vocab_size}")
    print(f"Sequence length: {num_patches} tokens (like text!)")
    
    # Build patch vocabulary (codebook) from training data
    print("\n" + "="*70)
    kmeans = build_patch_vocabulary(
        dataset['train'],
        config.patch_size,
        config.vocab_size,
        config.seed
    )
    print("="*70)
    
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
            img = np.array(sample['img'])  # PIL Image -> numpy [H, W, 3]
            label = sample['label']
            
            # Extract patches
            patches = image_to_patches(img, config.patch_size)  # [num_patches, patch_dim]
            
            # Tokenize patches using codebook (like BPE for text!)
            patch_tokens = patches_to_tokens(patches, kmeans)  # [num_patches]
            
            inputs.append(patch_tokens)
            
            # Label: predict class at CLS token position (first token)
            label_seq = np.full(num_patches, -100, dtype=np.int32)
            label_seq[0] = label  # Only first token predicts class
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
        # CRITICAL FIX: vocab_size must be num_classes for lm_head output!
        # embed_tokens vocab will be patch_vocab_size (stored separately)
        metadata = PuzzleDatasetMetadata(
            seq_len=num_patches,  # Number of patch tokens per image
            vocab_size=num_classes,  # OUTPUT vocab (10 for CIFAR-10)
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
    
    # Save codebook
    codebook_path = os.path.join(config.output_dir, "patch_codebook.npy")
    np.save(codebook_path, kmeans.cluster_centers_)
    print(f"Saved codebook to: {codebook_path}")
    
    # Save dataset info
    dataset_info = {
        "dataset_name": config.dataset_name,
        "num_classes": num_classes,
        "image_size": config.image_size,
        "patch_size": config.patch_size,
        "num_patches": num_patches,
        "patch_vocab_size": config.vocab_size,  # Input embedding vocab (2048 patch tokens)
        "output_vocab_size": num_classes,  # Output head vocab (10 classes)
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
