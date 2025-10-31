"""
Unified base class for all dataset builders.
Supports multimodal data: text, images, mazes, puzzles.
"""

import sys
import json
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
import numpy as np
import torch
from pydantic import BaseModel
from dataset.common import PuzzleDatasetMetadata


class ModalityType:
    """Supported data modalities."""
    TEXT = "text"
    IMAGE = "image"
    MAZE = "maze"
    GRID = "grid"  # Generic 2D grid (ARC, Sudoku, etc.)
    MULTIMODAL = "multimodal"  # Mixed modalities


class DataSample(BaseModel):
    """Single training sample with optional multiple modalities."""
    
    # Unique identifier
    sample_id: str
    
    # Primary modality
    modality: str  # ModalityType
    
    # Data fields (at least one must be present)
    text: Optional[str] = None
    image: Optional[Any] = None  # PIL Image or np.ndarray
    grid: Optional[np.ndarray] = None  # 2D array for mazes/puzzles
    
    # Labels/targets
    label: Optional[Union[str, int, np.ndarray]] = None
    
    # Metadata
    metadata: Dict[str, Any] = {}
    
    class Config:
        arbitrary_types_allowed = True


class BaseDatasetBuilder(ABC):
    """
    Abstract base class for all dataset builders.
    
    Implements common functionality:
    - Capsule encoding (HESC)
    - Metadata generation
    - Train/test splitting
    - Augmentation
    - Saving to disk
    """
    
    def __init__(self, config: BaseModel):
        self.config = config
        self.encoder = None
        self.samples: List[DataSample] = []
    
    @abstractmethod
    def load_raw_data(self) -> List[DataSample]:
        """Load raw data from source (files, HF datasets, etc.)."""
        pass
    
    @abstractmethod
    def preprocess_sample(self, sample: DataSample) -> DataSample:
        """Apply modality-specific preprocessing."""
        pass
    
    def augment_sample(self, sample: DataSample) -> List[DataSample]:
        """
        Generate augmented versions of sample.
        Override for modality-specific augmentation.
        """
        return [sample]  # No augmentation by default
    
    def build_dataset(self) -> Dict[str, Any]:
        """
        Main pipeline: load → preprocess → augment → encode → save.
        """
        print(f"\n🔨 Building dataset...")
        
        # Load raw data
        print(f"📥 Loading raw data...")
        self.samples = self.load_raw_data()
        print(f"   Loaded {len(self.samples)} samples")
        
        # Preprocess
        print(f"⚙️  Preprocessing...")
        self.samples = [self.preprocess_sample(s) for s in self.samples]
        
        # Augment
        if getattr(self.config, 'augment', False):
            print(f"🔄 Augmenting...")
            augmented = []
            for sample in self.samples:
                augmented.extend(self.augment_sample(sample))
            self.samples = augmented
            print(f"   Augmented to {len(self.samples)} samples")
        
        # Split train/test
        train_split = getattr(self.config, 'train_split', 0.9)
        split_idx = int(len(self.samples) * train_split)
        train_samples = self.samples[:split_idx]
        test_samples = self.samples[split_idx:]
        
        print(f"   Train: {len(train_samples)}, Test: {len(test_samples)}")
        
        # Encode to capsules
        use_capsules = getattr(self.config, 'use_capsules', True)
        if use_capsules:
            print(f"🧬 Encoding to HESC capsules...")
            train_data = self.encode_to_capsules(train_samples)
            test_data = self.encode_to_capsules(test_samples)
        else:
            print(f"🔢 Encoding to tokens...")
            train_data = self.encode_to_tokens(train_samples)
            test_data = self.encode_to_tokens(test_samples)
        
        return {
            'train': train_data,
            'test': test_data,
            'metadata': self.create_metadata(train_data)
        }
    
    def encode_to_capsules(self, samples: List[DataSample]) -> Dict[str, torch.Tensor]:
        """
        Unified encoding: text/image/grid → capsules.
        Single pass through CapsuleEncoder.
        """
        if self.encoder is None:
            from models.capsule_encoder import CapsuleEncoder
            
            self.encoder = CapsuleEncoder(
                hidden_size=getattr(self.config, 'hidden_size', 768),
                target_capsules=getattr(self.config, 'target_capsules', 12),
                children_per_capsule=4,
                checksum_dim=32,
                freeze_encoder=True,
                encoder_model=getattr(self.config, 'encoder_model', 'openai/clip-vit-large-patch14'),
                use_spatial=getattr(self.config, 'use_spatial_capsules', True),
                capsule_grid_shape=getattr(self.config, 'capsule_grid_shape', (3, 4))
            )
            self.encoder.eval()
            self.encoder.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Unified conversion: all modalities → text representation
        texts = [self._sample_to_text(s) for s in samples]
        
        # Single batch encoding loop
        all_data = {'sketches': [], 'checksums': [], 'children': []}
        batch_size = 8
        
        from tqdm import tqdm
        import gc
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding capsules"):
            with torch.no_grad():
                result = self.encoder(texts[i:i+batch_size], return_children=True)
            
            for key in all_data:
                if key in result and result[key] is not None:
                    all_data[key].append(result[key].cpu())
            
            # Clear intermediate results
            del result
            
            # Periodic memory cleanup (every 100 batches)
            if i % (batch_size * 100) == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Concatenate all batches
        result = {k: torch.cat(v, dim=0) if v else torch.empty(0) for k, v in all_data.items()}
        
        # Ensure at least 'sketches' key exists for compatibility
        if 'sketches' not in result or result['sketches'].numel() == 0:
            result['sketches'] = torch.empty(0, getattr(self.config, 'target_capsules', 12), getattr(self.config, 'hidden_size', 768))
        
        return result
    
    def _sample_to_text(self, sample: DataSample) -> str:
        """Convert any sample type to text representation."""
        if sample.text:
            return sample.text
        elif sample.grid is not None:
            return self._grid_to_text(sample.grid)
        elif sample.image is not None:
            return f"Image: {type(sample.image).__name__}"  # Placeholder
        return ""
    
    def encode_to_tokens(self, samples: List[DataSample]) -> Dict[str, np.ndarray]:
        """Fallback: encode to discrete tokens (legacy mode)."""
        # Implement token-based encoding if needed
        raise NotImplementedError("Token encoding not implemented in base class")
    
    def _grid_to_text(self, grid: np.ndarray) -> str:
        """Convert 2D grid to text representation."""
        # Simple row-based serialization
        rows = []
        for row in grid:
            rows.append(' '.join(map(str, row)))
        return ' | '.join(rows)
    
    def create_metadata(self, train_data: Dict) -> PuzzleDatasetMetadata:
        """Generate dataset metadata."""
        num_concepts = getattr(self.config, 'num_concepts', 2048)
        vocab_size = num_concepts + 4  # Concept vocab
        
        if 'sketches' in train_data:
            seq_len = train_data['sketches'].shape[1]
        else:
            seq_len = 1024  # Default
        
        return PuzzleDatasetMetadata(
            seq_len=seq_len,
            vocab_size=vocab_size,
            pad_id=0,
            ignore_label_id=-100,
            blank_identifier_id=0,
            num_puzzle_identifiers=0,
            total_groups=len(train_data.get('sketches', [])),
            mean_puzzle_examples=1.0,
            total_puzzles=len(train_data.get('sketches', [])),
            sets=["all"]
        )
    
    def save(self, dataset: Dict, output_dir: str):
        """Save dataset to disk."""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n💾 Saving dataset to: {output_dir}")
        
        # Save train (with float16 compression for memory efficiency)
        train_path = os.path.join(output_dir, 'capsule_dataset.pt')
        train_compressed = {
            'sketches': dataset['train']['sketches'].half() if 'sketches' in dataset['train'] else dataset['train'].get('sketches'),
            'checksums': dataset['train'].get('checksums'),  # Keep checksums in float32 for precision
        }
        if 'children' in dataset['train']:
            train_compressed['children'] = dataset['train']['children'].half()  # 50% memory reduction
        # Copy any other keys
        for key in dataset['train']:
            if key not in train_compressed:
                train_compressed[key] = dataset['train'][key]
        
        torch.save(train_compressed, train_path)
        print(f"✅ Saved train (float16 compressed): {train_path}")
        
        # Save test
        test_path = os.path.join(output_dir, 'capsule_dataset_test.pt')
        test_compressed = {
            'sketches': dataset['test']['sketches'].half() if 'sketches' in dataset['test'] else dataset['test'].get('sketches'),
            'checksums': dataset['test'].get('checksums'),
        }
        if 'children' in dataset['test']:
            test_compressed['children'] = dataset['test']['children'].half()
        for key in dataset['test']:
            if key not in test_compressed:
                test_compressed[key] = dataset['test'][key]
        
        torch.save(test_compressed, test_path)
        print(f"✅ Saved test (float16 compressed): {test_path}")
        
        # Save metadata
        metadata_path = os.path.join(output_dir, 'dataset.json')
        with open(metadata_path, 'w') as f:
            json.dump(dataset.get('metadata', {}).__dict__ if hasattr(dataset.get('metadata'), '__dict__') else dataset.get('metadata', {}), f, indent=2)
        print(f"✅ Saved metadata: {metadata_path}")
