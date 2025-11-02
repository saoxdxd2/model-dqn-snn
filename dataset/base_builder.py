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
        """Main pipeline: load → preprocess → augment → encode (chunked for memory)."""
        print("\n🔨 Building dataset...")
        
        # Load
        print("📥 Loading raw data...")
        raw_samples = self.load_raw_data()
        print(f"   Loaded {len(raw_samples)} samples")
        
        # Preprocess with joblib (C++ backend, 2x faster than multiprocessing)
        print("⚙️  Preprocessing...")
        import os
        
        num_workers = min(8, os.cpu_count() or 4)  # Use up to 8 cores
        
        if num_workers > 1 and len(raw_samples) > 1000:
            # Parallel processing with joblib (faster than multiprocessing)
            try:
                from joblib import Parallel, delayed
                processed = Parallel(n_jobs=num_workers, backend='threading', verbose=0)(
                    delayed(self.preprocess_sample)(s) for s in raw_samples
                )
            except ImportError:
                # Fallback to list comprehension
                print("   ⚠️  joblib not found, using single-thread (install: pip install joblib)")
                processed = [self.preprocess_sample(s) for s in raw_samples]
        else:
            # Single-threaded for small datasets
            processed = [self.preprocess_sample(s) for s in raw_samples]
        
        del raw_samples  # Free original
        
        # Augment with streaming to prevent RAM explosion
        print("🔄 Augmenting...")
        if getattr(self.config, 'augment', False):
            augmented = []
            batch_size = 5000  # Process in batches to free memory
            
            for i in range(0, len(processed), batch_size):
                batch = processed[i:i+batch_size]
                for sample in batch:
                    augmented.extend(self.augment_sample(sample))
                
                # Free processed batch immediately
                del batch
                
                # Periodic cleanup
                if i % 10000 == 0 and i > 0:
                    import gc
                    gc.collect()
            
            del processed  # Free original
        else:
            augmented = processed
        print(f"   Augmented to {len(augmented)} samples")
        
        # Split (keep both splits in RAM)
        train_split = getattr(self.config, 'train_split', 0.9)
        split_idx = int(len(augmented) * train_split)
        train_data = augmented[:split_idx]
        test_data = augmented[split_idx:]
        del augmented  # Free original list
        print(f"   Train: {len(train_data)}, Test: {len(test_data)}")
        
        # Encode with direct streaming to prevent RAM leak
        use_capsules = getattr(self.config, 'use_capsules', True)
        if use_capsules:
            print("🧶 Encoding to HESC capsules...")
            
            # CRITICAL: Encode directly without wrapper function to allow immediate cleanup
            import gc
            
            # Train encoding (batch processing with DataLoader)
            print(f"   Encoding {len(train_data)} train samples...")
            train_encoded = self._stream_encode_capsules(train_data)
            del train_data  # Free after encoding
            
            # Test encoding (batch processing with DataLoader)
            print(f"   Encoding {len(test_data)} test samples...")
            test_encoded = self._stream_encode_capsules(test_data)
            del test_data  # Free after encoding
        else:
            # Non-capsule mode: keep raw samples
            train_encoded = train_data
            test_encoded = test_data
        
        return {
            'train': train_encoded,
            'test': test_encoded,
            'metadata': self.create_metadata(train_encoded)
        }
    
    def _stream_encode_capsules(self, samples: List[DataSample]) -> Dict[str, torch.Tensor]:
        """High-performance encoding with 10GB RAM budget using PyTorch DataLoader."""
        if len(samples) == 0:
            return {'sketches': torch.empty(0, getattr(self.config, 'target_capsules', 12), getattr(self.config, 'hidden_size', 768))}
        
        # Initialize encoder once
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
        
        from torch.utils.data import Dataset, DataLoader
        from tqdm import tqdm
        
        # Custom Dataset for efficient batching
        class SampleDataset(Dataset):
            def __init__(self, samples, text_converter):
                self.samples = samples
                self.text_converter = text_converter
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                return self.text_converter(self.samples[idx])
        
        # Create dataset and dataloader with optimized settings
        dataset = SampleDataset(samples, self._sample_to_text)
        
        # Large batch size for GPU efficiency (10GB RAM budget)
        batch_size = 48  # Increased from 32 (use more VRAM: 10.5GB / 15GB)
        num_workers = 2  # Reduced from 4 to save 1-2GB RAM
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),  # Faster GPU transfer
            prefetch_factor=2,  # Prefetch batches
            persistent_workers=True  # Keep workers alive
        )
        
        # Storage for results (on GPU to prevent CPU RAM explosion)
        result_chunks = {'sketches': [], 'checksums': [], 'children': []}
        temp_gpu_list = {'sketches': [], 'checksums': [], 'children': []}
        device = next(self.encoder.parameters()).device
        
        consolidate_every = 800  # Consolidate every 800 batches (reduced overhead)
        batch_count = 0
        
        # Process with DataLoader (automatic batching + parallel preprocessing)
        with torch.no_grad():
            for batch_texts in tqdm(dataloader, desc="Encoding"):
                # Encode batch (stays on GPU)
                batch_result = self.encoder(batch_texts, return_children=True)
                
                # Keep on GPU and accumulate (uses VRAM, not CPU RAM)
                for key in temp_gpu_list:
                    if key in batch_result and batch_result[key] is not None:
                        temp_gpu_list[key].append(batch_result[key])  # Stay on GPU
                
                del batch_result
                batch_count += 1
                
                # Periodic consolidation to prevent VRAM overflow
                # 800 batches × 48 samples × 12 capsules × 768 × 4 bytes = ~1.77 GB
                if batch_count % consolidate_every == 0:
                    for key in temp_gpu_list:
                        if temp_gpu_list[key]:
                            # Concatenate on GPU, convert to fp16, move to CPU
                            chunk = torch.cat(temp_gpu_list[key], dim=0).half().cpu()
                            result_chunks[key].append(chunk)
                            temp_gpu_list[key] = []  # Clear GPU memory
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        # Final consolidation of remaining batches
        for key in temp_gpu_list:
            if temp_gpu_list[key]:
                chunk = torch.cat(temp_gpu_list[key], dim=0).half().cpu()
                result_chunks[key].append(chunk)
        
        del temp_gpu_list
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Final concatenation on CPU (all chunks already in fp16)
        result = {}
        for key, chunk_list in result_chunks.items():
            if chunk_list:
                result[key] = torch.cat(chunk_list, dim=0)  # Concatenate fp16 chunks
            else:
                result[key] = torch.empty(0)
        
        del result_chunks
        
        return result
    
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
        
        # Process in chunks to prevent RAM accumulation from text list
        batch_size = 4
        chunk_size = 500  # Consolidate every 500 samples
        
        from tqdm import tqdm
        import gc
        
        # Initialize result tensors
        result_chunks = {'sketches': [], 'checksums': [], 'children': []}
        temp_data = {'sketches': [], 'checksums': [], 'children': []}
        
        # Process samples without creating full text list (prevents RAM leak)
        total_samples = len(samples)
        for i in tqdm(range(0, total_samples, batch_size), desc="Encoding capsules"):
            # Convert only current batch to text (critical for RAM)
            batch_texts = [self._sample_to_text(s) for s in samples[i:i+batch_size]]
            
            with torch.no_grad():
                batch_result = self.encoder(batch_texts, return_children=True)
            
            # Clear batch texts immediately
            del batch_texts
            
            # Move to CPU immediately and append
            for key in temp_data:
                if key in batch_result and batch_result[key] is not None:
                    temp_data[key].append(batch_result[key].cpu())
            
            del batch_result
            
            # Concatenate chunks periodically to free intermediate lists
            if len(temp_data['sketches']) >= chunk_size // batch_size:
                for key in temp_data:
                    if temp_data[key]:
                        result_chunks[key].append(torch.cat(temp_data[key], dim=0))
                        temp_data[key] = []  # Clear temp list
                
                gc.collect()  # Force garbage collection
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Final concatenation of remaining data
        for key in temp_data:
            if temp_data[key]:
                result_chunks[key].append(torch.cat(temp_data[key], dim=0))
        
        # Concatenate all chunks into final result
        result = {k: torch.cat(v, dim=0) if v else torch.empty(0) for k, v in result_chunks.items()}
        
        # Final cleanup
        del temp_data, result_chunks
        gc.collect()
        
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
        """Save dataset to disk with parallel I/O."""
        import os
        from concurrent.futures import ThreadPoolExecutor
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n💾 Saving dataset to: {output_dir}")
        
        # Prepare compressed datasets
        train_path = os.path.join(output_dir, 'capsule_dataset.pt')
        train_compressed = {
            'sketches': dataset['train']['sketches'].half() if 'sketches' in dataset['train'] else dataset['train'].get('sketches'),
            'checksums': dataset['train'].get('checksums'),
        }
        if 'children' in dataset['train']:
            train_compressed['children'] = dataset['train']['children'].half()
        for key in dataset['train']:
            if key not in train_compressed:
                train_compressed[key] = dataset['train'][key]
        
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
        
        # Parallel save with ThreadPoolExecutor + LZ4 compression (C++ backend)
        def save_worker(data, path, name):
            try:
                # Use LZ4 compression for 3-5x faster I/O
                try:
                    import lz4.frame
                    import pickle
                    # Serialize with pickle, compress with LZ4 (C++ backend)
                    with lz4.frame.open(path + '.lz4', 'wb') as f:
                        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                    # Rename to .pt for compatibility
                    import os
                    os.rename(path + '.lz4', path)
                except ImportError:
                    # Fallback to standard torch.save
                    torch.save(data, path)
                return name, path, None
            except Exception as e:
                return name, path, str(e)
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_train = executor.submit(save_worker, train_compressed, train_path, "train")
            future_test = executor.submit(save_worker, test_compressed, test_path, "test")
            
            # Wait for both to complete and check errors
            train_name, train_saved, train_error = future_train.result()
            test_name, test_saved, test_error = future_test.result()
            
            if train_error:
                raise RuntimeError(f"Failed to save {train_name}: {train_error}")
            if test_error:
                raise RuntimeError(f"Failed to save {test_name}: {test_error}")
        
        print(f"✅ Saved {train_name} (float16 compressed): {train_saved}")
        print(f"✅ Saved {test_name} (float16 compressed): {test_saved}")
        
        # Save metadata
        metadata_path = os.path.join(output_dir, 'dataset.json')
        with open(metadata_path, 'w') as f:
            json.dump(dataset.get('metadata', {}).__dict__ if hasattr(dataset.get('metadata'), '__dict__') else dataset.get('metadata', {}), f, indent=2)
        print(f"✅ Saved metadata: {metadata_path}")
