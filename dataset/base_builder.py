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
    Abstract base class for vision-unified dataset builders.
    
    Pipeline:
    1. Load raw data (text/images/grids)
    2. Preprocess (render text to images if needed)
    3. Augment (rotations, flips, etc.)
    4. Encode via TRM vision encoder → capsules
    5. Save capsule dataset
    
    All data goes through TRM encoder - no separate text/vision paths.
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
        
        # Note: Parallel processing disabled for text rendering (PIL font pickling issues)
        # Process sequentially with progress indicator
        from tqdm import tqdm
        processed = [self.preprocess_sample(s) for s in tqdm(raw_samples, desc="   Processing", ncols=70)]
        
        del raw_samples  # Free original
        
        # Augment with streaming to prevent RAM explosion
        print("🔄 Augmenting...")
        if getattr(self.config, 'augment', False):
            # OPTIMIZATION: Skip augmentation for text-only datasets
            # Check first sample to determine dataset type
            has_visual_data = False
            if len(processed) > 0:
                first_sample = processed[0]
                has_visual_data = (first_sample.image is not None or first_sample.grid is not None)
            
            if not has_visual_data:
                print("   Skipping augmentation (text-only dataset)")
                augmented = processed
            else:
                augmented = []
                batch_size = 5000  # Process in batches to free memory
                
                from tqdm import tqdm
                for i in tqdm(range(0, len(processed), batch_size), desc="   Augmenting", ncols=70):
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
        
        # Encode to capsules (vision-unified pipeline - always uses TRM encoder)
        print("🧶 Encoding to HESC capsules (TRM vision encoder)...")
        import gc
        
        # Train encoding (batch processing with DataLoader)
        print(f"   Encoding {len(train_data)} train samples...")
        train_encoded = self._stream_encode_capsules(train_data)
        del train_data  # Free after encoding
        
        # Test encoding (batch processing with DataLoader)
        print(f"   Encoding {len(test_data)} test samples...")
        test_encoded = self._stream_encode_capsules(test_data)
        del test_data  # Free after encoding
        
        return {
            'train': train_encoded,
            'test': test_encoded,
            'metadata': self.create_metadata(train_encoded)
        }
    
    def _init_encoder(self):
        """Initialize CapsuleEncoder (shared by all encoding methods)."""
        if self.encoder is None:
            from models.capsule_encoder import CapsuleEncoder
            self.encoder = CapsuleEncoder(
                hidden_size=getattr(self.config, 'hidden_size', 768),
                target_capsules=getattr(self.config, 'target_capsules', 12),
                children_per_capsule=4,
                checksum_dim=32,
                num_layers=getattr(self.config, 'encoder_num_layers', 2),
                H_cycles=getattr(self.config, 'encoder_H_cycles', 2),
                L_cycles=getattr(self.config, 'encoder_L_cycles', 3),
                capsule_grid_shape=getattr(self.config, 'capsule_grid_shape', (3, 4))
            )
            self.encoder.eval()
            self.encoder.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _stream_encode_capsules(self, samples: List[DataSample]) -> Dict[str, torch.Tensor]:
        """High-performance encoding with 10GB RAM budget using PyTorch DataLoader."""
        if len(samples) == 0:
            return {'sketches': torch.empty(0, getattr(self.config, 'target_capsules', 12), getattr(self.config, 'hidden_size', 768))}
        
        # Initialize encoder once
        self._init_encoder()
        
        from torch.utils.data import Dataset, DataLoader
        from tqdm import tqdm
        
        # Custom collate function to handle numpy arrays -> PyTorch tensors (CHW format)
        def collate_images(batch):
            """Convert batch of numpy images [H,W,C] to tensor [B,C,H,W]."""
            import torch
            # Stack numpy arrays and convert to tensor
            batch_array = np.stack(batch, axis=0)  # [B, H, W, C]
            # Convert to tensor and transpose to CHW format
            batch_tensor = torch.from_numpy(batch_array).permute(0, 3, 1, 2)  # [B, C, H, W]
            return batch_tensor.float() / 255.0  # Normalize to [0, 1]
        
        # Custom Dataset for efficient batching
        class SampleDataset(Dataset):
            def __init__(self, samples, image_converter):
                self.samples = samples
                self.image_converter = image_converter
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                return self.image_converter(self.samples[idx])
        
        # Create dataset and dataloader with optimized settings
        dataset = SampleDataset(samples, self._sample_to_image)
        
        # Large batch size for GPU efficiency (10GB RAM budget)
        batch_size = 48  # Increased from 32 (use more VRAM: 10.5GB / 15GB)
        num_workers = 2  # Reduced from 4 to save 1-2GB RAM
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_images,  # Custom collate for numpy -> tensor conversion
            pin_memory=torch.cuda.is_available(),  # Faster GPU transfer
            prefetch_factor=2,  # Prefetch batches
            persistent_workers=True  # Keep workers alive
        )
        
        # Storage for results (on GPU to prevent CPU RAM explosion)
        result_chunks = {'sketches': [], 'checksums': [], 'children': []}
        temp_gpu_list = {'sketches': [], 'checksums': [], 'children': []}
        device = next(self.encoder.parameters()).device
        
        consolidate_every = 800  # Consolidate every 800 batches (reduced overhead)
        checkpoint_every = 2000  # Save checkpoint every 2000 batches (~100k samples)
        batch_count = 0
        
        # Check for existing checkpoint to resume
        checkpoint_path = self.config.output_dir / "encoding_checkpoint.pt"
        start_batch = 0
        if checkpoint_path.exists():
            print(f"📂 Found checkpoint, attempting resume...")
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                result_chunks = checkpoint['result_chunks']
                start_batch = checkpoint['batch_count']
                print(f"✅ Resumed from batch {start_batch}/{len(dataloader)}")
            except Exception as e:
                print(f"⚠️  Checkpoint load failed: {e}, starting fresh")
        
        # Process with DataLoader (automatic batching + parallel preprocessing)
        with torch.no_grad():
            for batch_idx, batch_images in enumerate(tqdm(dataloader, desc="Encoding", initial=start_batch)):
                # Skip already processed batches
                if batch_idx < start_batch:
                    continue
                
                # Move batch to GPU device
                batch_images = batch_images.to(device)
                
                # Encode batch (stays on GPU) - pass as keyword argument
                batch_result = self.encoder(images=batch_images, return_children=True)
                
                # Keep on GPU and accumulate (uses VRAM, not CPU RAM)
                for key in temp_gpu_list:
                    if key in batch_result and batch_result[key] is not None:
                        temp_gpu_list[key].append(batch_result[key])  # Stay on GPU
                
                del batch_result
                batch_count += 1
                
                # Periodic consolidation to prevent VRAM overflow
                if batch_count >= consolidate_every:
                    # Consolidate GPU tensors
                    for key in temp_gpu_list:
                        if temp_gpu_list[key]:
                            consolidated = torch.cat(temp_gpu_list[key], dim=0)
                            # Convert to fp16 immediately (2x memory reduction)
                            result_chunks[key].append(consolidated.half().cpu())
                            temp_gpu_list[key].clear()
                    
                    batch_count = 0
                    torch.cuda.empty_cache()
                    
                    # Save checkpoint periodically (every 2000 batches)
                    if batch_idx > 0 and batch_idx % checkpoint_every == 0:
                        print(f"\n💾 Saving checkpoint at batch {batch_idx}...")
                        torch.save({
                            'result_chunks': result_chunks,
                            'batch_count': batch_idx,
                            'total_batches': len(dataloader)
                        }, checkpoint_path)
                        print(f"✅ Checkpoint saved to {checkpoint_path}")
                        
                        # Auto-sync to Google Drive if in Colab
                        try:
                            import sys
                            if 'google.colab' in sys.modules:
                                from pathlib import Path
                                drive_dir = Path("/content/drive/MyDrive/model-dqn-snn")
                                if drive_dir.exists():
                                    import shutil
                                    shutil.copy2(checkpoint_path, drive_dir / checkpoint_path.name)
                                    print(f"☁️  Synced to Google Drive")
                        except:
                            pass
                        
                        torch.cuda.empty_cache()
        
        # Final consolidation of remaining batches
        for key in temp_gpu_list:
            if temp_gpu_list[key]:
                consolidated = torch.cat(temp_gpu_list[key], dim=0)
                # Convert to fp16 immediately (2x memory reduction)
                result_chunks[key].append(consolidated.half().cpu())
        
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
        # Initialize encoder once (shared method)
        self._init_encoder()
        
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
            # Convert only current batch to images (critical for RAM)
            batch_images = [self._sample_to_image(s) for s in samples[i:i+batch_size]]
            
            with torch.no_grad():
                batch_result = self.encoder(images=batch_images, return_children=True)
            
            # Clear batch images immediately
            del batch_images
            
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
    
    def _sample_to_image(self, sample: DataSample):
        """Convert any sample type to numpy array image for vision-unified encoder."""
        # Import TextRenderer lazily
        if not hasattr(self, 'text_renderer'):
            from models.text_renderer import TextRenderer
            self.text_renderer = TextRenderer(
                width=getattr(self.config, 'text_image_width', 224),
                height=getattr(self.config, 'text_image_height', 224)
            )
        
        # Text → render to image → convert to numpy
        if sample.text:
            pil_img = self.text_renderer.render_plain_text(sample.text)
            return np.array(pil_img)
        
        # Grid → render to image → convert to numpy
        elif sample.grid is not None:
            grid_text = self._grid_to_text(sample.grid)
            pil_img = self.text_renderer.render_plain_text(grid_text)
            return np.array(pil_img)
        
        # Image → return as numpy array
        elif sample.image is not None:
            # Already numpy array
            if isinstance(sample.image, np.ndarray):
                return sample.image.astype(np.uint8)
            # Convert PIL to numpy
            return np.array(sample.image)
        
        # Fallback: blank image as numpy array
        return np.ones((224, 224, 3), dtype=np.uint8) * 255
    
    # Legacy token encoding removed - vision-unified only uses capsules
    
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
        
        if isinstance(train_data, dict):
            if 'sketches' in train_data:
                seq_len = train_data['sketches'].shape[1]
            else:
                seq_len = 1024  # Default
        else:
            seq_len = 1024  # Default
        
        if isinstance(train_data, dict):
            total_samples = len(train_data.get('sketches', []))
        else:
            total_samples = len(train_data) if train_data else 0
        
        return PuzzleDatasetMetadata(
            seq_len=seq_len,
            vocab_size=vocab_size,
            pad_id=0,
            ignore_label_id=-100,
            blank_identifier_id=0,
            num_puzzle_identifiers=0,
            total_groups=total_samples,
            mean_puzzle_examples=1.0,
            total_puzzles=total_samples,
            sets=["all"]
        )

# ... (rest of the code remains the same)
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
