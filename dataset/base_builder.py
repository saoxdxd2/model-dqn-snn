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
        
        # Preprocessing handled by ImageCache during streaming
        # No need for sequential bottleneck here - just pass through
        print("⚙️  Preparing samples...")
        processed = raw_samples  # ImageCache will handle rendering in parallel
        
        # No augmentation, splitting, or pre-encoding - just return samples
        # TRM will encode images to capsules during training (on-the-fly)
        # Pre-encoding is pure overhead - we encode twice otherwise!
        print(f"   Ready: {len(processed)} samples (images cached)")
        
        result = {
            'train': processed,
            'test': []  # No separate test set for pretraining
        }
        
        result['metadata'] = {'num_samples': len(processed)}
        
        return result
    
    def _sample_is_cached(self, sample, cache):
        """Check if sample is already cached."""
        if hasattr(sample, 'text'):
            text = sample.text
        elif hasattr(sample, 'question'):
            text = sample.question
        elif isinstance(sample, dict):
            text = sample.get('text', '') or sample.get('question', '')
        else:
            text = str(sample)
        
        return cache.get(text, 224, 224) is not None
    
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
    
    def _stream_encode_capsules(self, samples: List[DataSample]):
        """Fast encoding with disk-cached text images (streaming mode available)."""
        if len(samples) == 0:
            return {'sketches': torch.empty(0, getattr(self.config, 'target_capsules', 12), getattr(self.config, 'hidden_size', 768))}
        
        # Initialize encoder once
        self._init_encoder()
        device = next(self.encoder.parameters()).device
        
        from torch.utils.data import Dataset, DataLoader
        from tqdm import tqdm
        import gc
        import os
        
        # Initialize image cache (on Drive so it syncs to local 2TB SSD)
        from dataset.image_cache import ImageCache
        from pathlib import Path
        import os
        
        # Use Drive path if available, otherwise local
        if os.path.exists("/content/drive/MyDrive"):
            cache_dir = "/content/drive/MyDrive/model_checkpoints/text_cache"
            print(f"💾 Using Drive for text cache (will sync to 2TB SSD)")
            print(f"   Cache: {cache_dir}")
        else:
            cache_dir = str(Path(self.config.output_dir) / "text_cache")
            print(f"⚠️  Drive not mounted, using local cache (session-only)")
        
        cache = ImageCache(cache_dir=cache_dir)
        
        # Check if we should use streaming mode
        use_streaming = getattr(self.config, 'streaming_mode', True)  # Default ON
        
        print("🗂️  Checking image cache...")
        
        # Sample across entire dataset to check cache status (not just first 1000)
        import random
        sample_size = min(1000, len(samples))
        sample_indices = random.sample(range(len(samples)), sample_size)
        existing_cached = sum(1 for idx in sample_indices if self._sample_is_cached(samples[idx], cache))
        cache_hit_rate = existing_cached / sample_size
        
        print(f"📊 Cache status: {cache_hit_rate*100:.0f}% ({existing_cached}/{sample_size} samples)")
        
        # Always use streaming for first run to utilize GPU immediately
        if use_streaming and cache_hit_rate < 0.95:
            print(f"🌊 Using STREAMING mode (cache incomplete)")
            print(f"   Strategy: CPU renders + GPU encodes simultaneously")
            if cache_hit_rate * 100 < 5:
                print(f"   GPU starts immediately (no cache found)")
            else:
                print(f"   GPU starts after 50k samples cached")
            
            if not getattr(self.config, 'drive_dir', None):
                print(f"⚠️  Drive not mounted, progress won't be saved to Drive")
            
            from dataset.streaming_builder import StreamingCacheEncoder
            device = next(self.encoder.parameters()).device
            encoder = StreamingCacheEncoder(
                cache=cache,
                encoder=self.encoder,
                device=device,
                batch_size=256,
                checkpoint_dir=str(Path(self.config.output_dir) / "stream_checkpoints"),
                drive_dir=getattr(self.config, 'drive_dir', None)
            )
            result = encoder.stream_build(samples, self.text_renderer, initial_cache_percent=cache_hit_rate * 100)
            
            # Drive path for progress saves (if available)
            drive_dir = "/content/drive/MyDrive/model_checkpoints/encoding_progress"
            if os.path.exists("/content/drive/MyDrive"):
                Path(drive_dir).mkdir(parents=True, exist_ok=True)
            else:
                drive_dir = None
                print("⚠️  Drive not mounted, progress won't be saved to Drive")
            
            streamer = StreamingCacheEncoder(cache, self.encoder, device, batch_size=256, 
                                            checkpoint_dir=checkpoint_dir, drive_dir=drive_dir)
            return streamer.stream_build(samples, self.text_renderer, start_threshold=50000)
        else:
            print(f"💾 Using STANDARD mode (cache complete)")
            cached_count, rendered_count = cache.populate_cache(samples, self.text_renderer)
        
            if rendered_count > 0:
                print(f"✅ Rendered {rendered_count} new images (one-time cost)")
            print(f"✅ Using {cached_count + rendered_count} cached images (10x faster than re-rendering)")
        
        # Force garbage collection before starting
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Custom collate function to handle numpy arrays -> PyTorch tensors (CHW format)
        def collate_images(batch):
            """Convert batch of numpy images [H,W,C] to tensor [B,C,H,W]."""
            batch_array = np.stack(batch, axis=0)  # [B, H, W, C]
            batch_tensor = torch.from_numpy(batch_array).permute(0, 3, 1, 2)  # [B, C, H, W]
            return batch_tensor.float() / 255.0  # Normalize to [0, 1]
        
        # Custom Dataset with cache support
        class SampleDataset(Dataset):
            def __init__(self, samples, image_converter, cache):
                self.samples = samples
                self.image_converter = image_converter
                self.cache = cache
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                sample = self.samples[idx]
                
                # Handle both dict and Pydantic DataSample objects
                if hasattr(sample, 'text'):
                    text = sample.text
                elif hasattr(sample, 'question'):
                    text = sample.question
                elif isinstance(sample, dict):
                    text = sample.get('text', '') or sample.get('question', '') or str(sample)
                else:
                    text = str(sample)
                
                # Try cache first (instant)
                cached_img = self.cache.get(text, 224, 224)
                if cached_img is not None:
                    return cached_img
                
                # Fallback to rendering (shouldn't happen after pre-population)
                return self.image_converter(sample)
        
        # Create dataset and dataloader with cache
        dataset = SampleDataset(samples, self._sample_to_image, cache)
        
        # OPTIMIZED: Parallel disk I/O with workers (safe with cached .npy files)
        # - Workers now just load .npy files (fast, no memory leak)
        # - Prefetching overlaps disk I/O with GPU encoding
        # - Larger batch maxes out GPU
        batch_size = 144  # Max VRAM (model 5GB + batch 9GB + buffer 1GB = 15GB)
        num_workers = 4  # Parallel disk I/O (4 workers prefetch while GPU encodes)
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_images,
            pin_memory=True,  # Fast CPU->GPU transfer
            prefetch_factor=3,  # Each worker prefetches 3 batches (12 total ready)
            persistent_workers=True  # Keep workers alive between epochs
        )
        
        print(f"🚀 OPTIMIZED Pipeline (parallel disk I/O):")
        print(f"   Batch size: {batch_size} | Workers: {num_workers} | Prefetch: 3")
        print(f"   Strategy: 4 workers prefetch .npy files while GPU encodes")
        print(f"   Disk I/O: Fully parallelized (no GPU idle time)")
        print(f"   Memory: Consolidate every 400 batches (~1 min)")
        print(f"   Checkpoints: Every 2000 batches (~5 min)")
        print(f"   Speed: ~10 batches/sec (0.1s/batch, 144 samples/batch = ~1440 samples/sec)")
        print(f"   Total ETA: ~22 minutes for 1.9M samples! (1 session)")
        
        # Storage for results
        result_chunks = {'sketches': [], 'checksums': [], 'children': []}
        temp_gpu_list = {'sketches': [], 'checksums': [], 'children': []}
        
        consolidate_every = 400  # Consolidate every 400 batches (~40 sec, less overhead)
        checkpoint_every = 2000  # Checkpoint every 2000 batches (~3min at 0.1s/batch)
        batch_count = 0
        
        # Memory management: track and limit result_chunks size
        max_cpu_chunks = 3  # Keep max 3 chunks in CPU RAM before saving intermediate
        
        # Check for existing checkpoint to resume
        from pathlib import Path
        checkpoint_path = Path(self.config.output_dir) / "encoding_checkpoint.pt"
        start_batch = 0
        
        if checkpoint_path.exists():
            print(f"📂 Found checkpoint, attempting resume...")
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                result_chunks = checkpoint['result_chunks']
                start_batch = checkpoint['batch_count']
                total_batches = len(dataloader)
                print(f"✅ Resumed from batch {start_batch}/{total_batches} ({start_batch*batch_size}/{len(samples)} samples)")
            except Exception as e:
                print(f"⚠️  Checkpoint load failed: {e}, starting fresh")
        
        # Hybrid CPU/GPU processing
        with torch.no_grad():
            for batch_idx, batch_images in enumerate(tqdm(dataloader, desc="Encoding", initial=start_batch)):
                # Skip already processed batches
                if batch_idx < start_batch:
                    continue
                
                # Move batch to GPU (workers already rendered on CPU)
                batch_images = batch_images.to(device, non_blocking=True)
                
                # Encode on GPU
                batch_result = self.encoder(images=batch_images, return_children=True)
                
                # Accumulate on GPU
                for key in temp_gpu_list:
                    if key in batch_result and batch_result[key] is not None:
                        temp_gpu_list[key].append(batch_result[key])
                
                del batch_images, batch_result
                batch_count += 1
                
                # Periodic light cleanup (every 100 batches, less overhead)
                if batch_idx % 100 == 0:
                    gc.collect()
                
                # Periodic consolidation (GPU -> CPU to free VRAM)
                if batch_count >= consolidate_every:
                    # Consolidate GPU tensors to CPU
                    for key in temp_gpu_list:
                        if temp_gpu_list[key]:
                            consolidated = torch.cat(temp_gpu_list[key], dim=0)
                            result_chunks[key].append(consolidated.half().cpu())
                            temp_gpu_list[key].clear()
                            del consolidated  # Explicit cleanup
                    
                    batch_count = 0
                    
                    # Aggressive memory cleanup
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # If CPU RAM chunks getting too large, consolidate them too
                    for key in result_chunks:
                        if len(result_chunks[key]) > max_cpu_chunks:
                            # Merge CPU chunks to free fragmented memory
                            merged = torch.cat(result_chunks[key], dim=0)
                            result_chunks[key] = [merged]
                            gc.collect()
                    
                    # Save checkpoint
                    if batch_idx > 0 and batch_idx % checkpoint_every == 0:
                        # Force memory cleanup before checkpoint
                        gc.collect()
                        
                        print(f"\n💾 Checkpoint at batch {batch_idx}/{len(dataloader)}...")
                        torch.save({
                            'result_chunks': result_chunks,
                            'batch_count': batch_idx,
                            'total_batches': len(dataloader)
                        }, checkpoint_path)
                        print(f"✅ Saved")
                        
                        # Sync to Drive
                        try:
                            import sys
                            if 'google.colab' in sys.modules:
                                drive_dir = Path("/content/drive/MyDrive/model-dqn-snn")
                                if drive_dir.exists():
                                    import shutil
                                    shutil.copy2(checkpoint_path, drive_dir / checkpoint_path.name)
                                    print(f"☁️  Synced to Drive")
                        except:
                            pass
                        
                        # Aggressive cleanup after checkpoint
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
        
        # Final consolidation of remaining batches
        for key in temp_gpu_list:
            if temp_gpu_list[key]:
                consolidated = torch.cat(temp_gpu_list[key], dim=0)
                result_chunks[key].append(consolidated.half().cpu())
                del consolidated
        
        del temp_gpu_list
        gc.collect()
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
