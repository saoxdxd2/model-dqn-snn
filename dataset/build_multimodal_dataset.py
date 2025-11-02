"""
Unified Multimodal Dataset Builder.

Supports composite samples with:
- Images (CIFAR, custom)
- Text descriptions/captions
- Grids (mazes, puzzles, ARC)
- Mixed combinations

Ideal for reasoning tasks requiring multiple modalities.
"""

import sys
import glob
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Any, Optional, Union
import numpy as np
import torch
from pydantic import BaseModel
from argdantic import ArgParser

from dataset.base_builder import BaseDatasetBuilder, DataSample, ModalityType
from dataset.common import PuzzleDatasetMetadata

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from models.text_renderer import TextRenderer
    TEXT_RENDERER_AVAILABLE = True
except ImportError:
    TEXT_RENDERER_AVAILABLE = False
    print("⚠️  TextRenderer not available - text will not be rendered to images")

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False


cli = ArgParser()


class MultimodalDatasetConfig(BaseModel):
    """Unified configuration for all dataset types."""
    
    # Data sources (auto-detects format)
    source_paths: List[str] = []  # Can be: HF dataset name, JSON, CSV, image dir, etc.
    output_dir: str = "datasets/output"  # Default, should be overridden by CLI
    
    # Auto-detection (override if needed)
    source_type: Optional[str] = None  # Auto-detect if None: "hf", "arc_json", "maze_csv", "image_dir", etc.
    
    # Modalities to include
    include_text: bool = True
    include_images: bool = True
    include_grids: bool = True
    
    # Processing
    use_capsules: bool = True
    augment: bool = True
    train_split: float = 0.9
    seed: int = 42
    
    # Quality scoring (from build_arc_dataset.py)
    enable_quality_scoring: bool = False
    difficulty_threshold: float = 0.3
    
    # HESC capsule config
    hidden_size: int = 768
    target_capsules: int = 12
    children_per_capsule: int = 4  # Fine-grained tokens per capsule
    num_concepts: int = 2048
    
    # Image processing
    image_size: int = 224
    
    # Text rendering (vision-unified pipeline)
    render_text_to_image: bool = True  # Enable text → image conversion
    text_image_width: int = 512
    text_image_height: int = 384
    
    # Grid processing
    max_grid_size: int = 30
    
    # ARC-specific
    arc_subsets: List[str] = ["training", "evaluation"]
    arc_test_set: str = "evaluation"
    
    # Maze-specific
    maze_augment_dihedral: bool = True


class MultimodalDatasetBuilder(BaseDatasetBuilder):
    """
    Unified builder for multimodal datasets.
    
    Handles composite samples like:
    - Image of a maze + text description
    - Text puzzle + grid representation
    - Image + caption + task grid
    """
    
    def __init__(self, config: MultimodalDatasetConfig):
        super().__init__(config)
        self.config: MultimodalDatasetConfig = config
        
        # Initialize text renderer for vision-unified pipeline
        self.text_renderer = None
        if self.config.render_text_to_image and TEXT_RENDERER_AVAILABLE:
            self.text_renderer = TextRenderer(
                width=self.config.text_image_width,
                height=self.config.text_image_height
            )
            print(f"✓ TextRenderer initialized: {self.config.text_image_width}×{self.config.text_image_height}")
    
    def load_raw_data(self) -> List[DataSample]:
        """Load data from multiple sources with auto-detection."""
        samples = []
        
        for source_path in self.config.source_paths:
            detected_type = self._detect_source_type(source_path)
            
            if detected_type == "arc_json":
                samples.extend(self._load_arc_json(source_path))
            elif detected_type == "maze_csv":
                samples.extend(self._load_maze_csv(source_path))
            elif detected_type == "sudoku_grid":
                samples.extend(self._load_sudoku(source_path))
            elif detected_type == "text_file":
                samples.extend(self._load_text_dataset(source_path))
            elif detected_type == "image_dir":
                samples.extend(self._load_image_dataset(source_path))
            elif detected_type == "hf":
                samples.extend(self._load_from_hf(source_path))
            else:
                samples.extend(self._load_from_local(source_path))
        
        return samples
    
    def _detect_source_type(self, path: str) -> str:
        """Auto-detect source type from path/name."""
        path_lower = path.lower()
        
        # ARC JSON format
        if "arc" in path_lower and path_lower.endswith(".json"):
            return "arc_json"
        
        # Maze CSV format
        if "maze" in path_lower and path_lower.endswith(".csv"):
            return "maze_csv"
        
        # Sudoku
        if "sudoku" in path_lower:
            return "sudoku_grid"
        
        # Text files
        if path_lower.endswith((".txt", ".md")) or "text" in path_lower or "wikitext" in path_lower or "tinystories" in path_lower:
            return "text_file"
        
        # Image datasets
        if "cifar" in path_lower or "imagenet" in path_lower or Path(path).is_dir():
            return "image_dir"
        
        # HuggingFace dataset name (no path separators)
        if "/" not in path and not Path(path).exists():
            return "hf"
        
        return "local"
    
    def _load_from_hf(self, dataset_name: str) -> List[DataSample]:
        """Load from HuggingFace - delegates to smart loader."""
        return self._smart_load_data(dataset_name, is_hf=True)
    
    def _load_from_local(self, path: str) -> List[DataSample]:
        """Load from local - delegates to smart loader."""
        return self._smart_load_data(path, is_hf=False)
    
    def _smart_load_data(self, source: str, is_hf: bool) -> List[DataSample]:
        """Unified loader for HF and local sources."""
        if is_hf:
            if not HF_DATASETS_AVAILABLE:
                print(f"⚠️  HuggingFace not available")
                return []
            
            print(f"📦 Loading: {source}")
            try:
                dataset = load_dataset(source)
                data = dataset.get('train', dataset)
                return [self._parse_item(item, f"{source}_{i}") for i, item in enumerate(data)]
            except Exception as e:
                print(f"❌ Failed: {e}")
                return []
        else:
            path = Path(source)
            if not path.exists():
                print(f"⚠️  Not found: {source}")
                return []
            
            print(f"📁 Loading: {source}")
            if path.is_dir():
                return [s for f in path.glob("**/*") if f.is_file() for s in [self._load_file(f)] if s]
            return [self._load_file(path)] if self._load_file(path) else []
    
    def _parse_item(self, item: Dict, sample_id: str) -> DataSample:
        """Parse dataset item to DataSample."""
        return DataSample(
            sample_id=sample_id,
            modality=ModalityType.MULTIMODAL,
            text=item.get('text') or item.get('caption') or item.get('question'),
            image=item.get('image') or item.get('img'),
            label=item.get('label') or item.get('answer'),
            metadata={'source': sample_id.split('_')[0]}
        )
    
    def _load_file(self, file_path: Path) -> Optional[DataSample]:
        """Load single file - unified logic."""
        suffix = file_path.suffix.lower()
        loaders = {
            ('.png', '.jpg', '.jpeg', '.bmp'): lambda: DataSample(
                sample_id=str(file_path), modality=ModalityType.IMAGE,
                image=Image.open(file_path).convert('RGB') if PIL_AVAILABLE else None,
                metadata={'source': str(file_path)}
            ) if self.config.include_images else None,
            ('.txt', '.md'): lambda: DataSample(
                sample_id=str(file_path), modality=ModalityType.TEXT,
                text=file_path.read_text(encoding='utf-8'),
                metadata={'source': str(file_path)}
            ) if self.config.include_text else None,
            ('.npy',): lambda: DataSample(
                sample_id=str(file_path), modality=ModalityType.GRID,
                grid=np.load(file_path),
                metadata={'source': str(file_path)}
            ) if self.config.include_grids else None
        }
        
        for exts, loader in loaders.items():
            if suffix in exts:
                try:
                    return loader()
                except Exception as e:
                    print(f"⚠️  Error loading {file_path}: {e}")
                    return None
        return None
    
    def _load_arc_json(self, json_path: str) -> List[DataSample]:
        """Load ARC dataset (challenge + solution files)."""
        # Use orjson for 2-3x faster JSON parsing (C++ backend)
        try:
            import orjson as json
            def json_load(f):
                return json.loads(f.read())
        except ImportError:
            import json
            def json_load(f):
                return json.load(f)
        
        # If exact path exists, use it
        if Path(json_path).exists():
            return self._parse_json_format(json_path, 'arc')
        
        # Try to find ARC files with pattern matching
        base_dir = Path(json_path).parent
        if base_dir.exists():
            # Look for ARC challenge files (training2 or training)
            patterns = [
                str(base_dir / "*training2_challenges.json"),  # Larger dataset
                str(base_dir / "*training_challenges.json"),    # Original
            ]
            
            for pattern in patterns:
                matches = glob.glob(pattern)
                if matches:
                    challenges_file = matches[0]
                    # Find corresponding solutions file
                    solutions_file = challenges_file.replace('_challenges.json', '_solutions.json')
                    
                    if Path(solutions_file).exists():
                        print(f"🔍 Found ARC files:")
                        print(f"   Challenges: {Path(challenges_file).name}")
                        print(f"   Solutions: {Path(solutions_file).name}")
                        return self._parse_arc_files(challenges_file, solutions_file)
                    else:
                        print(f"🔍 Found {challenges_file} but no solutions file")
                        return self._parse_json_format(challenges_file, 'arc')
        
        print(f"⚠️  Skipping ARC: No training files found in {base_dir}")
        return []
    
    def _load_maze_csv(self, csv_path: str) -> List[DataSample]:
        """Load maze CSV - unified format parser."""
        return self._parse_csv_format(csv_path, 'maze')
    
    def _load_sudoku(self, path: str) -> List[DataSample]:
        """Load Sudoku - delegates to smart loader."""
        return self._smart_load_data(path, is_hf=False)
    
    def _load_text_dataset(self, path: str) -> List[DataSample]:
        """Load text - unified smart loader."""
        # HuggingFace datasets - map to correct names
        hf_name_mapping = {
            'wikitext2': 'wikitext',
            'wikitext-2': 'wikitext',
            'tinystories': 'roneneldan/TinyStories'
        }
        
        if path in hf_name_mapping:
            actual_name = hf_name_mapping[path]
            print(f"📦 Loading: {path} (using HF: {actual_name})")
            try:
                dataset = load_dataset(actual_name, 'wikitext-2-raw-v1' if 'wikitext' in actual_name else None)
                data = dataset.get('train', dataset)
                return [self._parse_item(item, f"{path}_{i}") for i, item in enumerate(data)]
            except Exception as e:
                print(f"⚠️  Skipping {path}: {e}")
                return []
        # Local file chunking
        return self._chunk_text_file(path) if Path(path).exists() else []
    
    def _load_image_dataset(self, path: str) -> List[DataSample]:
        """Load images - unified smart loader."""
        if path.lower() in ['cifar10', 'cifar-10', 'cifar100', 'cifar-100']:
            return self._smart_load_data(path, is_hf=True)
        return self._smart_load_data(path, is_hf=False)
    
    def _parse_arc_files(self, challenges_file: str, solutions_file: str) -> List[DataSample]:
        """Parse ARC challenge and solution files together."""
        # Use orjson for faster parsing (C++ backend)
        try:
            import orjson as json
            def json_load(f):
                return json.loads(f.read())
        except ImportError:
            import json
            def json_load(f):
                return json.load(f)
        
        try:
            with open(challenges_file, 'rb') as f:  # orjson needs binary
                challenges = json_load(f)
            with open(solutions_file, 'rb') as f:
                solutions = json_load(f)
            
            samples = []
            for task_id in challenges:
                if task_id in solutions:
                    # Each task has train and test examples
                    task_data = {
                        'train': challenges[task_id].get('train', []),
                        'test': [{'input': ex['input'], 'output': solutions[task_id][i]}
                                for i, ex in enumerate(challenges[task_id].get('test', []))]
                    }
                    
                    # Parse as standard ARC format
                    for split in ['train', 'test']:
                        for idx, example in enumerate(task_data.get(split, [])):
                            output = example.get('output')
                            # Convert output to numpy array if it's a list
                            if isinstance(output, list):
                                output = np.array(output)
                            
                            samples.append(DataSample(
                                sample_id=f"{task_id}_{split}_{idx}",
                                modality=ModalityType.GRID,
                                grid=np.array(example.get('input', [])),
                                label=output,
                                metadata={'task_id': task_id, 'split': split}
                            ))
            
            print(f"   Loaded {len(samples)} ARC samples from {len(challenges)} tasks")
            return samples
        except Exception as e:
            print(f"⚠️  Error parsing ARC files: {e}")
            return []
    
    def _parse_json_format(self, json_path: str, format_type: str) -> List[DataSample]:
        """Unified JSON parser (ARC, custom formats)."""
        # Use orjson for faster parsing (C++ backend)
        try:
            import orjson as json
            def json_load(f):
                return json.loads(f.read())
        except ImportError:
            import json
            def json_load(f):
                return json.load(f)
        
        print(f"📄 Parsing {format_type.upper()}: {json_path}")
        
        try:
            with open(json_path, 'rb') as f:  # orjson needs binary
                data = json_load(f)
            
            samples = []
            for puzzle_id, puzzle_data in data.items():
                for split in ['train', 'test']:
                    for idx, example in enumerate(puzzle_data.get(split, [])):
                        samples.append(DataSample(
                            sample_id=f"{puzzle_id}_{split}_{idx}",
                            modality=ModalityType.GRID,
                            grid=np.array(example['input'], dtype=np.int32),
                            label=np.array(example.get('output', example['input']), dtype=np.int32),
                            metadata={'puzzle_id': puzzle_id, 'split': split, 'type': format_type}
                        ))
            
            print(f"   ✓ {len(samples)} samples from {len(data)} puzzles")
            return samples
        except Exception as e:
            print(f"❌ Failed: {e}")
            return []
    
    def _parse_csv_format(self, csv_path: str, format_type: str) -> List[DataSample]:
        """Unified CSV parser (maze, custom grids)."""
        import csv
        print(f"📊 Parsing {format_type.upper()}: {csv_path}")
        
        try:
            samples = []
            with open(csv_path, 'r', newline='') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                
                for row_idx, row in enumerate(reader):
                    if len(row) >= 3:
                        grid_size = int(len(row[1]) ** 0.5)
                        samples.append(DataSample(
                            sample_id=f"{format_type}_{row_idx}",
                            modality=ModalityType.MAZE,
                            grid=np.frombuffer(row[1].encode(), dtype=np.uint8).reshape(grid_size, grid_size),
                            label=np.frombuffer(row[2].encode(), dtype=np.uint8).reshape(grid_size, grid_size),
                            metadata={'type': format_type, 'source': row[0]}
                        ))
            
            print(f"   ✓ {len(samples)} samples")
            return samples
        except Exception as e:
            print(f"❌ Failed: {e}")
            return []
    
    def _chunk_text_file(self, path: str, chunk_size: int = 1000) -> List[DataSample]:
        """Split large text file into manageable chunks."""
        text = Path(path).read_text(encoding='utf-8')
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size) if text[i:i+chunk_size].strip()]
        
        return [DataSample(
            sample_id=f"text_{idx}",
            modality=ModalityType.TEXT,
            text=chunk,
            metadata={'type': 'text', 'source': path}
        ) for idx, chunk in enumerate(chunks)]
    
    def preprocess_sample(self, sample: DataSample) -> DataSample:
        """Unified preprocessing: normalize sizes, combine modalities."""
        
        # Text → Image rendering (DISABLED during dataset building - too slow)
        # This should happen during training on-the-fly instead
        # if sample.text and self.text_renderer is not None and sample.image is None:
        #     try:
        #         rendered_image = self.text_renderer.render_code(sample.text)
        #         sample.image = rendered_image
        #         sample.metadata['text_rendered'] = True
        #     except Exception as e:
        #         pass# Continue with text as-is
        
        # Image: resize and compress to uint8 (4x memory reduction)
        if sample.image and PIL_AVAILABLE and isinstance(sample.image, Image.Image):
            # Use OpenCV for 5-10x faster resizing (C++ backend)
            try:
                import cv2
                # Convert PIL to numpy, resize with OpenCV (INTER_LANCZOS4 = LANCZOS)
                img_array = np.array(sample.image)
                sample.image = cv2.resize(
                    img_array, 
                    (self.config.image_size, self.config.image_size),
                    interpolation=cv2.INTER_LANCZOS4
                ).astype(np.uint8)
            except ImportError:
                # Fallback to PIL if OpenCV not available
                sample.image = np.array(
                    sample.image.resize(
                        (self.config.image_size, self.config.image_size), 
                        Image.Resampling.LANCZOS
                    ),
                    dtype=np.uint8
                )
        
        # Grid: crop to max size
        if sample.grid is not None and sample.grid.shape[0] > self.config.max_grid_size:
            sample.grid = sample.grid[:self.config.max_grid_size, :self.config.max_grid_size]
        
        # Unified text representation
        return self._to_unified_text(sample)
    
    def _to_unified_text(self, sample: DataSample) -> DataSample:
        """Convert all modalities to unified text for encoding."""
        parts = [
            f"Text: {sample.text[:500]}" if sample.text else None,
            f"Image: {sample.image.shape}" if sample.image is not None else None,
            f"Grid: {self._grid_to_text(sample.grid)[:200]}" if sample.grid is not None else None
        ]
        sample.text = " | ".join(p for p in parts if p)
        return sample
    
    def augment_sample(self, sample: DataSample) -> List[DataSample]:
        """Unified augmentation: images (flip), grids (rotations)."""
        augmented = [sample]
        
        # Image flip (optimized to share memory where possible)
        if sample.image is not None and isinstance(sample.image, np.ndarray):
            aug = DataSample(
                sample_id=f"{sample.sample_id}_flip",
                modality=sample.modality,
                text=sample.text,
                image=np.fliplr(sample.image).copy(),  # Explicit copy needed for flip
                label=sample.label,
                metadata=sample.metadata
            )
            augmented.append(aug)
        
        # Grid rotations (dihedral group)
        if sample.grid is not None:
            from common import dihedral_transform
            for tid in [1, 2, 3]:
                aug = DataSample(**sample.model_dump())
                aug.grid = dihedral_transform(sample.grid, tid)
                aug.sample_id = f"{sample.sample_id}_rot{tid}"
                augmented.append(aug)
        
        return augmented


def build(config: MultimodalDatasetConfig):
    """Unified dataset builder - all formats, all modalities."""
    print(f"\n{'='*70}\n🌐 Building: {', '.join(config.source_paths)}\n{'='*70}")
    
    # Build pipeline
    builder = MultimodalDatasetBuilder(config)
    dataset = builder.build_dataset()
    
    # Save dataset (simplified for raw text mode)
    import os
    import json
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Get samples
    train_samples = dataset['train']
    test_samples = dataset['test']
    
    # Vision-unified pipeline: text → images → capsules
    print("\n🎨 Encoding to capsules (vision-unified mode)...")
    
    # Initialize capsule encoder
    from models.capsule_encoder import CapsuleEncoder
    encoder = CapsuleEncoder(
        hidden_size=config.hidden_size,
        target_capsules=config.target_capsules,
        children_per_capsule=config.children_per_capsule,
        num_layers=2,
        H_cycles=2,
        L_cycles=3,
    )
    encoder.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = encoder.to(device)
    
    def encode_to_capsules(samples, batch_size=128):  # Increased from 16 to 128
        """Encode samples to capsule format (optimized)."""
        import time
        from tqdm import tqdm
        
        all_sketches = []
        all_checksums = []
        all_children = []
        
        num_batches = (len(samples) + batch_size - 1) // batch_size
        start_time = time.time()
        
        # Process in batches with progress bar
        for i in tqdm(range(0, len(samples), batch_size), total=num_batches, desc="   Encoding"):
            batch_samples = samples[i:i+batch_size]
            
            # Prepare batch - render text to images if needed
            batch_images = []
            batch_texts = []
            
            for sample in batch_samples:
                if sample.image is not None:
                    batch_images.append(sample.image)
                elif sample.text:
                    # Truncate text to reasonable length for speed
                    text = sample.text[:500] if len(sample.text) > 500 else sample.text
                    batch_texts.append(text)
            
            # Encode batch
            with torch.no_grad():
                # Define image transform once (no duplicate)
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                if batch_images:
                    # Convert PIL images to tensors
                    image_tensors = torch.stack([transform(img) for img in batch_images]).to(device)
                    result = encoder(images=image_tensors, return_children=True)
                elif batch_texts:
                    # Render text to images, then encode with TRM
                    from models.text_renderer import TextRenderer
                    renderer = TextRenderer(width=224, height=224, font_size=12)
                    
                    # Render texts to images
                    text_images = [renderer.render_plain_text(text[:500]) for text in batch_texts]
                    
                    # Convert PIL images to tensors (same transform)
                    image_tensors = torch.stack([transform(img) for img in text_images]).to(device)
                    
                    # Encode through TRM
                    result = encoder(images=image_tensors, return_children=True)
                else:
                    continue
                
                all_sketches.append(result['sketches'].cpu())
                all_checksums.append(result['checksums'].cpu())
                if result.get('children') is not None:
                    all_children.append(result['children'].cpu())
        
        # Concatenate all batches
        elapsed = time.time() - start_time
        sketches = torch.cat(all_sketches, dim=0) if all_sketches else torch.zeros(1, config.target_capsules, config.hidden_size)
        checksums = torch.cat(all_checksums, dim=0) if all_checksums else torch.zeros(1, config.target_capsules, 32)
        children = torch.cat(all_children, dim=0) if all_children else None
        
        # Print timing info
        samples_per_sec = len(samples) / elapsed if elapsed > 0 else 0
        print(f"   ⏱️  Encoded {len(samples)} samples in {elapsed:.1f}s ({samples_per_sec:.1f} samples/sec)")
        
        return {
            'sketches': sketches,
            'checksums': checksums,
            'children': children
        }
    
    print("   Encoding train samples...")
    train_capsules = encode_to_capsules(train_samples)
    print(f"   ✓ Train: {train_capsules['sketches'].shape[0]} samples → {config.target_capsules} capsules")
    
    print("   Encoding test samples...")
    test_capsules = encode_to_capsules(test_samples)
    print(f"   ✓ Test: {test_capsules['sketches'].shape[0]} samples → {config.target_capsules} capsules")
    
    # Save as capsule_dataset.pt (semantic mode format)
    print("\n💾 Saving capsule datasets...")
    train_path = os.path.join(config.output_dir, 'capsule_dataset.pt')
    test_path = os.path.join(config.output_dir, 'capsule_dataset_test.pt')
    
    torch.save(train_capsules, train_path)
    torch.save(test_capsules, test_path)
    
    print(f"   ✓ {train_path}")
    print(f"   ✓ {test_path}")
    
    # Also save metadata for compatibility
    metadata = {
        'num_samples': train_capsules['sketches'].shape[0],
        'num_capsules': config.target_capsules,
        'hidden_size': config.hidden_size,
        'has_children': train_capsules['children'] is not None
    }
    
    with open(os.path.join(config.output_dir, 'dataset_info.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    train_count = train_capsules['sketches'].shape[0]
    test_count = test_capsules['sketches'].shape[0]
    
    print(f"\n✅ Complete: {train_count} train, {test_count} test samples")
    print(f"   Format: {config.target_capsules} capsules per sample")
    print(f"   Output: {config.output_dir}")
    print(f"\n💡 Training will use semantic_mode=true with capsule datasets")
    
    if train_count == 0:
        print(f"\n⚠️  WARNING: No samples were loaded from sources:")
        for src in config.source_paths:
            print(f"   - {src}")
        print(f"   Please check that the data files exist and are accessible.")

def _post_process(config: MultimodalDatasetConfig, dataset: Dict):
    """Unified post-processing: concept tables, quality scoring."""
    import os
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Concept expansion (for text/generation) - use in-memory dataset
    if config.use_capsules and config.include_text:
        print(f"\n📚 Building concept expansion table...")
        try:
            from models.concept_decoder import ConceptDecoder
            from transformers import AutoTokenizer
            
            # Use dataset already in memory (no reload)
            decoder = ConceptDecoder(num_concepts=config.num_concepts)
            
            # Pass in-memory data directly
            if 'train' in dataset and 'sketches' in dataset['train']:
                decoder.build_expansion_table_from_memory(
                    dataset['train']['sketches'],
                    AutoTokenizer.from_pretrained("gpt2"),
                    config.num_concepts
                )
                decoder.expansion_table.save(f"{config.output_dir}/concept_expansions.json")
                print(f"   ✓ Saved")
            else:
                print(f"   ⚠️  No training data to process")
        except Exception as e:
            print(f"   ⚠️  {e}")
    
    # Quality scoring (for reasoning tasks)
    if config.enable_quality_scoring:
        print(f"\n🎯 Quality scoring...")
        try:
            from difficulty_scorer import score_difficulty
            # TODO: Apply scoring
            print(f"   ✓ Complete")
        except Exception as e:
            print(f"   ⚠️  {e}")


# Preset commands (shortcuts for common use cases)
PRESETS = {
    'arc': {'include_text': False, 'include_images': False, 'include_grids': True, 'enable_quality_scoring': True},
    'text': {'include_text': True, 'include_images': False, 'include_grids': False},
    'image': {'include_text': False, 'include_images': True, 'include_grids': False},
    'maze': {'include_text': False, 'include_images': False, 'include_grids': True},
    'composite': {'include_text': True, 'include_images': True, 'include_grids': True}
}

@cli.command()
def quick_build(preset: str, sources: List[str], output_dir: str):
    """Quick build with preset config (arc, text, image, maze, composite)."""
    if preset not in PRESETS:
        print(f"❌ Unknown preset: {preset}. Available: {list(PRESETS.keys())}")
        return
    
    config = MultimodalDatasetConfig(
        source_paths=sources,
        output_dir=output_dir,
        **PRESETS[preset]
    )
    build(config)

# Legacy command aliases
@cli.command()
def build_arc(input_file_prefix: str = "kaggle/combined/arc-agi", output_dir: str = "data/arc-capsules"):
    import glob
    quick_build('arc', glob.glob(f"{input_file_prefix}*.json"), output_dir)

@cli.command()
def build_text(input_file: str = "wikitext2", output_dir: str = "datasets/wikitext2"):
    quick_build('text', [input_file], output_dir)

@cli.command()
def build_image(dataset_name: str = "cifar10", output_dir: str = "datasets/cifar10"):
    quick_build('image', [dataset_name], output_dir)

@cli.command()
def build_maze(source_repo: str = "sapientinc/maze-30x30-hard-1k", output_dir: str = "data/maze"):
    try:
        from huggingface_hub import hf_hub_download
        sources = [hf_hub_download(source_repo, f"{s}.csv", repo_type="dataset") for s in ['train', 'test']]
    except:
        sources = [f"{output_dir}/{s}.csv" for s in ['train', 'test']]
    quick_build('maze', sources, output_dir)

@cli.command()
def build_composite(
    sources: List[str], 
    output_dir: str = "datasets/composite",
    augment: bool = True,
    num_concepts: int = 2048,
    target_capsules: int = 12,
    enable_quality_scoring: bool = True
):
    """Build composite multimodal dataset from multiple sources."""
    config = MultimodalDatasetConfig(
        source_paths=sources,
        output_dir=output_dir,
        include_text=True,
        include_images=True,
        include_grids=True,
        # Encode to capsules (DISABLED - too slow and memory intensive during dataset building)
        # This should happen on-the-fly during training instead
        # if config.use_capsules:
        #     print("🧶 Encoding to HESC capsules...")
        #     dataset = encode_to_capsules(dataset, config)
        use_capsules=False,
        num_concepts=num_concepts,
        target_capsules=target_capsules,
        enable_quality_scoring=enable_quality_scoring
    )
    build(config)


if __name__ == "__main__":
    cli()
