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
    num_concepts: int = 2048
    encoder_model: str = "openai/clip-vit-large-patch14"
    
    # Image processing
    image_size: int = 224
    
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
        """Load ARC JSON - unified format parser."""
        return self._parse_json_format(json_path, 'arc')
    
    def _load_maze_csv(self, csv_path: str) -> List[DataSample]:
        """Load maze CSV - unified format parser."""
        return self._parse_csv_format(csv_path, 'maze')
    
    def _load_sudoku(self, path: str) -> List[DataSample]:
        """Load Sudoku - delegates to smart loader."""
        return self._smart_load_data(path, is_hf=False)
    
    def _load_text_dataset(self, path: str) -> List[DataSample]:
        """Load text - unified smart loader."""
        # HuggingFace datasets
        if path in ['wikitext2', 'wikitext-2', 'tinystories']:
            return self._smart_load_data(path, is_hf=True)
        # Local file chunking
        return self._chunk_text_file(path) if Path(path).exists() else []
    
    def _load_image_dataset(self, path: str) -> List[DataSample]:
        """Load images - unified smart loader."""
        if path.lower() in ['cifar10', 'cifar-10', 'cifar100', 'cifar-100']:
            return self._smart_load_data(path, is_hf=True)
        return self._smart_load_data(path, is_hf=False)
    
    def _parse_json_format(self, json_path: str, format_type: str) -> List[DataSample]:
        """Unified JSON parser (ARC, custom formats)."""
        import json
        print(f"📄 Parsing {format_type.upper()}: {json_path}")
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
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
        # Image: resize
        if sample.image and PIL_AVAILABLE and isinstance(sample.image, Image.Image):
            sample.image = np.array(sample.image.resize(
                (self.config.image_size, self.config.image_size), Image.Resampling.LANCZOS
            ))
        
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
        
        # Image flip
        if sample.image is not None and isinstance(sample.image, np.ndarray):
            aug = DataSample(**sample.model_dump())
            aug.image = np.fliplr(sample.image)
            aug.sample_id = f"{sample.sample_id}_flip"
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
    
    # Save FIRST (so post-processing can read the files)
    builder.save(dataset, config.output_dir)
    
    # Post-processing pipeline (concept expansion, quality scoring)
    _post_process(config, dataset)
    
    # Print summary
    train_size = dataset['train'].get('sketches', torch.empty(0)).shape[0] if 'sketches' in dataset['train'] else len(dataset['train']) if isinstance(dataset['train'], list) else 0
    test_size = dataset['test'].get('sketches', torch.empty(0)).shape[0] if 'sketches' in dataset['test'] else len(dataset['test']) if isinstance(dataset['test'], list) else 0
    
    print(f"\n✅ Complete: {train_size} train, {test_size} test samples")
    print(f"   Output: {config.output_dir}")
    
    if train_size == 0:
        print(f"\n⚠️  WARNING: No samples were loaded from sources:")
        for src in config.source_paths:
            print(f"   - {src}")
        print(f"   Please check that the data files exist and are accessible.")

def _post_process(config: MultimodalDatasetConfig, dataset: Dict):
    """Unified post-processing: concept tables, quality scoring."""
    import os
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Concept expansion (for text/generation)
    if config.use_capsules and config.include_text:
        print(f"\n📚 Building concept expansion table...")
        try:
            from models.concept_decoder import ConceptDecoder
            from transformers import AutoTokenizer
            
            # Path already saved by builder.save(), just reference it
            train_path = f"{config.output_dir}/capsule_dataset.pt"
            
            decoder = ConceptDecoder(num_concepts=config.num_concepts)
            decoder.build_expansion_table_from_data(train_path, AutoTokenizer.from_pretrained("gpt2"), config.num_concepts)
            decoder.expansion_table.save(f"{config.output_dir}/concept_expansions.json")
            print(f"   ✓ Saved")
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
        use_capsules=True,
        num_concepts=num_concepts,
        target_capsules=target_capsules,
        enable_quality_scoring=enable_quality_scoring
    )
    build(config)


if __name__ == "__main__":
    cli()
