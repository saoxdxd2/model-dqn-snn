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
    train_split: float = 0.9
    seed: int = 42
    
    # Quality scoring removed - adds overhead without implemented benefit
    
    # HESC capsule config
    hidden_size: int = 768
    target_capsules: int = 12
    children_per_capsule: int = 4  # Fine-grained tokens per capsule
    num_concepts: int = 2048
    
    # Image processing
    image_size: int = 224
    
    # Text rendering (vision-unified pipeline)
    render_text_to_image: bool = True  # Enable text → image conversion
    text_image_width: int = 224  # Must match TRM vision encoder image_size
    text_image_height: int = 224  # Must match TRM vision encoder image_size
    
    # Grid processing
    max_grid_size: int = 30
    
    # ARC-specific
    arc_subsets: List[str] = ["training", "evaluation"]
    arc_test_set: str = "evaluation"
    
    # Re-ARC infinite synthetic data (Winner Strategy)
    use_rearc_infinite: bool = True  # Infinite synthetic ARC tasks
    rearc_path: str = "dataset/re-arc"  # Path to re-arc repo
    rearc_examples_per_task: int = 100  # Generate 100 examples per task
    
    # Problem Augmentation (Winner Strategy - Creates 10-20x more tasks)
    enable_problem_augmentation: bool = True  # Transform inputs/outputs separately
    problem_aug_prob: float = 0.5  # 50% of samples get problem augmentation
    
    # Multi-Task Training (Winner Strategy - Omni-ARC approach)
    enable_multitask: bool = True  # Train on 2 tasks: solve + learn distribution
    
    # Maze-specific
    maze_augment_dihedral: bool = True
    cache_images: bool = True


@cli.command()
def build_dataset(config: MultimodalDatasetConfig):
    """Build multimodal dataset from source files."""
    build(config)


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
    
    def load_raw_data(self):
        """Generator: Yield samples one at a time - NEVER accumulate (fixes OOM).
        
        Memory-efficient streaming pipeline:
        - Original sources → yield immediately
        - Re-ARC infinite → generate on-the-fly
        - Problem augmentation → 50% chance per sample
        - Multi-task → yield distribution task after solve task
        
        Constant memory: ~2KB per sample (vs 240MB for 120K samples)
        """
        import random
        
        # Track task groups for multi-task (minimal memory)
        task_buffer = {}  # Keep last N samples per task for distribution learning
        max_buffer_per_task = 5
        
        # 1. Yield from original sources (NO list accumulation)
        for source_path in self.config.source_paths:
            detected_type = self._detect_source_type(source_path)
            
            # Each loader is now a generator
            source_generator = None
            if detected_type == "puzzlevqa":
                source_generator = self._load_puzzlevqa_streaming(source_path)
            elif detected_type == "mathv":
                source_generator = self._load_mathv_streaming(source_path)
            elif detected_type == "raven":
                source_generator = self._load_raven_streaming(source_path)
            elif detected_type == "arc_json":
                source_generator = self._load_arc_json_streaming(source_path)
            elif detected_type == "maze_csv":
                source_generator = self._load_maze_csv_streaming(source_path)
            elif detected_type == "sudoku_grid":
                source_generator = self._load_sudoku_streaming(source_path)
            elif detected_type == "text_file":
                source_generator = self._load_text_dataset_streaming(source_path)
            elif detected_type == "image_dir":
                source_generator = self._load_image_dataset_streaming(source_path)
            elif detected_type == "hf":
                source_generator = self._load_from_hf_streaming(source_path)
            else:
                source_generator = self._load_from_local_streaming(source_path)
            
            if source_generator:
                for sample in source_generator:
                    # Yield original sample
                    yield sample
                    
                    # Problem augmentation (50% chance, on-the-fly)
                    if self.config.enable_problem_augmentation and random.random() < self.config.problem_aug_prob:
                        aug_sample = self._augment_sample_streaming(sample)
                        if aug_sample:
                            yield aug_sample
                    
                    # Multi-task: Buffer samples for distribution learning
                    if self.config.enable_multitask:
                        task_id = sample.metadata.get('task_id', sample.metadata.get('puzzle_id', 'unknown'))
                        if task_id not in task_buffer:
                            task_buffer[task_id] = []
                        task_buffer[task_id].append(sample)
                        
                        # Keep buffer small (only last N samples per task)
                        if len(task_buffer[task_id]) > max_buffer_per_task:
                            task_buffer[task_id].pop(0)
                        
                        # Yield distribution learning task (inputs → input)
                        if len(task_buffer[task_id]) >= 2 and sample.grid is not None:
                            yield DataSample(
                                sample_id=f"{sample.sample_id}_distlearn",
                                modality=sample.modality,
                                grid=sample.grid,
                                label=sample.grid,  # Learn distribution: output = input
                                text=sample.text,
                                image=sample.image,
                                metadata={**sample.metadata, 'task_type': 'distribution_learning'}
                            )
        
        # 2. Yield Re-ARC infinite synthetic data (Generator inside generator)
        if self.config.use_rearc_infinite:
            for sample in self._load_rearc_infinite_streaming():
                yield sample
                
                # Problem augmentation on Re-ARC too
                if self.config.enable_problem_augmentation and random.random() < self.config.problem_aug_prob:
                    aug_sample = self._augment_sample_streaming(sample)
                    if aug_sample:
                        yield aug_sample
    
    def _detect_source_type(self, path: str) -> str:
        """Auto-detect source type from path/name."""
        path_lower = path.lower()
        
        # Tier 1 Datasets (Research-grade reasoning)
        if "puzzlevqa" in path_lower or "puzzle_vqa" in path_lower:
            return "puzzlevqa"
        
        if "math-v" in path_lower or "mathv" in path_lower or "math_vision" in path_lower:
            return "mathv"
        
        if "raven" in path_lower and not "craven" in path_lower:
            return "raven"
        
        # ARC JSON format (file or directory)
        if "arc" in path_lower:
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
    
    def _load_arc_json_streaming(self, source_path: str):
        """Streaming generator: Yield ARC samples one at a time."""
        print(f"🔄 Streaming ARC JSON: {source_path}")
        
        try:
            import orjson as json
            def json_load(f):
                return json.loads(f.read())
        except ImportError:
            import json
            def json_load(f):
                return json.load(f)
        
        try:
            source_path = Path(source_path)
            
            if source_path.is_dir():
                challenges_file = source_path / "arc-agi_training_challenges.json"
                solutions_file = source_path / "arc-agi_training_solutions.json"
                
                if not challenges_file.exists():
                    challenges_file = source_path / "arc-agi_evaluation_challenges.json"
                    solutions_file = source_path / "arc-agi_evaluation_solutions.json"
                
                if challenges_file.exists() and solutions_file.exists():
                    with open(challenges_file, 'rb') as f:
                        challenges = json_load(f)
                    with open(solutions_file, 'rb') as f:
                        solutions = json_load(f)
                    
                    for task_id in challenges.keys():
                        task_data = {
                            'train': challenges[task_id].get('train', []),
                            'test': [{'input': ex['input'], 'output': solutions[task_id][i]}
                                    for i, ex in enumerate(challenges[task_id].get('test', []))]
                        }
                        
                        for split in ['train', 'test']:
                            for idx, example in enumerate(task_data.get(split, [])):
                                output = example.get('output')
                                if isinstance(output, list):
                                    output = np.array(output)
                                
                                yield DataSample(
                                    sample_id=f"{task_id}_{split}_{idx}",
                                    modality=ModalityType.GRID,
                                    grid=np.array(example.get('input', [])),
                                    label=output,
                                    metadata={'task_id': task_id, 'split': split}
                                )
        except Exception as e:
            print(f"⚠️  Error streaming ARC: {e}")
    
    def _load_rearc_infinite_streaming(self):
        """Streaming generator: Yield infinite Re-ARC samples on-the-fly."""
        if not self.config.use_rearc_infinite:
            return
        
        import sys
        rearc_path = Path(self.config.rearc_path)
        if not rearc_path.exists():
            print(f"⚠️  Re-ARC not found at {rearc_path}")
            return
        
        sys.path.insert(0, str(rearc_path))
        
        try:
            import generators
            
            print(f"🔄 Streaming Re-ARC infinite data (on-the-fly generation)...")
            
            generators_dict = {name[9:]: getattr(generators, name) 
                             for name in dir(generators) if name.startswith('generate_')}
            
            total_generated = 0
            for task_id, generator_fn in generators_dict.items():
                for i in range(self.config.rearc_examples_per_task):
                    try:
                        example = generator_fn(diff_lb=0, diff_ub=1)
                        
                        yield DataSample(
                            sample_id=f"rearc_{task_id}_{i}",
                            modality=ModalityType.GRID,
                            grid=np.array(example['input'], dtype=np.int32),
                            label=np.array(example['output'], dtype=np.int32),
                            metadata={'task_id': task_id, 'source': 're-arc', 'synthetic': True}
                        )
                        total_generated += 1
                    except Exception:
                        continue  # Skip failed generations
            
            print(f"   ✓ Generated {total_generated} Re-ARC samples (streaming)")
        except Exception as e:
            print(f"⚠️  Re-ARC streaming failed: {e}")
    
    def _load_rearc_infinite(self) -> List[DataSample]:
        """Load infinite synthetic ARC data from Re-ARC (Winner Strategy)."""
        if not self.config.use_rearc_infinite:
            return []
        
        import sys
        rearc_path = Path(self.config.rearc_path)
        if not rearc_path.exists():
            print(f"⚠️  Re-ARC not found at {rearc_path}. Clone with: git clone https://github.com/michaelhodel/re-arc.git {rearc_path}")
            return []
        
        # Add Re-ARC to path
        sys.path.insert(0, str(rearc_path))
        
        try:
            import generators
            from utils import format_example
            
            print(f"🔄 Generating infinite Re-ARC data ({self.config.rearc_examples_per_task} examples per task)...")
            
            # Get all generator functions
            generators_dict = {name[9:]: getattr(generators, name) 
                             for name in dir(generators) if name.startswith('generate_')}
            
            samples = []
            for task_id, generator_fn in generators_dict.items():
                for i in range(self.config.rearc_examples_per_task):
                    try:
                        # Generate example with difficulty range 0-1
                        example = generator_fn(diff_lb=0, diff_ub=1)
                        
                        samples.append(DataSample(
                            sample_id=f"rearc_{task_id}_{i}",
                            modality=ModalityType.GRID,
                            grid=np.array(example['input'], dtype=np.int32),
                            label=np.array(example['output'], dtype=np.int32),
                            metadata={'task_id': task_id, 'source': 're-arc', 'synthetic': True}
                        ))
                    except Exception as e:
                        continue  # Skip failed generations
            
            print(f"   ✓ Generated {len(samples)} synthetic ARC samples from {len(generators_dict)} tasks")
            return samples
        except Exception as e:
            print(f"⚠️  Re-ARC generation failed: {e}")
            return []
    
    def _augment_sample_streaming(self, sample: DataSample):
        """On-the-fly augmentation: Transform input OR output (Winner Strategy)."""
        import random
        
        if sample.modality not in [ModalityType.GRID, ModalityType.MAZE]:
            return None
        
        # Randomly transform ONLY input OR output (not both)
        transforms = ['rotate_90', 'rotate_180', 'rotate_270', 'flip_h', 'flip_v']
        transform = random.choice(transforms)
        transform_input = random.choice([True, False])
        
        new_grid = sample.grid.copy() if sample.grid is not None else None
        new_label = sample.label.copy() if isinstance(sample.label, np.ndarray) else sample.label
        
        # Apply transform
        if transform == 'rotate_90':
            if transform_input and new_grid is not None:
                new_grid = np.rot90(new_grid, k=1)
            elif not transform_input and isinstance(new_label, np.ndarray):
                new_label = np.rot90(new_label, k=1)
        elif transform == 'rotate_180':
            if transform_input and new_grid is not None:
                new_grid = np.rot90(new_grid, k=2)
            elif not transform_input and isinstance(new_label, np.ndarray):
                new_label = np.rot90(new_label, k=2)
        elif transform == 'rotate_270':
            if transform_input and new_grid is not None:
                new_grid = np.rot90(new_grid, k=3)
            elif not transform_input and isinstance(new_label, np.ndarray):
                new_label = np.rot90(new_label, k=3)
        elif transform == 'flip_h':
            if transform_input and new_grid is not None:
                new_grid = np.fliplr(new_grid)
            elif not transform_input and isinstance(new_label, np.ndarray):
                new_label = np.fliplr(new_label)
        elif transform == 'flip_v':
            if transform_input and new_grid is not None:
                new_grid = np.flipud(new_grid)
            elif not transform_input and isinstance(new_label, np.ndarray):
                new_label = np.flipud(new_label)
        
        return DataSample(
            sample_id=f"{sample.sample_id}_probaug_{transform}_{transform_input}",
            modality=sample.modality,
            grid=new_grid,
            label=new_label,
            text=sample.text,
            image=sample.image,
            metadata={**sample.metadata, 'problem_aug': transform, 'aug_input': transform_input}
        )
    
    # Stub streaming loaders (fallback to empty generators for unused types)
    def _load_puzzlevqa_streaming(self, path: str):
        return iter([])  # Stub: implement if needed
    
    def _load_mathv_streaming(self, path: str):
        return iter([])  # Stub: implement if needed
    
    def _load_raven_streaming(self, path: str):
        return iter([])  # Stub: implement if needed
    
    def _load_maze_csv_streaming(self, path: str):
        return iter([])  # Stub: implement if needed
    
    def _load_sudoku_streaming(self, path: str):
        return iter([])  # Stub: implement if needed
    
    def _load_text_dataset_streaming(self, path: str):
        return iter([])  # Stub: implement if needed
    
    def _load_image_dataset_streaming(self, path: str):
        return iter([])  # Stub: implement if needed
    
    def _load_from_hf_streaming(self, dataset_name: str):
        return iter([])  # Stub: implement if needed
    
    def _load_from_local_streaming(self, path: str):
        return iter([])  # Stub: implement if needed
    
    def _apply_problem_augmentation(self, samples: List[DataSample]) -> List[DataSample]:
        """Apply transformations to inputs OR outputs to create new tasks (Winner Strategy)."""
        import random
        
        augmented = samples.copy()
        transforms = ['rotate_90', 'rotate_180', 'rotate_270', 'flip_h', 'flip_v']
        
        for sample in samples:
            if random.random() > self.config.problem_aug_prob:
                continue
            
            if sample.modality not in [ModalityType.GRID, ModalityType.MAZE]:
                continue
            
            # Randomly transform ONLY input OR output (not both)
            transform = random.choice(transforms)
            transform_input = random.choice([True, False])
            
            new_grid = sample.grid.copy() if sample.grid is not None else None
            new_label = sample.label.copy() if isinstance(sample.label, np.ndarray) else sample.label
            
            # Apply transform
            if transform == 'rotate_90':
                if transform_input and new_grid is not None:
                    new_grid = np.rot90(new_grid, k=1)
                elif not transform_input and isinstance(new_label, np.ndarray):
                    new_label = np.rot90(new_label, k=1)
            elif transform == 'rotate_180':
                if transform_input and new_grid is not None:
                    new_grid = np.rot90(new_grid, k=2)
                elif not transform_input and isinstance(new_label, np.ndarray):
                    new_label = np.rot90(new_label, k=2)
            elif transform == 'rotate_270':
                if transform_input and new_grid is not None:
                    new_grid = np.rot90(new_grid, k=3)
                elif not transform_input and isinstance(new_label, np.ndarray):
                    new_label = np.rot90(new_label, k=3)
            elif transform == 'flip_h':
                if transform_input and new_grid is not None:
                    new_grid = np.fliplr(new_grid)
                elif not transform_input and isinstance(new_label, np.ndarray):
                    new_label = np.fliplr(new_label)
            elif transform == 'flip_v':
                if transform_input and new_grid is not None:
                    new_grid = np.flipud(new_grid)
                elif not transform_input and isinstance(new_label, np.ndarray):
                    new_label = np.flipud(new_label)
            
            # Create new augmented sample
            augmented.append(DataSample(
                sample_id=f"{sample.sample_id}_probaug_{transform}_{transform_input}",
                modality=sample.modality,
                grid=new_grid,
                label=new_label,
                text=sample.text,
                image=sample.image,
                metadata={**sample.metadata, 'problem_aug': transform, 'aug_input': transform_input}
            ))
        
        print(f"   ✓ Problem augmentation: {len(samples)} → {len(augmented)} samples (+{len(augmented)-len(samples)} new tasks)")
        return augmented
    
    def _add_multitask_samples(self, samples: List[DataSample]) -> List[DataSample]:
        """Add Task 2: Learn input distribution (Winner Strategy - Omni-ARC)."""
        import random
        
        multitask = samples.copy()
        
        # Task 2: inputs → input (learn distribution)
        # Group samples by task to create input-only examples
        task_groups = {}
        for sample in samples:
            task_id = sample.metadata.get('task_id', sample.metadata.get('puzzle_id', 'unknown'))
            if task_id not in task_groups:
                task_groups[task_id] = []
            task_groups[task_id].append(sample)
        
        # For each task with multiple examples, create distribution learning samples
        for task_id, task_samples in task_groups.items():
            if len(task_samples) < 2:
                continue
            
            # Create samples that predict a random input from other inputs
            for i, target_sample in enumerate(task_samples[:min(len(task_samples), 5)]):
                other_samples = [s for j, s in enumerate(task_samples) if j != i]
                if len(other_samples) > 0 and target_sample.grid is not None:
                    multitask.append(DataSample(
                        sample_id=f"{target_sample.sample_id}_distlearn",
                        modality=target_sample.modality,
                        grid=target_sample.grid,  # Input is same
                        label=target_sample.grid,  # Output is input (learn distribution)
                        text=target_sample.text,
                        image=target_sample.image,
                        metadata={**target_sample.metadata, 'task_type': 'distribution_learning'}
                    ))
        
        print(f"   ✓ Multi-task training: {len(samples)} → {len(multitask)} samples (+{len(multitask)-len(samples)} distribution tasks)")
        return multitask
    
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
    
    def _load_puzzlevqa(self, path: str) -> List[DataSample]:
        """Load PuzzleVQA dataset (2K abstract visual reasoning puzzles)."""
        print(f"🧩 Loading PuzzleVQA: {path}")
        import json
        from PIL import Image
        
        samples = []
        data_path = Path(path)
        
        # PuzzleVQA structure: images/ + annotations.json
        if data_path.is_dir():
            annot_file = data_path / "annotations.json"
            img_dir = data_path / "images"
        else:
            annot_file = Path(path)
            img_dir = annot_file.parent / "images"
        
        if not annot_file.exists():
            print(f"⚠️  PuzzleVQA annotations not found: {annot_file}")
            return []
        
        with open(annot_file) as f:
            data = json.load(f)
        
        for item in data:
            img_path = img_dir / item.get('image', '')
            if img_path.exists():
                samples.append(DataSample(
                    sample_id=f"puzzlevqa_{item['id']}",
                    modality=ModalityType.IMAGE,
                    image=np.array(Image.open(img_path).convert('RGB')),
                    text=item.get('question', ''),
                    label=item.get('answer'),
                    metadata={'concept': item.get('concept'), 'source': 'puzzlevqa'}
                ))
        
        print(f"   Loaded {len(samples)} PuzzleVQA puzzles")
        return samples
    
    def _load_mathv(self, path: str) -> List[DataSample]:
        """Load MATH-V dataset (3K mathematical reasoning with visual context)."""
        print(f"📐 Loading MATH-V: {path}")
        import json
        from PIL import Image
        import io
        
        samples = []
        data_path = Path(path)
        
        # MATH-V structure: testmini.json or test.json
        json_files = list(data_path.glob("*.json")) if data_path.is_dir() else [Path(path)]
        
        for json_file in json_files:
            with open(json_file) as f:
                data = json.load(f)
            
            for item in data:
                # MATH-V has base64 encoded images or image paths
                img = None
                if 'image' in item:
                    if isinstance(item['image'], str) and item['image'].startswith('data:image'):
                        # Base64 encoded
                        import base64
                        img_data = base64.b64decode(item['image'].split(',')[1])
                        img = np.array(Image.open(io.BytesIO(img_data)).convert('RGB'))
                    else:
                        # File path
                        img_path = data_path / item['image']
                        if img_path.exists():
                            img = np.array(Image.open(img_path).convert('RGB'))
                
                samples.append(DataSample(
                    sample_id=f"mathv_{item['pid']}",
                    modality=ModalityType.MULTIMODAL if img is not None else ModalityType.TEXT,
                    image=img,
                    text=item.get('question', ''),
                    label=item.get('answer'),
                    metadata={
                        'subject': item.get('subject'),
                        'level': item.get('level'),
                        'source': 'math-v'
                    }
                ))
        
        print(f"   Loaded {len(samples)} MATH-V problems")
        return samples
    
    def _load_raven(self, path: str) -> List[DataSample]:
        """Load RAVEN dataset (70K RPM-style relational reasoning puzzles)."""
        print(f"🧠 Loading RAVEN: {path}")
        import numpy as np
        from PIL import Image
        
        samples = []
        data_path = Path(path)
        
        # RAVEN structure: center_single/, distribute_four/, etc.
        # Each config has images as .npz files
        config_dirs = [d for d in data_path.iterdir() if d.is_dir()]
        
        for config_dir in config_dirs:
            config_name = config_dir.name
            npz_files = list(config_dir.glob("*.npz"))
            
            for npz_file in npz_files[:1000]:  # Limit per config to prevent overload
                data = np.load(npz_file)
                
                # RAVEN images are 160x160 grayscale, 8 context + 1 target
                # Combine into single 3x3 grid visualization
                context = data['image'].reshape(8, 160, 160)  # 8 context panels
                target = data['target']  # Answer index (0-7)
                
                # Create 3x3 grid image (480x480)
                grid_img = np.ones((480, 480), dtype=np.uint8) * 255
                for i in range(8):
                    row, col = i // 3, i % 3
                    grid_img[row*160:(row+1)*160, col*160:(col+1)*160] = context[i]
                
                # Convert to RGB PIL Image then numpy
                grid_pil = Image.fromarray(grid_img).convert('RGB')
                
                samples.append(DataSample(
                    sample_id=f"raven_{config_name}_{npz_file.stem}",
                    modality=ModalityType.IMAGE,
                    image=np.array(grid_pil),
                    label=int(target),
                    metadata={'config': config_name, 'source': 'raven'}
                ))
        
        print(f"   Loaded {len(samples)} RAVEN puzzles")
        return samples
    
    def preprocess_sample(self, sample: DataSample) -> DataSample:
        """Unified preprocessing: clean text, normalize images, validate grids."""
        # Text cleaning
        if sample.text:
            # Remove excessive whitespace
            sample.text = ' '.join(sample.text.split())
            # Truncate very long text (prevent memory issues)
            max_text_len = 10000
            if len(sample.text) > max_text_len:
                sample.text = sample.text[:max_text_len] + "..."
        
        # Image: resize and compress to uint8 (4x memory reduction)
        if sample.image is not None and isinstance(sample.image, np.ndarray):
            # Use OpenCV for 5-10x faster resizing (C++ backend)
            try:
                import cv2
                # Resize with OpenCV (INTER_LANCZOS4 = LANCZOS)
                sample.image = cv2.resize(
                    sample.image, 
                    (self.config.image_size, self.config.image_size),
                    interpolation=cv2.INTER_LANCZOS4
                ).astype(np.uint8)
            except ImportError:
                # Fallback to PIL if OpenCV not available
                from PIL import Image
                sample.image = np.array(
                    Image.fromarray(sample.image).resize(
                        (self.config.image_size, self.config.image_size), 
                        Image.Resampling.LANCZOS
                    ),
                    dtype=np.uint8
                )
        
        # Grid: crop to max size
        if sample.grid is not None and sample.grid.shape[0] > self.config.max_grid_size:
            sample.grid = sample.grid[:self.config.max_grid_size, :self.config.max_grid_size]
        
        # Combine metadata fields into text for multimodal context
        parts = []
        if sample.text:
            parts.append(sample.text)
        if sample.metadata:
            # Add structured metadata as text context
            if 'category' in sample.metadata:
                parts.append(f"Category: {sample.metadata['category']}")
            if 'difficulty' in sample.metadata:
                parts.append(f"Difficulty: {sample.metadata['difficulty']}")
        
        sample.text = " | ".join(p for p in parts if p)
        return sample
    
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
        """Unified augmentation: full dihedral group (8 transformations) for both images and grids.
        
        This enables TRM to learn rotation-invariant features by seeing each sample from
        all 8 perspectives: 4 rotations + 4 reflections.
        """
        augmented = [sample]
        
        # Images: Full dihedral group (8 transformations) - same as grids
        # This is crucial for TRM's spatial reasoning capabilities
        if sample.image is not None and isinstance(sample.image, np.ndarray):
            # Apply all 8 dihedral transformations
            # tid 0 = identity (original, already in list)
            # tid 1-3 = rotations (90°, 180°, 270°)
            # tid 4-7 = reflections (flip_lr, flip_ud, transpose, anti-diagonal)
            for tid in [1, 2, 3, 4, 5, 6, 7]:
                # Use PIL for image rotations (handles RGB channels correctly)
                aug_image = sample.image.copy()
                
                if tid == 1:
                    aug_image = np.rot90(aug_image, k=1)  # 90° CCW
                elif tid == 2:
                    aug_image = np.rot90(aug_image, k=2)  # 180°
                elif tid == 3:
                    aug_image = np.rot90(aug_image, k=3)  # 270° CCW
                elif tid == 4:
                    aug_image = np.fliplr(aug_image)  # Horizontal flip
                elif tid == 5:
                    aug_image = np.flipud(aug_image)  # Vertical flip
                elif tid == 6:
                    aug_image = np.transpose(aug_image, (1, 0, 2))  # Transpose (swap H,W, keep C)
                elif tid == 7:
                    aug_image = np.fliplr(np.rot90(aug_image, k=1))  # Anti-diagonal
                
                aug = DataSample(
                    sample_id=f"{sample.sample_id}_aug{tid}",
                    modality=sample.modality,
                    text=sample.text,
                    image=aug_image,
                    label=sample.label,
                    metadata=sample.metadata
                )
                augmented.append(aug)
        
        # Grid rotations (dihedral group) - keep existing implementation
        if sample.grid is not None:
            from common import dihedral_transform
            for tid in [1, 2, 3, 4, 5, 6, 7]:  # All 7 non-identity transforms
                aug = DataSample(**sample.model_dump())
                aug.grid = dihedral_transform(sample.grid, tid)
                aug.sample_id = f"{sample.sample_id}_aug{tid}"
                augmented.append(aug)
        
        return augmented


def build(config: MultimodalDatasetConfig):
    """Unified dataset builder - all formats, all modalities."""
    print(f"\n{'='*70}\n🌐 Building: {', '.join(config.source_paths)}\n{'='*70}")
    
    # Check if dataset already exists (final or in-progress)
    import os
    from pathlib import Path
    train_path = os.path.join(config.output_dir, 'capsule_dataset.pt')
    test_path = os.path.join(config.output_dir, 'capsule_dataset_test.pt')
    info_path = os.path.join(config.output_dir, 'dataset_info.json')
    checkpoint_dir = os.path.join(config.output_dir, 'stream_checkpoints')
    
    # Check for completed dataset
    if os.path.exists(train_path) and os.path.exists(test_path) and os.path.exists(info_path):
        print(f"\n✅ Dataset already exists at {config.output_dir}")
        print(f"   Skipping re-encoding to save time")
        print(f"   To rebuild: delete {config.output_dir} and re-run\n")
        
        # Load and display info
        import json
        with open(info_path) as f:
            info = json.load(f)
        print(f"   Train samples: {info.get('num_samples', 'unknown')}")
        print(f"   Capsules per sample: {info.get('num_capsules', 'unknown')}")
        print(f"   Hidden size: {info.get('hidden_size', 'unknown')}")
        return
    
    # Check for partial progress from streaming builder
    # Note: We don't return early - the streaming builder has its own resume logic
    # in _check_resume_state() that will skip already-encoded samples
    if os.path.exists(checkpoint_dir):
        batch_files = list(Path(checkpoint_dir).glob('batch_*.pt'))
        consolidated_files = list(Path(checkpoint_dir).glob('consolidated_*.pt'))
        
        if batch_files or consolidated_files:
            print(f"\n♻️  Found existing progress: {len(batch_files)} batches + {len(consolidated_files)} chunks")
            print(f"   Streaming encoder will resume from where it left off\n")
    
    # Build pipeline (only if not resuming)
    builder = MultimodalDatasetBuilder(config)
    dataset = builder.build_dataset()
    
    # Save dataset with proper train/test structure
    import os
    import json
    
    # Create train/test subdirectories
    train_dir = os.path.join(config.output_dir, 'train')
    test_dir = os.path.join(config.output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Get raw samples (no pre-encoding - TRM will encode during training)
    train_samples = dataset['train']
    test_samples = dataset.get('test', [])
    
    # Save raw samples with PuzzleDataset-compatible structure
    print("💾 Saving raw samples...")
    
    # Create proper metadata with all required fields
    num_concepts = config.num_concepts
    vocab_size = num_concepts + 4  # Concept vocab + control symbols
    
    train_metadata = {
        'pad_id': 0,
        'ignore_label_id': -100,
        'blank_identifier_id': 0,
        'vocab_size': vocab_size,
        'seq_len': 512,  # Will be set by model during training
        'num_puzzle_identifiers': 0,
        'total_groups': len(train_samples),
        'mean_puzzle_examples': 1.0,
        'total_puzzles': len(train_samples),
        'sets': ['train']
    }
    
    with open(os.path.join(train_dir, 'dataset.json'), 'w') as f:
        json.dump(train_metadata, f, indent=2)
    torch.save(train_samples, os.path.join(train_dir, 'raw_samples.pt'))
    
    # Save test split (optional - only if test samples exist)
    if test_samples:
        test_metadata = train_metadata.copy()
        test_metadata['total_groups'] = len(test_samples)
        test_metadata['total_puzzles'] = len(test_samples)
        test_metadata['sets'] = ['test']
        
        with open(os.path.join(test_dir, 'dataset.json'), 'w') as f:
            json.dump(test_metadata, f, indent=2)
        torch.save(test_samples, os.path.join(test_dir, 'raw_samples.pt'))
    
    # Save overall metadata
    with open(info_path, 'w') as f:
        json.dump({
            'num_train_samples': len(train_samples),
            'num_test_samples': len(test_samples),
            'format': 'raw_samples',
            'note': 'Images cached, TRM encodes during training'
        }, f, indent=2)
    
    print(f"✅ Dataset ready: {len(train_samples)} samples")
    print(f"   Images cached: ImageCache handles rendering + denoising")
    print(f"   TRM encoding: During training (on-the-fly)")


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
    
    # Quality scoring removed - not implemented and adds overhead


# ===== Vision-Unified Pipeline Only =====
# All data (text, images, grids) goes through TRM vision encoder
# Text → rendered to images → TRM encoder → capsules
# Images → TRM encoder → capsules
# Grids → rendered to images → TRM encoder → capsules

# Main entry point: use build(config) directly in your scripts
# Example:
#   config = MultimodalDatasetConfig(source_paths=['data/'], output_dir='datasets/out')
#   build(config)


if __name__ == "__main__":
    cli()
