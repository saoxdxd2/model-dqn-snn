import torch
from torch.utils.data import DataLoader
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
import numpy as np
import time

# Mock metadata (using minimal required fields based on PuzzleDataset usage)
class MockMetadata:
    def __init__(self):
        self.seq_len = 100
        self.vocab_size = 100
        self.pad_id = 0
        self.ignore_label_id = -1
        self.blank_identifier_id = 0
        self.num_puzzle_identifiers = 1000
        self.total_groups = 100
        self.mean_puzzle_examples = 1.0
        self.total_puzzles = 1000
        self.sets = ["train"]

# Subclass to mock _load_metadata and _lazy_load_dataset
class MockPuzzleDataset(PuzzleDataset):
    def _load_metadata(self, dataset_path):
        return MockMetadata()
        
    def _lazy_load_dataset(self):
        pass

def test_dataloader():
    print("Testing PuzzleDataset with num_workers=4...")
    
    # Mock config
    config = PuzzleDatasetConfig(
        seed=42,
        dataset_paths=["mock_path"], # Needs at least one path to trigger metadata loading
        global_batch_size=4,
        test_set_mode=False,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1
    )
    
    metadata = MockMetadata()
    
    # Initialize dataset
    dataset = MockPuzzleDataset(config)
    # dataset.metadata is now populated by __init__ using the mock _load_metadata
    
    # Mock data loading to avoid needing real files
    # Manually populate _data structure used by _iter_train
    dataset._data = {
        "train": {
            "raw_samples": [f"Sample {i}" for i in range(1000)],
            "puzzle_identifiers": np.arange(1000, dtype=np.int32),
            "puzzle_indices": np.linspace(0, 1000, 1001, dtype=np.int32),
            "group_indices": np.linspace(0, 1000, 101, dtype=np.int32)
        }
    }
    
    # Lazy loading is now disabled by the override in MockPuzzleDataset
    
    # Create DataLoader with multiple workers
    dataloader = DataLoader(
        dataset,
        batch_size=None, # Dataset yields batches
        num_workers=4,
        pin_memory=False
    )
    
    print("Starting iteration...")
    start_time = time.time()
    
    seen_samples = set()
    batch_count = 0
    
    # Iterate for a bit
    for i, (set_name, batch, bs) in enumerate(dataloader):
        if i >= 20: break
        
        samples = batch['raw_samples']
        # print(f"Batch {i}: {len(samples)} samples from worker ?")
        
        for s in samples:
            seen_samples.add(s)
            
        batch_count += 1
        
    duration = time.time() - start_time
    print(f"Processed {batch_count} batches in {duration:.2f}s")
    print(f"Unique samples seen: {len(seen_samples)}")
    
    # Verify we got data
    assert batch_count > 0
    assert len(seen_samples) > 0
    
    print("✅ Multi-worker iteration successful!")

if __name__ == "__main__":
    test_dataloader()
