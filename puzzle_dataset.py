import os
import json
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import numpy as np
import pydantic

import torch
from torch.utils.data import IterableDataset, get_worker_info

from models.losses import IGNORE_LABEL_ID
from dataset.common import PuzzleDatasetMetadata

from argdantic import ArgParser
from pydantic import BaseModel

def _sample_batch(rng: np.random.Generator, group_order: np.ndarray, puzzle_indices: np.ndarray, group_indices: np.ndarray, start_index: int, global_batch_size: int):
    # Pack examples into a full batch
    batch = []
    batch_puzzle_indices = []
    current_size = 0

    while (start_index < group_order.size) and (current_size < global_batch_size):
        # Pick a group and a puzzle from that group
        group_id = group_order[start_index]
        puzzle_id = rng.integers(group_indices[group_id], group_indices[group_id + 1])
        start_index += 1

        # Get range of the puzzle
        puzzle_start = puzzle_indices[puzzle_id]
        puzzle_size = int(puzzle_indices[puzzle_id + 1] - puzzle_start)

        append_size = min(puzzle_size, global_batch_size - current_size)

        # Put into batch
        batch_puzzle_indices.append(np.full(append_size, puzzle_id, dtype=np.int32))
        batch.append(puzzle_start + np.random.choice(puzzle_size, append_size, replace=False))

        current_size += append_size

    return start_index, np.concatenate(batch), np.concatenate(batch_puzzle_indices)


class PuzzleDatasetConfig(pydantic.BaseModel):
    seed: int
    dataset_paths: List[str]
    global_batch_size: int
    test_set_mode: bool
    epochs_per_iter: int  # Batch X epochs in an iteration to reduce overhead.
    rank: int
    num_replicas: int

class PuzzleDataset(IterableDataset):
    def __init__(self, config: PuzzleDatasetConfig, split: str = "train"):
        super().__init__()
        self.config = config
        self.split = split

        # Merge multiple metadata
        prev_seq_len = None
        prev_vocab_size = None
        prev_pad_id = None
        prev_ignore_label_id = None
        prev_blank_identifier_id = None
        prev_sets = None
        prev_num_identifiers = None
        mean_puzzle_examples = 0
        total_puzzles = 0
        total_groups = 0
        num_identifiers = 0
        for dataset_path in config.dataset_paths:
            current_metadata = self._load_metadata(dataset_path)
            if prev_seq_len is None:
                prev_seq_len = current_metadata.seq_len
                prev_vocab_size = current_metadata.vocab_size
                prev_pad_id = current_metadata.pad_id
                prev_ignore_label_id = current_metadata.ignore_label_id
                prev_blank_identifier_id = current_metadata.blank_identifier_id
                prev_sets = current_metadata.sets
                prev_num_identifiers = current_metadata.num_puzzle_identifiers
            else:
                assert prev_seq_len == current_metadata.seq_len
                assert prev_vocab_size == current_metadata.vocab_size
                assert prev_pad_id == current_metadata.pad_id
                assert prev_ignore_label_id == current_metadata.ignore_label_id
                assert prev_blank_identifier_id == current_metadata.blank_identifier_id
                assert prev_sets == current_metadata.sets
                assert prev_num_identifiers == current_metadata.num_puzzle_identifiers
            mean_puzzle_examples += current_metadata.mean_puzzle_examples*current_metadata.total_puzzles
            total_puzzles += current_metadata.total_puzzles
            total_groups += current_metadata.total_groups
            num_identifiers += current_metadata.num_puzzle_identifiers
        mean_puzzle_examples = mean_puzzle_examples / total_puzzles

        self.metadata = PuzzleDatasetMetadata(
            seq_len=prev_seq_len,
            vocab_size=prev_vocab_size,
            pad_id=prev_pad_id,
            ignore_label_id=prev_ignore_label_id,
            blank_identifier_id=prev_blank_identifier_id,
            num_puzzle_identifiers=num_identifiers,
            total_groups=total_groups,
            mean_puzzle_examples=mean_puzzle_examples,
            total_puzzles=total_puzzles,
            sets=prev_sets
        )

        # Checks
        assert self.config.global_batch_size % self.config.num_replicas == 0, f"Global batch size {self.config.global_batch_size} must be multiples of nodes {self.config.num_replicas}."
        self.local_batch_size = self.config.global_batch_size // self.config.num_replicas

        # State
        self._data = None
        self._iters = 0

    def _load_metadata(self, dataset_path) -> PuzzleDatasetMetadata:
        # Check if streaming format (consolidated_*.pt files)
        stream_dir = os.path.join(dataset_path, "stream_checkpoints")
        if os.path.exists(stream_dir):
            consolidated_files = sorted(Path(stream_dir).glob("consolidated_*.pt"))
            if consolidated_files:
                # Load streaming format metadata from first chunk
                chunk = torch.load(consolidated_files[0], map_location='cpu')
                num_samples = sum(torch.load(f, map_location='cpu')['sketches'].shape[0] 
                                 for f in consolidated_files)
                
                # Create metadata for streaming format
                return PuzzleDatasetMetadata(
                    seq_len=chunk['sketches'].shape[1],  # Assuming capsule format
                    vocab_size=0,  # Not applicable for vision
                    pad_id=0,
                    ignore_label_id=-100,
                    blank_identifier_id=0,
                    num_puzzle_identifiers=num_samples,
                    total_groups=1,
                    mean_puzzle_examples=1.0,
                    total_puzzles=num_samples,
                    sets=["default"]
                )
        
        # Standard format
        with open(os.path.join(dataset_path, self.split, "dataset.json"), "r") as f:
            return PuzzleDatasetMetadata(**json.load(f))

    def _lazy_load_dataset(self):
        if self._data is not None:
            return

        # Load data
        self._data = {}
        for set_name in self.metadata.sets: # Load subset
            for i, dataset_path in enumerate(self.config.dataset_paths):
                if i > 0:
                    set_name_ = set_name + str(i)
                else:
                    set_name_ = set_name
                
                # Check if streaming format
                stream_dir = os.path.join(dataset_path, "stream_checkpoints")
                if os.path.exists(stream_dir):
                    consolidated_files = sorted(Path(stream_dir).glob("consolidated_*.pt"))
                    if consolidated_files:
                        # Load and concatenate consolidated chunks
                        all_inputs = []
                        all_labels = []
                        
                        for chunk_file in consolidated_files:
                            chunk = torch.load(chunk_file, map_location='cpu')
                            if 'sketches' in chunk:
                                all_inputs.append(chunk['sketches'].numpy())
                            if 'checksums' in chunk:
                                all_labels.append(chunk['checksums'].numpy())
                        
                        inputs = np.concatenate(all_inputs, axis=0)
                        labels = np.concatenate(all_labels, axis=0)
                        num_samples = inputs.shape[0]
                        
                        # group_indices are boundary markers: [0, 1, 2, ..., N]
                        # Each sample is its own group (1 sample per puzzle)
                        # puzzle_indices also boundaries: [0, 1, 2, ..., N]
                        group_indices = np.arange(num_samples + 1, dtype=np.int32)
                        puzzle_indices = np.arange(num_samples + 1, dtype=np.int32)
                        
                        self._data[set_name_] = {
                            "inputs": inputs,
                            "labels": labels,
                            "puzzle_identifiers": np.arange(num_samples, dtype=np.int32),
                            "puzzle_indices": puzzle_indices,
                            "group_indices": group_indices
                        }
                        continue
                
                # Check for raw_samples.pt format (vision-unified pipeline)
                raw_samples_path = os.path.join(dataset_path, self.split, 'raw_samples.pt')
                if os.path.exists(raw_samples_path):
                    # Load raw samples (images + metadata)
                    # weights_only=False needed for custom DataSample class
                    raw_samples = torch.load(raw_samples_path, map_location='cpu', weights_only=False)
                    num_samples = len(raw_samples)
                    
                    # Convert to expected format
                    # Each sample is a dict with 'input_image', 'output_image', 'task_id'
                    # Store as list of dicts - will be processed on-the-fly during training
                    group_indices = np.arange(num_samples + 1, dtype=np.int32)
                    puzzle_indices = np.arange(num_samples + 1, dtype=np.int32)
                    
                    self._data[set_name_] = {
                        "raw_samples": raw_samples,  # List of dicts
                        "puzzle_identifiers": np.arange(num_samples, dtype=np.int32),
                        "puzzle_indices": puzzle_indices,
                        "group_indices": group_indices
                    }
                    continue
                
                # Standard format with .npy files
                field_mmap_modes = {
                    "inputs": "r",
                    "labels": "r",
                    "puzzle_identifiers": None,
                    "puzzle_indices": None,
                    "group_indices": None
                }
                
                self._data[set_name_] = {
                    field_name: np.load(os.path.join(dataset_path, self.split, f"{set_name}__{field_name}.npy"), mmap_mode=mmap_mode)
                    for field_name, mmap_mode in field_mmap_modes.items()
                }


    def _collate_batch(self, batch):
        # Convert dtype
        batch = {k: v.astype(np.int32) for k, v in batch.items()}

        # Convert ignore label IDs
        if self.metadata.ignore_label_id is not None:
            batch["labels"][batch["labels"] == self.metadata.ignore_label_id] = IGNORE_LABEL_ID

        # Pad
        if batch["puzzle_identifiers"].size < self.local_batch_size:
            pad_size = self.local_batch_size - batch["puzzle_identifiers"].size
            pad_values = {
                "inputs": self.metadata.pad_id,
                "labels": IGNORE_LABEL_ID,
                "puzzle_identifiers": self.metadata.blank_identifier_id
            }
            batch = {k: np.pad(v, ((0, pad_size), ) + ((0, 0), ) * (v.ndim - 1), constant_values=pad_values[k]) for k, v in batch.items()}

        # To tensor
        return {k: torch.from_numpy(v) for k, v in batch.items()}
    
    def _iter_test(self):
        for set_i, (set_name, dataset) in enumerate(self._data.items()):  # type: ignore
            # Handle raw_samples format
            if "raw_samples" in dataset:
                total_examples = len(dataset["raw_samples"])
            else:
                total_examples = len(dataset["inputs"])

            # Load examples one by one
            start_index = 0
            while start_index < total_examples:
                # Compute indices
                end_index = min(total_examples, start_index + self.config.global_batch_size)
                
                local_start = start_index + self.config.rank * self.local_batch_size
                local_end   = min(start_index + (self.config.rank + 1) * self.local_batch_size, end_index)
                
                # Get batch of examples, and also puzzle IDs
                puzzle_indices = []
                puzzle_index = np.searchsorted(dataset["puzzle_indices"], local_start, side="right") - 1
                for i in range(local_start, local_end):
                    while puzzle_index + 1 < len(dataset["puzzle_indices"]) and i >= dataset["puzzle_indices"][puzzle_index + 1]:
                        puzzle_index += 1

                    puzzle_indices.append(puzzle_index)
                
                # Handle raw_samples format (vision-unified)
                if "raw_samples" in dataset:
                    batch_samples = [dataset["raw_samples"][idx] for idx in range(local_start, local_end)]
                    batch = {
                        "raw_samples": batch_samples,
                        "puzzle_identifiers": torch.from_numpy(dataset["puzzle_identifiers"][puzzle_indices].astype(np.int32))
                    }
                else:
                    batch = self._collate_batch({
                        "inputs": dataset["inputs"][local_start: local_end],
                        "labels": dataset["labels"][local_start: local_end],
                        "puzzle_identifiers": dataset["puzzle_identifiers"][puzzle_indices]
                    })

                yield set_name, batch, end_index - start_index
                
                # Advance to next batch
                start_index += self.config.global_batch_size

    def _iter_train(self):
        for set_name, dataset in self._data.items():  # type: ignore
            # Increase epoch count
            self._iters += 1

            # Randomly shuffle groups
            rng = np.random.Generator(np.random.Philox(seed=self.config.seed + self._iters))

            group_order = np.concatenate([rng.permutation(dataset["group_indices"].size - 1) for _i in range(self.config.epochs_per_iter)])
            start_index = 0
            
            while start_index < group_order.size:
                start_index, batch_indices, batch_puzzle_indices = _sample_batch(
                    rng,
                    group_order=group_order,
                    puzzle_indices=dataset["puzzle_indices"],
                    group_indices=dataset["group_indices"],
                    start_index=start_index,
                    global_batch_size=self.config.global_batch_size,
                )

                # Select current rank and collate
                global_effective_batch_size = batch_puzzle_indices.size  # Global effective batch size, excluding pads

                # Drop last batch
                if global_effective_batch_size < self.config.global_batch_size:
                    break

                batch_indices        = batch_indices       [self.config.rank * self.local_batch_size: (self.config.rank + 1) * self.local_batch_size]
                batch_puzzle_indices = batch_puzzle_indices[self.config.rank * self.local_batch_size: (self.config.rank + 1) * self.local_batch_size]
                
                # Handle raw_samples format (vision-unified)
                if "raw_samples" in dataset:
                    # Extract raw samples and return as-is (will be processed by model)
                    batch_samples = [dataset["raw_samples"][idx] for idx in batch_indices]
                    batch = {
                        "raw_samples": batch_samples,
                        "puzzle_identifiers": torch.from_numpy(dataset["puzzle_identifiers"][batch_puzzle_indices].astype(np.int32))
                    }
                else:
                    # Standard format with pre-encoded inputs
                    batch = self._collate_batch({
                        "inputs": dataset["inputs"][batch_indices],
                        "labels": dataset["labels"][batch_indices],
                        "puzzle_identifiers": dataset["puzzle_identifiers"][batch_puzzle_indices]
                    })

                yield set_name, batch, global_effective_batch_size
                
    def __iter__(self):
        worker_info = get_worker_info()
        assert worker_info is None or worker_info.num_workers == 1, "Multithreaded data loading is not currently supported."
        
        self._lazy_load_dataset()
        
        # Iterate using specified mode
        if self.config.test_set_mode:
            yield from self._iter_test()
        else:
            yield from self._iter_train()

