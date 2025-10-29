"""
Text dataset builder for TRM language modeling.
Supports WikiText-2, TinyStories, and custom text files.
"""

from typing import List
import os
import json
import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm

try:
    from transformers import AutoTokenizer
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False
    print("WARNING: transformers not installed. Install with: pip install transformers")

from common import PuzzleDatasetMetadata


cli = ArgParser()


class TextDatasetConfig(BaseModel):
    input_file: str  # Path to text file
    output_dir: str
    tokenizer_name: str = "gpt2"  # GPT-2 BPE tokenizer (50257 vocab)
    max_seq_len: int = 512  # Maximum sequence length
    stride: int = 256  # Overlap between sequences (for context)
    train_split: float = 0.9  # 90% train, 10% test
    seed: int = 42


def load_text_file(file_path: str) -> str:
    """Load text from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def download_wikitext2():
    """Download WikiText-2 dataset if not present."""
    try:
        from datasets import load_dataset
        print("Downloading WikiText-2 from HuggingFace...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        return dataset
    except ImportError:
        print("❌ datasets library not found. Install with: pip install datasets")
        return None
    except Exception as e:
        print(f"❌ Failed to download WikiText-2: {e}")
        return None


def download_tinystories():
    """Download TinyStories dataset (smaller, great for testing)."""
    try:
        from datasets import load_dataset
        print("Downloading TinyStories from HuggingFace...")
        dataset = load_dataset("roneneldan/TinyStories")
        return dataset
    except ImportError:
        print("❌ datasets library not found. Install with: pip install datasets")
        return None
    except Exception as e:
        print(f"❌ Failed to download TinyStories: {e}")
        return None


def download_code_python():
    """Download The Stack Python code dataset (deduplicated)."""
    try:
        from datasets import load_dataset
        print("Downloading The Stack Python (this may take a while...)")
        # Use streaming to avoid downloading entire dataset
        dataset = load_dataset(
            "bigcode/the-stack-dedup",
            data_dir="data/python",
            split="train",
            streaming=True
        )
        return dataset
    except ImportError:
        print("❌ datasets library not found. Install with: pip install datasets")
        return None
    except Exception as e:
        print(f"❌ Failed to download The Stack: {e}")
        return None


def tokenize_and_chunk(text: str, tokenizer, max_seq_len: int, stride: int):
    """
    Tokenize text and create overlapping chunks.
    
    Args:
        text: Input text
        tokenizer: HuggingFace tokenizer
        max_seq_len: Maximum tokens per sequence
        stride: Overlap between sequences
    
    Returns:
        List of token sequences
    """
    # Tokenize entire text
    tokens = tokenizer.encode(text)
    
    # Create sliding window chunks
    sequences = []
    for i in range(0, len(tokens) - max_seq_len + 1, stride):
        chunk = tokens[i:i + max_seq_len]
        if len(chunk) == max_seq_len:  # Only keep full sequences
            sequences.append(chunk)
    
    return sequences


def create_text_dataset(config: TextDatasetConfig):
    """Create tokenized text dataset for TRM training."""
    
    if not TOKENIZERS_AVAILABLE:
        raise ImportError("transformers library required. Install with: pip install transformers")
    
    np.random.seed(config.seed)
    
    # Load tokenizer
    print(f"Loading tokenizer: {config.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    vocab_size = tokenizer.vocab_size
    
    # Handle special tokens
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.vocab_size - 1
    
    # Load text
    if config.input_file == "wikitext-2":
        dataset = download_wikitext2()
        if dataset is None:
            raise RuntimeError("Failed to download WikiText-2")
        train_text = "\n".join(dataset['train']['text'])
        test_text = "\n".join(dataset['test']['text'])
    elif config.input_file == "tinystories":
        dataset = download_tinystories()
        if dataset is None:
            raise RuntimeError("Failed to download TinyStories")
        # TinyStories has 'train' and 'validation' splits
        train_text = "\n".join(dataset['train']['text'][:100000])  # Use first 100k stories
        test_text = "\n".join(dataset['validation']['text'][:10000])  # Use first 10k for test
        print(f"Using 100k train stories, 10k validation stories")
    elif config.input_file == "code-python":
        dataset = download_code_python()
        if dataset is None:
            raise RuntimeError("Failed to download The Stack Python")
        print("Collecting code samples (this may take a few minutes)...")
        # Use streaming dataset - take first 50k samples for train, next 5k for test
        code_samples = []
        for i, sample in enumerate(dataset):
            if i >= 55000:
                break
            code_samples.append(sample['content'])
            if (i + 1) % 5000 == 0:
                print(f"  Collected {i + 1} samples...")
        
        train_text = "\n\n".join(code_samples[:50000])
        test_text = "\n\n".join(code_samples[50000:55000])
        print(f"Using 50k train files, 5k test files")
    else:
        print(f"Loading text from: {config.input_file}")
        if not os.path.exists(config.input_file):
            raise FileNotFoundError(f"File not found: {config.input_file}")
        full_text = load_text_file(config.input_file)
        
        # Split into train/test
        split_idx = int(len(full_text) * config.train_split)
        train_text = full_text[:split_idx]
        test_text = full_text[split_idx:]
    
    print(f"Train text length: {len(train_text)} chars")
    print(f"Test text length: {len(test_text)} chars")
    
    # Tokenize and chunk
    print("Tokenizing train set...")
    train_sequences = tokenize_and_chunk(train_text, tokenizer, config.max_seq_len, config.stride)
    print(f"Created {len(train_sequences)} training sequences")
    
    print("Tokenizing test set...")
    test_sequences = tokenize_and_chunk(test_text, tokenizer, config.max_seq_len, config.stride)
    print(f"Created {len(test_sequences)} test sequences")
    
    # Convert to TRM format (similar to ARC structure)
    os.makedirs(config.output_dir, exist_ok=True)
    
    for split_name, sequences in [("train", train_sequences), ("test", test_sequences)]:
        os.makedirs(os.path.join(config.output_dir, split_name), exist_ok=True)
        
        # Convert sequences to numpy arrays
        inputs = []
        labels = []
        puzzle_identifiers = []  # All 0 for text (no puzzle-specific embeddings)
        puzzle_indices = [0]  # Each sequence is a "puzzle"
        group_indices = [0]  # All sequences in one group
        
        for seq in sequences:
            # Input: tokens 0 to n-1
            # Label: tokens 1 to n (next token prediction)
            inputs.append(seq[:-1])
            labels.append(seq[1:])
            puzzle_identifiers.append(0)  # No puzzle-specific embedding for text
            puzzle_indices.append(len(inputs))
        
        group_indices.append(len(sequences))
        
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
        metadata = PuzzleDatasetMetadata(
            seq_len=config.max_seq_len - 1,  # -1 because we shift for next-token prediction
            vocab_size=vocab_size,
            pad_id=pad_token_id,
            ignore_label_id=-100,  # Standard for language modeling
            blank_identifier_id=0,
            num_puzzle_identifiers=1,  # No puzzle embeddings for text
            total_groups=1,
            mean_puzzle_examples=1.0,
            total_puzzles=len(sequences),
            sets=["all"]
        )
        
        with open(os.path.join(config.output_dir, split_name, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f, indent=2)
    
    # Save tokenizer info
    tokenizer_info = {
        "tokenizer_name": config.tokenizer_name,
        "vocab_size": vocab_size,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
        "max_seq_len": config.max_seq_len
    }
    
    with open(os.path.join(config.output_dir, "tokenizer_info.json"), "w") as f:
        json.dump(tokenizer_info, f, indent=2)
    
    print(f"\n✅ Dataset created successfully!")
    print(f"   Output: {config.output_dir}")
    print(f"   Vocab size: {vocab_size}")
    print(f"   Sequence length: {config.max_seq_len - 1}")
    print(f"   Train sequences: {len(train_sequences)}")
    print(f"   Test sequences: {len(test_sequences)}")


@cli.command(singleton=True)
def main(config: TextDatasetConfig):
    create_text_dataset(config)


if __name__ == "__main__":
    cli()
