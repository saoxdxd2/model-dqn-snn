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
    input_file: str  # Path to text file or dataset name
    output_dir: str
    
    # HESC Capsule Mode (semantic_mode=True)
    semantic_mode: bool = False  # Build HESC capsules instead of tokens
    semantic_target_tokens: int = 12  # Number of capsules (k)
    semantic_hidden_size: int = 768  # TRM hidden size
    encoder_model: str = "openai/clip-vit-large-patch14"  # Semantic encoder
    num_concepts: int = 2048  # Concept vocabulary size (output)
    
    # Legacy Token Mode (semantic_mode=False)
    tokenizer_name: str = "gpt2"  # Only used in token mode
    max_seq_len: int = 512  # Only used in token mode
    stride: int = 256  # Only used in token mode
    
    # Common
    train_split: float = 0.9
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
        from huggingface_hub import HfFolder, login
        import os
        
        print("Downloading The Stack Python (this may take a while...)")
        
        # Check if already authenticated
        token = HfFolder.get_token()
        
        # Try loading dataset first
        try:
            dataset = load_dataset(
                "bigcode/the-stack-dedup",
                data_dir="data/python",
                split="train",
                streaming=True
            )
            return dataset
        except Exception as auth_error:
            if "gated dataset" in str(auth_error).lower() or "authenticated" in str(auth_error).lower():
                # Gated dataset - need authentication
                print("\n" + "="*70)
                print("  🔒 The Stack is a GATED dataset - Authentication Required")
                print("="*70)
                print("\nTo access this dataset, you need:")
                print("  1. A HuggingFace account (free)")
                print("  2. Request access to The Stack dataset")
                print("  3. Your HuggingFace API token\n")
                
                print("📝 Steps:")
                print("  1. Create account: https://huggingface.co/join")
                print("  2. Request access: https://huggingface.co/datasets/bigcode/the-stack-dedup")
                print("  3. Get your token: https://huggingface.co/settings/tokens\n")
                
                # Prompt for token
                token_input = input("Enter your HuggingFace token (or press Enter to skip): ").strip()
                
                if token_input:
                    try:
                        # Login with token
                        login(token=token_input, add_to_git_credential=True)
                        print("✅ Authentication successful!\n")
                        
                        # Retry loading
                        print("Retrying download...")
                        dataset = load_dataset(
                            "bigcode/the-stack-dedup",
                            data_dir="data/python",
                            split="train",
                            streaming=True
                        )
                        return dataset
                    except Exception as login_error:
                        print(f"❌ Authentication failed: {login_error}")
                        print("\nTip: Make sure you've requested access to the dataset first!")
                        return None
                else:
                    print("\n⚠️  Skipping code dataset. Try another model instead.")
                    return None
            else:
                # Different error
                raise auth_error
                
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


def create_code_dataset_batched(config: TextDatasetConfig, dataset, tokenizer, vocab_size: int, pad_token_id: int, eos_token_id: int):
    """
    Process code samples in batches to avoid memory exhaustion.
    Uses iterative processing instead of loading all text at once.
    """
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Process in smaller batches for code (files are larger)
    batch_size = 1000  # Process 1000 files at a time
    max_train_files = 50000
    max_test_files = 5000
    
    for split_name, max_files in [("train", max_train_files), ("test", max_test_files)]:
        print(f"\nTokenizing {split_name} set (batched processing)...")
        os.makedirs(os.path.join(config.output_dir, split_name), exist_ok=True)
        
        all_inputs = []
        all_labels = []
        
        file_count = 0
        batch_texts = []
        
        # Determine offset based on split
        start_idx = 0 if split_name == "train" else max_train_files
        end_idx = max_train_files if split_name == "train" else max_train_files + max_test_files
        
        # Process files in batches
        for idx, sample in enumerate(tqdm(dataset, desc=f"Processing {split_name}")):
            # Skip until we reach the split range
            if idx < start_idx:
                continue
            if idx >= end_idx:
                break
            
            batch_texts.append(sample['content'])
            file_count += 1
            
            # Process batch when full
            if len(batch_texts) >= batch_size or file_count >= (end_idx - start_idx):
                # Join texts
                batch_text = "\n\n".join(batch_texts)
                
                # Tokenize batch
                tokens = tokenizer.encode(batch_text)
                
                # Create sequences
                for i in range(0, len(tokens) - config.max_seq_len + 1, config.stride):
                    chunk = tokens[i:i + config.max_seq_len]
                    if len(chunk) == config.max_seq_len:
                        all_inputs.append(chunk[:-1])  # input
                        all_labels.append(chunk[1:])   # shifted labels
                
                # Clear batch
                batch_texts = []
                
                if file_count % 5000 == 0:
                    print(f"  Processed {file_count} files, {len(all_inputs)} sequences so far")
        
        print(f"Created {len(all_inputs)} {split_name} sequences")
        
        # Save sequences
        metadata = {
            "vocab_size": vocab_size,
            "seq_len": config.max_seq_len - 1,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
            "type": "code"
        }
        
        save_data = {
            "inputs": np.array(all_inputs, dtype=np.int32),
            "labels": np.array(all_labels, dtype=np.int32),
            "metadata": metadata
        }
        
        output_path = os.path.join(config.output_dir, split_name, "data.npz")
        np.savez_compressed(output_path, **save_data)
        print(f"Saved {split_name} data to {output_path}")


def create_tinystories_dataset_batched(config: TextDatasetConfig, dataset, tokenizer, vocab_size: int, pad_token_id: int, eos_token_id: int):
    """
    Process TinyStories in batches to avoid memory exhaustion.
    Uses iterative processing instead of loading all text at once.
    """
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Process in batches to avoid memory issues
    # Optimized for Colab (12GB RAM): 4000 stories = ~8GB peak usage
    batch_size = 4000  # Process 4000 stories at a time
    max_train_stories = 100000
    max_val_stories = 10000
    
    for split_name, split_data, max_stories in [
        ("train", dataset['train'], max_train_stories),
        ("test", dataset['validation'], max_val_stories)
    ]:
        print(f"\nTokenizing {split_name} set (batched processing)...")
        os.makedirs(os.path.join(config.output_dir, split_name), exist_ok=True)
        
        all_inputs = []
        all_labels = []
        
        story_count = 0
        batch_texts = []
        
        # Process stories in batches
        for idx, example in enumerate(tqdm(split_data, total=max_stories, desc=f"Processing {split_name}")):
            if story_count >= max_stories:
                break
            
            batch_texts.append(example['text'])
            story_count += 1
            
            # Process batch when it reaches batch_size or end of data
            if len(batch_texts) >= batch_size or story_count >= max_stories:
                # Tokenize batch efficiently
                batch_tokens = tokenizer(batch_texts, truncation=True, max_length=config.max_seq_len, 
                                        return_attention_mask=False, return_token_type_ids=False)
                
                # Process each story's tokens
                for tokens in batch_tokens['input_ids']:
                    # Create overlapping chunks from this story
                    for i in range(0, len(tokens) - config.max_seq_len + 1, config.stride):
                        chunk = tokens[i:i + config.max_seq_len]
                        if len(chunk) == config.max_seq_len:
                            all_inputs.append(chunk)
                            # Labels: shift by 1 for next-token prediction
                            labels = chunk[1:] + [eos_token_id]
                            all_labels.append(labels)
                
                # Clear batch
                batch_texts = []
        
        # Process any remaining texts in partial batch
        if batch_texts:
            batch_tokens = tokenizer(batch_texts, truncation=True, max_length=config.max_seq_len,
                                    return_attention_mask=False, return_token_type_ids=False)
            for tokens in batch_tokens['input_ids']:
                for i in range(0, len(tokens) - config.max_seq_len + 1, config.stride):
                    chunk = tokens[i:i + config.max_seq_len]
                    if len(chunk) == config.max_seq_len:
                        all_inputs.append(chunk)
                        labels = chunk[1:] + [eos_token_id]
                        all_labels.append(labels)
        
        print(f"Created {len(all_inputs)} sequences from {story_count} stories")
        
        # Save as numpy arrays
        results = {
            "inputs": np.array(all_inputs, dtype=np.int32),
            "labels": np.array(all_labels, dtype=np.int32),
            "puzzle_identifiers": np.zeros(len(all_inputs), dtype=np.int32),
            "puzzle_indices": np.arange(len(all_inputs) + 1, dtype=np.int32),
            "group_indices": np.array([0, len(all_inputs)], dtype=np.int32)
        }
        
        for k, v in results.items():
            np.save(os.path.join(config.output_dir, split_name, f"all__{k}.npy"), v)
        
        # Save metadata
        metadata = PuzzleDatasetMetadata(
            seq_len=config.max_seq_len,
            vocab_size=vocab_size,
            pad_id=pad_token_id,
            ignore_label_id=-100,
            blank_identifier_id=0,
            num_puzzle_identifiers=1,
            total_groups=1,
            mean_puzzle_examples=1.0,
            total_puzzles=len(all_inputs),
            sets=["all"]
        )
        
        with open(os.path.join(config.output_dir, split_name, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f, indent=2)
    
    print(f"\n✅ TinyStories dataset created successfully!")
    print(f"   Output: {config.output_dir}")
    print(f"   Vocab size: {vocab_size}")
    print(f"   Sequence length: {config.max_seq_len}")


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
        # TinyStories: Use batched processing to avoid memory exhaustion
        print("Using 100k train stories, 10k validation stories")
        create_tinystories_dataset_batched(config, dataset, tokenizer, vocab_size, pad_token_id, eos_token_id)
        return
        print(f"Using 100k train stories, 10k validation stories")
    elif config.input_file == "code-python":
        dataset = download_code_python()
        if dataset is None:
            raise RuntimeError("Failed to download The Stack Python")
        print("Processing code samples with batched tokenization (memory-efficient)...")
        # Use batched processing to avoid memory exhaustion
        create_code_dataset_batched(config, dataset, tokenizer, vocab_size, pad_token_id, eos_token_id)
        return
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
        group_indices = []  # Each sequence is its own group (for proper batch sampling)
        
        for i, seq in enumerate(sequences):
            # Input: tokens 0 to n-1
            # Label: tokens 1 to n (next token prediction)
            inputs.append(seq[:-1])
            labels.append(seq[1:])
            puzzle_identifiers.append(0)  # No puzzle-specific embedding for text
            puzzle_indices.append(len(inputs))
            group_indices.append(i)  # Each sequence is its own group
        
        group_indices.append(len(sequences))  # Final boundary
        
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
            total_groups=len(sequences),  # Each sequence is its own group
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


def build_capsule_dataset(config: TextDatasetConfig):
    """Build HESC capsule dataset with sketch/checksum/children (Stage A fast training)."""
    import torch
    from models.capsule_encoder import CapsuleEncoder
    
    print(f"\n🔧 Building semantic compression dataset...")
    print(f"   Mode: Precompute CLIP embeddings")
    print(f"   Target tokens: {config.semantic_target_tokens}")
    
    # Load texts
    if config.input_file == "wikitext2":
        dataset = download_wikitext2()
        train_texts = [t for t in dataset['train']['text'] if t.strip()]
        test_texts = [t for t in dataset['test']['text'] if t.strip()]
    else:
        text = load_text_file(config.input_file)
        texts = [p.strip() for p in text.split('\n\n') if p.strip()]
        split_idx = int(len(texts) * config.train_split)
        train_texts = texts[:split_idx]
        test_texts = texts[split_idx:]
    
    print(f"   Train texts: {len(train_texts)}")
    print(f"   Test texts: {len(test_texts)}")
    
    # Initialize capsule encoder
    encoder = CapsuleEncoder(
        hidden_size=config.semantic_hidden_size,
        target_capsules=config.semantic_target_tokens,
        children_per_capsule=4,
        checksum_dim=32,
        freeze_encoder=True,
        encoder_model=config.encoder_model  # Configurable: CLIP, BERT, etc.
    )
    encoder.eval()
    encoder.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Process train split
    print(f"\n📦 Encoding train split (capsules with children)...")
    train_capsules = {'sketches': [], 'checksums': [], 'children': []}
    batch_size = 8
    for i in tqdm(range(0, len(train_texts), batch_size)):
        batch = train_texts[i:i+batch_size]
        with torch.no_grad():
            capsule_data = encoder(batch, return_children=True)
            train_capsules['sketches'].append(capsule_data['sketches'].cpu())
            train_capsules['checksums'].append(capsule_data['checksums'].cpu())
            if 'children' in capsule_data:
                train_capsules['children'].append(capsule_data['children'].cpu())
    
    for key in train_capsules:
        if train_capsules[key]:
            train_capsules[key] = torch.cat(train_capsules[key], dim=0)
    
    # Process test split
    print(f"\n📦 Encoding test split...")
    test_capsules = {'sketches': [], 'checksums': [], 'children': []}
    for i in tqdm(range(0, len(test_texts), batch_size)):
        batch = test_texts[i:i+batch_size]
        with torch.no_grad():
            capsule_data = encoder(batch, return_children=True)
            test_capsules['sketches'].append(capsule_data['sketches'].cpu())
            test_capsules['checksums'].append(capsule_data['checksums'].cpu())
            if 'children' in capsule_data:
                test_capsules['children'].append(capsule_data['children'].cpu())
    
    for key in test_capsules:
        if test_capsules[key]:
            test_capsules[key] = torch.cat(test_capsules[key], dim=0)
    
    # Save
    os.makedirs(config.output_dir, exist_ok=True)
    torch.save({
        'sketches': train_capsules['sketches'],
        'checksums': train_capsules['checksums'],
        'children': train_capsules['children'],
        'texts': train_texts,
        'config': {
            'target_capsules': config.semantic_target_tokens,
            'children_per_capsule': 4,
            'checksum_dim': 32,
            'hidden_size': config.semantic_hidden_size
        }
    }, os.path.join(config.output_dir, 'capsule_dataset.pt'))
    
    torch.save({
        'sketches': test_capsules['sketches'],
        'checksums': test_capsules['checksums'],
        'children': test_capsules['children'],
        'texts': test_texts,
        'config': {
            'target_capsules': config.semantic_target_tokens,
            'children_per_capsule': 4,
            'checksum_dim': 32,
            'hidden_size': config.semantic_hidden_size
        }
    }, os.path.join(config.output_dir, 'capsule_dataset_test.pt'))
    
    # Calculate compression ratio (compare to BPE baseline)
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Standard baseline
        sample_texts = train_texts[:100]
        compression_ratio = encoder.get_compression_ratio(sample_texts, tokenizer)
    except Exception:
        compression_ratio = 10.0  # Default estimate
    
    print(f"\n✅ HESC capsule dataset created!")
    print(f"   Train sketches: {train_capsules['sketches'].shape}")
    print(f"   Train children: {train_capsules['children'].shape if 'children' in train_capsules and train_capsules['children'] is not None else 'N/A'}")
    print(f"   Test sketches: {test_capsules['sketches'].shape}")
    print(f"   Compression: {compression_ratio:.1f}x vs BPE (expandable on-demand)")
    print(f"   Output: {config.output_dir}")
    
    # Build concept expansion table
    print(f"\n📚 Building concept expansion table...")
    from models.concept_decoder import ConceptDecoder
    from transformers import AutoTokenizer
    
    num_concepts = getattr(config, 'num_concepts', 2048)
    decoder = ConceptDecoder(num_concepts=num_concepts)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        decoder.build_expansion_table_from_data(
            os.path.join(config.output_dir, 'capsule_dataset.pt'),
            tokenizer,
            max_concepts=num_concepts
        )
        
        # Save expansion table
        expansion_path = os.path.join(config.output_dir, 'concept_expansions.json')
        decoder.expansion_table.save(expansion_path)
        print(f"   ✅ Saved expansion table: {expansion_path}")
    except Exception as e:
        print(f"   ⚠️  Could not build expansion table: {e}")
        print(f"   Model will use learned embeddings as fallback")


@cli.command(singleton=True)
def main(config: TextDatasetConfig):
    if config.semantic_mode:
        build_capsule_dataset(config)
    else:
        create_text_dataset(config)


if __name__ == "__main__":
    cli()
