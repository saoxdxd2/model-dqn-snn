
import os
from pathlib import Path
from dataset.build_multimodal_dataset import MultimodalDatasetBuilder, MultimodalDatasetConfig

def test_load_text():
    os.makedirs("test_data", exist_ok=True)
    with open("test_data/sample.txt", "w") as f:
        f.write("This is a test sample.\nAnother line.")
        
    config = MultimodalDatasetConfig(
        source_paths=["test_data/sample.txt"],
        output_dir="test_output"
    )
    builder = MultimodalDatasetBuilder(config)
    
    print("Testing _load_text_dataset_streaming...")
    samples = list(builder._load_text_dataset_streaming("test_data/sample.txt"))
    print(f"Loaded {len(samples)} samples")
    for s in samples:
        print(f"Sample: {s.text}")

if __name__ == "__main__":
    test_load_text()
