import torch
import numpy as np
from utils.data_processing import process_vision_batch
from dataclasses import dataclass
from typing import Optional, Union, Any

@dataclass
class MockSample:
    image: Optional[Union[np.ndarray, Any]] = None
    grid: Optional[np.ndarray] = None
    text: Optional[str] = None
    label: Optional[np.ndarray] = None

def test_process_vision_batch():
    print("Testing process_vision_batch...")
    
    # 1. Test Image
    print("1. Testing Image input...")
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    batch = {'raw_samples': [MockSample(image=img)]}
    out = process_vision_batch(batch)
    assert 'images' in out
    assert out['images'].shape == (1, 3, 224, 224)
    print("   Image input passed.")

    # 2. Test Grid
    print("2. Testing Grid input...")
    grid = np.random.randint(0, 10, (10, 10), dtype=np.int32)
    batch = {'raw_samples': [MockSample(grid=grid)]}
    out = process_vision_batch(batch)
    assert 'images' in out
    assert out['images'].shape == (1, 3, 224, 224)
    print("   Grid input passed.")

    # 3. Test Mixed Batch
    print("3. Testing Mixed Batch...")
    batch = {'raw_samples': [MockSample(image=img), MockSample(grid=grid)]}
    out = process_vision_batch(batch)
    assert out['images'].shape == (2, 3, 224, 224)
    print("   Mixed batch passed.")

    print("✅ All tests passed!")

if __name__ == "__main__":
    test_process_vision_batch()
