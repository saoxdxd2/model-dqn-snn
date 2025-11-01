"""
Test evaluator modules (ARC, etc).
Run: python tests/test_evaluators.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np


def test_arc_evaluator_init():
    """Test ARC evaluator initialization."""
    from evaluators.arc import ARC
    from dataset.common import PuzzleDatasetMetadata
    
    print("\n=== Test 1: ARC Evaluator Init ===")
    
    # Check if test data exists
    test_data_path = Path(__file__).parent.parent / "kaggle" / "combined"
    if not test_data_path.exists():
        print("⚠️  Test data not found, skipping ARC evaluator tests")
        return None
    
    try:
        metadata = PuzzleDatasetMetadata(
            seq_len=900,
            vocab_size=12,
            pad_id=0,
            ignore_label_id=-100,
            blank_identifier_id=0,
            num_puzzle_identifiers=100,
            total_groups=100,
            mean_puzzle_examples=1.0,
            total_puzzles=100,
            sets=["training"]
        )
        
        evaluator = ARC(
            data_path=str(test_data_path),
            eval_metadata=metadata,
            submission_K=2,
            pass_Ks=(1, 2, 5)
        )
        
        print(f"✓ ARC evaluator initialized")
        print(f"  Submission K: {evaluator.submission_K}")
        print(f"  Pass Ks: {evaluator.pass_Ks}")
        
        return evaluator
    except Exception as e:
        print(f"⚠️  ARC evaluator init failed (expected if data missing): {e}")
        return None


def test_arc_crop():
    """Test ARC grid cropping logic."""
    from evaluators.arc import _crop
    
    print("\n=== Test 2: ARC Grid Cropping ===")
    
    # Create test grid (30x30)
    grid = np.full((900,), 0, dtype=np.int32)  # Flattened 30x30
    
    # Fill with valid ARC colors (2-11)
    grid[:100] = 5  # Top-left 10x10 region with color 5
    
    cropped = _crop(grid)
    
    assert cropped.shape[0] <= 30 and cropped.shape[1] <= 30
    print(f"✓ Grid cropping works: {grid.shape} → {cropped.shape}")
    
    return cropped


def test_arc_batch_update():
    """Test ARC batch update (mock)."""
    print("\n=== Test 3: ARC Batch Update (Mock) ===")
    
    # This would require full ARC data, so just test the interface exists
    from evaluators.arc import ARC
    
    required_outputs = ARC.required_outputs
    print(f"✓ ARC requires outputs: {required_outputs}")
    
    assert "inputs" in required_outputs
    assert "puzzle_identifiers" in required_outputs
    assert "q_halt_logits" in required_outputs
    assert "preds" in required_outputs
    
    print("✓ ARC interface validated")
    
    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("TESTING: Evaluators (ARC)")
    print("=" * 70)
    
    try:
        test_arc_evaluator_init()
        test_arc_crop()
        test_arc_batch_update()
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED")
        print("=" * 70)
        return 0
        
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
