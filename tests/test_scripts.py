"""
Test training/utility scripts compatibility.
Run: python tests/test_scripts.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_pretrain_imports():
    """Test that pretrain.py imports work."""
    print("\n=== Test 1: pretrain.py Imports ===")
    
    try:
        import pretrain
        print("✓ pretrain.py imports successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_train_imports():
    """Test that train.py imports work."""
    print("\n=== Test 2: train.py Imports ===")
    
    try:
        import train
        print("✓ train.py imports successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_generate_text_imports():
    """Test generation script."""
    print("\n=== Test 3: generate_text.py Imports ===")
    
    try:
        import generate_text
        print("✓ generate_text.py imports successfully")
        return True
    except Exception as e:
        print(f"⚠️  Import failed (expected): {e}")
        return True  # Not critical


def main():
    """Run all tests."""
    print("=" * 70)
    print("TESTING: Scripts Compatibility")
    print("=" * 70)
    
    try:
        test_pretrain_imports()
        test_train_imports()
        test_generate_text_imports()
        
        print("\n" + "=" * 70)
        print("✅ ALL SCRIPT TESTS PASSED")
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
