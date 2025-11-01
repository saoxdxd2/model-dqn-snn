"""
Run all tests in sequence.
Run: python tests/run_all_tests.py
"""

import sys
import subprocess
from pathlib import Path

def run_test(test_file):
    """Run a single test file."""
    print(f"\n{'='*70}")
    print(f"Running: {test_file.name}")
    print(f"{'='*70}")
    
    result = subprocess.run(
        [sys.executable, str(test_file)],
        capture_output=False,
        text=True
    )
    
    return result.returncode == 0

def main():
    """Run all tests."""
    tests_dir = Path(__file__).parent
    
    test_files = [
        # Core Components
        tests_dir / "test_text_renderer.py",
        tests_dir / "test_capsule_encoder.py",
        tests_dir / "test_dataset_pipeline.py",
        
        # Models & Features
        tests_dir / "test_trm_model.py",
        tests_dir / "test_losses.py",
        tests_dir / "test_concept_vocab.py",
        
        # Evaluators & Utils
        tests_dir / "test_evaluators.py",
        tests_dir / "test_utils.py",
        
        # Integration
        tests_dir / "test_integration.py",
        
        # Scripts
        tests_dir / "test_scripts.py",
    ]
    
    print("\n" + "="*70)
    print("RUNNING ALL TESTS")
    print("="*70)
    
    results = {}
    for test_file in test_files:
        if test_file.exists():
            results[test_file.name] = run_test(test_file)
        else:
            print(f"⚠️  Test file not found: {test_file}")
            results[test_file.name] = False
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:10} {test_name}")
    
    all_passed = all(results.values())
    
    print("="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*70)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())
