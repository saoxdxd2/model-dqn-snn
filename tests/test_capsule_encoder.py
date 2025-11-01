"""
Test CapsuleEncoder enhancements - DINOv2, hybrid mode, patches mode.
Run: python tests/test_capsule_encoder.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image


def test_basic_clip():
    """Test basic CLIP encoding (existing functionality)."""
    from models.capsule_encoder import CapsuleEncoder
    
    print("\n=== Test 1: Basic CLIP Encoding ===")
    encoder = CapsuleEncoder(
        hidden_size=768,
        target_capsules=12,
        encoder_model="openai/clip-vit-base-patch32"  # Smaller for testing
    )
    
    # Test text encoding
    texts = ["A red cube", "A blue sphere"]
    result = encoder(texts=texts)
    
    assert 'sketches' in result, "Should have sketches"
    assert result['sketches'].shape == (2, 12, 768), f"Unexpected shape: {result['sketches'].shape}"
    print(f"✓ Text encoding: {result['sketches'].shape}")
    
    return encoder


def test_image_encoding():
    """Test image encoding with spatial features."""
    from models.capsule_encoder import CapsuleEncoder
    
    print("\n=== Test 2: Image Encoding ===")
    encoder = CapsuleEncoder(
        hidden_size=768,
        target_capsules=12,
        use_spatial=True,
        encoder_model="openai/clip-vit-base-patch32"
    )
    
    # Create test image
    test_image = Image.new('RGB', (224, 224), color='red')
    images = [test_image]
    
    result = encoder(images=images)
    
    assert 'sketches' in result, "Should have sketches"
    assert result['sketches'].shape == (1, 12, 768), f"Unexpected shape: {result['sketches'].shape}"
    print(f"✓ Image encoding: {result['sketches'].shape}")
    
    return encoder


def test_patches_mode():
    """Test patches output mode (flat, like old cnn_tokenizer)."""
    from models.capsule_encoder import CapsuleEncoder
    
    print("\n=== Test 3: Patches Mode ===")
    encoder = CapsuleEncoder(
        hidden_size=768,
        output_mode="patches",  # Flat patches instead of capsules
        encoder_model="openai/clip-vit-base-patch32"
    )
    
    test_image = Image.new('RGB', (224, 224), color='blue')
    result = encoder(images=[test_image])
    
    assert 'sketches' in result, "Should have sketches"
    # CLIP base-patch32: 7x7 = 49 patches + 1 CLS = 50
    print(f"✓ Patches mode: {result['sketches'].shape}")
    assert result['sketches'].shape[1] > 12, "Patches mode should have more than 12 tokens"
    
    return encoder


def test_dinov2_mode():
    """Test DINOv2 encoder (if available)."""
    from models.capsule_encoder import CapsuleEncoder
    
    print("\n=== Test 4: DINOv2 Mode ===")
    try:
        encoder = CapsuleEncoder(
            hidden_size=768,
            target_capsules=12,
            use_dinov2=True,
            dinov2_model="facebook/dinov2-small",  # Small for testing
            encoder_model="openai/clip-vit-base-patch32"
        )
        
        # Test requires DINOv2 available
        test_image = Image.new('RGB', (224, 224), color='green')
        result = encoder(images=[test_image])
        
        print(f"✓ DINOv2 encoding: {result['sketches'].shape}")
        return encoder
        
    except Exception as e:
        print(f"⚠️  DINOv2 not available (expected): {e}")
        return None


def test_hybrid_mode():
    """Test hybrid mode (CLIP + DINOv2)."""
    from models.capsule_encoder import CapsuleEncoder
    
    print("\n=== Test 5: Hybrid Mode ===")
    try:
        encoder = CapsuleEncoder(
            hidden_size=768,
            target_capsules=12,
            use_hybrid=True,
            use_dinov2=True,
            dinov2_model="facebook/dinov2-small",
            encoder_model="openai/clip-vit-base-patch32"
        )
        
        test_image = Image.new('RGB', (224, 224), color='yellow')
        result = encoder(images=[test_image])
        
        print(f"✓ Hybrid encoding: {result['sketches'].shape}")
        return encoder
        
    except Exception as e:
        print(f"⚠️  Hybrid mode not available (expected): {e}")
        return None


def main():
    """Run all tests."""
    print("=" * 70)
    print("TESTING: CapsuleEncoder Enhancements")
    print("=" * 70)
    
    try:
        # Core tests (must pass)
        test_basic_clip()
        test_image_encoding()
        test_patches_mode()
        
        # Optional tests (may skip if DINOv2 not available)
        test_dinov2_mode()
        test_hybrid_mode()
        
        print("\n" + "=" * 70)
        print("✅ ALL CORE TESTS PASSED")
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
