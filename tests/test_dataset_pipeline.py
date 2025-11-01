"""
Test dataset pipeline with text rendering integration.
Run: python tests/test_dataset_pipeline.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_text_rendering_integration():
    """Test that text samples are rendered to images in pipeline."""
    from dataset.build_multimodal_dataset import MultimodalDatasetBuilder, MultimodalDatasetConfig
    from dataset.base_builder import DataSample, ModalityType
    
    print("\n=== Test 1: Text Rendering in Pipeline ===")
    
    config = MultimodalDatasetConfig(
        source_paths=[],
        output_dir="test_output",
        render_text_to_image=True,
        text_image_width=512,
        text_image_height=384,
        image_size=224
    )
    
    builder = MultimodalDatasetBuilder(config)
    
    # Create test sample with text
    sample = DataSample(
        sample_id="test_001",
        modality=ModalityType.TEXT,
        text="Hello World! Testing text-to-image rendering in the dataset pipeline.",
        metadata={"test": True}
    )
    
    print(f"Before: text='{sample.text[:30]}...', image={sample.image}")
    
    # Preprocess should render text to image
    processed = builder.preprocess_sample(sample)
    
    print(f"After: text='{processed.text[:30] if processed.text else None}...', image={type(processed.image)}")
    
    assert processed.image is not None, "Image should be rendered"
    assert processed.metadata.get('text_rendered') == True, "Should mark as rendered"
    print("✓ Text rendered to image successfully")
    
    return processed


def test_image_only_sample():
    """Test that image-only samples are not affected by text rendering."""
    from dataset.build_multimodal_dataset import MultimodalDatasetBuilder, MultimodalDatasetConfig
    from dataset.base_builder import DataSample, ModalityType
    from PIL import Image
    import numpy as np
    
    print("\n=== Test 2: Image-Only Sample ===")
    
    config = MultimodalDatasetConfig(
        source_paths=[],
        output_dir="test_output",
        render_text_to_image=True,
        image_size=224
    )
    
    builder = MultimodalDatasetBuilder(config)
    
    # Create test sample with image (no text)
    test_img = Image.new('RGB', (512, 512), color='red')
    sample = DataSample(
        sample_id="test_002",
        modality=ModalityType.IMAGE,
        image=test_img,
        metadata={"test": True}
    )
    
    print(f"Before: image size={sample.image.size}")
    
    processed = builder.preprocess_sample(sample)
    
    print(f"After: image shape={processed.image.shape if hasattr(processed.image, 'shape') else processed.image.size}")
    
    assert processed.image is not None, "Image should be preserved"
    assert processed.metadata.get('text_rendered') != True, "Should not mark as text rendered"
    print("✓ Image-only sample processed correctly")
    
    return processed


def test_mixed_sample():
    """Test sample with both text and image (should keep image, not render text)."""
    from dataset.build_multimodal_dataset import MultimodalDatasetBuilder, MultimodalDatasetConfig
    from dataset.base_builder import DataSample, ModalityType
    from PIL import Image
    
    print("\n=== Test 3: Mixed Text+Image Sample ===")
    
    config = MultimodalDatasetConfig(
        source_paths=[],
        output_dir="test_output",
        render_text_to_image=True,
        image_size=224
    )
    
    builder = MultimodalDatasetBuilder(config)
    
    test_img = Image.new('RGB', (256, 256), color='blue')
    sample = DataSample(
        sample_id="test_003",
        modality=ModalityType.MULTIMODAL,
        text="Caption: A blue square",
        image=test_img,
        metadata={"test": True}
    )
    
    print(f"Before: text='{sample.text}', image={sample.image.size}")
    
    processed = builder.preprocess_sample(sample)
    
    print(f"After: image present={processed.image is not None}, text_rendered={processed.metadata.get('text_rendered')}")
    
    assert processed.image is not None, "Image should be preserved"
    assert processed.metadata.get('text_rendered') != True, "Should not render text when image exists"
    print("✓ Mixed sample processed correctly (image takes priority)")
    
    return processed


def test_text_rendering_disabled():
    """Test that text rendering can be disabled via config."""
    from dataset.build_multimodal_dataset import MultimodalDatasetBuilder, MultimodalDatasetConfig
    from dataset.base_builder import DataSample, ModalityType
    
    print("\n=== Test 4: Text Rendering Disabled ===")
    
    config = MultimodalDatasetConfig(
        source_paths=[],
        output_dir="test_output",
        render_text_to_image=False,  # Disabled
        image_size=224
    )
    
    builder = MultimodalDatasetBuilder(config)
    
    sample = DataSample(
        sample_id="test_004",
        modality=ModalityType.TEXT,
        text="This text should NOT be rendered to image.",
        metadata={"test": True}
    )
    
    print(f"Before: text='{sample.text[:30]}...', image={sample.image}")
    
    processed = builder.preprocess_sample(sample)
    
    print(f"After: image={processed.image}, text_rendered={processed.metadata.get('text_rendered')}")
    
    assert processed.image is None, "Image should remain None when rendering disabled"
    assert processed.metadata.get('text_rendered') != True, "Should not mark as rendered"
    print("✓ Text rendering correctly disabled")
    
    return processed


def main():
    """Run all tests."""
    print("=" * 70)
    print("TESTING: Dataset Pipeline with Text Rendering")
    print("=" * 70)
    
    try:
        test_text_rendering_integration()
        test_image_only_sample()
        test_mixed_sample()
        
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
