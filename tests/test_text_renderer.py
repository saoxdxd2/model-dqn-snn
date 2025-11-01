"""
Test text_renderer.py - Plain Python, no framework needed.
Run: python tests/test_text_renderer.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_text_renderer():
    """Test basic text rendering."""
    from models.text_renderer import TextRenderer
    
    print("\n=== Test 1: Basic Text Rendering ===")
    renderer = TextRenderer(width=512, height=384)
    
    text = "Hello World!\nThis is a test of the text rendering system.\n\nIt should wrap text nicely."
    image = renderer.render_code(text)
    
    assert image is not None, "Image should not be None"
    assert image.size == (512, 384), f"Image size should be (512, 384), got {image.size}"
    print("✓ Basic text rendering works")
    
    return image


def test_code_rendering():
    """Test syntax-highlighted code rendering."""
    from models.text_renderer import TextRenderer
    
    print("\n=== Test 2: Code Rendering ===")
    renderer = TextRenderer(width=512, height=384)
    
    code = """def hello_world():
    print("Hello, World!")
    return 42

result = hello_world()
"""
    
    image = renderer.render_code(code, language='python')
    
    assert image is not None, "Code image should not be None"
    assert image.size == (512, 384), f"Image size should be (512, 384), got {image.size}"
    print("✓ Code rendering works")
    
    return image


def test_auto_detection():
    """Test auto-detection of code vs text."""
    from models.text_renderer import TextRenderer
    
    print("\n=== Test 3: Auto-Detection ===")
    renderer = TextRenderer()
    
    # Should detect as code
    code_sample = "def foo():\n    return 42"
    image1 = renderer.render_code(code_sample)
    print("✓ Code auto-detection works")
    
    # Should detect as plain text
    text_sample = "This is just plain text without any code."
    image2 = renderer.render_code(text_sample)
    print("✓ Plain text rendering works")
    
    return image1, image2


def test_batch_rendering():
    """Test batch rendering."""
    from models.text_renderer import TextRenderer
    
    print("\n=== Test 4: Batch Rendering ===")
    renderer = TextRenderer()
    
    texts = [
        "Sample 1: Plain text",
        "def sample_2():\n    return 'code'",
        "Sample 3: More text\nWith multiple lines"
    ]
    
    images = renderer.render_batch(texts)
    
    assert len(images) == 3, f"Should have 3 images, got {len(images)}"
    print(f"✓ Batch rendering works: {len(images)} images")
    
    return images


def main():
    """Run all tests."""
    print("=" * 70)
    print("TESTING: Text Renderer Module")
    print("=" * 70)
    
    try:
        # Run tests
        test_text_renderer()
        test_code_rendering()
        test_auto_detection()
        test_batch_rendering()
        
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
