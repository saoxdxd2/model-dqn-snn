"""
Integration tests: End-to-end pipeline testing.
Run: python tests/test_integration.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from PIL import Image


def test_text_to_capsule_pipeline():
    """Test: Text → Render → Image → CapsuleEncoder → Capsules."""
    from models.text_renderer import TextRenderer
    from models.capsule_encoder import CapsuleEncoder
    
    print("\n=== Test 1: Text → Render → Capsule Pipeline ===")
    
    renderer = TextRenderer(width=512, height=384)
    text = "The quick brown fox jumps over the lazy dog."
    rendered_image = renderer.render_code(text)
    
    encoder = CapsuleEncoder(
        hidden_size=768,
        target_capsules=12,
        encoder_model="openai/clip-vit-base-patch32"
    )
    
    result = encoder(images=[rendered_image])
    
    assert 'sketches' in result
    assert result['sketches'].shape == (1, 12, 768)
    print("✓ Text-to-capsule pipeline works end-to-end")
    
    return result


def test_training_step_simulation():
    """Test: Simulate one training step with all features."""
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1, TinyRecursiveReasoningModel_ACTV1Config
    from models.losses import ACTLossHead
    
    print("\n=== Test 2: Training Step Simulation ===")
    
    config = TinyRecursiveReasoningModel_ACTV1Config(
        hidden_size=256,
        num_heads=4,
        H_layers=0,
        L_layers=2,
        H_cycles=2,
        L_cycles=1,
        seq_len=12,
        vocab_size=2048,
        use_semantic_encoder=True,
        enable_dqn=True,
        enable_memory=True,
        enable_mtp=True
    )
    
    model = TinyRecursiveReasoningModel_ACTV1(config)
    loss_head = ACTLossHead(model, loss_type='stablemax_cross_entropy', enable_dqn=True)
    
    batch_size = 2
    batch = {
        "inputs": torch.randn(batch_size, 12, 256),
        "labels": torch.randint(0, 2048, (batch_size, 12)),
        "puzzle_identifiers": torch.zeros(batch_size, dtype=torch.long)
    }
    
    model.train()
    loss_outputs = loss_head(batch)
    
    assert 'total_loss' in loss_outputs
    print(f"✓ Training step: loss={loss_outputs['total_loss'].item():.4f}")
    
    return loss_outputs


def test_inference_mode():
    """Test: Model in inference/eval mode."""
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1, TinyRecursiveReasoningModel_ACTV1Config
    
    print("\n=== Test 3: Inference Mode ===")
    
    config = TinyRecursiveReasoningModel_ACTV1Config(
        hidden_size=256,
        num_heads=4,
        H_layers=0,
        L_layers=2,
        seq_len=12,
        vocab_size=2048,
        use_semantic_encoder=True
    )
    
    model = TinyRecursiveReasoningModel_ACTV1(config)
    model.eval()
    
    batch = {
        "inputs": torch.randn(1, 12, 256),
        "puzzle_identifiers": torch.zeros(1, dtype=torch.long)
    }
    
    with torch.no_grad():
        output, _ = model(batch)
    
    print("✓ Inference mode works (no grad computation)")
    return output


def main():
    """Run all tests."""
    print("=" * 70)
    print("TESTING: End-to-End Integration")
    print("=" * 70)
    
    try:
        test_text_to_capsule_pipeline()
        test_training_step_simulation()
        test_inference_mode()
        
        print("\n" + "=" * 70)
        print("✅ ALL INTEGRATION TESTS PASSED")
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
