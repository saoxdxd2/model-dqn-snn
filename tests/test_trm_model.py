"""
Test TRM model with all advanced features enabled.
Run: python tests/test_trm_model.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from omegaconf import OmegaConf


def test_trm_initialization():
    """Test TRM model initialization with vision-unified mode."""
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1, TinyRecursiveReasoningModel_ACTV1Config
    
    print("\n=== Test 1: TRM Initialization (Vision-Unified) ===")
    
    config = TinyRecursiveReasoningModel_ACTV1Config(
        hidden_size=256,
        num_heads=8,
        num_key_value_heads=2,
        H_layers=0,
        L_layers=2,
        H_cycles=3,
        L_cycles=2,
        seq_len=12,
        vocab_size=2048,
        batch_size=2,
        num_puzzle_identifiers=0,
        expansion=4.0,
        pos_encodings='rope',
        halt_max_steps=16,
        halt_exploration_prob=0.1,
        use_semantic_encoder=True,  # Vision-unified
        encoder_model='openai/clip-vit-base-patch32',  # Smaller model for tests
        target_capsules=12,
        enable_dqn=False,
        enable_memory=False,
        enable_mtp=False
    )
    
    model = TinyRecursiveReasoningModel_ACTV1(config)
    
    assert model.capsule_encoder is not None, "Should have capsule encoder"
    assert model.embed_tokens is None, "Should not have token embeddings in vision-unified"
    print(f"✓ Model initialized with capsule_encoder")
    
    return model


def test_trm_with_dqn():
    """Test TRM with DQN enabled."""
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1, TinyRecursiveReasoningModel_ACTV1Config
    
    print("\n=== Test 2: TRM with DQN ===")
    
    config = TinyRecursiveReasoningModel_ACTV1Config(
        hidden_size=256,
        num_heads=4,
        H_layers=0,
        L_layers=2,
        seq_len=12,
        vocab_size=2048,
        enable_dqn=True,
        dqn_buffer_capacity=1000,
        use_semantic_encoder=True
    )
    
    model = TinyRecursiveReasoningModel_ACTV1(config)
    
    assert hasattr(model, 'q_head_target'), "Should have target Q-head for DQN"
    print("✓ DQN components initialized")
    
    return model


def test_trm_with_memory():
    """Test TRM with Memory Bank enabled."""
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1, TinyRecursiveReasoningModel_ACTV1Config
    
    print("\n=== Test 3: TRM with Memory Bank ===")
    
    config = TinyRecursiveReasoningModel_ACTV1Config(
        hidden_size=256,
        num_heads=4,
        H_layers=0,
        L_layers=2,
        seq_len=12,
        vocab_size=2048,
        enable_memory=True,
        memory_capacity=128,
        use_semantic_encoder=True
    )
    
    model = TinyRecursiveReasoningModel_ACTV1(config)
    
    assert model.memory is not None, "Should have memory bank"
    print(f"✓ Memory Bank initialized: {model.memory.capacity} slots")
    
    return model


def test_trm_with_mtp():
    """Test TRM with Multi-Token Prediction (MTP) enabled."""
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1, TinyRecursiveReasoningModel_ACTV1Config
    
    print("\n=== Test 4: TRM with MTP (Fixed) ===")
    
    config = TinyRecursiveReasoningModel_ACTV1Config(
        hidden_size=256,
        num_heads=4,
        H_layers=0,
        L_layers=2,
        seq_len=12,
        vocab_size=2048,
        enable_mtp=True,
        mtp_num_depths=2,
        use_semantic_encoder=True
    )
    
    model = TinyRecursiveReasoningModel_ACTV1(config)
    
    assert model.mtp_modules is not None, "Should have MTP modules"
    assert len(model.mtp_modules) == 2, f"Should have 2 MTP modules, got {len(model.mtp_modules)}"
    
    # Test MTP module has concept vocab support
    mtp_module = model.mtp_modules[0]
    assert hasattr(mtp_module, 'use_concept_vocab'), "MTP should support concept vocab"
    print(f"✓ MTP initialized with {len(model.mtp_modules)} depth modules")
    print(f"  Concept vocab support: {mtp_module.use_concept_vocab}")
    
    return model


def test_trm_all_features():
    """Test TRM with ALL features enabled."""
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1, TinyRecursiveReasoningModel_ACTV1Config
    
    print("\n=== Test 5: TRM with ALL Features ===")
    
    config = TinyRecursiveReasoningModel_ACTV1Config(
        hidden_size=256,
        num_heads=4,
        H_layers=0,
        L_layers=2,
        seq_len=12,
        vocab_size=2048,
        use_semantic_encoder=True,
        enable_dqn=True,
        enable_memory=True,
        enable_mtp=True,
        mtp_num_depths=2
    )
    
    model = TinyRecursiveReasoningModel_ACTV1(config)
    
    assert model.capsule_encoder is not None, "Should have capsule encoder"
    assert hasattr(model, 'q_head_target'), "Should have DQN"
    assert model.memory is not None, "Should have memory"
    assert model.mtp_modules is not None, "Should have MTP"
    
    print("✓ All features initialized successfully:")
    print("  - Vision-unified (capsule_encoder)")
    print("  - DQN halting")
    print("  - Memory Bank")
    print("  - Multi-Token Prediction")
    
    return model


def test_trm_forward_pass():
    """Test forward pass with synthetic capsule input."""
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1, TinyRecursiveReasoningModel_ACTV1Config
    
    print("\n=== Test 6: Forward Pass ===")
    
    config = TinyRecursiveReasoningModel_ACTV1Config(
        hidden_size=256,
        num_heads=4,
        H_layers=0,
        L_layers=2,
        H_cycles=1,
        L_cycles=1,
        seq_len=12,
        vocab_size=2048,
        use_semantic_encoder=True,
        enable_dqn=False,
        enable_memory=False,
        enable_mtp=False
    )
    
    model = TinyRecursiveReasoningModel_ACTV1(config)
    model.eval()
    
    # Create synthetic input (simulating capsule encoder output)
    batch_size = 2
    seq_len = 12
    inputs = torch.randn(batch_size, seq_len, config.hidden_size)
    puzzle_ids = torch.zeros(batch_size, dtype=torch.long)
    
    batch = {
        "inputs": inputs,
        "puzzle_identifiers": puzzle_ids
    }
    
    with torch.no_grad():
        output, carry = model(batch)
    
    assert "logits" in output or isinstance(output, torch.Tensor), "Should have logits output"
    print(f"✓ Forward pass successful")
    
    return model, output


def test_config_from_yaml():
    """Test loading config from actual YAML file."""
    print("\n=== Test 7: Load Config from YAML ===")
    
    config_path = Path(__file__).parent.parent / "config" / "arch" / "multimodal_hesc.yaml"
    
    if not config_path.exists():
        print("⚠️  Config file not found, skipping")
        return None
    
    cfg = OmegaConf.load(config_path)
    
    assert cfg.enable_dqn == True, "DQN should be enabled in config"
    assert cfg.enable_memory == True, "Memory should be enabled"
    assert cfg.enable_mtp == True, "MTP should be enabled"
    
    print("✓ Config loaded successfully:")
    print(f"  DQN: {cfg.enable_dqn}")
    print(f"  Memory: {cfg.enable_memory}")
    print(f"  MTP: {cfg.enable_mtp}")
    print(f"  Curiosity: {cfg.enable_count_curiosity}")
    
    return cfg


def main():
    """Run all tests."""
    print("=" * 70)
    print("TESTING: TRM Model with All Features")
    print("=" * 70)
    
    try:
        test_trm_initialization()
        test_trm_with_dqn()
        test_trm_with_memory()
        test_trm_with_mtp()
        test_trm_all_features()
        test_trm_forward_pass()
        test_config_from_yaml()
        
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
