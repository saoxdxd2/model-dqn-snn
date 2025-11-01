"""
Test loss functions and advanced features (DQN, memory, curiosity).
Run: python tests/test_losses.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


def test_act_loss_head_basic():
    """Test basic ACTLossHead initialization."""
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1, TinyRecursiveReasoningModel_ACTV1Config
    from models.losses import ACTLossHead
    
    print("\n=== Test 1: ACTLossHead Basic ===")
    
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
        use_semantic_encoder=True,
        enable_dqn=False
    )
    
    model = TinyRecursiveReasoningModel_ACTV1(config)
    loss_head = ACTLossHead(model, loss_type='stablemax_cross_entropy', enable_dqn=False)
    
    assert loss_head is not None
    assert loss_head.enable_dqn == False
    print("✓ ACTLossHead initialized (DQN disabled)")
    
    return loss_head


def test_act_loss_head_with_dqn():
    """Test ACTLossHead with DQN enabled."""
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1, TinyRecursiveReasoningModel_ACTV1Config
    from models.losses import ACTLossHead
    
    print("\n=== Test 2: ACTLossHead with DQN ===")
    
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
        use_semantic_encoder=True,
        enable_dqn=True,
        dqn_buffer_capacity=1000
    )
    
    model = TinyRecursiveReasoningModel_ACTV1(config)
    loss_head = ACTLossHead(model, loss_type='stablemax_cross_entropy', enable_dqn=True)
    
    assert loss_head.enable_dqn == True
    assert hasattr(loss_head, 'replay_buffer'), "Should have replay buffer"
    assert hasattr(loss_head, 'intrinsic_reward'), "Should have intrinsic reward module"
    print(f"✓ DQN components initialized")
    print(f"  Replay buffer capacity: {loss_head.replay_buffer.capacity}")
    
    return loss_head


def test_intrinsic_reward():
    """Test intrinsic reward module (curiosity)."""
    from models.intrinsic_reward import IntrinsicRewardModule
    
    print("\n=== Test 3: Intrinsic Reward (Curiosity) ===")
    
    reward_module = IntrinsicRewardModule(
        hidden_size=256,
        enable_count=True,
        enable_rnd=True,
        count_beta=0.05,
        rnd_weight=0.1
    )
    
    # Test with synthetic states
    batch_size = 4
    states = torch.randn(batch_size, 256)
    
    rewards = reward_module.compute_intrinsic_reward(states)
    
    assert rewards.shape == (batch_size,), f"Expected shape ({batch_size},), got {rewards.shape}"
    assert rewards.min() >= 0, "Rewards should be non-negative"
    print(f"✓ Intrinsic rewards computed: {rewards.mean():.4f} avg")
    
    return reward_module


def test_memory_bank():
    """Test memory bank read/write."""
    from models.memory_bank import AssociativeMemoryBank
    
    print("\n=== Test 4: Memory Bank ===")
    
    memory = AssociativeMemoryBank(
        capacity=128,
        hidden_size=256,
        num_heads=4
    )
    
    # Test write
    batch_size = 2
    states = torch.randn(batch_size, 256)
    rewards = torch.tensor([0.5, 0.8])
    
    memory.write(states, rewards, threshold=0.3)
    
    # Test read
    query = torch.randn(1, 256)
    retrieved = memory.read(query)
    
    assert retrieved.shape == (1, 256), f"Expected shape (1, 256), got {retrieved.shape}"
    print("✓ Memory bank write/read successful")
    
    # Test cache
    memory.eval()  # Enable cache
    _ = memory.read(query)
    stats = memory.get_cache_stats()
    print(f"  Cache: {stats['hit_rate']:.2%} hit rate")
    
    return memory


def test_replay_buffer():
    """Test DQN replay buffer."""
    from models.losses import DQNReplayBuffer
    
    print("\n=== Test 5: DQN Replay Buffer ===")
    
    buffer = DQNReplayBuffer(capacity=1000, min_size=10)
    
    # Add samples
    for i in range(20):
        state = torch.randn(1, 256)
        action = torch.tensor([0 if i % 2 == 0 else 1])
        reward = torch.tensor([0.5])
        next_state = torch.randn(1, 256)
        done = torch.tensor([False])
        
        buffer.add(state, action, reward, next_state, done)
    
    assert len(buffer) == 20, f"Buffer should have 20 samples, has {len(buffer)}"
    assert buffer.can_sample(), "Buffer should be ready for sampling"
    
    # Test sampling
    batch = buffer.sample(batch_size=4)
    assert batch['states'].shape == (4, 1, 256)
    print(f"✓ Replay buffer: {len(buffer)} samples, sampling works")
    
    return buffer


def test_q_heads():
    """Test different Q-head architectures."""
    from models.q_heads import create_q_head
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1Config
    
    print("\n=== Test 6: Q-Head Architectures ===")
    
    for q_type in ['mlp', 'rnn', 'mini_attention']:
        config = TinyRecursiveReasoningModel_ACTV1Config(
            hidden_size=256,
            q_head_type=q_type,
            q_head_hidden_size=128,
            q_head_num_layers=1
        )
        
        q_head = create_q_head(config)
        
        # Test forward pass
        state = torch.randn(2, 256)
        q_values = q_head(state)
        
        assert q_values.shape == (2, 2), f"Expected shape (2, 2), got {q_values.shape}"
        print(f"  ✓ {q_type:15} Q-head works")
    
    print("✓ All Q-head types functional")


def test_deep_supervision():
    """Test deep supervision with multiple checkpoints."""
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1, TinyRecursiveReasoningModel_ACTV1Config
    from models.losses import ACTLossHead
    
    print("\n=== Test 7: Deep Supervision ===")
    
    config = TinyRecursiveReasoningModel_ACTV1Config(
        hidden_size=256,
        num_heads=4,
        H_layers=0,
        L_layers=2,
        H_cycles=3,  # More cycles for supervision
        seq_len=12,
        vocab_size=2048,
        use_semantic_encoder=True
    )
    
    model = TinyRecursiveReasoningModel_ACTV1(config)
    loss_head = ACTLossHead(
        model,
        loss_type='stablemax_cross_entropy',
        enable_dqn=False,
        deep_supervision_steps=2  # Supervise every 2 steps
    )
    
    assert loss_head.deep_supervision_steps == 2
    print("✓ Deep supervision configured (2-step checkpoints)")
    
    return loss_head


def main():
    """Run all tests."""
    print("=" * 70)
    print("TESTING: Losses and Advanced Features")
    print("=" * 70)
    
    try:
        test_act_loss_head_basic()
        test_act_loss_head_with_dqn()
        test_intrinsic_reward()
        test_memory_bank()
        test_replay_buffer()
        test_q_heads()
        test_deep_supervision()
        
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
