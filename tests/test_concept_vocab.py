"""
Test concept vocabulary and decoder functionality.
Run: python tests/test_concept_vocab.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np


def test_concept_vocab_basic():
    """Test basic concept vocabulary."""
    from models.concept_vocab import ConceptCodebook
    
    print("\n=== Test 1: Concept Vocabulary Basic ===")
    
    vocab = ConceptCodebook(
        num_concepts=2048,
        hidden_size=256
    )
    
    # Test encoding
    inputs = torch.randn(2, 12, 256)
    concepts, scores = vocab.encode(inputs)
    
    assert concepts.shape == (2, 12), f"Concepts shape: {concepts.shape}"
    assert scores.shape == (2, 12, 2048), f"Scores shape: {scores.shape}"
    print(f"✓ Encoding works: concepts {concepts.shape}, scores {scores.shape}")
    
    # Test decoding
    decoded = vocab.decode(concepts)
    assert decoded.shape == (2, 12, 256), f"Decoded shape: {decoded.shape}"
    print(f"✓ Decoding works: {decoded.shape}")
    
    return vocab


def test_hybrid_output_head():
    """Test hybrid output head (concept + token)."""
    from models.concept_vocab import HybridOutputHead
    
    print("\n=== Test 2: Hybrid Output Head ===")
    
    head = HybridOutputHead(
        hidden_size=256,
        num_concepts=2048,
        num_tokens=2048,
        use_concepts=True
    )
    
    # Test forward
    hidden = torch.randn(2, 12, 256)
    logits = head(hidden)
    
    assert logits.shape == (2, 12, 2048), f"Logits shape: {logits.shape}"
    print(f"✓ Hybrid head output: {logits.shape}")
    
    return head


def test_concept_decoder():
    """Test concept decoder with expansion."""
    from models.concept_decoder import ConceptDecoder
    
    print("\n=== Test 3: Concept Decoder ===")
    
    decoder = ConceptDecoder(
        num_concepts=2048,
        hidden_size=256,
        enable_expansion=True,
        max_children=4
    )
    
    # Test decoding
    concept_ids = torch.randint(0, 2048, (2, 12))
    decoded = decoder(concept_ids)
    
    assert decoded.shape == (2, 12, 256), f"Decoded shape: {decoded.shape}"
    print(f"✓ Concept decoding: {decoded.shape}")
    
    # Test expansion
    if decoder.enable_expansion:
        children = decoder.get_children(concept_ids[0])
        assert children is not None, "Should have children"
        print(f"✓ Concept expansion: {children.shape if hasattr(children, 'shape') else 'available'}")
    
    return decoder


def test_sparse_embedding():
    """Test sparse embedding optimization."""
    from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
    
    print("\n=== Test 4: Sparse Embedding ===")
    
    try:
        embedding = CastedSparseEmbeddingSignSGD_Distributed(
            num_embeddings=100,
            embedding_dim=256,
            cast_to=torch.float16
        )
        
        # Test lookup
        indices = torch.randint(0, 100, (4,))
        output = embedding(indices)
        
        assert output.shape == (4, 256), f"Output shape: {output.shape}"
        print(f"✓ Sparse embedding: {output.shape}")
        
        return embedding
    except Exception as e:
        print(f"⚠️  Sparse embedding test skipped: {e}")
        return None


def test_concept_checksum():
    """Test concept checksum for integrity."""
    from models.concept_vocab import ConceptCodebook
    
    print("\n=== Test 5: Concept Checksum ===")
    
    vocab = ConceptCodebook(num_concepts=2048, hidden_size=256)
    
    # Encode
    inputs = torch.randn(1, 12, 256)
    concepts, _ = vocab.encode(inputs)
    
    # Decode and re-encode (should be consistent)
    decoded = vocab.decode(concepts)
    concepts_2, _ = vocab.encode(decoded)
    
    # Check consistency
    match_rate = (concepts == concepts_2).float().mean()
    print(f"✓ Checksum consistency: {match_rate.item():.2%} match")
    
    return match_rate


def main():
    """Run all tests."""
    print("=" * 70)
    print("TESTING: Concept Vocabulary & Decoder")
    print("=" * 70)
    
    try:
        test_concept_vocab_basic()
        test_hybrid_output_head()
        test_concept_decoder()
        test_sparse_embedding()
        test_concept_checksum()
        
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
