#!/usr/bin/env python3
"""
Generation 1 Standalone Test - Direct imports to avoid dependency issues.
Tests the core MPC transformer functionality with minimal dependencies.
"""

import sys
import time
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_transformer():
    """Test basic transformer functionality."""
    print("=" * 60)
    print("Generation 1: Basic Transformer Test")
    print("=" * 60)
    
    try:
        # Direct import
        sys.path.insert(0, str(Path(__file__).parent / "src" / "secure_mpc_transformer" / "models"))
        from basic_transformer import (
            BasicSecureTransformer, 
            BasicTransformerConfig
        )
        
        # Create configuration
        config = BasicTransformerConfig(
            model_name="bert-base-test",
            max_sequence_length=128,
            hidden_size=256,
            num_parties=3,
            party_id=0,
            security_level=128
        )
        
        print(f"✓ Configuration created: {config.model_name}")
        
        # Initialize transformer
        transformer = BasicSecureTransformer(config)
        print(f"✓ Transformer initialized")
        
        # Test model info
        model_info = transformer.get_model_info()
        print(f"✓ Model info retrieved: {model_info['generation']}")
        
        # Test basic text processing
        test_texts = [
            "Hello world",
            "This is a test of secure computation",
            "Multi-party computation enables privacy"
        ]
        
        print(f"\n📝 Testing with {len(test_texts)} texts...")
        
        # Test preprocessing
        inputs = transformer.preprocess_text(test_texts)
        print(f"✓ Preprocessing completed: {len(inputs['input_ids'])} sequences")
        
        # Test secure inference
        start_time = time.time()
        results = transformer.predict_secure(test_texts)
        inference_time = time.time() - start_time
        
        print(f"✓ Secure inference completed in {inference_time:.3f}s")
        print(f"✓ Generated {len(results['predictions'])} predictions")
        
        # Validate results
        assert len(results['predictions']) == len(test_texts), "Prediction count mismatch"
        assert 'security_info' in results, "Missing security info"
        assert results['security_info']['protocol'] == 'basic_mpc', "Wrong protocol"
        assert results['security_info']['generation'] == '1_basic', "Wrong generation"
        
        print(f"✓ Results validation passed")
        
        # Display sample result
        sample_pred = results['predictions'][0]
        print(f"\n📊 Sample Prediction:")
        print(f"   Text: {sample_pred['text']}")
        print(f"   Shape: {sample_pred['embedding_shape']}")
        print(f"   Tokens: {sample_pred['processed_tokens']}")
        print(f"   Secure: {sample_pred['secure_computation']}")
        
        # Test cleanup
        transformer.cleanup()
        print(f"✓ Cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tokenizer():
    """Test basic tokenizer functionality."""
    print("\n" + "=" * 60)
    print("Generation 1: Tokenizer Test")
    print("=" * 60)
    
    try:
        # Direct import
        sys.path.insert(0, str(Path(__file__).parent / "src" / "secure_mpc_transformer" / "models"))
        from basic_transformer import BasicTokenizer
        
        # Create tokenizer
        tokenizer = BasicTokenizer(vocab_size=1000, max_length=64)
        print(f"✓ Tokenizer created: vocab_size={len(tokenizer.vocab)}")
        
        # Test tokenization
        test_text = "Hello world, this is a test!"
        tokens = tokenizer.tokenize(test_text)
        print(f"✓ Tokenization: '{test_text}' -> {len(tokens)} tokens")
        
        # Test encoding
        encoded = tokenizer.encode([test_text])
        assert "input_ids" in encoded, "Missing input_ids"
        assert "attention_mask" in encoded, "Missing attention_mask"
        print(f"✓ Encoding: {len(encoded['input_ids'][0])} token IDs")
        
        # Test decoding
        decoded = tokenizer.decode(encoded["input_ids"])
        print(f"✓ Decoding: '{decoded[0][:50]}...'")
        
        # Test batch processing
        batch_texts = ["First text", "Second text", "Third text"]
        batch_encoded = tokenizer.encode(batch_texts)
        assert len(batch_encoded["input_ids"]) == 3, "Batch size mismatch"
        print(f"✓ Batch processing: {len(batch_texts)} texts")
        
        return True
        
    except Exception as e:
        print(f"❌ Tokenizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mpc_protocol():
    """Test basic MPC protocol functionality."""
    print("\n" + "=" * 60)
    print("Generation 1: MPC Protocol Test")
    print("=" * 60)
    
    try:
        # Direct import
        sys.path.insert(0, str(Path(__file__).parent / "src" / "secure_mpc_transformer" / "models"))
        from basic_transformer import BasicMPCProtocol
        
        # Create protocol
        protocol = BasicMPCProtocol(party_id=0, num_parties=3, security_level=128)
        print(f"✓ Protocol created: {protocol.num_parties} parties")
        
        # Test data sharing
        test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        shares = protocol.share_tensor(test_data)
        assert len(shares) == 3, "Wrong number of shares"
        print(f"✓ Data sharing: {len(test_data)} values -> {len(shares)} shares")
        
        # Test reconstruction
        reconstructed = protocol.reconstruct(shares)
        print(f"✓ Reconstruction: {len(reconstructed)} values")
        
        # Test nested data (batch format)
        batch_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        batch_shares = protocol.share_tensor(batch_data)
        batch_reconstructed = protocol.reconstruct(batch_shares)
        print(f"✓ Batch sharing: {len(batch_data)} sequences")
        
        # Test secure addition
        shares_a = protocol.share_tensor([1.0, 2.0, 3.0])
        shares_b = protocol.share_tensor([4.0, 5.0, 6.0])
        sum_shares = protocol.secure_add(shares_a, shares_b)
        sum_result = protocol.reconstruct(sum_shares)
        print(f"✓ Secure addition: {len(sum_result)} results")
        
        # Cleanup
        protocol.cleanup()
        print(f"✓ Protocol cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"❌ MPC protocol test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_generation_1_tests():
    """Run all Generation 1 tests."""
    print("🚀 Generation 1: Basic MPC Transformer Functionality Tests")
    print("🔒 Testing secure multi-party computation with minimal dependencies")
    print("⚡ Focus: Core functionality, basic server, configuration")
    print()
    
    tests = [
        ("Basic Tokenizer", test_tokenizer),
        ("MPC Protocol", test_mpc_protocol),
        ("Basic Transformer", test_basic_transformer),
    ]
    
    results = {}
    total_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        start_time = time.time()
        
        try:
            success = test_func()
            test_time = time.time() - start_time
            results[test_name] = {
                "success": success,
                "time": test_time,
                "status": "✅ PASS" if success else "❌ FAIL"
            }
        except Exception as e:
            test_time = time.time() - start_time
            results[test_name] = {
                "success": False,
                "time": test_time,
                "status": "💥 ERROR",
                "error": str(e)
            }
    
    total_time = time.time() - total_time
    
    # Print summary
    print("\n" + "=" * 80)
    print("🏁 GENERATION 1 TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for r in results.values() if r["success"])
    total = len(results)
    
    for test_name, result in results.items():
        status_icon = "✅" if result["success"] else "❌"
        print(f"{status_icon} {test_name:25} | {result['time']:6.3f}s | {result['status']}")
        if not result["success"] and "error" in result:
            print(f"    Error: {result['error']}")
    
    print("-" * 80)
    print(f"📊 Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"⏱️  Total time: {total_time:.3f}s")
    print(f"🎯 Generation 1 Status: {'COMPLETE' if passed == total else 'PARTIAL'}")
    
    if passed == total:
        print("\n🎉 Generation 1 implementation is working correctly!")
        print("✨ Ready to proceed to Generation 2: Robustness & Security")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Review and fix issues before proceeding.")
    
    return passed == total


if __name__ == "__main__":
    success = run_generation_1_tests()
    sys.exit(0 if success else 1)