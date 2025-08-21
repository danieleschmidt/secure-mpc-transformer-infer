#!/usr/bin/env python3
"""Simple test of basic transformer functionality."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "secure_mpc_transformer" / "models"))

print("Testing basic transformer imports...")

try:
    from basic_transformer import BasicTokenizer
    print("✓ BasicTokenizer imported successfully")
    
    tokenizer = BasicTokenizer(vocab_size=100, max_length=16)
    print(f"✓ Tokenizer created with vocab size: {len(tokenizer.vocab)}")
    
    # Test simple tokenization
    text = "Hello world"
    tokens = tokenizer.tokenize(text)
    print(f"✓ Tokenized '{text}' -> {tokens}")
    
    # Test encoding
    encoded = tokenizer.encode([text])
    print(f"✓ Encoded to {len(encoded['input_ids'][0])} tokens")
    
except Exception as e:
    print(f"❌ Tokenizer test failed: {e}")
    import traceback
    traceback.print_exc()

try:
    from basic_transformer import BasicMPCProtocol
    print("✓ BasicMPCProtocol imported successfully")
    
    protocol = BasicMPCProtocol(0, 3, 128)
    print(f"✓ Protocol created with {protocol.num_parties} parties")
    
    # Test simple sharing
    data = [1, 2, 3]
    shares = protocol.share_tensor(data)
    print(f"✓ Shared data into {len(shares)} shares")
    
    # Test reconstruction
    result = protocol.reconstruct(shares)
    print(f"✓ Reconstructed data: {len(result)} items")
    
except Exception as e:
    print(f"❌ Protocol test failed: {e}")
    import traceback
    traceback.print_exc()

try:
    from basic_transformer import BasicTransformerConfig, BasicSecureTransformer
    print("✓ BasicSecureTransformer imported successfully")
    
    config = BasicTransformerConfig(
        model_name="test-model",
        max_sequence_length=64,
        hidden_size=128,
        num_parties=3
    )
    print(f"✓ Config created: {config.model_name}")
    
    transformer = BasicSecureTransformer(config)
    print("✓ Transformer created successfully")
    
    # Test simple prediction
    result = transformer.predict_secure(["Hello world"])
    print(f"✓ Prediction completed: {len(result['predictions'])} results")
    
    transformer.cleanup()
    print("✓ Cleanup completed")
    
except Exception as e:
    print(f"❌ Transformer test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n🎉 Basic functionality test completed!")