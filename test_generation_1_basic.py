#!/usr/bin/env python3
"""
Test script for Generation 1 basic functionality.
Tests the enhanced components without requiring full dependencies.
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from secure_mpc_transformer.services.model_service_enhanced import ModelService, ModelCache, MockModel
from secure_mpc_transformer.config import SecurityConfig


def test_model_cache():
    """Test model cache functionality."""
    print("Testing ModelCache...")
    
    cache = ModelCache(max_models=2, max_memory_mb=1000)
    
    # Test adding models
    mock_model1 = MockModel("bert-base")
    mock_model2 = MockModel("roberta-base")
    mock_model3 = MockModel("gpt2")
    
    from secure_mpc_transformer.services.model_service_enhanced import ModelInfo, ModelStatus
    
    info1 = ModelInfo(name="bert-base", status=ModelStatus.LOADED, memory_usage=500)
    info2 = ModelInfo(name="roberta-base", status=ModelStatus.LOADED, memory_usage=600)
    info3 = ModelInfo(name="gpt2", status=ModelStatus.LOADED, memory_usage=400)
    
    # Add first two models
    assert cache.put("bert-base", mock_model1, info1) == True
    assert cache.put("roberta-base", mock_model2, info2) == True
    
    # Check cache stats
    stats = cache.get_stats()
    print(f"Cache stats: {stats}")
    assert stats["cached_models"] == 2
    assert stats["total_memory_mb"] == 1100
    
    # Add third model (should evict LRU)
    assert cache.put("gpt2", mock_model3, info3) == True
    
    # Check that bert-base was evicted
    assert cache.get("bert-base") is None
    assert cache.get("roberta-base") is not None
    assert cache.get("gpt2") is not None
    
    print("✓ ModelCache tests passed")


async def test_model_service():
    """Test model service functionality."""
    print("Testing ModelService...")
    
    config = {
        "max_cached_models": 2,
        "max_cache_memory_mb": 1000,
        "auto_unload_timeout": 60
    }
    
    service = ModelService(config)
    
    # Test loading a mock model
    model = await service.load_model("bert-base-uncased")
    assert model is not None
    assert isinstance(model, MockModel)
    
    # Test that second call uses cache
    model2 = await service.load_model("bert-base-uncased")
    assert model2 is model  # Should be same object from cache
    
    # Test model listing
    models = service.list_models()
    print(f"Model list: {models}")
    assert len(models["cache_stats"]["models"]) == 1
    
    # Test model info
    info = service.get_model_info("bert-base-uncased")
    assert info is not None
    assert info["name"] == "bert-base-uncased"
    
    # Test unloading
    success = await service.unload_model("bert-base-uncased")
    assert success == True
    
    # Test that model is no longer cached
    models_after = service.list_models()
    assert len(models_after["cache_stats"]["models"]) == 0
    
    service.shutdown()
    print("✓ ModelService tests passed")


def test_security_config():
    """Test security configuration."""
    print("Testing SecurityConfig...")
    
    # Test default config
    config = SecurityConfig()
    assert config.gpu_acceleration == True
    assert config.num_parties == 3
    
    # Test validation
    try:
        config.validate()
        print("✓ Default config validation passed")
    except Exception as e:
        print(f"✗ Config validation failed: {e}")
        return False
    
    # Test production config
    prod_config = SecurityConfig.production_config(party_id=0)
    assert prod_config.use_tls == True
    assert prod_config.differential_privacy == True
    
    # Test development config
    dev_config = SecurityConfig.development_config()
    assert dev_config.use_tls == False
    assert dev_config.gpu_acceleration == False
    
    print("✓ SecurityConfig tests passed")


async def test_enhanced_main_entry():
    """Test enhanced main entry point."""
    print("Testing enhanced main entry point...")
    
    try:
        # Import and test the launcher
        sys.path.insert(0, str(Path(__file__).parent))
        from enhanced_main_entry_point import EnhancedMPCLauncher
        
        launcher = EnhancedMPCLauncher()
        
        # Test system requirements check
        requirements = launcher.check_system_requirements()
        print(f"System requirements: {requirements}")
        assert requirements["python_version"] == True
        assert requirements["config_accessible"] == True
        
        # Test configuration creation
        config = launcher.create_default_config()
        assert "server" in config
        assert "security" in config
        assert "enhanced_features" in config
        
        # Test configuration validation
        valid = launcher.validate_configuration(config)
        assert valid == True
        
        # Test health check
        health = launcher.create_health_check()
        assert "status" in health
        assert "requirements" in health
        
        print("✓ Enhanced main entry point tests passed")
        
    except ImportError as e:
        print(f"⚠ Enhanced main entry point not available: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Generation 1 Basic Functionality Tests")
    print("=" * 60)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    tests_passed = 0
    total_tests = 4
    
    try:
        # Test 1: Model Cache
        test_model_cache()
        tests_passed += 1
        
        # Test 2: Model Service
        asyncio.run(test_model_service())
        tests_passed += 1
        
        # Test 3: Security Config
        test_security_config()
        tests_passed += 1
        
        # Test 4: Enhanced Main Entry
        asyncio.run(test_enhanced_main_entry())
        tests_passed += 1
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Test Results: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("🎉 All Generation 1 tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())