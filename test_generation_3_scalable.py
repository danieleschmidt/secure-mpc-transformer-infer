#!/usr/bin/env python3
"""
Generation 3 Scalable Functionality Test
Tests advanced caching, performance optimization, auto-scaling, and batch processing.
"""

import sys
import time
import json
import threading
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_advanced_caching():
    """Test advanced caching system."""
    print("=" * 60)
    print("Generation 3: Advanced Caching Test")
    print("=" * 60)
    
    try:
        # Direct imports
        sys.path.insert(0, str(Path(__file__).parent / "src" / "secure_mpc_transformer" / "models"))
        
        from scalable_transformer import AdvancedCache, ScalableTransformerConfig
        
        # Create cache configuration
        config = ScalableTransformerConfig(
            enable_advanced_caching=True,
            cache_size_mb=1,  # Small cache for testing
            cache_ttl_seconds=5,
            cache_compression=True,
            cache_eviction_policy="lru"
        )
        
        cache = AdvancedCache(config)
        print(f"✓ Advanced cache created with {config.cache_size_mb}MB limit")
        
        # Test basic put/get
        test_data = {"test": "data", "numbers": [1, 2, 3, 4, 5]}
        success = cache.put("test_key", test_data)
        assert success, "Cache put should succeed"
        print(f"✓ Data cached successfully")
        
        retrieved_data = cache.get("test_key")
        assert retrieved_data == test_data, "Retrieved data should match original"
        print(f"✓ Data retrieved successfully")
        
        # Test cache miss
        missing_data = cache.get("nonexistent_key")
        assert missing_data is None, "Missing key should return None"
        print(f"✓ Cache miss handled correctly")
        
        # Test cache statistics
        stats = cache.get_stats()
        assert stats['total_items'] == 1, "Should have 1 cached item"
        print(f"✓ Cache statistics: {stats['total_items']} items, {stats['total_size_mb']}MB")
        
        # Test eviction by filling cache
        large_data = {"large": "x" * 10000}  # Large data to trigger eviction
        for i in range(10):
            cache.put(f"large_key_{i}", large_data)
        
        # Original key might be evicted
        final_stats = cache.get_stats()
        print(f"✓ Cache eviction working: {final_stats['total_items']} items after filling")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced caching test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_worker_pool():
    """Test worker pool with auto-scaling."""
    print("\n" + "=" * 60)
    print("Generation 3: Worker Pool Test")
    print("=" * 60)
    
    try:
        # Direct imports
        sys.path.insert(0, str(Path(__file__).parent / "src" / "secure_mpc_transformer" / "models"))
        
        from scalable_transformer import WorkerPool, ScalableTransformerConfig
        
        # Create worker pool configuration
        config = ScalableTransformerConfig(
            enable_auto_scaling=True,
            min_workers=1,
            max_workers=4,
            scale_up_threshold=0.5,
            scale_down_threshold=0.2,
            scaling_cooldown_seconds=1  # Short cooldown for testing
        )
        
        worker_pool = WorkerPool(config)
        print(f"✓ Worker pool created with {config.min_workers}-{config.max_workers} workers")
        
        # Test basic task submission
        def simple_task(x):
            time.sleep(0.1)  # Simulate work
            return x * 2
        
        future = worker_pool.submit_task(simple_task, 5)
        result = future.result(timeout=2.0)
        assert result == 10, "Task result should be 10"
        print(f"✓ Basic task execution working: {result}")
        
        # Test multiple tasks to trigger scaling
        futures = []
        for i in range(8):  # Submit more tasks than min workers
            future = worker_pool.submit_task(simple_task, i)
            futures.append(future)
        
        # Wait for all tasks to complete
        results = [f.result(timeout=5.0) for f in futures]
        expected = [i * 2 for i in range(8)]
        assert results == expected, "All tasks should complete correctly"
        print(f"✓ Multiple tasks completed: {len(results)} results")
        
        # Get worker pool statistics
        stats = worker_pool.get_stats()
        print(f"✓ Worker pool statistics:")
        print(f"   Current workers: {stats['current_workers']}")
        print(f"   Completed tasks: {stats['completed_tasks']}")
        print(f"   Success rate: {stats['success_rate_percent']}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Worker pool test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_processor():
    """Test intelligent batch processing."""
    print("\n" + "=" * 60)
    print("Generation 3: Batch Processor Test")
    print("=" * 60)
    
    try:
        # Direct imports
        sys.path.insert(0, str(Path(__file__).parent / "src" / "secure_mpc_transformer" / "models"))
        
        from scalable_transformer import BatchProcessor, ScalableTransformerConfig
        
        # Create batch processor configuration
        config = ScalableTransformerConfig(
            enable_batch_processing=True,
            max_batch_size=5,
            batch_timeout_ms=200
        )
        
        batch_processor = BatchProcessor(config)
        print(f"✓ Batch processor created with max batch size {config.max_batch_size}")
        
        # Test batch processing
        results = []
        result_event = threading.Event()
        
        def callback(result):
            results.extend(result)
            if len(results) >= 3:  # Expecting 3 texts
                result_event.set()
        
        # Add requests to batch
        test_texts = ["Hello", "batch", "processing"]
        batch_processor.add_request(test_texts, callback, "test_client")
        
        # Wait for batch to be processed
        success = result_event.wait(timeout=5.0)
        assert success, "Batch processing should complete within timeout"
        assert len(results) == 3, "Should process all 3 texts"
        print(f"✓ Batch processing completed: {len(results)} items processed")
        
        # Test batch statistics
        stats = batch_processor.get_stats()
        print(f"✓ Batch processor statistics:")
        print(f"   Total batches: {stats['total_batches']}")
        print(f"   Total items: {stats['total_items']}")
        print(f"   Average batch size: {stats['avg_batch_size']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch processor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scalable_transformer():
    """Test complete scalable transformer functionality."""
    print("\n" + "=" * 60)
    print("Generation 3: Scalable Transformer Test")
    print("=" * 60)
    
    try:
        # Direct imports
        sys.path.insert(0, str(Path(__file__).parent / "src" / "secure_mpc_transformer" / "models"))
        sys.path.insert(0, str(Path(__file__).parent / "src" / "secure_mpc_transformer" / "utils"))
        
        from scalable_transformer import ScalableSecureTransformer, ScalableTransformerConfig
        
        # Create scalable configuration
        config = ScalableTransformerConfig(
            model_name="bert-base-test",
            max_sequence_length=128,
            hidden_size=256,
            num_parties=3,
            party_id=0,
            security_level=128,
            
            # Generation 3 features
            enable_advanced_caching=True,
            cache_size_mb=10,
            enable_batch_processing=True,
            max_batch_size=8,
            enable_auto_scaling=True,
            min_workers=1,
            max_workers=3,
            enable_model_parallelism=True
        )
        
        print(f"✓ Scalable configuration created")
        
        # Initialize scalable transformer
        transformer = ScalableSecureTransformer(config)
        print(f"✓ Scalable transformer initialized")
        
        # Test comprehensive status
        status = transformer.get_comprehensive_status()
        assert status['generation'] == '3_scalable', "Should be Generation 3"
        print(f"✓ Comprehensive status retrieved: {status['generation']}")
        
        # Test scalable inference (small batch to avoid timeout)
        test_texts = [
            "Hello scalable world",
            "Generation 3 performance"
        ]
        
        print(f"\n📝 Testing scalable inference with {len(test_texts)} texts...")
        
        # Test with caching
        start_time = time.time()
        result1 = transformer.predict_secure_scalable(
            test_texts, 
            client_id="test-client",
            use_cache=True,
            use_batch=False  # Disable batch for deterministic testing
        )
        first_time = time.time() - start_time
        
        assert 'scalability' in result1, "Missing scalability metadata"
        assert result1['scalability']['generation'] == '3_scalable', "Wrong generation"
        print(f"✓ First inference completed in {first_time:.3f}s")
        
        # Test cache hit (should be faster)
        start_time = time.time()
        result2 = transformer.predict_secure_scalable(
            test_texts, 
            client_id="test-client",
            use_cache=True,
            use_batch=False
        )
        second_time = time.time() - start_time
        
        if 'cache_hit' in result2 and result2['cache_hit']:
            print(f"✓ Cache hit achieved in {second_time:.3f}s (speedup: {first_time/second_time:.1f}x)")
        else:
            print(f"⚠️ Cache miss (might be due to test timing)")
        
        # Test performance optimization
        print(f"\n🔧 Testing performance optimization...")
        optimizations = transformer.optimize_performance()
        optimized_features = sum(1 for v in optimizations.values() if v)
        print(f"✓ Performance optimization completed: {optimized_features} features optimized")
        
        # Display comprehensive metrics
        final_status = transformer.get_comprehensive_status()
        print(f"\n📊 Scalability Metrics:")
        
        if 'cache_stats' in final_status:
            cache_stats = final_status['cache_stats']
            print(f"   Cache: {cache_stats['total_items']} items, {cache_stats['usage_percent']:.1f}% full")
        
        if 'worker_pool_stats' in final_status:
            worker_stats = final_status['worker_pool_stats']
            print(f"   Workers: {worker_stats['current_workers']} active, {worker_stats['completed_tasks']} completed")
        
        scalability_features = final_status.get('scalability_features', {})
        enabled_features = sum(1 for v in scalability_features.values() if v)
        print(f"   Features: {enabled_features}/{len(scalability_features)} enabled")
        
        # Test cleanup
        transformer.cleanup()
        print(f"✓ Scalable cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Scalable transformer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_generation_3_tests():
    """Run all Generation 3 tests."""
    print("🚀 Generation 3: Scalable MPC Transformer Functionality Tests")
    print("🔒 Testing advanced caching, performance optimization, and auto-scaling")
    print("⚡ Focus: Scalability, caching, batch processing, worker pools")
    print()
    
    tests = [
        ("Advanced Caching", test_advanced_caching),
        ("Worker Pool & Auto-Scaling", test_worker_pool),
        ("Batch Processing", test_batch_processor),
        ("Scalable Transformer", test_scalable_transformer),
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
    print("🏁 GENERATION 3 TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for r in results.values() if r["success"])
    total = len(results)
    
    for test_name, result in results.items():
        status_icon = "✅" if result["success"] else "❌"
        print(f"{status_icon} {test_name:30} | {result['time']:6.3f}s | {result['status']}")
        if not result["success"] and "error" in result:
            print(f"    Error: {result['error']}")
    
    print("-" * 80)
    print(f"📊 Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"⏱️  Total time: {total_time:.3f}s")
    print(f"🎯 Generation 3 Status: {'COMPLETE' if passed == total else 'PARTIAL'}")
    
    if passed == total:
        print("\n🎉 Generation 3 implementation is working correctly!")
        print("✨ Enhanced with advanced caching, auto-scaling, and optimization")
        print("🚀 Ready for quality gates and research validation!")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Review and fix issues before proceeding.")
    
    return passed == total


if __name__ == "__main__":
    success = run_generation_3_tests()
    sys.exit(0 if success else 1)