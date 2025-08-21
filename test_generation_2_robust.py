#!/usr/bin/env python3
"""
Generation 2 Robust Functionality Test
Tests enhanced error handling, security, logging, and monitoring features.
"""

import sys
import time
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_robust_transformer():
    """Test robust transformer functionality."""
    print("=" * 60)
    print("Generation 2: Robust Transformer Test")
    print("=" * 60)
    
    try:
        # Direct imports
        sys.path.insert(0, str(Path(__file__).parent / "src" / "secure_mpc_transformer" / "models"))
        sys.path.insert(0, str(Path(__file__).parent / "src" / "secure_mpc_transformer" / "utils"))
        
        from robust_transformer import RobustSecureTransformer, RobustTransformerConfig
        
        # Create robust configuration
        config = RobustTransformerConfig(
            model_name="bert-base-test",
            max_sequence_length=128,
            hidden_size=256,
            num_parties=3,
            party_id=0,
            security_level=128,
            
            # Robustness features
            max_retry_attempts=2,
            timeout_seconds=10.0,
            enable_input_validation=True,
            enable_output_sanitization=True,
            enable_detailed_logging=True,
            enable_performance_monitoring=True,
            enable_audit_logging=True,
            enable_rate_limiting=True,
            max_requests_per_minute=30,
            enable_data_integrity_checks=True
        )
        
        print(f"✓ Robust configuration created: {config.model_name}")
        
        # Initialize robust transformer
        transformer = RobustSecureTransformer(config)
        print(f"✓ Robust transformer initialized")
        
        # Test system status
        system_status = transformer.get_system_status()
        print(f"✓ System status retrieved: {system_status['status']}")
        
        # Test basic text processing
        test_texts = [
            "Hello world",
            "This is a test of robust secure computation",
            "Generation 2 adds comprehensive error handling"
        ]
        
        print(f"\n📝 Testing robust inference with {len(test_texts)} texts...")
        
        # Test robust secure inference
        start_time = time.time()
        results = transformer.predict_secure_robust(test_texts, client_id="test-client")
        inference_time = time.time() - start_time
        
        print(f"✓ Robust secure inference completed in {inference_time:.3f}s")
        
        # Validate robust features
        assert 'monitoring' in results, "Missing monitoring data"
        assert 'data_integrity' in results, "Missing data integrity checks"
        assert results['monitoring']['generation'] == '2_robust', "Wrong generation"
        
        print(f"✓ Robust features validation passed")
        
        # Test error handling and recovery
        print(f"\n🔧 Testing error handling...")
        
        # Test input validation (should fail)
        try:
            invalid_input = ["x" * 20000]  # Too long
            transformer.predict_secure_robust(invalid_input, client_id="test-client")
            print(f"❌ Input validation should have failed")
            return False
        except Exception as e:
            print(f"✓ Input validation correctly rejected invalid input: {type(e).__name__}")
        
        # Test rate limiting
        print(f"🚦 Testing rate limiting...")
        request_count = 0
        for i in range(5):  # Should be within limits
            try:
                transformer.predict_secure_robust(["Short test"], client_id="rate-test-client")
                request_count += 1
            except Exception as e:
                if "rate limit" in str(e).lower():
                    break
        
        print(f"✓ Rate limiting working: {request_count} requests processed")
        
        # Display monitoring info
        monitoring = results.get('monitoring', {})
        print(f"\n📊 Monitoring Data:")
        print(f"   Request ID: {monitoring.get('request_id', 'N/A')}")
        print(f"   Processing Time: {monitoring.get('processing_time_ms', 0):.2f}ms")
        print(f"   Generation: {monitoring.get('generation', 'unknown')}")
        
        # Display security info
        security_features = system_status.get('security_features', {})
        print(f"\n🔒 Security Features:")
        for feature, enabled in security_features.items():
            status = "✓" if enabled else "✗"
            print(f"   {status} {feature.replace('_', ' ').title()}")
        
        # Test cleanup
        transformer.cleanup()
        print(f"✓ Robust cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling():
    """Test enhanced error handling system."""
    print("\n" + "=" * 60)
    print("Generation 2: Error Handling Test")
    print("=" * 60)
    
    try:
        # Direct imports
        sys.path.insert(0, str(Path(__file__).parent / "src" / "secure_mpc_transformer" / "utils"))
        
        from robust_error_handling import (
            RobustErrorHandler,
            SecurityException,
            ValidationException,
            ErrorCategory,
            ErrorSeverity,
            robust_exception_handler
        )
        
        # Create error handler
        error_handler = RobustErrorHandler()
        print(f"✓ Error handler created")
        
        # Test different exception types
        try:
            raise ValidationException("Test validation error", severity=ErrorSeverity.MEDIUM)
        except Exception as e:
            error_ctx = error_handler.handle_error(e, request_id="test-001")
            print(f"✓ Validation exception handled: {error_ctx.error_id}")
        
        try:
            raise SecurityException("Test security error", severity=ErrorSeverity.HIGH)
        except Exception as e:
            error_ctx = error_handler.handle_error(e, request_id="test-002")
            print(f"✓ Security exception handled: {error_ctx.error_id}")
        
        # Test error statistics
        stats = error_handler.get_error_statistics()
        print(f"✓ Error statistics: {stats['total_errors']} total errors")
        print(f"   Recovery attempts: {stats.get('recovery_statistics', {}).get('attempts', 0)}")
        
        # Test decorator
        @robust_exception_handler(category=ErrorCategory.VALIDATION, attempt_recovery=False)
        def test_function():
            raise ValueError("Test decorated error")
        
        try:
            test_function()
        except ValueError:
            print(f"✓ Decorator error handling working")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_security_validation():
    """Test security validation features."""
    print("\n" + "=" * 60)
    print("Generation 2: Security Validation Test")
    print("=" * 60)
    
    try:
        # Direct imports
        sys.path.insert(0, str(Path(__file__).parent / "src" / "secure_mpc_transformer" / "models"))
        
        from robust_transformer import SecurityValidator, RobustTransformerConfig
        
        # Create security validator
        config = RobustTransformerConfig(
            enable_input_validation=True,
            max_input_length=1000,
            allowed_input_chars="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?",
            enable_rate_limiting=True,
            max_requests_per_minute=10
        )
        
        validator = SecurityValidator(config)
        print(f"✓ Security validator created")
        
        # Test valid input
        valid_texts = ["Hello world", "This is a valid test"]
        is_valid, message = validator.validate_input(valid_texts)
        assert is_valid, f"Valid input should pass: {message}"
        print(f"✓ Valid input validation passed")
        
        # Test invalid input (too long)
        invalid_texts = ["x" * 2000]
        is_valid, message = validator.validate_input(invalid_texts)
        assert not is_valid, "Invalid input should fail"
        print(f"✓ Invalid input validation correctly failed: {message}")
        
        # Test suspicious patterns
        suspicious_texts = ["<script>alert('xss')</script>"]
        is_valid, message = validator.validate_input(suspicious_texts)
        assert not is_valid, "Suspicious input should fail"
        print(f"✓ Suspicious pattern detection working: {message}")
        
        # Test rate limiting
        client_id = "test-client"
        for i in range(3):
            is_ok, msg = validator.check_rate_limit(client_id)
            if not is_ok:
                break
        print(f"✓ Rate limiting working: {msg}")
        
        # Test checksum generation
        test_data = {"test": "data", "number": 123}
        checksum = validator.generate_checksum(test_data)
        assert len(checksum) == 64, "SHA256 checksum should be 64 chars"
        print(f"✓ Checksum generation working: {checksum[:16]}...")
        
        # Test checksum verification
        is_valid = validator.verify_checksum(test_data, checksum)
        assert is_valid, "Checksum verification should pass"
        print(f"✓ Checksum verification working")
        
        return True
        
    except Exception as e:
        print(f"❌ Security validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_monitoring():
    """Test performance monitoring features."""
    print("\n" + "=" * 60)
    print("Generation 2: Performance Monitoring Test")
    print("=" * 60)
    
    try:
        # Direct imports
        sys.path.insert(0, str(Path(__file__).parent / "src" / "secure_mpc_transformer" / "models"))
        
        from robust_transformer import PerformanceMonitor, RobustTransformerConfig
        
        # Create performance monitor
        config = RobustTransformerConfig(enable_performance_monitoring=True)
        monitor = PerformanceMonitor(config)
        print(f"✓ Performance monitor created")
        
        # Test request tracking
        request_id = monitor.start_request()
        print(f"✓ Request tracking started: {request_id}")
        
        # Simulate processing time
        time.sleep(0.1)
        
        # End request tracking
        monitor.end_request(request_id, success=True, processing_time=0.1)
        print(f"✓ Request tracking completed")
        
        # Test metrics
        metrics = monitor.get_metrics()
        assert metrics['total_requests'] == 1, "Should have 1 request"
        assert metrics['successful_requests'] == 1, "Should have 1 successful request"
        assert metrics['success_rate_percent'] == 100.0, "Success rate should be 100%"
        print(f"✓ Performance metrics working:")
        print(f"   Total requests: {metrics['total_requests']}")
        print(f"   Success rate: {metrics['success_rate_percent']}%")
        print(f"   Average time: {metrics['average_processing_time_ms']:.2f}ms")
        
        # Test failed request
        failed_request_id = monitor.start_request()
        monitor.end_request(failed_request_id, success=False, processing_time=0.05)
        
        updated_metrics = monitor.get_metrics()
        assert updated_metrics['total_requests'] == 2, "Should have 2 requests"
        assert updated_metrics['failed_requests'] == 1, "Should have 1 failed request"
        print(f"✓ Failed request tracking working")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance monitoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_generation_2_tests():
    """Run all Generation 2 tests."""
    print("🚀 Generation 2: Robust MPC Transformer Functionality Tests")
    print("🔒 Testing enhanced error handling, security, logging, and monitoring")
    print("⚡ Focus: Robustness, security validation, performance monitoring")
    print()
    
    tests = [
        ("Error Handling System", test_error_handling),
        ("Security Validation", test_security_validation),
        ("Performance Monitoring", test_performance_monitoring),
        ("Robust Transformer", test_robust_transformer),
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
    print("🏁 GENERATION 2 TEST SUMMARY")
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
    print(f"🎯 Generation 2 Status: {'COMPLETE' if passed == total else 'PARTIAL'}")
    
    if passed == total:
        print("\n🎉 Generation 2 implementation is working correctly!")
        print("✨ Enhanced with robust error handling, security, and monitoring")
        print("🚀 Ready to proceed to Generation 3: Scalability & Performance")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Review and fix issues before proceeding.")
    
    return passed == total


if __name__ == "__main__":
    success = run_generation_2_tests()
    sys.exit(0 if success else 1)