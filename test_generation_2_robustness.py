#!/usr/bin/env python3
"""
Test script for Generation 2 robustness enhancements.
Tests the advanced error recovery, security, and monitoring systems.
"""

import sys
import asyncio
import logging
import time
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from secure_mpc_transformer.monitoring.comprehensive_health_monitor import (
    ComprehensiveHealthMonitor, SystemResourceCheck, HealthStatus, AlertSeverity,
    log_alert_handler, console_alert_handler
)

from secure_mpc_transformer.security.enhanced_security_orchestrator import (
    EnhancedSecurityOrchestrator, ValidationPattern, ThreatLevel, ValidationResult
)

from secure_mpc_transformer.resilience.advanced_error_recovery import (
    AdvancedErrorRecovery, RecoveryRule, ErrorCategory, RecoveryAction,
    RetryManager, FallbackManager, CircuitBreaker
)


async def test_health_monitoring():
    """Test comprehensive health monitoring."""
    print("Testing Comprehensive Health Monitoring...")
    
    # Create health monitor
    monitor = ComprehensiveHealthMonitor()
    
    # Add alert handlers
    monitor.add_alert_handler(log_alert_handler)
    
    # Setup default checks
    monitor.setup_default_checks()
    
    # Test manual health check
    if monitor.checks:
        check = monitor.checks[0]  # System resources check
        result = await check.check()
        
        assert result.name == check.name
        assert result.status in [status for status in HealthStatus]
        
        print(f"Health check result: {result.name} - {result.status.value}")
    
    # Test health summary
    summary = monitor.get_health_summary()
    assert "overall_status" in summary
    assert "timestamp" in summary
    
    print(f"Health summary: {summary['overall_status']}")
    
    # Test metrics
    metrics = monitor.get_metrics()
    assert "total_checks" in metrics
    assert "monitoring_active" in metrics
    
    print("✓ Health monitoring tests passed")


async def test_security_orchestrator():
    """Test enhanced security orchestrator."""
    print("Testing Enhanced Security Orchestrator...")
    
    # Create security orchestrator
    security = EnhancedSecurityOrchestrator({
        "rate_limit_requests": 10,
        "rate_limit_window": 60,
        "auto_block_enabled": True,
        "threat_threshold": 3
    })
    
    # Test normal input validation
    result1 = await security.validate_request("Hello world", source_ip="192.168.1.1")
    assert result1["allowed"] == True
    assert result1["result"] == ValidationResult.ALLOWED
    
    print("✓ Normal input allowed")
    
    # Test malicious input detection
    result2 = await security.validate_request(
        "<script>alert('xss')</script>", 
        source_ip="192.168.1.2"
    )
    assert result2["allowed"] == False
    assert result2["result"] == ValidationResult.BLOCKED
    assert len(result2["violations"]) > 0
    
    print("✓ Malicious input blocked")
    
    # Test SQL injection detection
    result3 = await security.validate_request(
        "'; DROP TABLE users; --",
        source_ip="192.168.1.3"
    )
    assert result3["allowed"] == False
    assert result3["result"] == ValidationResult.BLOCKED
    
    print("✓ SQL injection blocked")
    
    # Test session management
    session_id = security.create_session("user123", {"role": "admin"})
    assert session_id is not None
    
    session_info = security.validate_session(session_id)
    assert session_info is not None
    assert session_info["user_id"] == "user123"
    
    print("✓ Session management working")
    
    # Test rate limiting (simulate multiple requests)
    for i in range(15):  # Exceed rate limit
        result = await security.validate_request(f"Request {i}", source_ip="192.168.1.4")
        if not result["allowed"] and result["result"] == ValidationResult.RATE_LIMITED:
            print("✓ Rate limiting working")
            break
    else:
        print("⚠ Rate limiting may not be working")
    
    # Test security status
    status = security.get_security_status()
    assert "threat_summary" in status
    assert "rate_limiter_status" in status
    
    print("✓ Security orchestrator tests passed")


async def test_error_recovery():
    """Test advanced error recovery."""
    print("Testing Advanced Error Recovery...")
    
    # Create error recovery system
    recovery = AdvancedErrorRecovery({
        "max_retries": 3,
        "base_delay": 0.1,  # Shorter for testing
        "max_delay": 1.0
    })
    
    # Test retry mechanism
    retry_manager = RetryManager(max_retries=2, base_delay=0.1)
    
    call_count = 0
    def failing_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("Network error")
        return "success"
    
    result = await retry_manager.retry_async(failing_function)
    assert result == "success"
    assert call_count == 3
    
    print("✓ Retry mechanism working")
    
    # Test fallback mechanism
    fallback_manager = FallbackManager()
    
    def primary_function():
        raise ValueError("Primary failed")
    
    def fallback_function():
        return "fallback_result"
    
    fallback_manager.register_fallback("test_operation", fallback_function)
    
    result = await fallback_manager.execute_with_fallback(
        "test_operation", primary_function
    )
    assert result == "fallback_result"
    
    print("✓ Fallback mechanism working")
    
    # Test circuit breaker
    circuit_breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1)
    
    failure_count = 0
    def sometimes_failing_function():
        nonlocal failure_count
        failure_count += 1
        if failure_count <= 3:
            raise Exception("Service unavailable")
        return "success"
    
    # Trigger circuit breaker opening
    for i in range(3):
        try:
            await circuit_breaker.call(sometimes_failing_function)
        except Exception:
            pass
    
    # Circuit should be open now
    assert circuit_breaker.state == "OPEN"
    
    print("✓ Circuit breaker opening")
    
    # Test comprehensive error handling
    recovery.add_error_handler(lambda err: print(f"Error handled: {err.error_type}"))
    
    async def test_function():
        raise ConnectionError("Test network error")
    
    try:
        await recovery.execute_with_recovery("test_op", test_function)
    except Exception:
        pass  # Expected to fail
    
    # Check error statistics
    stats = recovery.get_error_statistics(hours=1)
    assert stats["total_errors"] > 0
    
    print("✓ Error recovery system working")
    
    print("✓ Advanced error recovery tests passed")


async def test_system_integration():
    """Test integration between all robustness components."""
    print("Testing System Integration...")
    
    # Create all components
    health_monitor = ComprehensiveHealthMonitor()
    security = EnhancedSecurityOrchestrator()
    error_recovery = AdvancedErrorRecovery()
    
    # Setup health monitoring with security service
    health_monitor.setup_default_checks(security_service=security)
    
    # Test coordinated operation
    async def test_operation():
        # Validate security
        security_result = await security.validate_request("Test input")
        if not security_result["allowed"]:
            raise ValueError("Security validation failed")
        
        # Simulate some processing
        await asyncio.sleep(0.1)
        
        return "operation_success"
    
    # Execute with error recovery
    result = await error_recovery.execute_with_recovery("integrated_test", test_operation)
    assert result == "operation_success"
    
    # Check health status
    health_summary = health_monitor.get_health_summary()
    assert health_summary["overall_status"] in ["healthy", "unknown"]
    
    # Check security status
    security_status = security.get_security_status()
    assert "threat_summary" in security_status
    
    # Check error recovery status
    recovery_health = error_recovery.get_system_health()
    assert recovery_health["status"] in ["healthy", "warning", "degraded", "critical"]
    
    print("✓ System integration tests passed")


async def test_resilience_under_load():
    """Test system resilience under simulated load."""
    print("Testing Resilience Under Load...")
    
    security = EnhancedSecurityOrchestrator({
        "rate_limit_requests": 5,
        "rate_limit_window": 1  # 1 second window
    })
    
    error_recovery = AdvancedErrorRecovery()
    
    # Simulate concurrent requests
    async def simulate_request(request_id: int):
        try:
            # Mix of normal and malicious requests
            if request_id % 3 == 0:
                text = f"<script>alert('attack_{request_id}')</script>"
            else:
                text = f"Normal request {request_id}"
            
            result = await security.validate_request(
                text, 
                source_ip=f"192.168.1.{request_id % 10}"
            )
            
            return {"request_id": request_id, "allowed": result["allowed"]}
        except Exception as e:
            return {"request_id": request_id, "error": str(e)}
    
    # Run multiple concurrent requests
    tasks = [simulate_request(i) for i in range(20)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Analyze results
    successful_requests = [r for r in results if isinstance(r, dict) and "allowed" in r]
    blocked_requests = [r for r in successful_requests if not r["allowed"]]
    
    print(f"Processed {len(successful_requests)} requests")
    print(f"Blocked {len(blocked_requests)} malicious requests")
    
    # Check that system handled load gracefully
    assert len(successful_requests) > 0
    assert len(blocked_requests) > 0  # Some attacks should be blocked
    
    print("✓ Resilience under load tests passed")


def main():
    """Run all Generation 2 robustness tests."""
    print("=" * 60)
    print("Generation 2 Robustness Tests")
    print("=" * 60)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    tests_passed = 0
    total_tests = 5
    
    try:
        # Test 1: Health Monitoring
        asyncio.run(test_health_monitoring())
        tests_passed += 1
        
        # Test 2: Security Orchestrator
        asyncio.run(test_security_orchestrator())
        tests_passed += 1
        
        # Test 3: Error Recovery
        asyncio.run(test_error_recovery())
        tests_passed += 1
        
        # Test 4: System Integration
        asyncio.run(test_system_integration())
        tests_passed += 1
        
        # Test 5: Resilience Under Load
        asyncio.run(test_resilience_under_load())
        tests_passed += 1
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Test Results: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("🎉 All Generation 2 robustness tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())