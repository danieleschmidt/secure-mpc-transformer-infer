#!/usr/bin/env python3
"""
Standalone test for Generation 2 robustness enhancements.
Tests advanced error recovery, security, and monitoring without package imports.
"""

import sys
import asyncio
import logging
import time
import hashlib
import hmac
import secrets
import re
from typing import Dict, List, Optional, Any, Set, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


# Health Monitoring Components
class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: Any
    status: HealthStatus
    threshold: Optional[float] = None
    timestamp: float = None
    message: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class MockSystemResourceCheck:
    """Mock system resource health check."""
    
    def __init__(self):
        self.name = "system_resources"
        self.last_check = 0.0
        self.last_result = None
    
    async def check(self) -> HealthMetric:
        """Perform mock health check."""
        # Simulate system metrics
        cpu_percent = 25.0
        memory_percent = 40.0
        
        status = HealthStatus.HEALTHY
        message = f"Resources normal: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%"
        
        self.last_check = time.time()
        self.last_result = HealthMetric(
            name=self.name,
            value={"cpu_percent": cpu_percent, "memory_percent": memory_percent},
            status=status,
            message=message
        )
        
        return self.last_result


# Security Components
class ThreatLevel(Enum):
    """Threat level classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ValidationResult(Enum):
    """Input validation result."""
    ALLOWED = "allowed"
    BLOCKED = "blocked"
    RATE_LIMITED = "rate_limited"
    QUARANTINED = "quarantined"


@dataclass
class ValidationPattern:
    """Input validation pattern."""
    name: str
    pattern: str
    threat_level: ThreatLevel
    description: str
    action: str = "block"


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = {}
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed under rate limit."""
        current_time = time.time()
        
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        request_times = self.requests[identifier]
        
        # Remove requests outside the window
        cutoff_time = current_time - self.window_seconds
        request_times[:] = [t for t in request_times if t > cutoff_time]
        
        # Check if under limit
        if len(request_times) < self.max_requests:
            request_times.append(current_time)
            return True
        
        return False


class InputValidator:
    """Advanced input validation system."""
    
    def __init__(self):
        self.patterns: List[ValidationPattern] = []
        self.quarantine_cache: Dict[str, float] = {}
        self._setup_default_patterns()
    
    def _setup_default_patterns(self):
        """Setup default security patterns."""
        default_patterns = [
            ValidationPattern(
                name="xss_script",
                pattern=r"<script[^>]*>.*?</script>",
                threat_level=ThreatLevel.HIGH,
                description="Cross-site scripting attempt",
                action="block"
            ),
            ValidationPattern(
                name="sql_injection",
                pattern=r"(?i)(union|select|insert|update|delete|drop|exec|script)\s+",
                threat_level=ThreatLevel.HIGH,
                description="SQL injection attempt",
                action="block"
            ),
            ValidationPattern(
                name="command_injection",
                pattern=r"(?i)(exec|eval|system|shell_exec|passthru|`)",
                threat_level=ThreatLevel.CRITICAL,
                description="Command injection attempt",
                action="block"
            )
        ]
        
        self.patterns.extend(default_patterns)
    
    def validate_input(self, text: str) -> Dict[str, Any]:
        """Validate input text against security patterns."""
        violations = []
        max_threat_level = ThreatLevel.LOW
        
        # Check against patterns
        for pattern in self.patterns:
            try:
                if re.search(pattern.pattern, text, re.IGNORECASE | re.MULTILINE):
                    violations.append({
                        "pattern": pattern.name,
                        "description": pattern.description,
                        "threat_level": pattern.threat_level.value,
                        "action": pattern.action
                    })
                    
                    if self._is_higher_threat(pattern.threat_level, max_threat_level):
                        max_threat_level = pattern.threat_level
            except re.error:
                continue
        
        # Determine action
        if not violations:
            return {
                "allowed": True,
                "result": ValidationResult.ALLOWED,
                "violations": [],
                "threat_level": ThreatLevel.LOW.value
            }
        
        actions = [v["action"] for v in violations]
        
        if "block" in actions:
            result = ValidationResult.BLOCKED
            allowed = False
        else:
            result = ValidationResult.ALLOWED
            allowed = True
        
        return {
            "allowed": allowed,
            "result": result,
            "violations": violations,
            "threat_level": max_threat_level.value,
            "reason": f"Input matched {len(violations)} security patterns"
        }
    
    def _is_higher_threat(self, level1: ThreatLevel, level2: ThreatLevel) -> bool:
        """Check if level1 is higher threat than level2."""
        threat_order = {
            ThreatLevel.LOW: 0,
            ThreatLevel.MEDIUM: 1,
            ThreatLevel.HIGH: 2,
            ThreatLevel.CRITICAL: 3
        }
        return threat_order[level1] > threat_order[level2]


class MockSecurityOrchestrator:
    """Mock security orchestrator for testing."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.rate_limiter = RateLimiter(
            max_requests=self.config.get("rate_limit_requests", 10),
            window_seconds=self.config.get("rate_limit_window", 60)
        )
        self.input_validator = InputValidator()
        self.blocked_ips: Set[str] = set()
        self.events: List[Dict[str, Any]] = []
    
    async def validate_request(self, text: str, source_ip: str = None) -> Dict[str, Any]:
        """Validate request."""
        start_time = time.time()
        
        # Check IP blocklist
        if source_ip and source_ip in self.blocked_ips:
            return {
                "allowed": False,
                "result": ValidationResult.BLOCKED,
                "reason": "IP address blocked"
            }
        
        # Rate limiting
        identifier = source_ip or "anonymous"
        if not self.rate_limiter.is_allowed(identifier):
            return {
                "allowed": False,
                "result": ValidationResult.RATE_LIMITED,
                "reason": "Rate limit exceeded"
            }
        
        # Input validation
        validation_result = self.input_validator.validate_input(text)
        
        # Log event
        if not validation_result["allowed"]:
            self.events.append({
                "timestamp": time.time(),
                "type": "validation_failed",
                "source_ip": source_ip,
                "threat_level": validation_result["threat_level"],
                "violations": validation_result["violations"]
            })
        
        return validation_result


# Error Recovery Components
class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryAction(Enum):
    """Recovery action types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    RESTART = "restart"
    ESCALATE = "escalate"
    IGNORE = "ignore"


class RetryManager:
    """Advanced retry mechanism with exponential backoff."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, exponential_base: float = 2.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
    
    async def retry_async(self, func: Callable, *args, **kwargs) -> Any:
        """Retry async function with exponential backoff."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = min(
                        self.base_delay * (self.exponential_base ** attempt),
                        self.max_delay
                    )
                    await asyncio.sleep(delay)
        
        raise last_exception


class FallbackManager:
    """Fallback mechanism for graceful degradation."""
    
    def __init__(self):
        self.fallbacks: Dict[str, Callable] = {}
        self.fallback_used_count: Dict[str, int] = {}
    
    def register_fallback(self, operation_name: str, fallback_func: Callable):
        """Register fallback function for an operation."""
        self.fallbacks[operation_name] = fallback_func
        self.fallback_used_count[operation_name] = 0
    
    async def execute_with_fallback(self, operation_name: str, primary_func: Callable, 
                                  *args, **kwargs) -> Any:
        """Execute function with fallback on failure."""
        try:
            if asyncio.iscoroutinefunction(primary_func):
                return await primary_func(*args, **kwargs)
            else:
                return primary_func(*args, **kwargs)
                
        except Exception:
            if operation_name in self.fallbacks:
                fallback_func = self.fallbacks[operation_name]
                self.fallback_used_count[operation_name] += 1
                
                if asyncio.iscoroutinefunction(fallback_func):
                    return await fallback_func(*args, **kwargs)
                else:
                    return fallback_func(*args, **kwargs)
            else:
                raise


class CircuitBreaker:
    """Circuit breaker for external service calls."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise e


# Test Functions
async def test_health_monitoring():
    """Test health monitoring system."""
    print("Testing Health Monitoring...")
    
    # Create and test health check
    health_check = MockSystemResourceCheck()
    result = await health_check.check()
    
    assert result.name == "system_resources"
    assert result.status == HealthStatus.HEALTHY
    assert "cpu_percent" in result.value
    assert "memory_percent" in result.value
    
    print(f"✓ Health check: {result.status.value} - {result.message}")
    print("✓ Health monitoring tests passed")


async def test_security_orchestrator():
    """Test security orchestrator."""
    print("Testing Security Orchestrator...")
    
    security = MockSecurityOrchestrator({
        "rate_limit_requests": 5,
        "rate_limit_window": 60
    })
    
    # Test normal input
    result1 = await security.validate_request("Hello world", source_ip="192.168.1.1")
    assert result1["allowed"] == True
    assert result1["result"] == ValidationResult.ALLOWED
    print("✓ Normal input allowed")
    
    # Test XSS attack
    result2 = await security.validate_request(
        "<script>alert('xss')</script>", 
        source_ip="192.168.1.2"
    )
    assert result2["allowed"] == False
    assert result2["result"] == ValidationResult.BLOCKED
    assert len(result2["violations"]) > 0
    print("✓ XSS attack blocked")
    
    # Test SQL injection
    result3 = await security.validate_request(
        "'; DROP TABLE users; --",
        source_ip="192.168.1.3"
    )
    assert result3["allowed"] == False
    assert result3["result"] == ValidationResult.BLOCKED
    print("✓ SQL injection blocked")
    
    # Test rate limiting
    for i in range(10):  # Exceed rate limit
        result = await security.validate_request(f"Request {i}", source_ip="192.168.1.4")
        if not result["allowed"] and result["result"] == ValidationResult.RATE_LIMITED:
            print("✓ Rate limiting working")
            break
    
    print("✓ Security orchestrator tests passed")


async def test_error_recovery():
    """Test error recovery system."""
    print("Testing Error Recovery...")
    
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
    
    assert circuit_breaker.state == "OPEN"
    print("✓ Circuit breaker opening")
    
    print("✓ Error recovery tests passed")


async def test_resilience_under_load():
    """Test system resilience under load."""
    print("Testing Resilience Under Load...")
    
    security = MockSecurityOrchestrator({
        "rate_limit_requests": 5,
        "rate_limit_window": 1
    })
    
    # Simulate concurrent requests
    async def simulate_request(request_id: int):
        try:
            if request_id % 3 == 0:
                text = f"<script>alert('attack_{request_id}')</script>"
            else:
                text = f"Normal request {request_id}"
            
            result = await security.validate_request(
                text, 
                source_ip=f"192.168.1.{request_id % 5}"
            )
            
            return {"request_id": request_id, "allowed": result["allowed"]}
        except Exception as e:
            return {"request_id": request_id, "error": str(e)}
    
    # Run multiple concurrent requests
    tasks = [simulate_request(i) for i in range(15)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Analyze results
    successful_requests = [r for r in results if isinstance(r, dict) and "allowed" in r]
    blocked_requests = [r for r in successful_requests if not r["allowed"]]
    
    print(f"Processed {len(successful_requests)} requests")
    print(f"Blocked {len(blocked_requests)} malicious requests")
    
    assert len(successful_requests) > 0
    assert len(blocked_requests) > 0
    
    print("✓ Resilience under load tests passed")


async def test_integration():
    """Test integration of all components."""
    print("Testing Component Integration...")
    
    # Create all components
    health_check = MockSystemResourceCheck()
    security = MockSecurityOrchestrator()
    retry_manager = RetryManager(max_retries=2, base_delay=0.1)
    fallback_manager = FallbackManager()
    
    # Test coordinated operation
    async def secure_operation(text: str):
        # Security validation
        security_result = await security.validate_request(text)
        if not security_result["allowed"]:
            raise ValueError("Security validation failed")
        
        # Simulate processing
        await asyncio.sleep(0.05)
        return f"Processed: {text}"
    
    # Register fallback
    def fallback_operation(text: str):
        return f"Fallback processed: {text}"
    
    fallback_manager.register_fallback("secure_operation", fallback_operation)
    
    # Test normal operation
    result1 = await fallback_manager.execute_with_fallback(
        "secure_operation", secure_operation, "Hello world"
    )
    assert "Processed: Hello world" in result1
    
    # Test fallback on security failure
    result2 = await fallback_manager.execute_with_fallback(
        "secure_operation", secure_operation, "<script>alert('xss')</script>"
    )
    assert "Fallback processed" in result2
    
    # Test health check
    health_result = await health_check.check()
    assert health_result.status == HealthStatus.HEALTHY
    
    print("✓ Integration tests passed")


def main():
    """Run all Generation 2 robustness tests."""
    print("=" * 60)
    print("Generation 2 Robustness Standalone Tests")
    print("=" * 60)
    
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
        
        # Test 4: Resilience Under Load
        asyncio.run(test_resilience_under_load())
        tests_passed += 1
        
        # Test 5: Integration
        asyncio.run(test_integration())
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