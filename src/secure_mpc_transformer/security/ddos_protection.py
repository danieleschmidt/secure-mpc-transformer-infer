"""Advanced DDoS protection and rate limiting system."""

import time
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum
import threading
import ipaddress
import hashlib
import secrets
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ProtectionLevel(Enum):
    """DDoS protection levels."""
    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EMERGENCY = "emergency"


class AttackType(Enum):
    """Types of detected attacks."""
    VOLUMETRIC = "volumetric"
    PROTOCOL = "protocol"
    APPLICATION = "application"
    SLOW_LORIS = "slow_loris"
    HTTP_FLOOD = "http_flood"
    SYN_FLOOD = "syn_flood"
    UDP_FLOOD = "udp_flood"


@dataclass
class RateLimitRule:
    """Rate limiting rule configuration."""
    
    name: str
    requests_per_second: int
    requests_per_minute: int
    requests_per_hour: int
    burst_limit: int
    block_duration: int  # seconds
    priority: int
    conditions: Dict[str, Any]  # Conditions for applying this rule


@dataclass
class TrafficMetrics:
    """Traffic metrics for analysis."""
    
    timestamp: float
    requests_per_second: float
    bytes_per_second: float
    unique_ips: int
    error_rate: float
    avg_response_time: float
    connection_count: int
    suspicious_patterns: List[str]


@dataclass
class AttackEvent:
    """DDoS attack event."""
    
    event_id: str
    attack_type: AttackType
    start_time: float
    end_time: Optional[float]
    source_ips: Set[str]
    target_endpoints: Set[str]
    peak_rps: float
    total_requests: int
    mitigation_actions: List[str]
    severity: str
    confidence: float


class TokenBucket:
    """Token bucket rate limiter."""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.time()
        self._lock = threading.Lock()
    
    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens from bucket."""
        with self._lock:
            now = time.time()
            
            # Add tokens based on elapsed time
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def get_tokens(self) -> float:
        """Get current token count."""
        with self._lock:
            return self.tokens


class SlidingWindowCounter:
    """Sliding window rate counter."""
    
    def __init__(self, window_size: int, bucket_count: int = 60):
        self.window_size = window_size  # seconds
        self.bucket_count = bucket_count
        self.bucket_duration = window_size / bucket_count
        self.buckets = deque([0] * bucket_count, maxlen=bucket_count)
        self.last_update = time.time()
        self._lock = threading.Lock()
    
    def add_request(self, count: int = 1):
        """Add request to current bucket."""
        with self._lock:
            self._update_buckets()
            self.buckets[-1] += count
    
    def get_count(self) -> int:
        """Get total count in current window."""
        with self._lock:
            self._update_buckets()
            return sum(self.buckets)
    
    def _update_buckets(self):
        """Update buckets based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update
        
        if elapsed >= self.bucket_duration:
            buckets_to_advance = min(self.bucket_count, int(elapsed / self.bucket_duration))
            
            # Add new empty buckets
            for _ in range(buckets_to_advance):
                self.buckets.append(0)
            
            self.last_update = now


class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on traffic patterns."""
    
    def __init__(self, base_limit: int, max_limit: int, adaptation_rate: float = 0.1):
        self.base_limit = base_limit
        self.max_limit = max_limit
        self.current_limit = base_limit
        self.adaptation_rate = adaptation_rate
        
        self.success_rate_tracker = SlidingWindowCounter(60)  # 1 minute window
        self.total_requests_tracker = SlidingWindowCounter(60)
        
        self.token_bucket = TokenBucket(base_limit, base_limit / 60.0)  # per second
        self._lock = threading.Lock()
    
    def is_allowed(self, request_size: int = 1) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is allowed and return metadata."""
        with self._lock:
            self.total_requests_tracker.add_request()
            
            allowed = self.token_bucket.consume(request_size)
            
            if allowed:
                self.success_rate_tracker.add_request()
            
            # Adapt rate limit based on success rate
            self._adapt_rate_limit()
            
            metadata = {
                'current_limit': self.current_limit,
                'tokens_remaining': self.token_bucket.get_tokens(),
                'success_rate': self._get_success_rate(),
                'total_requests': self.total_requests_tracker.get_count()
            }
            
            return allowed, metadata
    
    def _adapt_rate_limit(self):
        """Adapt rate limit based on traffic patterns."""
        success_rate = self._get_success_rate()
        
        if success_rate > 0.95:  # High success rate, can increase limit
            new_limit = min(self.max_limit, self.current_limit * (1 + self.adaptation_rate))
        elif success_rate < 0.8:  # Low success rate, decrease limit
            new_limit = max(self.base_limit, self.current_limit * (1 - self.adaptation_rate))
        else:
            new_limit = self.current_limit
        
        if new_limit != self.current_limit:
            self.current_limit = new_limit
            # Update token bucket capacity
            self.token_bucket.capacity = int(new_limit)
            self.token_bucket.refill_rate = new_limit / 60.0
    
    def _get_success_rate(self) -> float:
        """Calculate current success rate."""
        total = self.total_requests_tracker.get_count()
        successful = self.success_rate_tracker.get_count()
        
        if total == 0:
            return 1.0
        
        return successful / total


class GeolocationFilter:
    """Geographic filtering for DDoS protection."""
    
    def __init__(self):
        self.allowed_countries = set()
        self.blocked_countries = set()
        self.blocked_regions = set()
        self.ip_whitelist = set()
        self.ip_blacklist = set()
    
    def is_allowed(self, ip: str, country: str = None, region: str = None) -> Tuple[bool, str]:
        """Check if IP/location is allowed."""
        # Check IP whitelist first
        if ip in self.ip_whitelist:
            return True, "whitelisted_ip"
        
        # Check IP blacklist
        if ip in self.ip_blacklist:
            return False, "blacklisted_ip"
        
        # Check country restrictions
        if self.allowed_countries and country and country not in self.allowed_countries:
            return False, "country_not_allowed"
        
        if country and country in self.blocked_countries:
            return False, "blocked_country"
        
        # Check region restrictions
        if region and region in self.blocked_regions:
            return False, "blocked_region"
        
        return True, "allowed"


class TrafficAnalyzer:
    """Analyze traffic patterns for anomaly detection."""
    
    def __init__(self, window_size: int = 300):  # 5 minutes
        self.window_size = window_size
        self.metrics_history = deque(maxlen=1000)
        self.baseline_metrics = None
        self.anomaly_threshold = 3.0  # Standard deviations
        self._lock = threading.Lock()
    
    def add_metrics(self, metrics: TrafficMetrics):
        """Add traffic metrics for analysis."""
        with self._lock:
            self.metrics_history.append(metrics)
            
            # Update baseline if we have enough data
            if len(self.metrics_history) >= 100:
                self._update_baseline()
    
    def detect_anomalies(self, current_metrics: TrafficMetrics) -> List[str]:
        """Detect traffic anomalies."""
        anomalies = []
        
        if not self.baseline_metrics:
            return anomalies
        
        # Request rate anomaly
        if current_metrics.requests_per_second > self.baseline_metrics['avg_rps'] + (
            self.baseline_metrics['std_rps'] * self.anomaly_threshold):
            anomalies.append("abnormal_request_rate")
        
        # Error rate anomaly
        if current_metrics.error_rate > self.baseline_metrics['avg_error_rate'] + 0.2:
            anomalies.append("high_error_rate")
        
        # Response time anomaly
        if current_metrics.avg_response_time > self.baseline_metrics['avg_response_time'] * 3:
            anomalies.append("slow_response_time")
        
        # Unique IP anomaly
        if current_metrics.unique_ips > self.baseline_metrics['avg_unique_ips'] * 5:
            anomalies.append("unusual_ip_diversity")
        
        return anomalies
    
    def _update_baseline(self):
        """Update baseline metrics from historical data."""
        if len(self.metrics_history) < 50:
            return
        
        # Calculate statistics from recent data
        recent_metrics = list(self.metrics_history)[-100:]  # Last 100 data points
        
        rps_values = [m.requests_per_second for m in recent_metrics]
        error_rates = [m.error_rate for m in recent_metrics]
        response_times = [m.avg_response_time for m in recent_metrics]
        unique_ips = [m.unique_ips for m in recent_metrics]
        
        self.baseline_metrics = {
            'avg_rps': sum(rps_values) / len(rps_values),
            'std_rps': self._calculate_std_dev(rps_values),
            'avg_error_rate': sum(error_rates) / len(error_rates),
            'avg_response_time': sum(response_times) / len(response_times),
            'avg_unique_ips': sum(unique_ips) / len(unique_ips)
        }
    
    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5


class DDosProtectionSystem:
    """Comprehensive DDoS protection system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Protection configuration
        self.protection_level = ProtectionLevel(self.config.get('protection_level', 'medium'))
        self.auto_mitigation = self.config.get('auto_mitigation', True)
        self.learning_mode = self.config.get('learning_mode', True)
        
        # Rate limiting rules
        self.rate_limit_rules = self._initialize_rate_limit_rules()
        
        # Components
        self.traffic_analyzer = TrafficAnalyzer(self.config.get('analysis_window', 300))
        self.geo_filter = GeolocationFilter()
        
        # Per-IP rate limiters
        self.ip_limiters = {}
        self.ip_last_access = {}
        
        # Attack tracking
        self.active_attacks = {}
        self.attack_history = deque(maxlen=1000)
        
        # Mitigation state
        self.blocked_ips = set()
        self.rate_limited_ips = {}  # ip -> until_timestamp
        self.emergency_mode = False
        self.emergency_start_time = None
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'blocked_requests': 0,
            'rate_limited_requests': 0,
            'attacks_detected': 0,
            'false_positives': 0
        }
        
        self._lock = threading.Lock()
        
        # Start cleanup task
        self._start_cleanup_task()
        
        logger.info(f"DDoS protection system initialized with level: {self.protection_level.value}")
    
    def _initialize_rate_limit_rules(self) -> List[RateLimitRule]:
        """Initialize default rate limiting rules."""
        rules = []
        
        # API endpoint rules
        rules.append(RateLimitRule(
            name="api_inference",
            requests_per_second=10,
            requests_per_minute=100,
            requests_per_hour=1000,
            burst_limit=20,
            block_duration=60,
            priority=1,
            conditions={"endpoint_prefix": "/api/inference"}
        ))
        
        rules.append(RateLimitRule(
            name="api_general",
            requests_per_second=20,
            requests_per_minute=300,
            requests_per_hour=3000,
            burst_limit=50,
            block_duration=30,
            priority=2,
            conditions={"endpoint_prefix": "/api/"}
        ))
        
        # Global rate limit
        rules.append(RateLimitRule(
            name="global",
            requests_per_second=50,
            requests_per_minute=1000,
            requests_per_hour=10000,
            burst_limit=100,
            block_duration=60,
            priority=3,
            conditions={}
        ))
        
        return rules
    
    async def check_request(self, ip: str, endpoint: str, user_agent: str = "",
                          request_size: int = 1, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Check if request should be allowed."""
        current_time = time.time()
        metadata = metadata or {}
        
        with self._lock:
            self.stats['total_requests'] += 1
        
        # Check if IP is blocked
        if ip in self.blocked_ips:
            with self._lock:
                self.stats['blocked_requests'] += 1
            return {
                'allowed': False,
                'reason': 'ip_blocked',
                'retry_after': None,
                'protection_level': self.protection_level.value
            }
        
        # Check rate limiting
        rate_limit_result = await self._check_rate_limits(ip, endpoint, current_time, request_size)
        if not rate_limit_result['allowed']:
            with self._lock:
                self.stats['rate_limited_requests'] += 1
            return rate_limit_result
        
        # Geographic filtering
        geo_allowed, geo_reason = self.geo_filter.is_allowed(
            ip, 
            metadata.get('country'),
            metadata.get('region')
        )
        if not geo_allowed:
            return {
                'allowed': False,
                'reason': geo_reason,
                'retry_after': None,
                'protection_level': self.protection_level.value
            }
        
        # Emergency mode checks
        if self.emergency_mode:
            emergency_result = await self._emergency_mode_check(ip, endpoint, metadata)
            if not emergency_result['allowed']:
                return emergency_result
        
        # Update traffic metrics
        self._update_traffic_metrics(ip, endpoint, current_time, metadata)
        
        return {
            'allowed': True,
            'reason': 'approved',
            'protection_level': self.protection_level.value,
            'rate_limit_info': rate_limit_result.get('metadata', {})
        }
    
    async def _check_rate_limits(self, ip: str, endpoint: str, current_time: float,
                               request_size: int) -> Dict[str, Any]:
        """Check rate limits for the request."""
        
        # Find applicable rate limit rule
        applicable_rule = None
        for rule in sorted(self.rate_limit_rules, key=lambda r: r.priority):
            if self._rule_matches(rule, endpoint):
                applicable_rule = rule
                break
        
        if not applicable_rule:
            return {'allowed': True}
        
        # Get or create rate limiter for this IP
        if ip not in self.ip_limiters:
            self.ip_limiters[ip] = AdaptiveRateLimiter(
                base_limit=applicable_rule.requests_per_minute,
                max_limit=applicable_rule.requests_per_minute * 2
            )
        
        limiter = self.ip_limiters[ip]
        allowed, metadata = limiter.is_allowed(request_size)
        
        if not allowed:
            # Rate limit exceeded, add to temporary block list
            block_until = current_time + applicable_rule.block_duration
            self.rate_limited_ips[ip] = block_until
            
            return {
                'allowed': False,
                'reason': 'rate_limit_exceeded',
                'retry_after': applicable_rule.block_duration,
                'rule_name': applicable_rule.name,
                'metadata': metadata
            }
        
        return {'allowed': True, 'metadata': metadata}
    
    def _rule_matches(self, rule: RateLimitRule, endpoint: str) -> bool:
        """Check if rule applies to the endpoint."""
        conditions = rule.conditions
        
        if 'endpoint_prefix' in conditions:
            if not endpoint.startswith(conditions['endpoint_prefix']):
                return False
        
        if 'endpoint_exact' in conditions:
            if endpoint != conditions['endpoint_exact']:
                return False
        
        return True
    
    async def _emergency_mode_check(self, ip: str, endpoint: str,
                                  metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Additional checks during emergency mode."""
        
        # Only allow essential endpoints during emergency
        essential_endpoints = ['/health', '/api/status']
        
        if endpoint not in essential_endpoints:
            return {
                'allowed': False,
                'reason': 'emergency_mode_restriction',
                'retry_after': 300,  # 5 minutes
                'protection_level': self.protection_level.value
            }
        
        # Stricter rate limiting during emergency
        current_time = time.time()
        if ip not in self.ip_limiters:
            self.ip_limiters[ip] = AdaptiveRateLimiter(base_limit=10, max_limit=10)
        
        limiter = self.ip_limiters[ip]
        allowed, limit_metadata = limiter.is_allowed(1)
        
        if not allowed:
            return {
                'allowed': False,
                'reason': 'emergency_rate_limit',
                'retry_after': 60,
                'protection_level': self.protection_level.value
            }
        
        return {'allowed': True}
    
    def _update_traffic_metrics(self, ip: str, endpoint: str, timestamp: float,
                               metadata: Dict[str, Any]):
        """Update traffic metrics for analysis."""
        
        # Update IP last access time
        self.ip_last_access[ip] = timestamp
        
        # Collect metrics (simplified version)
        current_metrics = TrafficMetrics(
            timestamp=timestamp,
            requests_per_second=self._calculate_current_rps(),
            bytes_per_second=metadata.get('request_size', 0),
            unique_ips=len(self.ip_last_access),
            error_rate=metadata.get('error_rate', 0.0),
            avg_response_time=metadata.get('response_time', 0.0),
            connection_count=len(self.ip_limiters),
            suspicious_patterns=[]
        )
        
        # Add to analyzer
        self.traffic_analyzer.add_metrics(current_metrics)
        
        # Check for anomalies
        anomalies = self.traffic_analyzer.detect_anomalies(current_metrics)
        if anomalies and self.auto_mitigation:
            asyncio.create_task(self._handle_traffic_anomalies(anomalies, current_metrics))
    
    def _calculate_current_rps(self) -> float:
        """Calculate current requests per second."""
        current_time = time.time()
        recent_requests = 0
        
        # Count requests in last second from all IPs
        for limiter in self.ip_limiters.values():
            if hasattr(limiter, 'total_requests_tracker'):
                recent_requests += limiter.total_requests_tracker.get_count()
        
        return recent_requests
    
    async def _handle_traffic_anomalies(self, anomalies: List[str], metrics: TrafficMetrics):
        """Handle detected traffic anomalies."""
        logger.warning(f"Traffic anomalies detected: {anomalies}")
        
        # Escalate protection level based on anomalies
        critical_anomalies = ['abnormal_request_rate', 'unusual_ip_diversity']
        
        if any(anomaly in critical_anomalies for anomaly in anomalies):
            await self.escalate_protection_level()
        
        # Create attack event
        attack_event = AttackEvent(
            event_id=secrets.token_hex(8),
            attack_type=AttackType.HTTP_FLOOD,
            start_time=metrics.timestamp,
            end_time=None,
            source_ips=set(self.ip_last_access.keys()),
            target_endpoints=set(),
            peak_rps=metrics.requests_per_second,
            total_requests=0,
            mitigation_actions=anomalies,
            severity="medium",
            confidence=0.7
        )
        
        self.active_attacks[attack_event.event_id] = attack_event
        with self._lock:
            self.stats['attacks_detected'] += 1
    
    async def escalate_protection_level(self):
        """Escalate protection level during attack."""
        current_level = self.protection_level
        
        if current_level == ProtectionLevel.LOW:
            self.protection_level = ProtectionLevel.MEDIUM
        elif current_level == ProtectionLevel.MEDIUM:
            self.protection_level = ProtectionLevel.HIGH
        elif current_level == ProtectionLevel.HIGH:
            self.protection_level = ProtectionLevel.EMERGENCY
            self.emergency_mode = True
            self.emergency_start_time = time.time()
        
        logger.warning(f"Protection level escalated to: {self.protection_level.value}")
    
    async def de_escalate_protection_level(self):
        """De-escalate protection level when threat subsides."""
        current_level = self.protection_level
        
        if current_level == ProtectionLevel.EMERGENCY:
            self.protection_level = ProtectionLevel.HIGH
            self.emergency_mode = False
            self.emergency_start_time = None
        elif current_level == ProtectionLevel.HIGH:
            self.protection_level = ProtectionLevel.MEDIUM
        elif current_level == ProtectionLevel.MEDIUM:
            self.protection_level = ProtectionLevel.LOW
        
        logger.info(f"Protection level de-escalated to: {self.protection_level.value}")
    
    def block_ip(self, ip: str, reason: str, duration: Optional[int] = None):
        """Block an IP address."""
        self.blocked_ips.add(ip)
        
        if duration:
            # Schedule unblock
            async def unblock_after_duration():
                await asyncio.sleep(duration)
                self.unblock_ip(ip)
            
            asyncio.create_task(unblock_after_duration())
        
        logger.warning(f"IP {ip} blocked: {reason}")
    
    def unblock_ip(self, ip: str):
        """Unblock an IP address."""
        self.blocked_ips.discard(ip)
        if ip in self.rate_limited_ips:
            del self.rate_limited_ips[ip]
        
        logger.info(f"IP {ip} unblocked")
    
    def get_protection_stats(self) -> Dict[str, Any]:
        """Get protection system statistics."""
        current_time = time.time()
        
        with self._lock:
            base_stats = self.stats.copy()
        
        # Add runtime statistics
        active_rate_limited = sum(
            1 for until_time in self.rate_limited_ips.values()
            if until_time > current_time
        )
        
        return {
            **base_stats,
            'protection_level': self.protection_level.value,
            'emergency_mode': self.emergency_mode,
            'blocked_ips': len(self.blocked_ips),
            'rate_limited_ips': active_rate_limited,
            'active_attacks': len(self.active_attacks),
            'unique_clients': len(self.ip_limiters),
            'learning_mode': self.learning_mode
        }
    
    def _start_cleanup_task(self):
        """Start background cleanup task."""
        async def cleanup_loop():
            while True:
                try:
                    await self._cleanup_expired_data()
                    await asyncio.sleep(60)  # Run every minute
                except Exception as e:
                    logger.error(f"DDoS protection cleanup error: {e}")
                    await asyncio.sleep(60)
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(cleanup_loop())
        except RuntimeError:
            # No event loop, start manual cleanup
            threading.Timer(60, self._manual_cleanup).start()
    
    async def _cleanup_expired_data(self):
        """Clean up expired data."""
        current_time = time.time()
        
        # Remove expired rate limits
        expired_ips = [
            ip for ip, until_time in self.rate_limited_ips.items()
            if until_time <= current_time
        ]
        
        for ip in expired_ips:
            del self.rate_limited_ips[ip]
        
        # Clean up old IP limiters (inactive for > 1 hour)
        inactive_cutoff = current_time - 3600
        inactive_ips = [
            ip for ip, last_time in self.ip_last_access.items()
            if last_time < inactive_cutoff
        ]
        
        for ip in inactive_ips:
            if ip in self.ip_limiters:
                del self.ip_limiters[ip]
            del self.ip_last_access[ip]
        
        # End emergency mode after 30 minutes
        if (self.emergency_mode and self.emergency_start_time and 
            current_time - self.emergency_start_time > 1800):
            await self.de_escalate_protection_level()
        
        logger.debug(f"Cleaned up {len(expired_ips)} expired rate limits and {len(inactive_ips)} inactive IPs")
    
    def _manual_cleanup(self):
        """Manual cleanup for non-async environments."""
        try:
            asyncio.run(self._cleanup_expired_data())
        except Exception as e:
            logger.error(f"Manual DDoS cleanup error: {e}")
        finally:
            threading.Timer(60, self._manual_cleanup).start()


# Global DDoS protection instance
ddos_protection = DDosProtectionSystem()