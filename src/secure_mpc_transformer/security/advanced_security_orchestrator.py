#!/usr/bin/env python3
"""
Advanced Security Orchestrator for Secure MPC Transformer

High-performance security orchestration system with intelligent load balancing,
adaptive caching, auto-scaling security controls, and comprehensive threat correlation
for large-scale deployment scenarios.
"""

import asyncio
import hashlib
import logging
import time
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np

from ..monitoring.security_dashboard import SecurityMetricsDashboard
from ..utils.error_handling import SecurityError
from ..utils.metrics import MetricsCollector
from .enhanced_validator import (
    EnhancedSecurityValidator,
    ValidationContext,
)
from .incident_response import AIIncidentResponseSystem
from .quantum_monitor import QuantumSecurityMonitor

logger = logging.getLogger(__name__)


class ScalingStrategy(Enum):
    """Auto-scaling strategies for security components."""
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    ADAPTIVE = "adaptive"


class LoadBalancingMethod(Enum):
    """Load balancing methods for security processing."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_RESPONSE_TIME = "weighted_response_time"
    RESOURCE_AWARE = "resource_aware"
    THREAT_AWARE = "threat_aware"


class CachingStrategy(Enum):
    """Caching strategies for security validation."""
    LRU = "lru"
    LFU = "lfu"
    ADAPTIVE = "adaptive"
    THREAT_AWARE = "threat_aware"


@dataclass
class SecurityWorkerPool:
    """Configuration for security worker pool."""
    pool_type: str  # "thread" or "process"
    min_workers: int
    max_workers: int
    current_workers: int
    worker_capacity: int
    load_balancer: LoadBalancingMethod
    executor: ThreadPoolExecutor | ProcessPoolExecutor | None = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for security operations."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    current_rps: float = 0.0  # Requests per second
    cache_hit_rate: float = 0.0
    threat_detection_rate: float = 0.0
    false_positive_rate: float = 0.0


@dataclass
class ThreatCorrelationRule:
    """Rule for correlating related security threats."""
    rule_id: str
    name: str
    description: str
    conditions: list[dict[str, Any]]
    correlation_window: int  # seconds
    threshold: int  # minimum occurrences
    severity_boost: float  # severity multiplier when correlated
    enabled: bool = True


class AdaptiveSecurityCache:
    """
    High-performance adaptive caching system for security validations.
    
    Features:
    - Multiple cache layers (L1, L2, distributed)
    - Intelligent cache warming
    - Threat-aware eviction policies
    - Performance optimization
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.cache_strategy = CachingStrategy(config.get("strategy", "adaptive"))
        self.max_size = config.get("max_size", 100000)
        self.ttl = config.get("ttl", 3600)  # 1 hour default

        # Multi-layer cache
        self.l1_cache = {}  # In-memory fast cache
        self.l2_cache = {}  # Secondary cache with larger capacity
        self.access_count = defaultdict(int)
        self.access_time = {}
        self.threat_scores = {}  # Cache threat scores for intelligent eviction

        # Performance tracking
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "entries": 0
        }

        # Background maintenance
        self._maintenance_task = None

    async def start_cache(self) -> None:
        """Start the adaptive cache system."""
        logger.info("Starting adaptive security cache")
        self._maintenance_task = asyncio.create_task(self._cache_maintenance_loop())

    async def get(self, key: str) -> dict[str, Any] | None:
        """Get item from cache with adaptive access tracking."""
        try:
            current_time = time.time()

            # Check L1 cache first
            if key in self.l1_cache:
                entry = self.l1_cache[key]
                if current_time - entry["timestamp"] < self.ttl:
                    self.access_count[key] += 1
                    self.access_time[key] = current_time
                    self.cache_stats["hits"] += 1
                    return entry["data"]
                else:
                    # Expired entry
                    del self.l1_cache[key]

            # Check L2 cache
            if key in self.l2_cache:
                entry = self.l2_cache[key]
                if current_time - entry["timestamp"] < self.ttl:
                    # Promote to L1 cache
                    self.l1_cache[key] = entry
                    self.access_count[key] += 1
                    self.access_time[key] = current_time
                    self.cache_stats["hits"] += 1
                    return entry["data"]
                else:
                    # Expired entry
                    del self.l2_cache[key]

            # Cache miss
            self.cache_stats["misses"] += 1
            return None

        except Exception as e:
            logger.error(f"Cache get operation failed: {e}")
            self.cache_stats["misses"] += 1
            return None

    async def set(
        self,
        key: str,
        data: dict[str, Any],
        threat_score: float = 0.0,
        priority: float = 1.0
    ) -> bool:
        """Set item in cache with intelligent placement."""
        try:
            current_time = time.time()
            entry = {
                "data": data,
                "timestamp": current_time,
                "threat_score": threat_score,
                "priority": priority
            }

            # Determine cache layer based on priority and threat score
            if priority > 0.8 or threat_score > 0.7:
                # High priority or high threat - store in L1
                target_cache = self.l1_cache
                cache_limit = self.max_size // 4  # 25% for L1
            else:
                # Normal priority - store in L2
                target_cache = self.l2_cache
                cache_limit = self.max_size * 3 // 4  # 75% for L2

            # Evict if necessary
            if len(target_cache) >= cache_limit:
                await self._evict_entries(target_cache, cache_limit // 10)  # Evict 10%

            target_cache[key] = entry
            self.access_count[key] = 1
            self.access_time[key] = current_time
            self.threat_scores[key] = threat_score

            self.cache_stats["entries"] = len(self.l1_cache) + len(self.l2_cache)

            return True

        except Exception as e:
            logger.error(f"Cache set operation failed: {e}")
            return False

    async def invalidate(self, key: str) -> bool:
        """Invalidate cache entry."""
        try:
            found = False

            if key in self.l1_cache:
                del self.l1_cache[key]
                found = True

            if key in self.l2_cache:
                del self.l2_cache[key]
                found = True

            if key in self.access_count:
                del self.access_count[key]
            if key in self.access_time:
                del self.access_time[key]
            if key in self.threat_scores:
                del self.threat_scores[key]

            if found:
                self.cache_stats["entries"] = len(self.l1_cache) + len(self.l2_cache)

            return found

        except Exception as e:
            logger.error(f"Cache invalidation failed: {e}")
            return False

    async def _evict_entries(self, cache: dict[str, Any], count: int) -> None:
        """Evict entries using adaptive strategy."""
        try:
            if not cache or count <= 0:
                return

            current_time = time.time()

            if self.cache_strategy == CachingStrategy.LRU:
                # Least Recently Used
                candidates = [(key, self.access_time.get(key, 0)) for key in cache]
                candidates.sort(key=lambda x: x[1])

            elif self.cache_strategy == CachingStrategy.LFU:
                # Least Frequently Used
                candidates = [(key, self.access_count.get(key, 0)) for key in cache]
                candidates.sort(key=lambda x: x[1])

            elif self.cache_strategy == CachingStrategy.THREAT_AWARE:
                # Evict low-threat, old entries first
                candidates = []
                for key in cache:
                    threat_score = self.threat_scores.get(key, 0.0)
                    age = current_time - self.access_time.get(key, current_time)
                    # Lower score = more likely to be evicted
                    eviction_score = (1.0 - threat_score) * age
                    candidates.append((key, eviction_score))

                candidates.sort(key=lambda x: x[1], reverse=True)

            else:  # Adaptive
                # Combine multiple factors
                candidates = []
                for key in cache:
                    access_freq = self.access_count.get(key, 0)
                    last_access = self.access_time.get(key, 0)
                    threat_score = self.threat_scores.get(key, 0.0)
                    age = current_time - last_access

                    # Adaptive score (higher = more likely to evict)
                    adaptive_score = age / max(access_freq, 1) / max(threat_score, 0.1)
                    candidates.append((key, adaptive_score))

                candidates.sort(key=lambda x: x[1], reverse=True)

            # Evict selected entries
            evicted = 0
            for key, _ in candidates[:count]:
                if key in cache:
                    del cache[key]
                    evicted += 1

                    # Clean up related data
                    if key in self.access_count:
                        del self.access_count[key]
                    if key in self.access_time:
                        del self.access_time[key]
                    if key in self.threat_scores:
                        del self.threat_scores[key]

            self.cache_stats["evictions"] += evicted
            logger.debug(f"Evicted {evicted} entries using {self.cache_strategy.value} strategy")

        except Exception as e:
            logger.error(f"Cache eviction failed: {e}")

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / max(total_requests, 1)

        return {
            "strategy": self.cache_strategy.value,
            "l1_entries": len(self.l1_cache),
            "l2_entries": len(self.l2_cache),
            "total_entries": self.cache_stats["entries"],
            "max_size": self.max_size,
            "hit_rate": hit_rate,
            "hits": self.cache_stats["hits"],
            "misses": self.cache_stats["misses"],
            "evictions": self.cache_stats["evictions"]
        }

    async def _cache_maintenance_loop(self) -> None:
        """Background cache maintenance loop."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes

                current_time = time.time()

                # Remove expired entries
                expired_keys = []

                for cache in [self.l1_cache, self.l2_cache]:
                    for key, entry in list(cache.items()):
                        if current_time - entry["timestamp"] > self.ttl:
                            expired_keys.append(key)

                for key in expired_keys:
                    await self.invalidate(key)

                # Update cache statistics
                self.cache_stats["entries"] = len(self.l1_cache) + len(self.l2_cache)

                logger.debug(f"Cache maintenance: removed {len(expired_keys)} expired entries")

            except Exception as e:
                logger.error(f"Cache maintenance loop error: {e}")
                await asyncio.sleep(600)  # Back off on error


class IntelligentLoadBalancer:
    """
    Intelligent load balancer for security processing workloads.
    
    Features:
    - Multiple load balancing algorithms
    - Real-time performance monitoring
    - Adaptive routing based on threat levels
    - Circuit breaker integration
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.method = LoadBalancingMethod(config.get("method", "threat_aware"))
        self.worker_pools = {}
        self.worker_stats = defaultdict(lambda: {
            "requests": 0,
            "response_times": deque(maxlen=1000),
            "error_rate": 0.0,
            "capacity_utilization": 0.0,
            "threat_processing_score": 1.0
        })
        self.circuit_breakers = {}
        self.routing_history = deque(maxlen=10000)

    async def register_worker_pool(
        self,
        pool_id: str,
        pool: SecurityWorkerPool
    ) -> None:
        """Register a worker pool for load balancing."""
        try:
            self.worker_pools[pool_id] = pool
            self.circuit_breakers[pool_id] = {
                "state": "CLOSED",  # CLOSED, OPEN, HALF_OPEN
                "failure_count": 0,
                "last_failure_time": 0,
                "reset_timeout": 60.0  # seconds
            }

            logger.info(f"Registered worker pool {pool_id} with {pool.current_workers} workers")

        except Exception as e:
            logger.error(f"Failed to register worker pool {pool_id}: {e}")

    async def select_worker_pool(
        self,
        request_context: dict[str, Any]
    ) -> tuple[str, SecurityWorkerPool] | None:
        """Select optimal worker pool for request processing."""
        try:
            available_pools = [
                (pool_id, pool) for pool_id, pool in self.worker_pools.items()
                if self._is_pool_available(pool_id)
            ]

            if not available_pools:
                logger.warning("No available worker pools for request processing")
                return None

            # Select based on load balancing method
            if self.method == LoadBalancingMethod.ROUND_ROBIN:
                selected = self._round_robin_selection(available_pools)
            elif self.method == LoadBalancingMethod.LEAST_CONNECTIONS:
                selected = self._least_connections_selection(available_pools)
            elif self.method == LoadBalancingMethod.WEIGHTED_RESPONSE_TIME:
                selected = self._weighted_response_time_selection(available_pools)
            elif self.method == LoadBalancingMethod.RESOURCE_AWARE:
                selected = self._resource_aware_selection(available_pools)
            elif self.method == LoadBalancingMethod.THREAT_AWARE:
                selected = self._threat_aware_selection(available_pools, request_context)
            else:
                # Fallback to round robin
                selected = self._round_robin_selection(available_pools)

            # Record routing decision
            if selected:
                self.routing_history.append({
                    "timestamp": time.time(),
                    "pool_id": selected[0],
                    "method": self.method.value,
                    "threat_score": request_context.get("threat_score", 0.0)
                })

            return selected

        except Exception as e:
            logger.error(f"Worker pool selection failed: {e}")
            return None

    async def record_request_result(
        self,
        pool_id: str,
        success: bool,
        response_time: float
    ) -> None:
        """Record the result of a request for performance tracking."""
        try:
            if pool_id not in self.worker_stats:
                return

            stats = self.worker_stats[pool_id]
            stats["requests"] += 1
            stats["response_times"].append(response_time)

            # Update circuit breaker state
            breaker = self.circuit_breakers.get(pool_id, {})

            if success:
                # Reset failure count on success
                breaker["failure_count"] = 0
                if breaker["state"] == "HALF_OPEN":
                    breaker["state"] = "CLOSED"
            else:
                # Increment failure count
                breaker["failure_count"] += 1
                breaker["last_failure_time"] = time.time()

                # Trip circuit breaker if too many failures
                if breaker["failure_count"] >= 5 and breaker["state"] == "CLOSED":
                    breaker["state"] = "OPEN"
                    logger.warning(f"Circuit breaker opened for pool {pool_id}")

            # Calculate error rate
            recent_requests = list(stats["response_times"])[-100:]  # Last 100
            if len(recent_requests) >= 10:
                # Estimate error rate based on outliers (rough approximation)
                avg_time = np.mean(recent_requests)
                std_time = np.std(recent_requests)
                outliers = [t for t in recent_requests if t > avg_time + 2 * std_time]
                stats["error_rate"] = len(outliers) / len(recent_requests)

        except Exception as e:
            logger.error(f"Failed to record request result for pool {pool_id}: {e}")

    def _is_pool_available(self, pool_id: str) -> bool:
        """Check if a worker pool is available for processing."""
        breaker = self.circuit_breakers.get(pool_id, {})

        if breaker["state"] == "OPEN":
            # Check if reset timeout has passed
            if time.time() - breaker["last_failure_time"] > breaker["reset_timeout"]:
                breaker["state"] = "HALF_OPEN"
                logger.info(f"Circuit breaker for pool {pool_id} moved to HALF_OPEN")
                return True
            return False

        return True

    def _round_robin_selection(self, pools: list[tuple[str, SecurityWorkerPool]]) -> tuple[str, SecurityWorkerPool]:
        """Round-robin pool selection."""
        # Simple round-robin based on total requests
        pool_requests = [(pool_id, self.worker_stats[pool_id]["requests"]) for pool_id, _ in pools]
        pool_requests.sort(key=lambda x: x[1])  # Select pool with fewest requests

        selected_id = pool_requests[0][0]
        return next((pool_id, pool) for pool_id, pool in pools if pool_id == selected_id)

    def _least_connections_selection(self, pools: list[tuple[str, SecurityWorkerPool]]) -> tuple[str, SecurityWorkerPool]:
        """Select pool with least connections (lowest utilization)."""
        pool_utilization = []

        for pool_id, pool in pools:
            stats = self.worker_stats[pool_id]
            utilization = stats.get("capacity_utilization", 0.0)
            pool_utilization.append((pool_id, pool, utilization))

        # Sort by utilization (ascending)
        pool_utilization.sort(key=lambda x: x[2])
        return pool_utilization[0][0], pool_utilization[0][1]

    def _weighted_response_time_selection(self, pools: list[tuple[str, SecurityWorkerPool]]) -> tuple[str, SecurityWorkerPool]:
        """Select pool based on weighted response times."""
        pool_scores = []

        for pool_id, pool in pools:
            stats = self.worker_stats[pool_id]
            response_times = list(stats["response_times"])

            if response_times:
                avg_response_time = np.mean(response_times[-50:])  # Last 50 requests
                # Lower response time = higher score
                score = 1.0 / max(avg_response_time, 0.001)
            else:
                score = 1.0  # Default score for new pools

            pool_scores.append((pool_id, pool, score))

        # Sort by score (descending)
        pool_scores.sort(key=lambda x: x[2], reverse=True)
        return pool_scores[0][0], pool_scores[0][1]

    def _resource_aware_selection(self, pools: list[tuple[str, SecurityWorkerPool]]) -> tuple[str, SecurityWorkerPool]:
        """Select pool based on current resource availability."""
        pool_scores = []

        for pool_id, pool in pools:
            stats = self.worker_stats[pool_id]

            # Calculate composite resource score
            utilization = stats.get("capacity_utilization", 0.0)
            error_rate = stats.get("error_rate", 0.0)

            # Higher available capacity and lower error rate = higher score
            resource_score = (1.0 - utilization) * (1.0 - error_rate)
            pool_scores.append((pool_id, pool, resource_score))

        # Sort by resource score (descending)
        pool_scores.sort(key=lambda x: x[2], reverse=True)
        return pool_scores[0][0], pool_scores[0][1]

    def _threat_aware_selection(
        self,
        pools: list[tuple[str, SecurityWorkerPool]],
        request_context: dict[str, Any]
    ) -> tuple[str, SecurityWorkerPool]:
        """Select pool based on threat level and processing capabilities."""
        threat_score = request_context.get("threat_score", 0.0)
        pool_scores = []

        for pool_id, pool in pools:
            stats = self.worker_stats[pool_id]

            # Calculate threat processing score
            threat_processing_score = stats.get("threat_processing_score", 1.0)
            utilization = stats.get("capacity_utilization", 0.0)
            error_rate = stats.get("error_rate", 0.0)

            # For high-threat requests, prefer pools with better threat processing
            if threat_score > 0.7:
                score = threat_processing_score * (1.0 - utilization) * (1.0 - error_rate)
            else:
                # For normal requests, standard resource-aware selection
                score = (1.0 - utilization) * (1.0 - error_rate)

            pool_scores.append((pool_id, pool, score))

        # Sort by threat-aware score (descending)
        pool_scores.sort(key=lambda x: x[2], reverse=True)
        return pool_scores[0][0], pool_scores[0][1]

    async def get_load_balancer_stats(self) -> dict[str, Any]:
        """Get load balancer performance statistics."""
        try:
            stats = {
                "method": self.method.value,
                "total_pools": len(self.worker_pools),
                "available_pools": sum(1 for pool_id in self.worker_pools.keys()
                                     if self._is_pool_available(pool_id)),
                "routing_decisions": len(self.routing_history),
                "circuit_breakers": {}
            }

            # Circuit breaker states
            for pool_id, breaker in self.circuit_breakers.items():
                stats["circuit_breakers"][pool_id] = breaker["state"]

            # Worker pool statistics
            stats["worker_pools"] = {}
            for pool_id, pool_stats in self.worker_stats.items():
                response_times = list(pool_stats["response_times"])
                stats["worker_pools"][pool_id] = {
                    "requests": pool_stats["requests"],
                    "avg_response_time": np.mean(response_times) if response_times else 0.0,
                    "p95_response_time": np.percentile(response_times, 95) if response_times else 0.0,
                    "error_rate": pool_stats["error_rate"],
                    "capacity_utilization": pool_stats["capacity_utilization"]
                }

            return stats

        except Exception as e:
            logger.error(f"Failed to get load balancer stats: {e}")
            return {"error": str(e)}


class ThreatCorrelationEngine:
    """
    Advanced threat correlation engine for identifying coordinated attacks
    and emerging threat patterns across multiple security events.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.correlation_rules = []
        self.correlation_buffer = deque(maxlen=100000)  # Large buffer for correlation
        self.correlated_incidents = {}
        self.pattern_analyzer = ThreatPatternAnalyzer()

    async def add_correlation_rule(self, rule: ThreatCorrelationRule) -> None:
        """Add a threat correlation rule."""
        try:
            self.correlation_rules.append(rule)
            logger.info(f"Added threat correlation rule: {rule.name}")

        except Exception as e:
            logger.error(f"Failed to add correlation rule {rule.name}: {e}")

    async def correlate_security_event(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Correlate a security event with historical events to identify patterns.
        
        Args:
            event: Security event data
            
        Returns:
            List of correlation results
        """
        try:
            current_time = time.time()

            # Add event to correlation buffer
            enhanced_event = {
                **event,
                "correlation_timestamp": current_time,
                "correlation_id": hashlib.md5(
                    f"{event.get('source_ip', '')}{event.get('timestamp', current_time)}".encode()
                ).hexdigest()[:16]
            }

            self.correlation_buffer.append(enhanced_event)

            correlations = []

            # Apply correlation rules
            for rule in self.correlation_rules:
                if not rule.enabled:
                    continue

                correlation_result = await self._apply_correlation_rule(rule, enhanced_event)
                if correlation_result:
                    correlations.append(correlation_result)

            # Pattern-based correlation
            pattern_correlations = await self.pattern_analyzer.analyze_event_patterns(
                enhanced_event, list(self.correlation_buffer)
            )
            correlations.extend(pattern_correlations)

            return correlations

        except Exception as e:
            logger.error(f"Event correlation failed: {e}")
            return []

    async def _apply_correlation_rule(
        self,
        rule: ThreatCorrelationRule,
        event: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Apply a specific correlation rule to an event."""
        try:
            current_time = event["correlation_timestamp"]
            correlation_window_start = current_time - rule.correlation_window

            # Find related events within correlation window
            related_events = [
                e for e in self.correlation_buffer
                if e["correlation_timestamp"] >= correlation_window_start
            ]

            # Check rule conditions
            matching_events = []
            for condition in rule.conditions:
                condition_matches = await self._evaluate_condition(condition, related_events)
                matching_events.extend(condition_matches)

            # Check if threshold is met
            if len(matching_events) >= rule.threshold:
                correlation_id = f"corr_{rule.rule_id}_{int(current_time)}"

                correlation_result = {
                    "correlation_id": correlation_id,
                    "rule_id": rule.rule_id,
                    "rule_name": rule.name,
                    "matching_events": len(matching_events),
                    "threshold": rule.threshold,
                    "severity_boost": rule.severity_boost,
                    "correlation_window": rule.correlation_window,
                    "correlated_events": [e["correlation_id"] for e in matching_events[:10]],  # Top 10
                    "timestamp": current_time,
                    "description": f"Correlated threat pattern detected: {rule.description}"
                }

                # Store correlation
                self.correlated_incidents[correlation_id] = correlation_result

                logger.warning(f"Threat correlation detected: {rule.name} - "
                              f"{len(matching_events)} events in {rule.correlation_window}s window")

                return correlation_result

            return None

        except Exception as e:
            logger.error(f"Failed to apply correlation rule {rule.rule_id}: {e}")
            return None

    async def _evaluate_condition(
        self,
        condition: dict[str, Any],
        events: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Evaluate a correlation condition against events."""
        try:
            condition_type = condition.get("type", "field_match")
            matching_events = []

            if condition_type == "field_match":
                # Simple field matching
                field = condition.get("field")
                value = condition.get("value")
                operator = condition.get("operator", "equals")

                for event in events:
                    event_value = event.get(field)

                    if operator == "equals" and event_value == value or operator == "contains" and value in str(event_value) or operator == "greater_than" and isinstance(event_value, (int, float)) and event_value > value or operator == "less_than" and isinstance(event_value, (int, float)) and event_value < value:
                        matching_events.append(event)

            elif condition_type == "pattern_match":
                # Pattern-based matching
                pattern = condition.get("pattern")
                field = condition.get("field", "content")

                import re
                for event in events:
                    event_value = str(event.get(field, ""))
                    if re.search(pattern, event_value, re.IGNORECASE):
                        matching_events.append(event)

            elif condition_type == "threshold":
                # Count-based threshold
                threshold_value = condition.get("value", 0)
                if len(events) >= threshold_value:
                    matching_events = events[:threshold_value]

            elif condition_type == "source_correlation":
                # Source IP correlation
                source_ips = defaultdict(list)
                for event in events:
                    source_ip = event.get("source_ip")
                    if source_ip:
                        source_ips[source_ip].append(event)

                min_events_per_source = condition.get("min_events", 3)
                for ip, ip_events in source_ips.items():
                    if len(ip_events) >= min_events_per_source:
                        matching_events.extend(ip_events)

            return matching_events

        except Exception as e:
            logger.error(f"Failed to evaluate correlation condition: {e}")
            return []

    async def get_correlation_stats(self) -> dict[str, Any]:
        """Get correlation engine statistics."""
        try:
            current_time = time.time()

            # Recent correlations (last hour)
            recent_correlations = [
                corr for corr in self.correlated_incidents.values()
                if current_time - corr["timestamp"] < 3600
            ]

            # Active rules
            active_rules = sum(1 for rule in self.correlation_rules if rule.enabled)

            return {
                "total_rules": len(self.correlation_rules),
                "active_rules": active_rules,
                "buffer_size": len(self.correlation_buffer),
                "recent_correlations": len(recent_correlations),
                "total_correlations": len(self.correlated_incidents),
                "correlation_rate": len(recent_correlations) / max(len(self.correlation_buffer), 1)
            }

        except Exception as e:
            logger.error(f"Failed to get correlation stats: {e}")
            return {"error": str(e)}


class ThreatPatternAnalyzer:
    """Analyze patterns in security events for advanced threat detection."""

    async def analyze_event_patterns(
        self,
        event: dict[str, Any],
        event_history: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Analyze event patterns for advanced threat detection."""
        try:
            correlations = []

            # Time-based pattern analysis
            time_correlations = await self._analyze_temporal_patterns(event, event_history)
            correlations.extend(time_correlations)

            # Behavioral pattern analysis
            behavioral_correlations = await self._analyze_behavioral_patterns(event, event_history)
            correlations.extend(behavioral_correlations)

            # Geographic pattern analysis
            geo_correlations = await self._analyze_geographic_patterns(event, event_history)
            correlations.extend(geo_correlations)

            return correlations

        except Exception as e:
            logger.error(f"Pattern analysis failed: {e}")
            return []

    async def _analyze_temporal_patterns(
        self,
        event: dict[str, Any],
        history: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Analyze temporal attack patterns."""
        try:
            correlations = []
            current_time = event.get("correlation_timestamp", time.time())

            # Look for rapid succession attacks (burst pattern)
            burst_window = 60  # 60 seconds
            burst_threshold = 10  # 10 events

            recent_events = [
                e for e in history
                if current_time - e.get("correlation_timestamp", 0) < burst_window
            ]

            if len(recent_events) >= burst_threshold:
                correlations.append({
                    "pattern_type": "burst_attack",
                    "description": f"Burst of {len(recent_events)} events in {burst_window} seconds",
                    "severity_boost": 1.5,
                    "confidence": min(1.0, len(recent_events) / (burst_threshold * 2))
                })

            # Look for regular interval attacks (systematic pattern)
            source_ip = event.get("source_ip")
            if source_ip:
                source_events = [
                    e for e in history
                    if e.get("source_ip") == source_ip and
                    current_time - e.get("correlation_timestamp", 0) < 3600  # Last hour
                ]

                if len(source_events) >= 5:
                    # Calculate intervals between events
                    timestamps = sorted([e.get("correlation_timestamp", 0) for e in source_events])
                    intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]

                    if intervals:
                        avg_interval = np.mean(intervals)
                        interval_std = np.std(intervals)

                        # Low standard deviation indicates regular intervals
                        if interval_std < avg_interval * 0.2:  # Within 20% variation
                            correlations.append({
                                "pattern_type": "systematic_attack",
                                "description": f"Regular interval attacks every {avg_interval:.1f}s",
                                "severity_boost": 1.3,
                                "confidence": 1.0 - (interval_std / max(avg_interval, 1))
                            })

            return correlations

        except Exception as e:
            logger.error(f"Temporal pattern analysis failed: {e}")
            return []

    async def _analyze_behavioral_patterns(
        self,
        event: dict[str, Any],
        history: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Analyze behavioral attack patterns."""
        try:
            correlations = []

            # Analyze attack vector diversity
            recent_events = [
                e for e in history
                if time.time() - e.get("correlation_timestamp", 0) < 1800  # Last 30 minutes
            ]

            if len(recent_events) >= 5:
                attack_types = set()
                for e in recent_events:
                    threat_indicators = e.get("threat_indicators", [])
                    attack_types.update(threat_indicators[:3])  # Top 3 indicators

                # High diversity suggests sophisticated attack
                if len(attack_types) >= 8:
                    correlations.append({
                        "pattern_type": "multi_vector_attack",
                        "description": f"Attack using {len(attack_types)} different vectors",
                        "severity_boost": 1.4,
                        "confidence": min(1.0, len(attack_types) / 12)
                    })

            # Analyze escalation patterns
            source_ip = event.get("source_ip")
            if source_ip:
                source_events = [
                    e for e in history
                    if e.get("source_ip") == source_ip and
                    time.time() - e.get("correlation_timestamp", 0) < 3600
                ]

                if len(source_events) >= 3:
                    # Look for increasing threat scores (escalation)
                    threat_scores = [e.get("threat_score", 0) for e in source_events]
                    if len(threat_scores) >= 3:
                        # Check if threat scores are generally increasing
                        increasing_count = sum(
                            1 for i in range(len(threat_scores)-1)
                            if threat_scores[i+1] > threat_scores[i]
                        )

                        if increasing_count >= len(threat_scores) * 0.6:  # 60% increasing
                            correlations.append({
                                "pattern_type": "escalating_attack",
                                "description": "Attack sophistication is escalating",
                                "severity_boost": 1.6,
                                "confidence": increasing_count / max(len(threat_scores)-1, 1)
                            })

            return correlations

        except Exception as e:
            logger.error(f"Behavioral pattern analysis failed: {e}")
            return []

    async def _analyze_geographic_patterns(
        self,
        event: dict[str, Any],
        history: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Analyze geographic attack patterns."""
        try:
            correlations = []

            # Analyze for distributed attacks from multiple IPs
            recent_events = [
                e for e in history
                if time.time() - e.get("correlation_timestamp", 0) < 1800  # Last 30 minutes
            ]

            if len(recent_events) >= 10:
                unique_ips = set(e.get("source_ip") for e in recent_events if e.get("source_ip"))

                # High number of unique IPs suggests distributed attack
                if len(unique_ips) >= 20:
                    correlations.append({
                        "pattern_type": "distributed_attack",
                        "description": f"Attack from {len(unique_ips)} different sources",
                        "severity_boost": 1.3,
                        "confidence": min(1.0, len(unique_ips) / 50)
                    })

            return correlations

        except Exception as e:
            logger.error(f"Geographic pattern analysis failed: {e}")
            return []


class AutoScalingSecurityManager:
    """
    Auto-scaling manager for security components based on load and threat levels.
    
    Features:
    - Predictive scaling based on historical patterns
    - Threat-aware scaling triggers
    - Resource optimization
    - Cost-aware scaling decisions
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.scaling_strategy = ScalingStrategy(config.get("strategy", "adaptive"))
        self.min_instances = config.get("min_instances", 2)
        self.max_instances = config.get("max_instances", 20)
        self.target_utilization = config.get("target_utilization", 0.7)

        # Scaling metrics
        self.scaling_history = deque(maxlen=1000)
        self.performance_predictor = PerformancePredictor()
        self.resource_optimizer = ResourceOptimizer()

    async def evaluate_scaling_decision(
        self,
        current_metrics: PerformanceMetrics,
        threat_level: float
    ) -> dict[str, Any]:
        """Evaluate whether scaling is needed based on current conditions."""
        try:
            current_time = time.time()

            scaling_decision = {
                "timestamp": current_time,
                "action": "no_change",
                "current_instances": self._get_current_instance_count(),
                "recommended_instances": self._get_current_instance_count(),
                "reasoning": [],
                "confidence": 0.0
            }

            # Reactive scaling based on current metrics
            reactive_recommendation = await self._reactive_scaling_evaluation(current_metrics, threat_level)
            scaling_decision.update(reactive_recommendation)

            # Predictive scaling based on trends
            if self.scaling_strategy in [ScalingStrategy.PREDICTIVE, ScalingStrategy.ADAPTIVE]:
                predictive_recommendation = await self._predictive_scaling_evaluation(current_metrics)

                # Combine reactive and predictive recommendations
                if predictive_recommendation["confidence"] > 0.7:
                    scaling_decision = self._combine_scaling_recommendations(
                        scaling_decision, predictive_recommendation
                    )

            # Resource optimization
            if self.scaling_strategy == ScalingStrategy.ADAPTIVE:
                optimized_recommendation = await self.resource_optimizer.optimize_resources(
                    scaling_decision, current_metrics
                )
                scaling_decision.update(optimized_recommendation)

            # Record scaling decision
            self.scaling_history.append(scaling_decision)

            return scaling_decision

        except Exception as e:
            logger.error(f"Scaling evaluation failed: {e}")
            return {
                "timestamp": time.time(),
                "action": "no_change",
                "error": str(e)
            }

    async def _reactive_scaling_evaluation(
        self,
        metrics: PerformanceMetrics,
        threat_level: float
    ) -> dict[str, Any]:
        """Evaluate reactive scaling based on current load."""
        try:
            current_instances = self._get_current_instance_count()
            reasoning = []

            # Calculate utilization-based scaling
            current_rps = metrics.current_rps
            avg_response_time = metrics.avg_response_time

            # Estimate required capacity
            target_rps_per_instance = 100  # Configurable
            required_instances = max(1, int(np.ceil(current_rps / target_rps_per_instance)))

            # Adjust for response time
            if avg_response_time > 1.0:  # 1 second threshold
                response_time_multiplier = min(2.0, avg_response_time / 1.0)
                required_instances = int(required_instances * response_time_multiplier)
                reasoning.append(f"High response time ({avg_response_time:.2f}s) requires scaling")

            # Adjust for threat level
            if threat_level > 0.7:
                threat_multiplier = 1.0 + (threat_level - 0.7) * 2.0  # Up to 3x for max threat
                required_instances = int(required_instances * threat_multiplier)
                reasoning.append(f"High threat level ({threat_level:.2f}) requires additional capacity")

            # Apply bounds
            required_instances = max(self.min_instances, min(self.max_instances, required_instances))

            # Determine action
            if required_instances > current_instances:
                action = "scale_up"
                reasoning.append(f"Scale up: {current_instances} -> {required_instances}")
            elif required_instances < current_instances:
                action = "scale_down"
                reasoning.append(f"Scale down: {current_instances} -> {required_instances}")
            else:
                action = "no_change"
                reasoning.append("Current capacity is adequate")

            confidence = min(1.0, abs(required_instances - current_instances) / max(current_instances, 1))

            return {
                "action": action,
                "recommended_instances": required_instances,
                "reasoning": reasoning,
                "confidence": confidence,
                "method": "reactive"
            }

        except Exception as e:
            logger.error(f"Reactive scaling evaluation failed: {e}")
            return {
                "action": "no_change",
                "recommended_instances": self._get_current_instance_count(),
                "reasoning": [f"Evaluation error: {str(e)}"],
                "confidence": 0.0,
                "method": "reactive"
            }

    async def _predictive_scaling_evaluation(self, metrics: PerformanceMetrics) -> dict[str, Any]:
        """Evaluate predictive scaling based on trends."""
        try:
            # Use performance predictor to forecast load
            forecast = await self.performance_predictor.predict_load(
                self.scaling_history, metrics
            )

            if not forecast:
                return {
                    "action": "no_change",
                    "recommended_instances": self._get_current_instance_count(),
                    "reasoning": ["Insufficient data for prediction"],
                    "confidence": 0.0,
                    "method": "predictive"
                }

            predicted_load = forecast["predicted_rps"]
            prediction_confidence = forecast["confidence"]

            # Calculate required instances for predicted load
            target_rps_per_instance = 100
            predicted_instances = max(1, int(np.ceil(predicted_load / target_rps_per_instance)))
            predicted_instances = max(self.min_instances, min(self.max_instances, predicted_instances))

            current_instances = self._get_current_instance_count()

            # Determine action
            if predicted_instances > current_instances:
                action = "scale_up"
            elif predicted_instances < current_instances:
                action = "scale_down"
            else:
                action = "no_change"

            return {
                "action": action,
                "recommended_instances": predicted_instances,
                "reasoning": [f"Predicted load: {predicted_load:.1f} RPS"],
                "confidence": prediction_confidence,
                "method": "predictive"
            }

        except Exception as e:
            logger.error(f"Predictive scaling evaluation failed: {e}")
            return {
                "action": "no_change",
                "recommended_instances": self._get_current_instance_count(),
                "reasoning": [f"Prediction error: {str(e)}"],
                "confidence": 0.0,
                "method": "predictive"
            }

    def _combine_scaling_recommendations(
        self,
        reactive: dict[str, Any],
        predictive: dict[str, Any]
    ) -> dict[str, Any]:
        """Combine reactive and predictive scaling recommendations."""
        try:
            # Weighted combination based on confidence
            reactive_weight = reactive["confidence"]
            predictive_weight = predictive["confidence"]
            total_weight = reactive_weight + predictive_weight

            if total_weight == 0:
                return reactive  # Fallback to reactive

            # Weighted average of recommended instances
            combined_instances = (
                (reactive["recommended_instances"] * reactive_weight +
                 predictive["recommended_instances"] * predictive_weight) / total_weight
            )

            combined_instances = int(np.round(combined_instances))
            combined_instances = max(self.min_instances, min(self.max_instances, combined_instances))

            current_instances = self._get_current_instance_count()

            # Determine combined action
            if combined_instances > current_instances:
                action = "scale_up"
            elif combined_instances < current_instances:
                action = "scale_down"
            else:
                action = "no_change"

            # Combine reasoning
            combined_reasoning = reactive["reasoning"] + predictive["reasoning"]
            combined_confidence = (reactive_weight + predictive_weight) / 2

            return {
                "action": action,
                "recommended_instances": combined_instances,
                "reasoning": combined_reasoning,
                "confidence": combined_confidence,
                "method": "adaptive"
            }

        except Exception as e:
            logger.error(f"Failed to combine scaling recommendations: {e}")
            return reactive

    def _get_current_instance_count(self) -> int:
        """Get current number of security processing instances."""
        # Placeholder - in production would query actual infrastructure
        return 5


class PerformancePredictor:
    """Predict future performance based on historical patterns."""

    async def predict_load(
        self,
        history: deque,
        current_metrics: PerformanceMetrics
    ) -> dict[str, Any] | None:
        """Predict future load based on historical patterns."""
        try:
            if len(history) < 10:
                return None

            # Extract time series data
            timestamps = [h.get("timestamp", 0) for h in history if "timestamp" in h]

            if len(timestamps) < 5:
                return None

            # Simple linear trend prediction
            x = np.array(timestamps)
            y = np.array([current_metrics.current_rps] * len(timestamps))  # Placeholder

            # Fit linear trend
            if len(x) >= 2:
                slope = np.polyfit(x, y, 1)[0]

                # Predict load 5 minutes ahead
                future_time = time.time() + 300  # 5 minutes
                predicted_rps = current_metrics.current_rps + slope * 300
                predicted_rps = max(0, predicted_rps)  # Ensure non-negative

                # Calculate confidence based on trend consistency
                confidence = min(1.0, 1.0 / (1.0 + abs(slope)))

                return {
                    "predicted_rps": predicted_rps,
                    "prediction_horizon": 300,  # seconds
                    "confidence": confidence,
                    "trend_slope": slope
                }

            return None

        except Exception as e:
            logger.error(f"Load prediction failed: {e}")
            return None


class ResourceOptimizer:
    """Optimize resource allocation for cost and performance."""

    async def optimize_resources(
        self,
        scaling_decision: dict[str, Any],
        metrics: PerformanceMetrics
    ) -> dict[str, Any]:
        """Optimize resource allocation decision."""
        try:
            optimizations = {}

            # Cost optimization
            current_instances = scaling_decision.get("current_instances", 0)
            recommended_instances = scaling_decision.get("recommended_instances", 0)

            # Avoid unnecessary scaling for small changes
            if abs(recommended_instances - current_instances) == 1:
                if metrics.avg_response_time < 0.5:  # Good performance
                    optimizations["recommended_instances"] = current_instances
                    optimizations["reasoning"] = scaling_decision.get("reasoning", []) + [
                        "Avoided minor scaling due to good performance"
                    ]

            # Performance optimization
            if metrics.p99_response_time > 2.0:  # Poor tail latency
                scale_boost = max(1, int(recommended_instances * 0.2))  # 20% boost
                optimizations["recommended_instances"] = recommended_instances + scale_boost
                optimizations["reasoning"] = scaling_decision.get("reasoning", []) + [
                    f"Added {scale_boost} instances for tail latency optimization"
                ]

            return optimizations

        except Exception as e:
            logger.error(f"Resource optimization failed: {e}")
            return {}


class AdvancedSecurityOrchestrator:
    """
    Advanced security orchestration system that coordinates all security components
    with high-performance processing, intelligent scaling, and comprehensive monitoring.
    
    Features:
    - Multi-layer adaptive caching
    - Intelligent load balancing
    - Auto-scaling security controls
    - Advanced threat correlation
    - Performance optimization
    - Comprehensive monitoring
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

        # Initialize core components
        self.adaptive_cache = AdaptiveSecurityCache(
            self.config.get("cache", {})
        )
        self.load_balancer = IntelligentLoadBalancer(
            self.config.get("load_balancer", {})
        )
        self.correlation_engine = ThreatCorrelationEngine(
            self.config.get("correlation", {})
        )
        self.autoscaler = AutoScalingSecurityManager(
            self.config.get("autoscaling", {})
        )

        # Security component instances
        self.security_validators = {}
        self.quantum_monitors = {}
        self.incident_responders = {}
        self.security_dashboard = SecurityMetricsDashboard()

        # Performance tracking
        self.performance_metrics = PerformanceMetrics()
        self.metrics_collector = MetricsCollector("security_orchestrator")

        # Worker pools
        self.worker_pools = {}

    async def start_orchestrator(self) -> None:
        """Start the advanced security orchestrator system."""
        logger.info("Starting advanced security orchestrator")

        try:
            # Start core components
            await self.adaptive_cache.start_cache()
            await self.security_dashboard.start_dashboard()

            # Initialize default worker pools
            await self._initialize_worker_pools()

            # Load default correlation rules
            await self._load_default_correlation_rules()

            # Start background tasks
            asyncio.create_task(self._performance_monitoring_loop())
            asyncio.create_task(self._autoscaling_loop())
            asyncio.create_task(self._health_check_loop())

            logger.info("Advanced security orchestrator started successfully")

        except Exception as e:
            logger.error(f"Failed to start security orchestrator: {e}")
            raise SecurityError(f"Orchestrator startup failed: {e}")

    async def process_security_request(
        self,
        request_data: dict[str, Any],
        request_context: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Process a security request through the orchestrated security pipeline.
        
        Args:
            request_data: The security request data
            request_context: Request context information
            
        Returns:
            Comprehensive security analysis result
        """
        start_time = time.time()

        try:
            # Generate request ID for tracking
            request_id = f"req_{int(start_time)}_{hash(str(request_data)) % 10000:04d}"

            logger.debug(f"Processing security request {request_id}")

            # Check cache first
            cache_key = self._generate_cache_key(request_data)
            cached_result = await self.adaptive_cache.get(cache_key)

            if cached_result:
                logger.debug(f"Cache hit for request {request_id}")
                await self._update_performance_metrics(start_time, True, cached=True)
                return cached_result

            # Select appropriate worker pool
            pool_selection = await self.load_balancer.select_worker_pool(request_context)
            if not pool_selection:
                raise SecurityError("No available worker pools for request processing")

            pool_id, worker_pool = pool_selection

            # Process request through security pipeline
            security_result = await self._execute_security_pipeline(
                request_data, request_context, worker_pool
            )

            # Correlate with other security events
            correlation_results = await self.correlation_engine.correlate_security_event({
                **request_data,
                **request_context,
                "security_result": security_result
            })

            # Add correlation information to result
            if correlation_results:
                security_result["correlations"] = correlation_results

                # Boost threat score if correlated
                original_threat_score = security_result.get("threat_score", 0.0)
                max_boost = max((c.get("severity_boost", 1.0) for c in correlation_results), default=1.0)
                security_result["threat_score"] = min(1.0, original_threat_score * max_boost)

            # Cache the result
            threat_score = security_result.get("threat_score", 0.0)
            cache_priority = 0.5 + (threat_score * 0.5)  # Higher priority for threats

            await self.adaptive_cache.set(
                cache_key,
                security_result,
                threat_score=threat_score,
                priority=cache_priority
            )

            # Record performance metrics
            processing_time = time.time() - start_time
            success = security_result.get("status") != "error"

            await self.load_balancer.record_request_result(pool_id, success, processing_time)
            await self._update_performance_metrics(start_time, success, cached=False)

            # Record in dashboard
            if "validation_result" in security_result:
                await self.security_dashboard.record_validation_result(
                    security_result["validation_result"]
                )

            if "incident" in security_result:
                await self.security_dashboard.record_security_incident(
                    security_result["incident"]
                )

            if "quantum_events" in security_result:
                for event in security_result["quantum_events"]:
                    await self.security_dashboard.record_quantum_event(event)

            logger.debug(f"Completed security request {request_id} in {processing_time:.3f}s")

            return security_result

        except Exception as e:
            logger.error(f"Security request processing failed: {e}")
            await self._update_performance_metrics(start_time, False, cached=False)

            return {
                "status": "error",
                "error": str(e),
                "request_id": f"error_{int(start_time)}",
                "processing_time": time.time() - start_time
            }

    async def _execute_security_pipeline(
        self,
        request_data: dict[str, Any],
        request_context: dict[str, Any],
        worker_pool: SecurityWorkerPool
    ) -> dict[str, Any]:
        """Execute the comprehensive security processing pipeline."""
        try:
            results = {
                "status": "success",
                "timestamp": time.time(),
                "processing_components": []
            }

            # Enhanced validation
            if "validation" in self.config.get("enabled_components", ["validation"]):
                validator = await self._get_security_validator(worker_pool)

                validation_context = ValidationContext(
                    client_ip=request_context.get("client_ip", "unknown"),
                    user_agent=request_context.get("user_agent", "unknown"),
                    session_id=request_context.get("session_id"),
                    request_timestamp=datetime.now(timezone.utc),
                    request_size=len(str(request_data)),
                    content_type=request_context.get("content_type", "application/json")
                )

                validation_result = await validator.validate_request_pipeline(
                    str(request_data), validation_context
                )

                results["validation_result"] = validation_result
                results["threat_score"] = validation_result.risk_score
                results["processing_components"].append("enhanced_validation")

            # Quantum monitoring (if applicable)
            if "quantum" in self.config.get("enabled_components", []):
                quantum_monitor = await self._get_quantum_monitor(worker_pool)
                # Quantum monitoring would be triggered based on request type
                # results["quantum_events"] = [...]
                results["processing_components"].append("quantum_monitoring")

            # Incident analysis
            if results.get("threat_score", 0.0) > 0.7:
                incident_responder = await self._get_incident_responder(worker_pool)

                incident = await incident_responder.process_security_event(
                    {**request_data, **request_context, "threat_score": results["threat_score"]},
                    {"processing_source": "orchestrator"}
                )

                results["incident"] = incident
                results["processing_components"].append("incident_response")

            return results

        except Exception as e:
            logger.error(f"Security pipeline execution failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "processing_components": []
            }

    async def get_orchestrator_status(self) -> dict[str, Any]:
        """Get comprehensive orchestrator status."""
        try:
            # Cache statistics
            cache_stats = await self.adaptive_cache.get_stats()

            # Load balancer statistics
            lb_stats = await self.load_balancer.get_load_balancer_stats()

            # Correlation engine statistics
            correlation_stats = await self.correlation_engine.get_correlation_stats()

            # Performance metrics
            perf_metrics = {
                "total_requests": self.performance_metrics.total_requests,
                "success_rate": self.performance_metrics.successful_requests / max(self.performance_metrics.total_requests, 1),
                "avg_response_time": self.performance_metrics.avg_response_time,
                "p95_response_time": self.performance_metrics.p95_response_time,
                "p99_response_time": self.performance_metrics.p99_response_time,
                "current_rps": self.performance_metrics.current_rps,
                "cache_hit_rate": cache_stats.get("hit_rate", 0.0),
                "threat_detection_rate": self.performance_metrics.threat_detection_rate
            }

            return {
                "status": "operational",
                "timestamp": time.time(),
                "cache": cache_stats,
                "load_balancer": lb_stats,
                "correlation": correlation_stats,
                "performance": perf_metrics,
                "worker_pools": len(self.worker_pools),
                "enabled_components": self.config.get("enabled_components", [])
            }

        except Exception as e:
            logger.error(f"Failed to get orchestrator status: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }

    def _generate_cache_key(self, request_data: dict[str, Any]) -> str:
        """Generate cache key for request."""
        content_hash = hashlib.md5(str(request_data).encode()).hexdigest()
        return f"sec_req_{content_hash[:16]}"

    async def _get_security_validator(self, worker_pool: SecurityWorkerPool) -> EnhancedSecurityValidator:
        """Get or create security validator instance."""
        pool_id = id(worker_pool)
        if pool_id not in self.security_validators:
            self.security_validators[pool_id] = EnhancedSecurityValidator()
        return self.security_validators[pool_id]

    async def _get_quantum_monitor(self, worker_pool: SecurityWorkerPool) -> QuantumSecurityMonitor:
        """Get or create quantum monitor instance."""
        pool_id = id(worker_pool)
        if pool_id not in self.quantum_monitors:
            monitor = QuantumSecurityMonitor()
            await monitor.start_monitoring()
            self.quantum_monitors[pool_id] = monitor
        return self.quantum_monitors[pool_id]

    async def _get_incident_responder(self, worker_pool: SecurityWorkerPool) -> AIIncidentResponseSystem:
        """Get or create incident responder instance."""
        pool_id = id(worker_pool)
        if pool_id not in self.incident_responders:
            responder = AIIncidentResponseSystem()
            await responder.start_system()
            self.incident_responders[pool_id] = responder
        return self.incident_responders[pool_id]

    async def _initialize_worker_pools(self) -> None:
        """Initialize default worker pools."""
        try:
            # Create default worker pools
            default_pools = [
                {
                    "id": "validation_pool",
                    "type": "thread",
                    "min_workers": 4,
                    "max_workers": 16,
                    "capacity": 1000
                },
                {
                    "id": "quantum_pool",
                    "type": "process",
                    "min_workers": 2,
                    "max_workers": 8,
                    "capacity": 500
                },
                {
                    "id": "incident_pool",
                    "type": "thread",
                    "min_workers": 2,
                    "max_workers": 8,
                    "capacity": 200
                }
            ]

            for pool_config in default_pools:
                pool = SecurityWorkerPool(
                    pool_type=pool_config["type"],
                    min_workers=pool_config["min_workers"],
                    max_workers=pool_config["max_workers"],
                    current_workers=pool_config["min_workers"],
                    worker_capacity=pool_config["capacity"],
                    load_balancer=LoadBalancingMethod.THREAT_AWARE
                )

                # Create executor
                if pool.pool_type == "thread":
                    pool.executor = ThreadPoolExecutor(max_workers=pool.max_workers)
                else:
                    pool.executor = ProcessPoolExecutor(max_workers=pool.max_workers)

                self.worker_pools[pool_config["id"]] = pool
                await self.load_balancer.register_worker_pool(pool_config["id"], pool)

            logger.info(f"Initialized {len(self.worker_pools)} worker pools")

        except Exception as e:
            logger.error(f"Failed to initialize worker pools: {e}")
            raise

    async def _load_default_correlation_rules(self) -> None:
        """Load default threat correlation rules."""
        try:
            default_rules = [
                ThreatCorrelationRule(
                    rule_id="burst_attack",
                    name="Burst Attack Detection",
                    description="Detect rapid succession of attacks from single source",
                    conditions=[
                        {
                            "type": "source_correlation",
                            "min_events": 10
                        },
                        {
                            "type": "threshold",
                            "value": 10
                        }
                    ],
                    correlation_window=60,  # 1 minute
                    threshold=10,
                    severity_boost=1.5
                ),
                ThreatCorrelationRule(
                    rule_id="distributed_attack",
                    name="Distributed Attack Detection",
                    description="Detect coordinated attacks from multiple sources",
                    conditions=[
                        {
                            "type": "field_match",
                            "field": "threat_indicators",
                            "value": "injection",
                            "operator": "contains"
                        }
                    ],
                    correlation_window=300,  # 5 minutes
                    threshold=20,
                    severity_boost=1.3
                ),
                ThreatCorrelationRule(
                    rule_id="escalating_attack",
                    name="Escalating Attack Pattern",
                    description="Detect attacks with increasing sophistication",
                    conditions=[
                        {
                            "type": "source_correlation",
                            "min_events": 5
                        }
                    ],
                    correlation_window=1800,  # 30 minutes
                    threshold=5,
                    severity_boost=1.4
                )
            ]

            for rule in default_rules:
                await self.correlation_engine.add_correlation_rule(rule)

            logger.info(f"Loaded {len(default_rules)} default correlation rules")

        except Exception as e:
            logger.error(f"Failed to load correlation rules: {e}")

    async def _update_performance_metrics(self, start_time: float, success: bool, cached: bool) -> None:
        """Update performance metrics."""
        try:
            processing_time = time.time() - start_time

            self.performance_metrics.total_requests += 1
            if success:
                self.performance_metrics.successful_requests += 1
            else:
                self.performance_metrics.failed_requests += 1

            # Update response time metrics (simple running average)
            if self.performance_metrics.avg_response_time == 0:
                self.performance_metrics.avg_response_time = processing_time
            else:
                alpha = 0.1  # Exponential moving average factor
                self.performance_metrics.avg_response_time = (
                    alpha * processing_time +
                    (1 - alpha) * self.performance_metrics.avg_response_time
                )

            # Update cache hit rate
            if cached:
                cache_stats = await self.adaptive_cache.get_stats()
                self.performance_metrics.cache_hit_rate = cache_stats.get("hit_rate", 0.0)

        except Exception as e:
            logger.error(f"Failed to update performance metrics: {e}")

    async def _performance_monitoring_loop(self) -> None:
        """Background performance monitoring loop."""
        while True:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds

                # Collect performance metrics
                await self.metrics_collector.record_gauge(
                    "orchestrator_total_requests", self.performance_metrics.total_requests
                )
                await self.metrics_collector.record_gauge(
                    "orchestrator_avg_response_time", self.performance_metrics.avg_response_time
                )
                await self.metrics_collector.record_gauge(
                    "orchestrator_cache_hit_rate", self.performance_metrics.cache_hit_rate
                )

            except Exception as e:
                logger.error(f"Performance monitoring loop error: {e}")
                await asyncio.sleep(60)  # Back off on error

    async def _autoscaling_loop(self) -> None:
        """Background autoscaling evaluation loop."""
        while True:
            try:
                await asyncio.sleep(60)  # Evaluate every minute

                # Calculate current threat level (placeholder)
                threat_level = 0.3  # Would be calculated from recent events

                # Evaluate scaling decision
                scaling_decision = await self.autoscaler.evaluate_scaling_decision(
                    self.performance_metrics, threat_level
                )

                # Log scaling decisions
                if scaling_decision["action"] != "no_change":
                    logger.info(f"Autoscaling decision: {scaling_decision}")

                # In production, would execute scaling actions here

            except Exception as e:
                logger.error(f"Autoscaling loop error: {e}")
                await asyncio.sleep(120)  # Back off on error

    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while True:
            try:
                await asyncio.sleep(120)  # Health check every 2 minutes

                # Check component health
                status = await self.get_orchestrator_status()

                # Log any issues
                if status["status"] != "operational":
                    logger.warning(f"Orchestrator health issue detected: {status}")

            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(300)  # Back off significantly on error


# Export main classes for advanced security orchestration
__all__ = [
    "AdvancedSecurityOrchestrator",
    "AdaptiveSecurityCache",
    "IntelligentLoadBalancer",
    "ThreatCorrelationEngine",
    "AutoScalingSecurityManager",
    "SecurityWorkerPool",
    "PerformanceMetrics",
    "ThreatCorrelationRule",
    "ScalingStrategy",
    "LoadBalancingMethod",
    "CachingStrategy"
]
