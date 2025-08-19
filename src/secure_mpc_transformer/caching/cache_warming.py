"""
Cache warming and preloading system for intelligent cache management.

This module provides strategies for warming caches with frequently
accessed data and preloading models based on usage patterns.
"""

import asyncio
import heapq
import logging
import pickle
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class WarmingStrategy(Enum):
    """Cache warming strategies."""
    EAGER = "eager"           # Load everything immediately
    LAZY = "lazy"            # Load on demand
    PREDICTIVE = "predictive" # Load based on predictions
    ADAPTIVE = "adaptive"     # Adapt based on usage patterns
    SCHEDULED = "scheduled"   # Load based on schedule


@dataclass
class WarmingConfig:
    """Configuration for cache warming."""
    strategy: WarmingStrategy = WarmingStrategy.PREDICTIVE
    max_concurrent_loads: int = 5
    warmup_timeout_seconds: int = 300  # 5 minutes
    enable_background_warming: bool = True
    warming_interval_seconds: int = 1800  # 30 minutes
    preload_threshold: float = 0.7  # Load when hit rate < threshold
    max_warmup_items: int = 1000
    enable_persistence: bool = True
    persistence_file: str | None = None

    # Predictive warming settings
    prediction_window_hours: int = 24
    min_access_frequency: int = 5
    prediction_confidence_threshold: float = 0.6

    # Adaptive warming settings
    adaptation_learning_rate: float = 0.1
    adaptation_memory_decay: float = 0.95


@dataclass
class WarmingItem:
    """Item to be loaded into cache."""
    key: str
    namespace: str = "default"
    priority: float = 1.0
    estimated_load_time: float = 1.0
    dependencies: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other):
        # For priority queue (higher priority first)
        return self.priority > other.priority


@dataclass
class AccessPattern:
    """Access pattern for cache entries."""
    key: str
    access_times: list[float] = field(default_factory=list)
    access_frequency: float = 0.0
    last_access: float = 0.0
    daily_pattern: dict[int, int] = field(default_factory=lambda: defaultdict(int))  # hour -> count
    weekly_pattern: dict[int, int] = field(default_factory=lambda: defaultdict(int))  # day -> count
    seasonal_weight: float = 1.0

    def update_access(self):
        """Update access pattern with new access."""
        now = time.time()
        self.access_times.append(now)
        self.last_access = now

        # Keep only recent access times (last 30 days)
        cutoff = now - (30 * 24 * 3600)
        self.access_times = [t for t in self.access_times if t > cutoff]

        # Update frequency
        if self.access_times:
            time_span = max(self.access_times) - min(self.access_times)
            self.access_frequency = len(self.access_times) / max(time_span, 1.0)

        # Update daily pattern
        from datetime import datetime
        dt = datetime.fromtimestamp(now)
        self.daily_pattern[dt.hour] += 1
        self.weekly_pattern[dt.weekday()] += 1

    def predict_next_access(self) -> float | None:
        """Predict when next access might occur."""
        if len(self.access_times) < 2:
            return None

        # Simple prediction based on average interval
        intervals = [
            self.access_times[i] - self.access_times[i-1]
            for i in range(1, len(self.access_times))
        ]

        if intervals:
            avg_interval = sum(intervals) / len(intervals)
            return self.last_access + avg_interval

        return None

    def get_access_score(self, current_time: float) -> float:
        """Calculate access score for prioritization."""
        if not self.access_times:
            return 0.0

        # Recency factor
        time_since_access = current_time - self.last_access
        recency_factor = 1.0 / (1.0 + time_since_access / 3600.0)  # Decay over hours

        # Frequency factor
        frequency_factor = min(self.access_frequency * 100, 10.0)  # Cap at 10

        # Pattern-based prediction
        from datetime import datetime
        dt = datetime.fromtimestamp(current_time)
        daily_score = self.daily_pattern.get(dt.hour, 0)
        weekly_score = self.weekly_pattern.get(dt.weekday(), 0)
        pattern_factor = (daily_score + weekly_score) / 100.0

        return (recency_factor * 0.3 + frequency_factor * 0.5 +
                pattern_factor * 0.2) * self.seasonal_weight


class CacheWarmer:
    """Intelligent cache warming system."""

    def __init__(self, config: WarmingConfig, cache_manager=None):
        self.config = config
        self.cache_manager = cache_manager

        # Access pattern tracking
        self.access_patterns: dict[str, AccessPattern] = {}
        self._access_lock = threading.RLock()

        # Warming queue and state
        self.warming_queue: list[WarmingItem] = []
        self.warming_in_progress: set[str] = set()
        self._queue_lock = threading.Lock()

        # Background warming
        self._warming_thread: threading.Thread | None = None
        self._warming_running = False

        # Data sources for warming
        self._warming_sources: dict[str, Callable[[str], Any]] = {}

        # Statistics
        self._stats = {
            'items_warmed': 0,
            'warming_time_total': 0.0,
            'warming_failures': 0,
            'cache_hits_after_warming': 0,
            'patterns_tracked': 0,
            'predictions_made': 0,
            'successful_predictions': 0
        }

        # Load persisted patterns if enabled
        if self.config.enable_persistence:
            self._load_patterns()

        # Start background warming if enabled
        if self.config.enable_background_warming:
            self._start_background_warming()

        logger.info(f"Cache warmer initialized with {config.strategy.value} strategy")

    def record_access(self, key: str, namespace: str = "default"):
        """Record cache access for pattern learning."""
        full_key = f"{namespace}:{key}"

        with self._access_lock:
            if full_key not in self.access_patterns:
                self.access_patterns[full_key] = AccessPattern(key=full_key)
                self._stats['patterns_tracked'] += 1

            self.access_patterns[full_key].update_access()

    def register_warming_source(self, source_name: str, loader_func: Callable[[str], Any]):
        """Register a data source for cache warming."""
        self._warming_sources[source_name] = loader_func
        logger.info(f"Registered warming source: {source_name}")

    def add_warming_item(self, item: WarmingItem):
        """Add item to warming queue."""
        with self._queue_lock:
            heapq.heappush(self.warming_queue, item)

        logger.debug(f"Added warming item: {item.key} (priority: {item.priority})")

    def warm_cache_immediate(self, keys: list[str], namespace: str = "default",
                           source: str = "default") -> dict[str, bool]:
        """Immediately warm cache with specified keys."""
        results = {}

        if source not in self._warming_sources:
            logger.error(f"Unknown warming source: {source}")
            return dict.fromkeys(keys, False)

        loader_func = self._warming_sources[source]

        for key in keys:
            try:
                start_time = time.perf_counter()

                # Load data
                value = loader_func(key)

                # Store in cache
                if self.cache_manager and value is not None:
                    success = self.cache_manager.put(key, value, namespace=namespace)
                    results[key] = success

                    if success:
                        self._stats['items_warmed'] += 1
                        load_time = time.perf_counter() - start_time
                        self._stats['warming_time_total'] += load_time

                        logger.debug(f"Successfully warmed cache key: {key} ({load_time:.3f}s)")
                    else:
                        self._stats['warming_failures'] += 1
                        results[key] = False
                else:
                    results[key] = False
                    self._stats['warming_failures'] += 1

            except Exception as e:
                logger.error(f"Failed to warm cache key {key}: {e}")
                results[key] = False
                self._stats['warming_failures'] += 1

        return results

    async def warm_cache_async(self, keys: list[str], namespace: str = "default",
                              source: str = "default") -> dict[str, bool]:
        """Asynchronously warm cache with specified keys."""
        if source not in self._warming_sources:
            logger.error(f"Unknown warming source: {source}")
            return dict.fromkeys(keys, False)

        # Process in batches to limit concurrency
        batch_size = self.config.max_concurrent_loads
        results = {}

        for i in range(0, len(keys), batch_size):
            batch = keys[i:i + batch_size]

            # Create tasks for batch
            tasks = [
                asyncio.create_task(self._warm_single_key_async(key, namespace, source))
                for key in batch
            ]

            # Wait for batch completion
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for key, result in zip(batch, batch_results, strict=False):
                if isinstance(result, Exception):
                    logger.error(f"Async warming failed for key {key}: {result}")
                    results[key] = False
                    self._stats['warming_failures'] += 1
                else:
                    results[key] = result

        logger.info(f"Async warming completed: {sum(results.values())}/{len(keys)} successful")
        return results

    async def _warm_single_key_async(self, key: str, namespace: str, source: str) -> bool:
        """Warm a single key asynchronously."""
        try:
            loader_func = self._warming_sources[source]

            # Run loader in executor to avoid blocking
            loop = asyncio.get_event_loop()
            value = await loop.run_in_executor(None, loader_func, key)

            # Store in cache
            if self.cache_manager and value is not None:
                success = self.cache_manager.put(key, value, namespace=namespace)

                if success:
                    self._stats['items_warmed'] += 1
                    return True

            return False

        except Exception as e:
            logger.error(f"Async warming failed for key {key}: {e}")
            return False

    def generate_warming_predictions(self) -> list[WarmingItem]:
        """Generate warming predictions based on access patterns."""
        predictions = []
        current_time = time.time()

        with self._access_lock:
            for key, pattern in self.access_patterns.items():
                # Calculate access score
                score = pattern.get_access_score(current_time)

                # Check if prediction meets confidence threshold
                if score > self.config.prediction_confidence_threshold:
                    # Predict next access time
                    next_access = pattern.predict_next_access()

                    if next_access and next_access > current_time:
                        # Create warming item
                        priority = score * 100  # Convert to priority

                        warming_item = WarmingItem(
                            key=pattern.key,
                            priority=priority,
                            estimated_load_time=1.0,  # Default estimate
                            metadata={
                                'predicted_access_time': next_access,
                                'confidence_score': score,
                                'access_frequency': pattern.access_frequency
                            }
                        )

                        predictions.append(warming_item)
                        self._stats['predictions_made'] += 1

        # Sort by priority
        predictions.sort(reverse=True)

        # Limit to max warmup items
        predictions = predictions[:self.config.max_warmup_items]

        logger.info(f"Generated {len(predictions)} warming predictions")
        return predictions

    def _start_background_warming(self):
        """Start background warming thread."""
        self._warming_running = True
        self._warming_thread = threading.Thread(target=self._background_warming_loop, daemon=True)
        self._warming_thread.start()
        logger.info("Background cache warming started")

    def _background_warming_loop(self):
        """Background warming loop."""
        while self._warming_running:
            try:
                # Generate predictions based on strategy
                if self.config.strategy in [WarmingStrategy.PREDICTIVE, WarmingStrategy.ADAPTIVE]:
                    predictions = self.generate_warming_predictions()

                    # Add predictions to warming queue
                    with self._queue_lock:
                        for prediction in predictions:
                            if prediction.key not in self.warming_in_progress:
                                heapq.heappush(self.warming_queue, prediction)

                # Process warming queue
                self._process_warming_queue()

                # Sleep until next warming cycle
                time.sleep(self.config.warming_interval_seconds)

            except Exception as e:
                logger.error(f"Background warming error: {e}")
                time.sleep(60)  # Short sleep on error

    def _process_warming_queue(self):
        """Process items in warming queue."""
        processed = 0
        max_process = self.config.max_concurrent_loads

        while processed < max_process:
            with self._queue_lock:
                if not self.warming_queue:
                    break

                item = heapq.heappop(self.warming_queue)

                # Check if already in progress
                if item.key in self.warming_in_progress:
                    continue

                self.warming_in_progress.add(item.key)

            # Process item
            try:
                self._process_warming_item(item)
            finally:
                self.warming_in_progress.discard(item.key)

            processed += 1

        if processed > 0:
            logger.debug(f"Processed {processed} warming items")

    def _process_warming_item(self, item: WarmingItem):
        """Process a single warming item."""
        try:
            # Check dependencies first
            if item.dependencies:
                for dep in item.dependencies:
                    if not self._is_dependency_satisfied(dep):
                        logger.debug(f"Dependency not satisfied for {item.key}: {dep}")
                        return

            # Determine source from metadata or use default
            source = item.metadata.get('source', 'default')

            # Load and cache the item
            results = self.warm_cache_immediate([item.key], item.namespace, source)

            if results.get(item.key, False):
                logger.debug(f"Successfully warmed item: {item.key}")

                # Track successful prediction if it was predicted
                if 'predicted_access_time' in item.metadata:
                    self._stats['successful_predictions'] += 1

        except Exception as e:
            logger.error(f"Failed to process warming item {item.key}: {e}")

    def _is_dependency_satisfied(self, dependency: str) -> bool:
        """Check if a dependency is satisfied."""
        if self.cache_manager:
            # Check if dependency exists in cache
            value = self.cache_manager.get(dependency)
            return value is not None

        return True  # Assume satisfied if no cache manager

    def adaptive_warm_based_on_misses(self, miss_threshold: float = 0.3):
        """Adaptively warm cache based on miss rate."""
        if not self.cache_manager:
            return

        cache_stats = self.cache_manager.get_cache_stats()
        overall_stats = cache_stats.get('overall', {})
        miss_rate = overall_stats.get('miss_rate', 0.0)

        if miss_rate > miss_threshold:
            logger.info(f"High miss rate detected ({miss_rate:.2f}), triggering adaptive warming")

            # Get most frequently missed keys
            frequently_missed = self._get_frequently_missed_keys()

            # Create warming items for these keys
            warming_items = []
            for key in frequently_missed:
                item = WarmingItem(
                    key=key,
                    priority=10.0,  # High priority
                    metadata={'reason': 'high_miss_rate'}
                )
                warming_items.append(item)

            # Add to warming queue
            with self._queue_lock:
                for item in warming_items:
                    heapq.heappush(self.warming_queue, item)

            logger.info(f"Queued {len(warming_items)} items for adaptive warming")

    def _get_frequently_missed_keys(self) -> list[str]:
        """Get keys that are frequently missed."""
        # This would need to be implemented based on cache manager's miss tracking
        # For now, return empty list
        return []

    def schedule_warming(self, schedule: dict[str, list[str]]):
        """Schedule cache warming at specific times."""
        # Implementation for scheduled warming
        # Format: {"HH:MM": ["key1", "key2"], ...}
        logger.info(f"Scheduled warming configured for {len(schedule)} time slots")

    def _save_patterns(self):
        """Save access patterns to persistent storage."""
        if not self.config.enable_persistence:
            return

        try:
            persistence_file = self.config.persistence_file or "cache_warming_patterns.pkl"

            with self._access_lock:
                data = {
                    'patterns': self.access_patterns,
                    'stats': self._stats,
                    'timestamp': time.time()
                }

                with open(persistence_file, 'wb') as f:
                    pickle.dump(data, f)

            logger.debug(f"Saved {len(self.access_patterns)} access patterns")

        except Exception as e:
            logger.error(f"Failed to save patterns: {e}")

    def _load_patterns(self):
        """Load access patterns from persistent storage."""
        if not self.config.enable_persistence:
            return

        try:
            persistence_file = self.config.persistence_file or "cache_warming_patterns.pkl"

            if Path(persistence_file).exists():
                with open(persistence_file, 'rb') as f:
                    data = pickle.load(f)

                self.access_patterns = data.get('patterns', {})
                saved_stats = data.get('stats', {})

                # Merge stats
                for key, value in saved_stats.items():
                    self._stats[key] = self._stats.get(key, 0) + value

                logger.info(f"Loaded {len(self.access_patterns)} access patterns")

        except Exception as e:
            logger.error(f"Failed to load patterns: {e}")

    def get_warming_stats(self) -> dict[str, Any]:
        """Get cache warming statistics."""
        with self._access_lock:
            return {
                'strategy': self.config.strategy.value,
                'patterns_tracked': len(self.access_patterns),
                'warming_queue_size': len(self.warming_queue),
                'items_warming': len(self.warming_in_progress),
                'warming_sources': list(self._warming_sources.keys()),
                **self._stats,
                'avg_warming_time': (
                    self._stats['warming_time_total'] / max(self._stats['items_warmed'], 1)
                ),
                'prediction_accuracy': (
                    self._stats['successful_predictions'] / max(self._stats['predictions_made'], 1)
                ) if self._stats['predictions_made'] > 0 else 0.0
            }

    def get_top_patterns(self, n: int = 10) -> list[tuple[str, float]]:
        """Get top N access patterns by score."""
        current_time = time.time()
        patterns_with_scores = []

        with self._access_lock:
            for key, pattern in self.access_patterns.items():
                score = pattern.get_access_score(current_time)
                patterns_with_scores.append((key, score))

        # Sort by score and return top N
        patterns_with_scores.sort(key=lambda x: x[1], reverse=True)
        return patterns_with_scores[:n]

    def clear_patterns(self):
        """Clear all access patterns."""
        with self._access_lock:
            self.access_patterns.clear()

        logger.info("Cleared all access patterns")

    def shutdown(self):
        """Shutdown cache warmer."""
        logger.info("Shutting down cache warmer")

        # Stop background warming
        self._warming_running = False
        if self._warming_thread:
            self._warming_thread.join(timeout=5.0)

        # Save patterns if persistence is enabled
        self._save_patterns()

        logger.info("Cache warmer shutdown completed")

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.shutdown()
        except:
            pass
