#!/usr/bin/env python3
"""
Quantum Security Monitor for Secure MPC Transformer

Specialized monitoring system for quantum-enhanced operations with comprehensive
security analysis for quantum computing components and side-channel attack detection.
"""

import asyncio
import logging
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np

from ..utils.metrics import MetricsCollector

logger = logging.getLogger(__name__)


class QuantumThreatLevel(Enum):
    """Quantum-specific threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class QuantumSecurityEvent:
    """Quantum security event data structure."""
    event_id: str
    event_type: str
    timestamp: datetime
    threat_level: QuantumThreatLevel
    quantum_state_id: str
    coherence_level: float
    entanglement_metrics: dict[str, float]
    security_metrics: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuantumOperationContext:
    """Context for quantum operations being monitored."""
    operation_id: str
    operation_type: str
    start_time: datetime
    quantum_circuit_depth: int
    qubit_count: int
    gate_count: int
    measurement_basis: list[str]
    expected_coherence: float


class QuantumCoherenceMonitor:
    """Monitor quantum coherence and detect anomalies."""

    def __init__(self, coherence_threshold: float = 0.1):
        self.coherence_threshold = coherence_threshold
        self.coherence_history = deque(maxlen=1000)
        self.baseline_coherence = {}
        self.anomaly_detector = QuantumAnomalyDetector()

    async def monitor_coherence(self, quantum_state: dict[str, Any]) -> tuple[bool, float, list[str]]:
        """
        Monitor quantum coherence and detect security-relevant anomalies.
        
        Args:
            quantum_state: Current quantum state information
            
        Returns:
            Tuple of (is_secure, coherence_score, alerts)
        """
        alerts = []

        try:
            coherence_level = quantum_state.get("coherence", 0.0)
            state_id = quantum_state.get("state_id", "unknown")

            # Record coherence measurement
            self.coherence_history.append({
                "timestamp": time.time(),
                "coherence": coherence_level,
                "state_id": state_id
            })

            # Check coherence drop
            if coherence_level < self.coherence_threshold:
                alerts.append(f"coherence_below_threshold_{coherence_level:.3f}")

            # Detect sudden coherence drops (potential decoherence attacks)
            if len(self.coherence_history) >= 2:
                recent_coherence = [h["coherence"] for h in list(self.coherence_history)[-10:]]
                if len(recent_coherence) >= 2:
                    coherence_drop = max(recent_coherence) - min(recent_coherence)
                    if coherence_drop > 0.5:  # Sudden drop > 50%
                        alerts.append(f"sudden_coherence_drop_{coherence_drop:.3f}")

            # Anomaly detection
            is_anomaly, anomaly_score = await self.anomaly_detector.detect_coherence_anomaly(
                coherence_level, self.coherence_history
            )

            if is_anomaly:
                alerts.append(f"coherence_anomaly_score_{anomaly_score:.3f}")

            # Calculate overall coherence security score
            coherence_score = min(1.0, coherence_level / self.coherence_threshold)
            is_secure = len(alerts) == 0 and coherence_score > 0.8

            logger.debug(f"Coherence monitoring: level={coherence_level:.3f}, "
                        f"score={coherence_score:.3f}, alerts={alerts}")

            return is_secure, coherence_score, alerts

        except Exception as e:
            logger.error(f"Coherence monitoring failed: {e}")
            return False, 0.0, ["coherence_monitoring_error"]


class SideChannelDetector:
    """Detect side-channel attacks on quantum operations."""

    def __init__(self):
        self.timing_history = defaultdict(deque)
        self.power_consumption_history = deque(maxlen=1000)
        self.cache_access_patterns = defaultdict(list)
        self.statistical_analyzer = StatisticalSecurityAnalyzer()

    async def detect_timing_attacks(self, operation_context: QuantumOperationContext) -> tuple[bool, list[str]]:
        """Detect timing-based side-channel attacks."""
        alerts = []

        try:
            operation_time = (datetime.now(timezone.utc) - operation_context.start_time).total_seconds()
            operation_type = operation_context.operation_type

            # Store timing measurement
            self.timing_history[operation_type].append({
                "timestamp": time.time(),
                "duration": operation_time,
                "operation_id": operation_context.operation_id,
                "circuit_depth": operation_context.quantum_circuit_depth,
                "qubit_count": operation_context.qubit_count
            })

            # Analyze timing patterns
            if len(self.timing_history[operation_type]) >= 10:
                recent_timings = [
                    t["duration"] for t in list(self.timing_history[operation_type])[-20:]
                ]

                # Statistical analysis
                timing_variance = statistics.variance(recent_timings) if len(recent_timings) > 1 else 0
                timing_mean = statistics.mean(recent_timings)

                # Detect timing anomalies
                if timing_variance > timing_mean * 0.5:  # High variance indicates potential attack
                    alerts.append(f"high_timing_variance_{timing_variance:.6f}")

                # Detect timing correlation attacks
                correlation_score = await self._detect_timing_correlations(operation_context)
                if correlation_score > 0.7:
                    alerts.append(f"timing_correlation_attack_{correlation_score:.3f}")

            # Detect timing side-channel information leakage
            if operation_time > 0:
                info_leakage_score = await self._analyze_timing_information_leakage(
                    operation_context, operation_time
                )
                if info_leakage_score > 0.6:
                    alerts.append(f"timing_info_leakage_{info_leakage_score:.3f}")

            is_attack_detected = len(alerts) > 0

            if is_attack_detected:
                logger.warning(f"Timing attack indicators detected: operation={operation_type}, "
                              f"time={operation_time:.6f}s, alerts={alerts}")

            return is_attack_detected, alerts

        except Exception as e:
            logger.error(f"Timing attack detection failed: {e}")
            return True, ["timing_detection_error"]  # Fail secure

    async def detect_power_analysis_attacks(self, power_measurements: list[float]) -> tuple[bool, list[str]]:
        """Detect power analysis side-channel attacks."""
        alerts = []

        try:
            if not power_measurements:
                return False, []

            # Store power measurements
            self.power_consumption_history.extend(power_measurements)

            # Statistical analysis of power consumption
            power_variance = statistics.variance(power_measurements) if len(power_measurements) > 1 else 0
            power_mean = statistics.mean(power_measurements)

            # Detect differential power analysis (DPA) patterns
            if power_variance > power_mean * 0.3:  # Suspicious power variance
                alerts.append(f"suspicious_power_variance_{power_variance:.3f}")

            # Detect correlation power analysis (CPA) patterns
            if len(self.power_consumption_history) >= 100:
                recent_power = list(self.power_consumption_history)[-100:]
                correlation_patterns = await self._analyze_power_correlations(recent_power)

                for pattern, correlation in correlation_patterns.items():
                    if correlation > 0.8:
                        alerts.append(f"power_correlation_{pattern}_{correlation:.3f}")

            # Detect simple power analysis (SPA) attacks
            spa_indicators = await self._detect_spa_patterns(power_measurements)
            alerts.extend(spa_indicators)

            is_attack_detected = len(alerts) > 0

            if is_attack_detected:
                logger.warning(f"Power analysis attack indicators: mean_power={power_mean:.3f}, "
                              f"variance={power_variance:.3f}, alerts={alerts}")

            return is_attack_detected, alerts

        except Exception as e:
            logger.error(f"Power analysis detection failed: {e}")
            return True, ["power_analysis_error"]  # Fail secure

    async def detect_cache_timing_attacks(self, cache_access_log: list[dict[str, Any]]) -> tuple[bool, list[str]]:
        """Detect cache timing side-channel attacks."""
        alerts = []

        try:
            if not cache_access_log:
                return False, []

            # Analyze cache access patterns
            for access in cache_access_log:
                cache_line = access.get("cache_line", "unknown")
                access_time = access.get("access_time", 0)

                self.cache_access_patterns[cache_line].append({
                    "timestamp": time.time(),
                    "access_time": access_time
                })

            # Detect flush+reload attacks
            flush_reload_score = await self._detect_flush_reload_patterns()
            if flush_reload_score > 0.7:
                alerts.append(f"flush_reload_attack_{flush_reload_score:.3f}")

            # Detect prime+probe attacks
            prime_probe_score = await self._detect_prime_probe_patterns()
            if prime_probe_score > 0.7:
                alerts.append(f"prime_probe_attack_{prime_probe_score:.3f}")

            is_attack_detected = len(alerts) > 0

            return is_attack_detected, alerts

        except Exception as e:
            logger.error(f"Cache timing attack detection failed: {e}")
            return True, ["cache_timing_error"]  # Fail secure

    async def _detect_timing_correlations(self, context: QuantumOperationContext) -> float:
        """Detect correlations in timing that could indicate attacks."""
        try:
            operation_type = context.operation_type
            if len(self.timing_history[operation_type]) < 10:
                return 0.0

            recent_ops = list(self.timing_history[operation_type])[-20:]

            # Analyze correlation between circuit depth and timing
            depths = [op["circuit_depth"] for op in recent_ops]
            timings = [op["duration"] for op in recent_ops]

            if len(depths) >= 2 and len(timings) >= 2:
                correlation = np.corrcoef(depths, timings)[0, 1]
                return abs(correlation) if not np.isnan(correlation) else 0.0

            return 0.0

        except Exception as e:
            logger.error(f"Timing correlation analysis failed: {e}")
            return 0.0

    async def _analyze_timing_information_leakage(self, context: QuantumOperationContext, timing: float) -> float:
        """Analyze potential information leakage through timing."""
        try:
            # Simple heuristic: timing should not correlate with sensitive parameters
            expected_timing = context.quantum_circuit_depth * 0.001  # 1ms per circuit depth unit
            timing_deviation = abs(timing - expected_timing) / max(expected_timing, 0.001)

            # High deviation could indicate information leakage
            return min(1.0, timing_deviation)

        except Exception as e:
            logger.error(f"Information leakage analysis failed: {e}")
            return 0.0

    async def _analyze_power_correlations(self, power_data: list[float]) -> dict[str, float]:
        """Analyze power consumption correlations."""
        correlations = {}

        try:
            if len(power_data) < 10:
                return correlations

            # Analyze autocorrelation
            power_array = np.array(power_data)
            autocorr = np.corrcoef(power_array[:-1], power_array[1:])[0, 1]
            if not np.isnan(autocorr):
                correlations["autocorrelation"] = abs(autocorr)

            # Analyze periodic patterns
            fft_result = np.fft.fft(power_array)
            dominant_freq_power = np.max(np.abs(fft_result[1:len(fft_result)//2]))
            total_power = np.sum(np.abs(fft_result))

            if total_power > 0:
                correlations["frequency_concentration"] = dominant_freq_power / total_power

            return correlations

        except Exception as e:
            logger.error(f"Power correlation analysis failed: {e}")
            return correlations

    async def _detect_spa_patterns(self, power_data: list[float]) -> list[str]:
        """Detect Simple Power Analysis attack patterns."""
        alerts = []

        try:
            if len(power_data) < 5:
                return alerts

            # Look for distinctive power patterns
            power_array = np.array(power_data)

            # Detect sudden power spikes (potential key operations)
            mean_power = np.mean(power_array)
            std_power = np.std(power_array)

            spike_threshold = mean_power + 3 * std_power
            spikes = np.where(power_array > spike_threshold)[0]

            if len(spikes) > len(power_array) * 0.1:  # More than 10% spikes
                alerts.append(f"excessive_power_spikes_{len(spikes)}")

            # Detect power pattern regularity (could indicate key-dependent operations)
            if std_power > mean_power * 0.5:  # High variance
                alerts.append(f"high_power_variance_{std_power:.3f}")

            return alerts

        except Exception as e:
            logger.error(f"SPA pattern detection failed: {e}")
            return ["spa_detection_error"]

    async def _detect_flush_reload_patterns(self) -> float:
        """Detect flush+reload cache attack patterns."""
        try:
            suspicious_score = 0.0

            for cache_line, accesses in self.cache_access_patterns.items():
                if len(accesses) < 10:
                    continue

                # Look for alternating fast/slow access patterns
                recent_accesses = accesses[-20:]
                access_times = [a["access_time"] for a in recent_accesses]

                if len(access_times) >= 4:
                    # Check for alternating pattern
                    fast_threshold = statistics.median(access_times) * 0.5
                    slow_threshold = statistics.median(access_times) * 2.0

                    alternating_count = 0
                    for i in range(len(access_times) - 1):
                        if ((access_times[i] < fast_threshold and access_times[i+1] > slow_threshold) or
                            (access_times[i] > slow_threshold and access_times[i+1] < fast_threshold)):
                            alternating_count += 1

                    if alternating_count > len(access_times) * 0.3:  # 30% alternating
                        suspicious_score = max(suspicious_score, alternating_count / len(access_times))

            return suspicious_score

        except Exception as e:
            logger.error(f"Flush+reload detection failed: {e}")
            return 0.0

    async def _detect_prime_probe_patterns(self) -> float:
        """Detect prime+probe cache attack patterns."""
        try:
            # Look for synchronized access patterns across cache lines
            if len(self.cache_access_patterns) < 2:
                return 0.0

            cache_lines = list(self.cache_access_patterns.keys())
            sync_score = 0.0

            for i in range(len(cache_lines)):
                for j in range(i + 1, len(cache_lines)):
                    line1_accesses = [a["timestamp"] for a in self.cache_access_patterns[cache_lines[i]][-10:]]
                    line2_accesses = [a["timestamp"] for a in self.cache_access_patterns[cache_lines[j]][-10:]]

                    if len(line1_accesses) >= 3 and len(line2_accesses) >= 3:
                        # Check for synchronized access patterns
                        time_correlation = await self._calculate_temporal_correlation(
                            line1_accesses, line2_accesses
                        )
                        sync_score = max(sync_score, time_correlation)

            return sync_score

        except Exception as e:
            logger.error(f"Prime+probe detection failed: {e}")
            return 0.0

    async def _calculate_temporal_correlation(self, timestamps1: list[float], timestamps2: list[float]) -> float:
        """Calculate temporal correlation between two timestamp sequences."""
        try:
            if len(timestamps1) != len(timestamps2):
                return 0.0

            # Simple correlation based on time differences
            diffs1 = [timestamps1[i+1] - timestamps1[i] for i in range(len(timestamps1)-1)]
            diffs2 = [timestamps2[i+1] - timestamps2[i] for i in range(len(timestamps2)-1)]

            if len(diffs1) >= 2 and len(diffs2) >= 2:
                correlation = np.corrcoef(diffs1, diffs2)[0, 1]
                return abs(correlation) if not np.isnan(correlation) else 0.0

            return 0.0

        except Exception as e:
            logger.error(f"Temporal correlation calculation failed: {e}")
            return 0.0


class QuantumAnomalyDetector:
    """Advanced anomaly detection for quantum operations."""

    def __init__(self):
        self.baseline_models = {}
        self.anomaly_threshold = 2.5  # Standard deviations

    async def detect_coherence_anomaly(
        self,
        current_coherence: float,
        coherence_history: deque
    ) -> tuple[bool, float]:
        """Detect anomalies in quantum coherence."""
        try:
            if len(coherence_history) < 10:
                return False, 0.0  # Insufficient data

            # Extract recent coherence values
            recent_coherence = [h["coherence"] for h in list(coherence_history)[-50:]]

            # Statistical analysis
            mean_coherence = statistics.mean(recent_coherence)
            std_coherence = statistics.stdev(recent_coherence) if len(recent_coherence) > 1 else 0.1

            # Z-score based anomaly detection
            if std_coherence > 0:
                z_score = abs(current_coherence - mean_coherence) / std_coherence
                is_anomaly = z_score > self.anomaly_threshold
                anomaly_score = min(1.0, z_score / self.anomaly_threshold)
            else:
                is_anomaly = False
                anomaly_score = 0.0

            return is_anomaly, anomaly_score

        except Exception as e:
            logger.error(f"Coherence anomaly detection failed: {e}")
            return True, 1.0  # Fail secure


class StatisticalSecurityAnalyzer:
    """Statistical analysis for security monitoring."""

    def __init__(self):
        self.entropy_analyzer = EntropyAnalyzer()

    async def analyze_entropy(self, data_sequence: list[float]) -> dict[str, float]:
        """Analyze entropy of data sequence."""
        return await self.entropy_analyzer.calculate_entropy_metrics(data_sequence)


class EntropyAnalyzer:
    """Entropy analysis for detecting non-random patterns."""

    async def calculate_entropy_metrics(self, data: list[float]) -> dict[str, float]:
        """Calculate various entropy metrics."""
        metrics = {}

        try:
            if len(data) < 10:
                return metrics

            # Shannon entropy
            data_array = np.array(data)
            hist, _ = np.histogram(data_array, bins=min(50, len(data)//2))
            hist = hist[hist > 0]  # Remove zero counts

            if len(hist) > 0:
                probabilities = hist / np.sum(hist)
                shannon_entropy = -np.sum(probabilities * np.log2(probabilities))
                metrics["shannon_entropy"] = shannon_entropy
                metrics["max_entropy"] = np.log2(len(hist))
                metrics["entropy_ratio"] = shannon_entropy / np.log2(len(hist)) if len(hist) > 1 else 0.0

            # Approximate entropy
            if len(data) >= 20:
                approx_entropy = await self._calculate_approximate_entropy(data)
                metrics["approximate_entropy"] = approx_entropy

            return metrics

        except Exception as e:
            logger.error(f"Entropy analysis failed: {e}")
            return metrics

    async def _calculate_approximate_entropy(self, data: list[float], pattern_length: int = 2) -> float:
        """Calculate approximate entropy (ApEn)."""
        try:
            n = len(data)
            tolerance = 0.2 * statistics.stdev(data) if len(data) > 1 else 0.1

            def _maxdist(xi, xj, n, pattern_length):
                return max([abs(ua - va) for ua, va in zip(xi[k:k+pattern_length], xj[k:k+pattern_length], strict=False)
                           for k in range(n - pattern_length + 1)])

            def _phi(pattern_length):
                patterns = np.array([data[i:i+pattern_length] for i in range(n - pattern_length + 1)])
                c = np.zeros(n - pattern_length + 1)

                for i in range(n - pattern_length + 1):
                    template = patterns[i]
                    for j, candidate in enumerate(patterns):
                        if _maxdist(template, candidate, n, pattern_length) <= tolerance:
                            c[i] += 1.0

                phi = np.sum(np.log(c / (n - pattern_length + 1.0))) / (n - pattern_length + 1.0)
                return phi

            return _phi(pattern_length) - _phi(pattern_length + 1)

        except Exception as e:
            logger.error(f"Approximate entropy calculation failed: {e}")
            return 0.0


class QuantumSecurityMonitor:
    """
    Comprehensive quantum security monitoring system.
    
    Monitors quantum operations for:
    - Quantum coherence and state integrity
    - Side-channel attacks (timing, power, cache)
    - Quantum-specific threats and vulnerabilities
    - Security metrics and alerting
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.coherence_monitor = QuantumCoherenceMonitor(
            coherence_threshold=self.config.get("coherence_threshold", 0.1)
        )
        self.sidechannel_detector = SideChannelDetector()
        self.metrics_collector = MetricsCollector("quantum_security")
        self.active_operations = {}
        self.security_events = []

    async def start_monitoring(self) -> None:
        """Start the quantum security monitoring system."""
        logger.info("Starting quantum security monitoring")

        # Start background monitoring tasks
        asyncio.create_task(self._continuous_monitoring_loop())
        asyncio.create_task(self._security_metrics_collection_loop())

    async def monitor_quantum_operation(
        self,
        operation_context: QuantumOperationContext,
        quantum_state: dict[str, Any]
    ) -> QuantumSecurityEvent:
        """Monitor a specific quantum operation for security threats."""

        try:
            # Store active operation
            self.active_operations[operation_context.operation_id] = operation_context

            alerts = []
            security_metrics = {}
            threat_level = QuantumThreatLevel.LOW

            # Monitor quantum coherence
            coherence_secure, coherence_score, coherence_alerts = await self.coherence_monitor.monitor_coherence(
                quantum_state
            )
            alerts.extend(coherence_alerts)
            security_metrics["coherence_score"] = coherence_score

            if not coherence_secure:
                threat_level = max(threat_level, QuantumThreatLevel.MEDIUM)

            # Detect timing attacks
            timing_attack_detected, timing_alerts = await self.sidechannel_detector.detect_timing_attacks(
                operation_context
            )
            alerts.extend(timing_alerts)
            security_metrics["timing_attack_risk"] = 1.0 if timing_attack_detected else 0.0

            if timing_attack_detected:
                threat_level = max(threat_level, QuantumThreatLevel.HIGH)

            # Create security event
            security_event = QuantumSecurityEvent(
                event_id=f"qsec_{int(time.time())}_{operation_context.operation_id[:8]}",
                event_type="quantum_operation_monitoring",
                timestamp=datetime.now(timezone.utc),
                threat_level=threat_level,
                quantum_state_id=quantum_state.get("state_id", "unknown"),
                coherence_level=quantum_state.get("coherence", 0.0),
                entanglement_metrics=quantum_state.get("entanglement_metrics", {}),
                security_metrics=security_metrics,
                metadata={
                    "operation_context": operation_context,
                    "alerts": alerts,
                    "monitoring_timestamp": time.time()
                }
            )

            # Store security event
            self.security_events.append(security_event)

            # Collect metrics
            await self._collect_security_metrics(security_event)

            # Log high-threat events
            if threat_level.value in ["high", "critical"]:
                logger.warning(f"High-threat quantum operation detected: {security_event.event_id}, "
                              f"threat_level={threat_level.value}, alerts={alerts}")

            return security_event

        except Exception as e:
            logger.error(f"Quantum operation monitoring failed: {e}")

            # Return critical threat event on error
            return QuantumSecurityEvent(
                event_id=f"error_{int(time.time())}",
                event_type="monitoring_error",
                timestamp=datetime.now(timezone.utc),
                threat_level=QuantumThreatLevel.CRITICAL,
                quantum_state_id="error",
                coherence_level=0.0,
                entanglement_metrics={},
                security_metrics={"error": 1.0},
                metadata={"error": str(e)}
            )

        finally:
            # Clean up active operation
            if operation_context.operation_id in self.active_operations:
                del self.active_operations[operation_context.operation_id]

    async def analyze_power_consumption(self, power_measurements: list[float]) -> bool:
        """Analyze power consumption for side-channel attacks."""
        attack_detected, alerts = await self.sidechannel_detector.detect_power_analysis_attacks(
            power_measurements
        )

        if attack_detected:
            logger.warning(f"Power analysis attack detected: {alerts}")
            await self._record_security_alert("power_analysis_attack", alerts)

        return attack_detected

    async def analyze_cache_behavior(self, cache_access_log: list[dict[str, Any]]) -> bool:
        """Analyze cache access patterns for timing attacks."""
        attack_detected, alerts = await self.sidechannel_detector.detect_cache_timing_attacks(
            cache_access_log
        )

        if attack_detected:
            logger.warning(f"Cache timing attack detected: {alerts}")
            await self._record_security_alert("cache_timing_attack", alerts)

        return attack_detected

    async def get_security_status(self) -> dict[str, Any]:
        """Get current quantum security status."""
        try:
            # Calculate overall threat level
            recent_events = [e for e in self.security_events[-100:]
                           if (datetime.now(timezone.utc) - e.timestamp).seconds < 3600]  # Last hour

            threat_counts = {level.value: 0 for level in QuantumThreatLevel}
            for event in recent_events:
                threat_counts[event.threat_level.value] += 1

            overall_threat = "low"
            if threat_counts["critical"] > 0:
                overall_threat = "critical"
            elif threat_counts["high"] > 0 or threat_counts["medium"] > 5:
                overall_threat = "high"
            elif threat_counts["medium"] > 0:
                overall_threat = "medium"

            return {
                "overall_threat_level": overall_threat,
                "active_operations": len(self.active_operations),
                "recent_events_count": len(recent_events),
                "threat_distribution": threat_counts,
                "coherence_monitoring": {
                    "threshold": self.coherence_monitor.coherence_threshold,
                    "history_size": len(self.coherence_monitor.coherence_history)
                },
                "sidechannel_monitoring": {
                    "timing_patterns": len(self.sidechannel_detector.timing_history),
                    "power_history_size": len(self.sidechannel_detector.power_consumption_history),
                    "cache_patterns": len(self.sidechannel_detector.cache_access_patterns)
                }
            }

        except Exception as e:
            logger.error(f"Failed to get security status: {e}")
            return {"error": str(e), "overall_threat_level": "unknown"}

    async def _continuous_monitoring_loop(self) -> None:
        """Continuous background monitoring loop."""
        while True:
            try:
                await asyncio.sleep(10)  # Monitor every 10 seconds

                # Check for stale operations (potential DoS)
                current_time = datetime.now(timezone.utc)
                stale_operations = []

                for op_id, context in self.active_operations.items():
                    if (current_time - context.start_time).seconds > 300:  # 5 minutes
                        stale_operations.append(op_id)

                if stale_operations:
                    logger.warning(f"Detected stale quantum operations: {stale_operations}")
                    await self._record_security_alert("stale_operations", stale_operations)

                # Clean up old events (keep last 10000)
                if len(self.security_events) > 10000:
                    self.security_events = self.security_events[-5000:]

            except Exception as e:
                logger.error(f"Continuous monitoring loop error: {e}")
                await asyncio.sleep(30)  # Back off on error

    async def _security_metrics_collection_loop(self) -> None:
        """Collect security metrics periodically."""
        while True:
            try:
                await asyncio.sleep(60)  # Collect metrics every minute

                status = await self.get_security_status()

                # Record metrics
                self.metrics_collector.record_gauge("active_operations", status["active_operations"])
                self.metrics_collector.record_gauge("recent_events", status["recent_events_count"])

                threat_level_values = {"low": 0, "medium": 1, "high": 2, "critical": 3}
                self.metrics_collector.record_gauge(
                    "overall_threat_level",
                    threat_level_values.get(status["overall_threat_level"], 0)
                )

            except Exception as e:
                logger.error(f"Security metrics collection error: {e}")
                await asyncio.sleep(120)  # Back off on error

    async def _collect_security_metrics(self, event: QuantumSecurityEvent) -> None:
        """Collect metrics from security event."""
        try:
            # Record threat level
            threat_values = {"low": 0, "medium": 1, "high": 2, "critical": 3}
            self.metrics_collector.record_counter(
                f"quantum_threats_{event.threat_level.value}", 1
            )

            # Record coherence metrics
            self.metrics_collector.record_gauge(
                "quantum_coherence_level", event.coherence_level
            )

            # Record security scores
            for metric, value in event.security_metrics.items():
                self.metrics_collector.record_gauge(f"security_metric_{metric}", value)

        except Exception as e:
            logger.error(f"Failed to collect security metrics: {e}")

    async def _record_security_alert(self, alert_type: str, details: Any) -> None:
        """Record a security alert."""
        try:
            alert_event = QuantumSecurityEvent(
                event_id=f"alert_{int(time.time())}",
                event_type=alert_type,
                timestamp=datetime.now(timezone.utc),
                threat_level=QuantumThreatLevel.HIGH,
                quantum_state_id="alert",
                coherence_level=0.0,
                entanglement_metrics={},
                security_metrics={"alert": 1.0},
                metadata={"alert_details": details}
            )

            self.security_events.append(alert_event)
            await self._collect_security_metrics(alert_event)

        except Exception as e:
            logger.error(f"Failed to record security alert: {e}")


# Export main classes for defensive quantum security monitoring
__all__ = [
    "QuantumSecurityMonitor",
    "QuantumCoherenceMonitor",
    "SideChannelDetector",
    "QuantumSecurityEvent",
    "QuantumOperationContext",
    "QuantumThreatLevel",
    "QuantumAnomalyDetector",
    "StatisticalSecurityAnalyzer"
]
