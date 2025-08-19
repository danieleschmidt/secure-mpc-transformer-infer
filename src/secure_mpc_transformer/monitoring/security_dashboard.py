#!/usr/bin/env python3
"""
Security Metrics Dashboard for Secure MPC Transformer

Comprehensive real-time security monitoring dashboard with threat landscape
visualization, attack vector analysis, and security control effectiveness tracking.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

import numpy as np

from ..security.enhanced_validator import ValidationResult
from ..security.incident_response import (
    IncidentSeverity,
    SecurityIncident,
)
from ..security.quantum_monitor import QuantumSecurityEvent, QuantumThreatLevel

logger = logging.getLogger(__name__)


class DashboardTheme(Enum):
    """Dashboard visual themes."""
    LIGHT = "light"
    DARK = "dark"
    HIGH_CONTRAST = "high_contrast"


class MetricType(Enum):
    """Types of security metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    RATE = "rate"
    PERCENTAGE = "percentage"


@dataclass
class SecurityMetric:
    """Individual security metric data."""
    name: str
    value: int | float
    metric_type: MetricType
    timestamp: datetime
    labels: dict[str, str] = field(default_factory=dict)
    threshold: float | None = None
    status: str = "normal"  # normal, warning, critical
    description: str = ""


@dataclass
class ThreatLandscapeData:
    """Threat landscape visualization data."""
    timestamp: datetime
    threat_categories: dict[str, int]
    severity_distribution: dict[str, int]
    attack_vectors: dict[str, int]
    geographic_distribution: dict[str, int]
    temporal_patterns: dict[str, list[float]]
    trending_threats: list[dict[str, Any]]


@dataclass
class SecurityControlEffectiveness:
    """Security control effectiveness metrics."""
    control_name: str
    detection_rate: float
    false_positive_rate: float
    mean_response_time: float
    blocked_threats: int
    allowed_threats: int
    effectiveness_score: float
    last_updated: datetime


class RealTimeMetricsCollector:
    """Real-time collection and aggregation of security metrics."""

    def __init__(self):
        self.metrics_buffer = {}
        self.aggregated_metrics = {}
        self.time_series_data = defaultdict(lambda: deque(maxlen=1440))  # 24 hours at 1-min intervals
        self.alert_thresholds = {}
        self.metric_history = defaultdict(lambda: deque(maxlen=10000))

    async def collect_security_metric(
        self,
        metric_name: str,
        value: int | float,
        metric_type: MetricType,
        labels: dict[str, str] | None = None,
        threshold: float | None = None
    ) -> SecurityMetric:
        """Collect and store a security metric."""
        try:
            labels = labels or {}
            timestamp = datetime.now(timezone.utc)

            # Determine metric status based on threshold
            status = "normal"
            if threshold is not None:
                if value > threshold:
                    status = "critical"
                elif value > threshold * 0.8:  # 80% of threshold
                    status = "warning"

            metric = SecurityMetric(
                name=metric_name,
                value=value,
                metric_type=metric_type,
                timestamp=timestamp,
                labels=labels,
                threshold=threshold,
                status=status
            )

            # Store in buffer for real-time access
            metric_key = f"{metric_name}_{hash(str(labels))}"
            self.metrics_buffer[metric_key] = metric

            # Add to time series
            time_point = int(timestamp.timestamp() // 60) * 60  # Round to minute
            self.time_series_data[metric_key].append((time_point, value))

            # Add to history
            self.metric_history[metric_key].append(metric)

            # Update aggregated metrics
            await self._update_aggregated_metrics(metric)

            return metric

        except Exception as e:
            logger.error(f"Failed to collect security metric {metric_name}: {e}")
            return SecurityMetric(
                name=metric_name,
                value=0,
                metric_type=MetricType.COUNTER,
                timestamp=datetime.now(timezone.utc),
                status="error"
            )

    async def get_real_time_metrics(self, metric_pattern: str | None = None) -> dict[str, SecurityMetric]:
        """Get current real-time security metrics."""
        try:
            if metric_pattern:
                return {k: v for k, v in self.metrics_buffer.items()
                       if metric_pattern in k}
            return self.metrics_buffer.copy()

        except Exception as e:
            logger.error(f"Failed to get real-time metrics: {e}")
            return {}

    async def get_time_series_data(
        self,
        metric_name: str,
        duration_minutes: int = 60
    ) -> list[tuple[int, float]]:
        """Get time series data for a specific metric."""
        try:
            current_time = time.time()
            cutoff_time = current_time - (duration_minutes * 60)

            # Find all matching metric keys
            matching_keys = [k for k in self.time_series_data.keys()
                           if metric_name in k]

            if not matching_keys:
                return []

            # Use the first matching key
            metric_key = matching_keys[0]
            time_series = self.time_series_data[metric_key]

            # Filter by time range
            filtered_data = [(timestamp, value) for timestamp, value in time_series
                           if timestamp >= cutoff_time]

            return filtered_data

        except Exception as e:
            logger.error(f"Failed to get time series data for {metric_name}: {e}")
            return []

    async def _update_aggregated_metrics(self, metric: SecurityMetric) -> None:
        """Update aggregated metrics based on new metric."""
        try:
            metric_base_name = metric.name.split('_')[0]  # Base metric name

            if metric_base_name not in self.aggregated_metrics:
                self.aggregated_metrics[metric_base_name] = {
                    "count": 0,
                    "sum": 0.0,
                    "min": float('inf'),
                    "max": float('-inf'),
                    "avg": 0.0,
                    "last_updated": metric.timestamp
                }

            agg = self.aggregated_metrics[metric_base_name]
            agg["count"] += 1
            agg["sum"] += metric.value
            agg["min"] = min(agg["min"], metric.value)
            agg["max"] = max(agg["max"], metric.value)
            agg["avg"] = agg["sum"] / agg["count"]
            agg["last_updated"] = metric.timestamp

        except Exception as e:
            logger.error(f"Failed to update aggregated metrics: {e}")


class ThreatLandscapeAnalyzer:
    """Analyze and visualize the current threat landscape."""

    def __init__(self):
        self.threat_data = deque(maxlen=1000)
        self.geographic_analyzer = GeographicThreatAnalyzer()
        self.temporal_analyzer = TemporalThreatAnalyzer()
        self.trend_analyzer = ThreatTrendAnalyzer()

    async def analyze_threat_landscape(
        self,
        incidents: list[SecurityIncident],
        quantum_events: list[QuantumSecurityEvent]
    ) -> ThreatLandscapeData:
        """Analyze current threat landscape from incidents and events."""
        try:
            timestamp = datetime.now(timezone.utc)

            # Analyze threat categories
            threat_categories = {}
            for incident in incidents:
                category = incident.category.value
                threat_categories[category] = threat_categories.get(category, 0) + 1

            # Analyze severity distribution
            severity_distribution = {}
            for incident in incidents:
                severity = incident.severity.value
                severity_distribution[severity] = severity_distribution.get(severity, 0) + 1

            # Analyze attack vectors
            attack_vectors = {}
            for incident in incidents:
                for indicator in incident.threat_indicators[:5]:  # Top 5 indicators
                    attack_vectors[indicator] = attack_vectors.get(indicator, 0) + 1

            # Analyze quantum threats
            quantum_threat_levels = {}
            for event in quantum_events:
                level = event.threat_level.value
                quantum_threat_levels[f"quantum_{level}"] = quantum_threat_levels.get(f"quantum_{level}", 0) + 1

            # Combine attack vectors
            attack_vectors.update(quantum_threat_levels)

            # Geographic distribution
            geographic_distribution = await self.geographic_analyzer.analyze_geographic_threats(incidents)

            # Temporal patterns
            temporal_patterns = await self.temporal_analyzer.analyze_temporal_patterns(incidents)

            # Trending threats
            trending_threats = await self.trend_analyzer.identify_trending_threats(incidents)

            landscape_data = ThreatLandscapeData(
                timestamp=timestamp,
                threat_categories=threat_categories,
                severity_distribution=severity_distribution,
                attack_vectors=attack_vectors,
                geographic_distribution=geographic_distribution,
                temporal_patterns=temporal_patterns,
                trending_threats=trending_threats
            )

            # Store for historical analysis
            self.threat_data.append(landscape_data)

            return landscape_data

        except Exception as e:
            logger.error(f"Threat landscape analysis failed: {e}")
            return ThreatLandscapeData(
                timestamp=datetime.now(timezone.utc),
                threat_categories={},
                severity_distribution={},
                attack_vectors={},
                geographic_distribution={},
                temporal_patterns={},
                trending_threats=[]
            )

    async def get_historical_landscape_data(self, hours: int = 24) -> list[ThreatLandscapeData]:
        """Get historical threat landscape data."""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            return [data for data in self.threat_data if data.timestamp >= cutoff_time]

        except Exception as e:
            logger.error(f"Failed to get historical landscape data: {e}")
            return []


class GeographicThreatAnalyzer:
    """Analyze geographic distribution of threats."""

    async def analyze_geographic_threats(self, incidents: list[SecurityIncident]) -> dict[str, int]:
        """Analyze geographic distribution of threat sources."""
        try:
            geo_distribution = {}

            for incident in incidents:
                # Placeholder for IP geolocation
                # In production, would use actual geolocation service
                source_ip = incident.source_ip

                if source_ip.startswith("192.168.") or source_ip.startswith("10."):
                    country = "Private_Network"
                elif source_ip.startswith("127."):
                    country = "Localhost"
                else:
                    # Simulate geolocation based on IP patterns
                    country = self._simulate_geolocation(source_ip)

                geo_distribution[country] = geo_distribution.get(country, 0) + 1

            return geo_distribution

        except Exception as e:
            logger.error(f"Geographic threat analysis failed: {e}")
            return {}

    def _simulate_geolocation(self, ip: str) -> str:
        """Simulate IP geolocation for demonstration."""
        # Simple simulation based on IP hash
        ip_hash = hash(ip) % 10
        countries = ["US", "CN", "RU", "DE", "GB", "FR", "JP", "KR", "IN", "BR"]
        return countries[ip_hash]


class TemporalThreatAnalyzer:
    """Analyze temporal patterns in threats."""

    async def analyze_temporal_patterns(self, incidents: list[SecurityIncident]) -> dict[str, list[float]]:
        """Analyze temporal patterns in threat activity."""
        try:
            patterns = {
                "hourly": [0.0] * 24,
                "daily": [0.0] * 7,
                "monthly": [0.0] * 12
            }

            for incident in incidents:
                timestamp = incident.timestamp

                # Hourly pattern
                hour = timestamp.hour
                patterns["hourly"][hour] += 1.0

                # Daily pattern (0=Monday, 6=Sunday)
                weekday = timestamp.weekday()
                patterns["daily"][weekday] += 1.0

                # Monthly pattern
                month = timestamp.month - 1  # 0-indexed
                patterns["monthly"][month] += 1.0

            # Normalize patterns
            for pattern_type, values in patterns.items():
                total = sum(values)
                if total > 0:
                    patterns[pattern_type] = [v / total for v in values]

            return patterns

        except Exception as e:
            logger.error(f"Temporal pattern analysis failed: {e}")
            return {"hourly": [], "daily": [], "monthly": []}


class ThreatTrendAnalyzer:
    """Analyze trending threats and emerging patterns."""

    async def identify_trending_threats(self, incidents: list[SecurityIncident]) -> list[dict[str, Any]]:
        """Identify trending threats and attack patterns."""
        try:
            trending_threats = []

            # Analyze recent vs historical threat patterns
            recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
            older_cutoff = datetime.now(timezone.utc) - timedelta(days=7)

            recent_incidents = [i for i in incidents if i.timestamp >= recent_cutoff]
            historical_incidents = [i for i in incidents if older_cutoff <= i.timestamp < recent_cutoff]

            # Count threat categories
            recent_categories = {}
            historical_categories = {}

            for incident in recent_incidents:
                category = incident.category.value
                recent_categories[category] = recent_categories.get(category, 0) + 1

            for incident in historical_incidents:
                category = incident.category.value
                historical_categories[category] = historical_categories.get(category, 0) + 1

            # Calculate trend scores
            for category in recent_categories:
                recent_count = recent_categories[category]
                historical_count = historical_categories.get(category, 1)  # Avoid division by zero

                # Calculate growth rate
                growth_rate = (recent_count - historical_count) / historical_count

                if growth_rate > 0.5:  # 50% increase is considered trending
                    trending_threats.append({
                        "category": category,
                        "recent_count": recent_count,
                        "historical_count": historical_count,
                        "growth_rate": growth_rate,
                        "trend_status": "increasing"
                    })

            # Sort by growth rate
            trending_threats.sort(key=lambda x: x["growth_rate"], reverse=True)

            return trending_threats[:10]  # Top 10 trending threats

        except Exception as e:
            logger.error(f"Threat trend analysis failed: {e}")
            return []


class SecurityControlAnalyzer:
    """Analyze effectiveness of security controls."""

    def __init__(self):
        self.control_metrics = {}
        self.performance_history = defaultdict(lambda: deque(maxlen=1000))

    async def analyze_control_effectiveness(
        self,
        validation_results: list[ValidationResult],
        incidents: list[SecurityIncident],
        quantum_events: list[QuantumSecurityEvent]
    ) -> list[SecurityControlEffectiveness]:
        """Analyze effectiveness of various security controls."""
        try:
            control_effectiveness = []

            # Analyze validation system effectiveness
            validation_effectiveness = await self._analyze_validation_effectiveness(validation_results)
            control_effectiveness.append(validation_effectiveness)

            # Analyze incident response effectiveness
            response_effectiveness = await self._analyze_incident_response_effectiveness(incidents)
            control_effectiveness.append(response_effectiveness)

            # Analyze quantum monitoring effectiveness
            quantum_effectiveness = await self._analyze_quantum_monitoring_effectiveness(quantum_events)
            control_effectiveness.append(quantum_effectiveness)

            return control_effectiveness

        except Exception as e:
            logger.error(f"Control effectiveness analysis failed: {e}")
            return []

    async def _analyze_validation_effectiveness(
        self,
        validation_results: list[ValidationResult]
    ) -> SecurityControlEffectiveness:
        """Analyze validation system effectiveness."""
        try:
            if not validation_results:
                return SecurityControlEffectiveness(
                    control_name="Input Validation",
                    detection_rate=0.0,
                    false_positive_rate=0.0,
                    mean_response_time=0.0,
                    blocked_threats=0,
                    allowed_threats=0,
                    effectiveness_score=0.0,
                    last_updated=datetime.now(timezone.utc)
                )

            # Calculate metrics
            total_validations = len(validation_results)
            blocked_requests = sum(1 for r in validation_results if not r.is_valid)
            high_risk_detections = sum(1 for r in validation_results if r.risk_score > 0.7)

            # Calculate detection rate (high-risk detections / total)
            detection_rate = high_risk_detections / total_validations if total_validations > 0 else 0.0

            # Calculate false positive rate (estimate based on low-risk blocks)
            low_risk_blocks = sum(1 for r in validation_results
                                if not r.is_valid and r.risk_score < 0.3)
            false_positive_rate = low_risk_blocks / blocked_requests if blocked_requests > 0 else 0.0

            # Calculate mean response time
            response_times = [r.validation_time for r in validation_results]
            mean_response_time = np.mean(response_times) if response_times else 0.0

            # Calculate effectiveness score
            effectiveness_score = (detection_rate * 0.4 +
                                 (1 - false_positive_rate) * 0.4 +
                                 (1 / max(mean_response_time, 0.001)) * 0.2)
            effectiveness_score = min(1.0, effectiveness_score)

            return SecurityControlEffectiveness(
                control_name="Input Validation",
                detection_rate=detection_rate,
                false_positive_rate=false_positive_rate,
                mean_response_time=mean_response_time,
                blocked_threats=blocked_requests,
                allowed_threats=total_validations - blocked_requests,
                effectiveness_score=effectiveness_score,
                last_updated=datetime.now(timezone.utc)
            )

        except Exception as e:
            logger.error(f"Validation effectiveness analysis failed: {e}")
            return SecurityControlEffectiveness(
                control_name="Input Validation",
                detection_rate=0.0,
                false_positive_rate=1.0,
                mean_response_time=0.0,
                blocked_threats=0,
                allowed_threats=0,
                effectiveness_score=0.0,
                last_updated=datetime.now(timezone.utc)
            )

    async def _analyze_incident_response_effectiveness(
        self,
        incidents: list[SecurityIncident]
    ) -> SecurityControlEffectiveness:
        """Analyze incident response system effectiveness."""
        try:
            if not incidents:
                return SecurityControlEffectiveness(
                    control_name="Incident Response",
                    detection_rate=0.0,
                    false_positive_rate=0.0,
                    mean_response_time=0.0,
                    blocked_threats=0,
                    allowed_threats=0,
                    effectiveness_score=0.0,
                    last_updated=datetime.now(timezone.utc)
                )

            # Calculate metrics
            total_incidents = len(incidents)
            high_confidence_incidents = sum(1 for i in incidents if i.confidence_score > 0.8)

            # Detection rate based on high-confidence incidents
            detection_rate = high_confidence_incidents / total_incidents if total_incidents > 0 else 0.0

            # False positive rate based on false positive likelihood
            total_fp_likelihood = sum(i.false_positive_likelihood for i in incidents)
            false_positive_rate = total_fp_likelihood / total_incidents if total_incidents > 0 else 0.0

            # Response time (placeholder - would need actual response time data)
            mean_response_time = 30.0  # 30 seconds average

            # Count critical threats handled
            critical_threats = sum(1 for i in incidents
                                 if i.severity in [IncidentSeverity.HIGH, IncidentSeverity.CRITICAL])

            # Effectiveness score
            effectiveness_score = (detection_rate * 0.5 +
                                 (1 - false_positive_rate) * 0.3 +
                                 (critical_threats / max(total_incidents, 1)) * 0.2)
            effectiveness_score = min(1.0, effectiveness_score)

            return SecurityControlEffectiveness(
                control_name="Incident Response",
                detection_rate=detection_rate,
                false_positive_rate=false_positive_rate,
                mean_response_time=mean_response_time,
                blocked_threats=critical_threats,
                allowed_threats=total_incidents - critical_threats,
                effectiveness_score=effectiveness_score,
                last_updated=datetime.now(timezone.utc)
            )

        except Exception as e:
            logger.error(f"Incident response effectiveness analysis failed: {e}")
            return SecurityControlEffectiveness(
                control_name="Incident Response",
                detection_rate=0.0,
                false_positive_rate=1.0,
                mean_response_time=0.0,
                blocked_threats=0,
                allowed_threats=0,
                effectiveness_score=0.0,
                last_updated=datetime.now(timezone.utc)
            )

    async def _analyze_quantum_monitoring_effectiveness(
        self,
        quantum_events: list[QuantumSecurityEvent]
    ) -> SecurityControlEffectiveness:
        """Analyze quantum monitoring system effectiveness."""
        try:
            if not quantum_events:
                return SecurityControlEffectiveness(
                    control_name="Quantum Monitoring",
                    detection_rate=0.0,
                    false_positive_rate=0.0,
                    mean_response_time=0.0,
                    blocked_threats=0,
                    allowed_threats=0,
                    effectiveness_score=0.0,
                    last_updated=datetime.now(timezone.utc)
                )

            total_events = len(quantum_events)
            high_threat_events = sum(1 for e in quantum_events
                                   if e.threat_level in [QuantumThreatLevel.HIGH, QuantumThreatLevel.CRITICAL])

            # Detection rate based on high-threat quantum events
            detection_rate = high_threat_events / total_events if total_events > 0 else 0.0

            # Estimate false positive rate based on low-coherence events
            low_coherence_events = sum(1 for e in quantum_events if e.coherence_level < 0.3)
            false_positive_rate = low_coherence_events / total_events if total_events > 0 else 0.0
            false_positive_rate = min(false_positive_rate, 0.5)  # Cap at 50%

            # Response time (quantum operations are typically fast)
            mean_response_time = 0.1  # 100ms average

            # Effectiveness score
            effectiveness_score = (detection_rate * 0.4 +
                                 (1 - false_positive_rate) * 0.4 +
                                 (1 / max(mean_response_time, 0.001)) * 0.2)
            effectiveness_score = min(1.0, effectiveness_score * 0.01)  # Scale down for quantum

            return SecurityControlEffectiveness(
                control_name="Quantum Monitoring",
                detection_rate=detection_rate,
                false_positive_rate=false_positive_rate,
                mean_response_time=mean_response_time,
                blocked_threats=high_threat_events,
                allowed_threats=total_events - high_threat_events,
                effectiveness_score=effectiveness_score,
                last_updated=datetime.now(timezone.utc)
            )

        except Exception as e:
            logger.error(f"Quantum monitoring effectiveness analysis failed: {e}")
            return SecurityControlEffectiveness(
                control_name="Quantum Monitoring",
                detection_rate=0.0,
                false_positive_rate=1.0,
                mean_response_time=0.0,
                blocked_threats=0,
                allowed_threats=0,
                effectiveness_score=0.0,
                last_updated=datetime.now(timezone.utc)
            )


class DashboardGenerator:
    """Generate dashboard HTML and data for security visualization."""

    def __init__(self, theme: DashboardTheme = DashboardTheme.DARK):
        self.theme = theme
        self.template_cache = {}

    async def generate_dashboard_html(
        self,
        metrics: dict[str, SecurityMetric],
        threat_landscape: ThreatLandscapeData,
        control_effectiveness: list[SecurityControlEffectiveness],
        time_series_data: dict[str, list[tuple[int, float]]]
    ) -> str:
        """Generate comprehensive security dashboard HTML."""
        try:
            # Dashboard HTML template
            dashboard_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Security Dashboard - Secure MPC Transformer</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        {self._get_css_styles()}
    </style>
</head>
<body class="theme-{self.theme.value}">
    <div class="dashboard-container">
        <header class="dashboard-header">
            <h1>🛡️ Security Dashboard - Secure MPC Transformer</h1>
            <div class="timestamp">Last Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</div>
        </header>
        
        <div class="dashboard-grid">
            <div class="metrics-overview">
                <h2>📊 Security Metrics Overview</h2>
                {self._generate_metrics_cards(metrics)}
            </div>
            
            <div class="threat-landscape">
                <h2>🗺️ Threat Landscape</h2>
                {self._generate_threat_landscape_section(threat_landscape)}
            </div>
            
            <div class="control-effectiveness">
                <h2>🛠️ Security Control Effectiveness</h2>
                {self._generate_control_effectiveness_section(control_effectiveness)}
            </div>
            
            <div class="time-series">
                <h2>📈 Time Series Analysis</h2>
                {self._generate_time_series_section(time_series_data)}
            </div>
        </div>
    </div>
    
    <script>
        {self._generate_javascript_code(threat_landscape, control_effectiveness, time_series_data)}
    </script>
</body>
</html>
            """

            return dashboard_html

        except Exception as e:
            logger.error(f"Dashboard HTML generation failed: {e}")
            return f"<html><body><h1>Dashboard Generation Error</h1><p>{str(e)}</p></body></html>"

    def _get_css_styles(self) -> str:
        """Get CSS styles for dashboard."""
        if self.theme == DashboardTheme.DARK:
            return """
                body { 
                    background: #1a1a1a; 
                    color: #ffffff; 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                }
                .dashboard-container { 
                    max-width: 1400px; 
                    margin: 0 auto; 
                }
                .dashboard-header { 
                    text-align: center; 
                    margin-bottom: 30px; 
                    border-bottom: 2px solid #333;
                    padding-bottom: 20px;
                }
                .dashboard-grid { 
                    display: grid; 
                    grid-template-columns: 1fr 1fr; 
                    grid-gap: 20px; 
                }
                .metrics-overview, .threat-landscape, .control-effectiveness, .time-series {
                    background: #2d2d2d;
                    border-radius: 10px;
                    padding: 20px;
                    border: 1px solid #404040;
                }
                .metric-card {
                    background: #3d3d3d;
                    border-radius: 8px;
                    padding: 15px;
                    margin: 10px 0;
                    border-left: 4px solid #00ff88;
                }
                .metric-card.warning { border-left-color: #ffaa00; }
                .metric-card.critical { border-left-color: #ff4444; }
                .chart-container { 
                    position: relative; 
                    height: 300px; 
                    margin: 20px 0; 
                }
                .timestamp { 
                    color: #888; 
                    font-size: 0.9em; 
                }
                h2 { 
                    color: #00ff88; 
                    border-bottom: 1px solid #404040; 
                    padding-bottom: 10px; 
                }
                .effectiveness-bar {
                    background: #404040;
                    border-radius: 10px;
                    height: 20px;
                    margin: 10px 0;
                    overflow: hidden;
                }
                .effectiveness-fill {
                    height: 100%;
                    background: linear-gradient(90deg, #ff4444, #ffaa00, #00ff88);
                    transition: width 0.5s ease;
                }
                .threat-category {
                    display: inline-block;
                    background: #404040;
                    border-radius: 15px;
                    padding: 5px 10px;
                    margin: 5px;
                    font-size: 0.9em;
                }
            """
        else:  # Light theme
            return """
                body { 
                    background: #f8f9fa; 
                    color: #333333; 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                }
                .dashboard-container { 
                    max-width: 1400px; 
                    margin: 0 auto; 
                }
                .dashboard-header { 
                    text-align: center; 
                    margin-bottom: 30px; 
                    border-bottom: 2px solid #dee2e6;
                    padding-bottom: 20px;
                }
                .dashboard-grid { 
                    display: grid; 
                    grid-template-columns: 1fr 1fr; 
                    grid-gap: 20px; 
                }
                .metrics-overview, .threat-landscape, .control-effectiveness, .time-series {
                    background: #ffffff;
                    border-radius: 10px;
                    padding: 20px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    border: 1px solid #dee2e6;
                }
                .metric-card {
                    background: #f8f9fa;
                    border-radius: 8px;
                    padding: 15px;
                    margin: 10px 0;
                    border-left: 4px solid #28a745;
                }
                .metric-card.warning { border-left-color: #ffc107; }
                .metric-card.critical { border-left-color: #dc3545; }
                .chart-container { 
                    position: relative; 
                    height: 300px; 
                    margin: 20px 0; 
                }
                .timestamp { 
                    color: #6c757d; 
                    font-size: 0.9em; 
                }
                h2 { 
                    color: #007bff; 
                    border-bottom: 1px solid #dee2e6; 
                    padding-bottom: 10px; 
                }
                .effectiveness-bar {
                    background: #e9ecef;
                    border-radius: 10px;
                    height: 20px;
                    margin: 10px 0;
                    overflow: hidden;
                }
                .effectiveness-fill {
                    height: 100%;
                    background: linear-gradient(90deg, #dc3545, #ffc107, #28a745);
                    transition: width 0.5s ease;
                }
                .threat-category {
                    display: inline-block;
                    background: #e9ecef;
                    border-radius: 15px;
                    padding: 5px 10px;
                    margin: 5px;
                    font-size: 0.9em;
                }
            """

    def _generate_metrics_cards(self, metrics: dict[str, SecurityMetric]) -> str:
        """Generate HTML for metrics cards."""
        if not metrics:
            return "<p>No metrics available</p>"

        cards_html = ""
        for metric_key, metric in metrics.items():
            status_class = metric.status
            cards_html += f"""
            <div class="metric-card {status_class}">
                <h3>{metric.name}</h3>
                <div class="metric-value">{metric.value}</div>
                <div class="metric-description">{metric.description or metric.metric_type.value}</div>
                <div class="metric-timestamp">{metric.timestamp.strftime('%H:%M:%S')}</div>
            </div>
            """

        return cards_html

    def _generate_threat_landscape_section(self, landscape: ThreatLandscapeData) -> str:
        """Generate threat landscape visualization section."""
        # Threat categories
        categories_html = ""
        for category, count in landscape.threat_categories.items():
            categories_html += f'<span class="threat-category">{category}: {count}</span>'

        # Severity distribution
        severity_html = ""
        for severity, count in landscape.severity_distribution.items():
            severity_html += f'<span class="threat-category">{severity}: {count}</span>'

        # Geographic distribution (top 5)
        geo_items = sorted(landscape.geographic_distribution.items(), key=lambda x: x[1], reverse=True)[:5]
        geo_html = ""
        for country, count in geo_items:
            geo_html += f'<span class="threat-category">{country}: {count}</span>'

        return f"""
        <div class="threat-categories">
            <h4>Threat Categories</h4>
            {categories_html}
        </div>
        
        <div class="severity-distribution">
            <h4>Severity Distribution</h4>
            {severity_html}
        </div>
        
        <div class="geographic-distribution">
            <h4>Geographic Distribution</h4>
            {geo_html}
        </div>
        
        <div class="chart-container">
            <canvas id="threatChart"></canvas>
        </div>
        """

    def _generate_control_effectiveness_section(self, controls: list[SecurityControlEffectiveness]) -> str:
        """Generate security control effectiveness section."""
        if not controls:
            return "<p>No control effectiveness data available</p>"

        controls_html = ""
        for control in controls:
            effectiveness_percent = control.effectiveness_score * 100
            controls_html += f"""
            <div class="control-item">
                <h4>{control.control_name}</h4>
                <div class="effectiveness-bar">
                    <div class="effectiveness-fill" style="width: {effectiveness_percent}%"></div>
                </div>
                <div class="control-metrics">
                    <span>Detection Rate: {control.detection_rate:.1%}</span>
                    <span>False Positive Rate: {control.false_positive_rate:.1%}</span>
                    <span>Response Time: {control.mean_response_time:.3f}s</span>
                    <span>Effectiveness: {effectiveness_percent:.1f}%</span>
                </div>
            </div>
            """

        return controls_html

    def _generate_time_series_section(self, time_series: dict[str, list[tuple[int, float]]]) -> str:
        """Generate time series charts section."""
        if not time_series:
            return "<p>No time series data available</p>"

        return """
        <div class="chart-container">
            <canvas id="timeSeriesChart"></canvas>
        </div>
        <div class="time-series-controls">
            <button onclick="updateTimeRange(60)">1 Hour</button>
            <button onclick="updateTimeRange(360)">6 Hours</button>
            <button onclick="updateTimeRange(1440)">24 Hours</button>
        </div>
        """

    def _generate_javascript_code(
        self,
        landscape: ThreatLandscapeData,
        controls: list[SecurityControlEffectiveness],
        time_series: dict[str, list[tuple[int, float]]]
    ) -> str:
        """Generate JavaScript code for interactive charts."""
        return f"""
        // Threat landscape chart
        const threatCtx = document.getElementById('threatChart');
        if (threatCtx) {{
            new Chart(threatCtx, {{
                type: 'doughnut',
                data: {{
                    labels: {list(landscape.threat_categories.keys())},
                    datasets: [{{
                        data: {list(landscape.threat_categories.values())},
                        backgroundColor: [
                            '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', 
                            '#9966FF', '#FF9F40', '#FF6384', '#C9CBCF'
                        ]
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'Threat Categories Distribution'
                        }}
                    }}
                }}
            }});
        }}
        
        // Time series chart
        const timeSeriesCtx = document.getElementById('timeSeriesChart');
        if (timeSeriesCtx) {{
            const timeSeriesData = {json.dumps({k: v for k, v in list(time_series.items())[:3]})};
            
            const datasets = Object.keys(timeSeriesData).map((key, index) => ({{
                label: key,
                data: timeSeriesData[key].map(([timestamp, value]) => ({{
                    x: timestamp * 1000,
                    y: value
                }})),
                borderColor: ['#FF6384', '#36A2EB', '#FFCE56'][index % 3],
                backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56'][index % 3] + '20',
                fill: false,
                tension: 0.1
            }}));
            
            new Chart(timeSeriesCtx, {{
                type: 'line',
                data: {{ datasets }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        x: {{
                            type: 'time',
                            time: {{
                                unit: 'minute'
                            }}
                        }}
                    }},
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'Security Metrics Over Time'
                        }}
                    }}
                }}
            }});
        }}
        
        // Auto-refresh functionality
        function refreshDashboard() {{
            window.location.reload();
        }}
        
        // Set auto-refresh every 60 seconds
        setInterval(refreshDashboard, 60000);
        
        // Time range update function
        function updateTimeRange(minutes) {{
            console.log('Updating time range to ' + minutes + ' minutes');
            // In production, this would trigger a data refresh
        }}
        """


class SecurityMetricsDashboard:
    """
    Comprehensive security metrics dashboard system.
    
    Provides real-time security monitoring with:
    - Threat landscape visualization
    - Security control effectiveness tracking
    - Time series analysis
    - Interactive dashboards
    - Automated alerting
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.metrics_collector = RealTimeMetricsCollector()
        self.landscape_analyzer = ThreatLandscapeAnalyzer()
        self.control_analyzer = SecurityControlAnalyzer()
        self.dashboard_generator = DashboardGenerator(
            theme=DashboardTheme(self.config.get("theme", "dark"))
        )

        # Data storage
        self.validation_results = deque(maxlen=10000)
        self.security_incidents = deque(maxlen=10000)
        self.quantum_events = deque(maxlen=10000)

        # Alerting
        self.alert_thresholds = self.config.get("alert_thresholds", {
            "critical_incidents_per_hour": 10,
            "high_false_positive_rate": 0.3,
            "low_detection_rate": 0.7,
            "high_response_time": 5.0
        })

    async def start_dashboard(self) -> None:
        """Start the security dashboard system."""
        logger.info("Starting security metrics dashboard")

        # Start background tasks
        asyncio.create_task(self._metrics_collection_loop())
        asyncio.create_task(self._alerting_loop())

    async def record_validation_result(self, result: ValidationResult) -> None:
        """Record a validation result for dashboard analysis."""
        try:
            self.validation_results.append(result)

            # Collect metrics
            await self.metrics_collector.collect_security_metric(
                "validation_requests_total",
                1,
                MetricType.COUNTER,
                labels={"valid": str(result.is_valid)}
            )

            await self.metrics_collector.collect_security_metric(
                "validation_risk_score",
                result.risk_score,
                MetricType.HISTOGRAM
            )

            await self.metrics_collector.collect_security_metric(
                "validation_response_time",
                result.validation_time,
                MetricType.HISTOGRAM,
                threshold=1.0  # 1 second threshold
            )

        except Exception as e:
            logger.error(f"Failed to record validation result: {e}")

    async def record_security_incident(self, incident: SecurityIncident) -> None:
        """Record a security incident for dashboard analysis."""
        try:
            self.security_incidents.append(incident)

            # Collect metrics
            await self.metrics_collector.collect_security_metric(
                "security_incidents_total",
                1,
                MetricType.COUNTER,
                labels={
                    "category": incident.category.value,
                    "severity": incident.severity.value
                }
            )

            await self.metrics_collector.collect_security_metric(
                "incident_confidence_score",
                incident.confidence_score,
                MetricType.HISTOGRAM
            )

            await self.metrics_collector.collect_security_metric(
                "false_positive_likelihood",
                incident.false_positive_likelihood,
                MetricType.HISTOGRAM
            )

        except Exception as e:
            logger.error(f"Failed to record security incident: {e}")

    async def record_quantum_event(self, event: QuantumSecurityEvent) -> None:
        """Record a quantum security event for dashboard analysis."""
        try:
            self.quantum_events.append(event)

            # Collect metrics
            await self.metrics_collector.collect_security_metric(
                "quantum_events_total",
                1,
                MetricType.COUNTER,
                labels={"threat_level": event.threat_level.value}
            )

            await self.metrics_collector.collect_security_metric(
                "quantum_coherence_level",
                event.coherence_level,
                MetricType.GAUGE,
                threshold=0.1  # Coherence threshold
            )

        except Exception as e:
            logger.error(f"Failed to record quantum event: {e}")

    async def generate_dashboard(self) -> str:
        """Generate comprehensive security dashboard HTML."""
        try:
            # Get current metrics
            current_metrics = await self.metrics_collector.get_real_time_metrics()

            # Analyze threat landscape
            recent_incidents = list(self.security_incidents)[-1000:]  # Last 1000 incidents
            recent_quantum_events = list(self.quantum_events)[-1000:]  # Last 1000 events

            threat_landscape = await self.landscape_analyzer.analyze_threat_landscape(
                recent_incidents, recent_quantum_events
            )

            # Analyze control effectiveness
            recent_validations = list(self.validation_results)[-1000:]  # Last 1000 validations
            control_effectiveness = await self.control_analyzer.analyze_control_effectiveness(
                recent_validations, recent_incidents, recent_quantum_events
            )

            # Get time series data
            time_series_data = {}
            for metric_name in ["validation_requests_total", "security_incidents_total", "quantum_coherence_level"]:
                time_series_data[metric_name] = await self.metrics_collector.get_time_series_data(
                    metric_name, duration_minutes=60
                )

            # Generate dashboard HTML
            dashboard_html = await self.dashboard_generator.generate_dashboard_html(
                current_metrics,
                threat_landscape,
                control_effectiveness,
                time_series_data
            )

            return dashboard_html

        except Exception as e:
            logger.error(f"Dashboard generation failed: {e}")
            return f"<html><body><h1>Dashboard Error</h1><p>{str(e)}</p></body></html>"

    async def get_dashboard_data_json(self) -> dict[str, Any]:
        """Get dashboard data in JSON format for API access."""
        try:
            # Get current metrics
            current_metrics = await self.metrics_collector.get_real_time_metrics()

            # Convert metrics to JSON-serializable format
            metrics_json = {}
            for key, metric in current_metrics.items():
                metrics_json[key] = {
                    "name": metric.name,
                    "value": metric.value,
                    "type": metric.metric_type.value,
                    "timestamp": metric.timestamp.isoformat(),
                    "status": metric.status,
                    "threshold": metric.threshold
                }

            # Analyze threat landscape
            recent_incidents = list(self.security_incidents)[-1000:]
            recent_quantum_events = list(self.quantum_events)[-1000:]

            threat_landscape = await self.landscape_analyzer.analyze_threat_landscape(
                recent_incidents, recent_quantum_events
            )

            # Analyze control effectiveness
            recent_validations = list(self.validation_results)[-1000:]
            control_effectiveness = await self.control_analyzer.analyze_control_effectiveness(
                recent_validations, recent_incidents, recent_quantum_events
            )

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metrics": metrics_json,
                "threat_landscape": {
                    "timestamp": threat_landscape.timestamp.isoformat(),
                    "threat_categories": threat_landscape.threat_categories,
                    "severity_distribution": threat_landscape.severity_distribution,
                    "attack_vectors": threat_landscape.attack_vectors,
                    "geographic_distribution": threat_landscape.geographic_distribution,
                    "temporal_patterns": threat_landscape.temporal_patterns,
                    "trending_threats": threat_landscape.trending_threats
                },
                "control_effectiveness": [
                    {
                        "control_name": control.control_name,
                        "detection_rate": control.detection_rate,
                        "false_positive_rate": control.false_positive_rate,
                        "mean_response_time": control.mean_response_time,
                        "blocked_threats": control.blocked_threats,
                        "allowed_threats": control.allowed_threats,
                        "effectiveness_score": control.effectiveness_score,
                        "last_updated": control.last_updated.isoformat()
                    }
                    for control in control_effectiveness
                ]
            }

        except Exception as e:
            logger.error(f"Dashboard JSON generation failed: {e}")
            return {"error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}

    async def _metrics_collection_loop(self) -> None:
        """Background loop for continuous metrics collection."""
        while True:
            try:
                await asyncio.sleep(30)  # Collect metrics every 30 seconds

                # Calculate derived metrics
                await self._calculate_derived_metrics()

            except Exception as e:
                logger.error(f"Metrics collection loop error: {e}")
                await asyncio.sleep(60)  # Back off on error

    async def _calculate_derived_metrics(self) -> None:
        """Calculate derived security metrics."""
        try:
            # Calculate incident rates
            recent_incidents = [i for i in self.security_incidents
                              if (datetime.now(timezone.utc) - i.timestamp).seconds < 3600]  # Last hour

            await self.metrics_collector.collect_security_metric(
                "incidents_per_hour",
                len(recent_incidents),
                MetricType.GAUGE,
                threshold=self.alert_thresholds.get("critical_incidents_per_hour", 10)
            )

            # Calculate average validation response time
            recent_validations = [v for v in self.validation_results
                                if len(self.validation_results) > 0][-100:]  # Last 100

            if recent_validations:
                avg_response_time = np.mean([v.validation_time for v in recent_validations])
                await self.metrics_collector.collect_security_metric(
                    "avg_validation_response_time",
                    avg_response_time,
                    MetricType.GAUGE,
                    threshold=self.alert_thresholds.get("high_response_time", 5.0)
                )

        except Exception as e:
            logger.error(f"Derived metrics calculation failed: {e}")

    async def _alerting_loop(self) -> None:
        """Background loop for security alerting."""
        while True:
            try:
                await asyncio.sleep(60)  # Check alerts every minute

                # Check for alert conditions
                await self._check_alert_conditions()

            except Exception as e:
                logger.error(f"Alerting loop error: {e}")
                await asyncio.sleep(120)  # Back off on error

    async def _check_alert_conditions(self) -> None:
        """Check for security alert conditions."""
        try:
            current_metrics = await self.metrics_collector.get_real_time_metrics()

            # Check for critical conditions
            for metric_key, metric in current_metrics.items():
                if metric.status == "critical":
                    logger.warning(f"SECURITY ALERT: Critical condition detected - "
                                 f"{metric.name} = {metric.value} (threshold: {metric.threshold})")

                    # In production, would trigger actual alerting system

        except Exception as e:
            logger.error(f"Alert condition checking failed: {e}")


# Export main classes for security dashboard
__all__ = [
    "SecurityMetricsDashboard",
    "SecurityMetric",
    "ThreatLandscapeData",
    "SecurityControlEffectiveness",
    "RealTimeMetricsCollector",
    "ThreatLandscapeAnalyzer",
    "SecurityControlAnalyzer",
    "DashboardGenerator",
    "MetricType",
    "DashboardTheme"
]
