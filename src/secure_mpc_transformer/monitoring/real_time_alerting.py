"""
Real-time Alerting System with ML-based Anomaly Detection
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import statistics
import threading
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertStatus(Enum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

class AlertChannel(Enum):
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    PAGERDUTY = "pagerduty"

@dataclass
class AlertRule:
    id: str
    name: str
    description: str
    condition: str  # Metric condition expression
    severity: AlertSeverity
    threshold: float
    comparison: str  # "gt", "lt", "eq", "ne", "gte", "lte"
    window_minutes: int = 5
    cooldown_minutes: int = 15
    enabled: bool = True
    channels: List[AlertChannel] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class Alert:
    id: str
    rule_id: str
    rule_name: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    metric_name: str
    metric_value: float
    threshold: float
    timestamp: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    context: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None

@dataclass
class NotificationChannel:
    type: AlertChannel
    config: Dict[str, Any]
    enabled: bool = True
    rate_limit_per_hour: int = 100
    last_sent_count: int = 0
    last_reset_time: float = field(default_factory=time.time)

class MetricAnomalyDetector:
    """ML-based anomaly detection for metrics"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metric_history = defaultdict(lambda: deque(maxlen=1000))
        self.baselines = {}
        self.anomaly_thresholds = {}
        self._ml_initialized = False
        
        # Initialize ML models if available
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler
            
            self.isolation_forest = IsolationForest(
                contamination=config.get("contamination", 0.05),
                random_state=42
            )
            self.scaler = StandardScaler()
            self._ml_available = True
            
        except ImportError:
            logger.warning("scikit-learn not available, using statistical anomaly detection")
            self._ml_available = False
    
    async def add_metric_point(self, metric_name: str, value: float) -> Dict[str, Any]:
        """Add metric point and check for anomalies"""
        timestamp = time.time()
        
        # Add to history
        self.metric_history[metric_name].append({
            'value': value,
            'timestamp': timestamp
        })
        
        # Check for anomalies
        anomaly_result = await self._detect_anomaly(metric_name, value)
        
        # Update baseline periodically
        if len(self.metric_history[metric_name]) % 100 == 0:
            await self._update_baseline(metric_name)
        
        return anomaly_result
    
    async def _detect_anomaly(self, metric_name: str, value: float) -> Dict[str, Any]:
        """Detect if metric value is anomalous"""
        try:
            history = self.metric_history[metric_name]
            
            if len(history) < 20:  # Need minimum history
                return {
                    'is_anomaly': False,
                    'confidence': 0.0,
                    'method': 'insufficient_data'
                }
            
            # Statistical detection (always available)
            statistical_result = await self._statistical_anomaly_detection(metric_name, value)
            
            # ML-based detection if available
            if self._ml_available and len(history) >= 100:
                ml_result = await self._ml_anomaly_detection(metric_name, value)
                
                # Combine results
                combined_confidence = (statistical_result['confidence'] + ml_result['confidence']) / 2
                is_anomaly = statistical_result['is_anomaly'] or ml_result['is_anomaly']
                
                return {
                    'is_anomaly': is_anomaly,
                    'confidence': combined_confidence,
                    'method': 'combined',
                    'statistical': statistical_result,
                    'ml': ml_result
                }
            
            return statistical_result
            
        except Exception as e:
            logger.error(f"Anomaly detection failed for {metric_name}: {e}")
            return {
                'is_anomaly': False,
                'confidence': 0.0,
                'method': 'error',
                'error': str(e)
            }
    
    async def _statistical_anomaly_detection(self, metric_name: str, value: float) -> Dict[str, Any]:
        """Statistical anomaly detection using z-score and IQR"""
        try:
            history = self.metric_history[metric_name]
            values = [point['value'] for point in history[-100:]]  # Last 100 points
            
            if len(values) < 10:
                return {'is_anomaly': False, 'confidence': 0.0, 'method': 'statistical'}
            
            # Z-score based detection
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0
            
            z_score = abs(value - mean_val) / max(std_val, 0.001)
            z_anomaly = z_score > 3.0  # 3-sigma rule
            z_confidence = min(1.0, z_score / 3.0)
            
            # IQR based detection
            sorted_values = sorted(values)
            q1 = sorted_values[len(sorted_values) // 4]
            q3 = sorted_values[3 * len(sorted_values) // 4]
            iqr = q3 - q1
            
            iqr_lower = q1 - 1.5 * iqr
            iqr_upper = q3 + 1.5 * iqr
            iqr_anomaly = value < iqr_lower or value > iqr_upper
            
            # Combine methods
            is_anomaly = z_anomaly or iqr_anomaly
            confidence = max(z_confidence, 0.5 if iqr_anomaly else 0.0)
            
            return {
                'is_anomaly': is_anomaly,
                'confidence': confidence,
                'method': 'statistical',
                'z_score': z_score,
                'iqr_anomaly': iqr_anomaly,
                'baseline_mean': mean_val,
                'baseline_std': std_val
            }
            
        except Exception as e:
            logger.error(f"Statistical anomaly detection failed: {e}")
            return {'is_anomaly': False, 'confidence': 0.0, 'method': 'statistical_error'}
    
    async def _ml_anomaly_detection(self, metric_name: str, value: float) -> Dict[str, Any]:
        """ML-based anomaly detection using Isolation Forest"""
        try:
            if not self._ml_available:
                return {'is_anomaly': False, 'confidence': 0.0, 'method': 'ml_unavailable'}
            
            history = self.metric_history[metric_name]
            
            if len(history) < 100:
                return {'is_anomaly': False, 'confidence': 0.0, 'method': 'ml_insufficient_data'}
            
            # Prepare training data
            import numpy as np
            
            # Extract features: value, time-based features, trend
            features = []
            values = []
            
            for i, point in enumerate(history[-100:]):
                val = point['value']
                timestamp = point['timestamp']
                
                # Basic features
                feature_vector = [
                    val,
                    timestamp % 86400,  # Time of day
                    len(history) - i,   # Recency
                ]
                
                # Trend features (if enough history)
                if i >= 5:
                    recent_values = [history[j]['value'] for j in range(max(0, i-5), i)]
                    trend = (val - statistics.mean(recent_values)) / max(statistics.stdev(recent_values), 0.001)
                    feature_vector.append(trend)
                else:
                    feature_vector.append(0.0)
                
                features.append(feature_vector)
                values.append(val)
            
            X = np.array(features)
            X_scaled = self.scaler.fit_transform(X)
            
            # Train or update model
            self.isolation_forest.fit(X_scaled)
            
            # Predict anomaly for current value
            current_features = np.array([[
                value,
                time.time() % 86400,
                0,  # Most recent
                0   # No trend for current point
            ]])
            
            current_scaled = self.scaler.transform(current_features)
            
            # Get anomaly score
            anomaly_score = self.isolation_forest.decision_function(current_scaled)[0]
            is_outlier = self.isolation_forest.predict(current_scaled)[0] == -1
            
            # Convert score to confidence
            confidence = max(0.0, min(1.0, (1 - anomaly_score) / 2))
            
            return {
                'is_anomaly': is_outlier,
                'confidence': confidence,
                'method': 'isolation_forest',
                'anomaly_score': anomaly_score
            }
            
        except Exception as e:
            logger.error(f"ML anomaly detection failed: {e}")
            return {'is_anomaly': False, 'confidence': 0.0, 'method': 'ml_error'}
    
    async def _update_baseline(self, metric_name: str) -> None:
        """Update baseline statistics for metric"""
        try:
            history = self.metric_history[metric_name]
            
            if len(history) < 20:
                return
            
            values = [point['value'] for point in history[-200:]]  # Last 200 points
            
            self.baselines[metric_name] = {
                'mean': statistics.mean(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0,
                'min': min(values),
                'max': max(values),
                'median': statistics.median(values),
                'updated_at': time.time()
            }
            
        except Exception as e:
            logger.error(f"Failed to update baseline for {metric_name}: {e}")

class RealTimeAlertingSystem:
    """
    Real-time alerting system with ML-based anomaly detection,
    intelligent notification routing, and alert correlation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Core components
        self.anomaly_detector = MetricAnomalyDetector(config.get("anomaly_detection", {}))
        
        # Alert management
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        
        # Notification channels
        self.notification_channels: Dict[AlertChannel, NotificationChannel] = {}
        
        # Metric tracking
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.metric_locks = defaultdict(threading.RLock)
        
        # Background tasks
        self._alert_evaluation_task = None
        self._notification_task = None
        self._cleanup_task = None
        
        # Alert correlation
        self.correlation_rules = []
        self.correlation_window = config.get("correlation_window", 300)  # 5 minutes
        
        # Rate limiting
        self.alert_rate_limiter = defaultdict(lambda: {'count': 0, 'reset_time': time.time()})
        
    async def start_alerting_system(self) -> None:
        """Start the real-time alerting system"""
        logger.info("Starting real-time alerting system")
        
        try:
            # Initialize default notification channels
            await self._initialize_notification_channels()
            
            # Load default alert rules
            await self._load_default_alert_rules()
            
            # Start background tasks
            self._alert_evaluation_task = asyncio.create_task(self._alert_evaluation_loop())
            self._notification_task = asyncio.create_task(self._notification_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            logger.info("Real-time alerting system started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start alerting system: {e}")
            raise
    
    async def add_metric_point(self, metric_name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Add a metric point and evaluate alerts"""
        try:
            timestamp = time.time()
            
            # Store metric with lock
            with self.metric_locks[metric_name]:
                self.metrics[metric_name].append({
                    'value': value,
                    'timestamp': timestamp,
                    'tags': tags or {}
                })
            
            # Check for anomalies
            anomaly_result = await self.anomaly_detector.add_metric_point(metric_name, value)
            
            # Create anomaly alert if detected
            if anomaly_result.get('is_anomaly') and anomaly_result.get('confidence', 0) > 0.7:
                await self._create_anomaly_alert(metric_name, value, anomaly_result, tags or {})
            
        except Exception as e:
            logger.error(f"Failed to add metric point {metric_name}: {e}")
    
    async def add_alert_rule(self, rule: AlertRule) -> None:
        """Add an alert rule"""
        try:
            self.alert_rules[rule.id] = rule
            logger.info(f"Added alert rule: {rule.name}")
            
        except Exception as e:
            logger.error(f"Failed to add alert rule {rule.id}: {e}")
    
    async def acknowledge_alert(self, alert_id: str, user: str = "system") -> bool:
        """Acknowledge an active alert"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = datetime.now(timezone.utc)
                alert.context['acknowledged_by'] = user
                
                logger.info(f"Alert {alert_id} acknowledged by {user}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to acknowledge alert {alert_id}: {e}")
            return False
    
    async def resolve_alert(self, alert_id: str, user: str = "system") -> bool:
        """Resolve an active alert"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.now(timezone.utc)
                alert.context['resolved_by'] = user
                
                # Move to history
                self.alert_history.append(alert)
                del self.active_alerts[alert_id]
                
                logger.info(f"Alert {alert_id} resolved by {user}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to resolve alert {alert_id}: {e}")
            return False
    
    async def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    async def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alerting system statistics"""
        try:
            current_time = time.time()
            one_hour_ago = current_time - 3600
            
            # Recent alerts (last hour)
            recent_alerts = [
                alert for alert in self.alert_history
                if alert.timestamp.timestamp() > one_hour_ago
            ]
            
            # Statistics by severity
            severity_counts = defaultdict(int)
            for alert in self.active_alerts.values():
                severity_counts[alert.severity.value] += 1
            
            # Notification statistics
            notification_stats = {}
            for channel_type, channel in self.notification_channels.items():
                notification_stats[channel_type.value] = {
                    'enabled': channel.enabled,
                    'rate_limit': channel.rate_limit_per_hour,
                    'sent_last_hour': channel.last_sent_count
                }
            
            return {
                'active_alerts': len(self.active_alerts),
                'alerts_last_hour': len(recent_alerts),
                'total_alert_rules': len(self.alert_rules),
                'enabled_rules': sum(1 for rule in self.alert_rules.values() if rule.enabled),
                'severity_breakdown': dict(severity_counts),
                'notification_channels': notification_stats,
                'anomaly_detection_enabled': True,
                'metrics_tracked': len(self.metrics)
            }
            
        except Exception as e:
            logger.error(f"Failed to get alert statistics: {e}")
            return {'error': str(e)}
    
    async def _create_anomaly_alert(
        self, 
        metric_name: str, 
        value: float, 
        anomaly_result: Dict[str, Any],
        tags: Dict[str, str]
    ) -> None:
        """Create alert for detected anomaly"""
        try:
            alert_id = f"anomaly_{metric_name}_{int(time.time())}"
            
            # Determine severity based on confidence
            confidence = anomaly_result.get('confidence', 0.0)
            if confidence > 0.9:
                severity = AlertSeverity.CRITICAL
            elif confidence > 0.8:
                severity = AlertSeverity.HIGH
            elif confidence > 0.7:
                severity = AlertSeverity.MEDIUM
            else:
                severity = AlertSeverity.LOW
            
            alert = Alert(
                id=alert_id,
                rule_id="anomaly_detection",
                rule_name="ML Anomaly Detection",
                description=f"Anomalous value detected for {metric_name}: {value:.2f} (confidence: {confidence:.2f})",
                severity=severity,
                status=AlertStatus.OPEN,
                metric_name=metric_name,
                metric_value=value,
                threshold=0.0,  # N/A for anomaly detection
                timestamp=datetime.now(timezone.utc),
                context={
                    'anomaly_result': anomaly_result,
                    'tags': tags,
                    'detection_method': anomaly_result.get('method', 'unknown')
                }
            )
            
            # Check for rate limiting
            if not await self._check_rate_limit(alert):
                return
            
            self.active_alerts[alert_id] = alert
            
            # Queue for notification
            await self._queue_notification(alert)
            
            logger.warning(f"Anomaly alert created: {alert.description}")
            
        except Exception as e:
            logger.error(f"Failed to create anomaly alert: {e}")
    
    async def _evaluate_alert_rules(self) -> None:
        """Evaluate all enabled alert rules"""
        try:
            current_time = time.time()
            
            for rule in self.alert_rules.values():
                if not rule.enabled:
                    continue
                
                try:
                    await self._evaluate_single_rule(rule, current_time)
                except Exception as e:
                    logger.error(f"Failed to evaluate rule {rule.id}: {e}")
            
        except Exception as e:
            logger.error(f"Alert rule evaluation failed: {e}")
    
    async def _evaluate_single_rule(self, rule: AlertRule, current_time: float) -> None:
        """Evaluate a single alert rule"""
        try:
            # Extract metric name from condition
            metric_name = self._extract_metric_name(rule.condition)
            if not metric_name or metric_name not in self.metrics:
                return
            
            # Get metric values in window
            window_start = current_time - (rule.window_minutes * 60)
            
            with self.metric_locks[metric_name]:
                window_values = [
                    point for point in self.metrics[metric_name]
                    if point['timestamp'] > window_start
                ]
            
            if not window_values:
                return
            
            # Calculate aggregate value (for now, use latest)
            latest_value = window_values[-1]['value']
            
            # Evaluate condition
            triggered = self._evaluate_condition(latest_value, rule.threshold, rule.comparison)
            
            if triggered:
                await self._create_rule_alert(rule, metric_name, latest_value)
                
        except Exception as e:
            logger.error(f"Failed to evaluate rule {rule.id}: {e}")
    
    def _extract_metric_name(self, condition: str) -> Optional[str]:
        """Extract metric name from condition expression"""
        # Simple parser - in production would use more sophisticated parsing
        if ' ' in condition:
            return condition.split(' ')[0]
        return condition
    
    def _evaluate_condition(self, value: float, threshold: float, comparison: str) -> bool:
        """Evaluate alert condition"""
        comparisons = {
            'gt': value > threshold,
            'gte': value >= threshold,
            'lt': value < threshold,
            'lte': value <= threshold,
            'eq': value == threshold,
            'ne': value != threshold
        }
        return comparisons.get(comparison, False)
    
    async def _create_rule_alert(self, rule: AlertRule, metric_name: str, value: float) -> None:
        """Create alert from rule trigger"""
        try:
            # Check cooldown
            cooldown_key = f"{rule.id}_{metric_name}"
            if not await self._check_cooldown(cooldown_key, rule.cooldown_minutes):
                return
            
            alert_id = f"rule_{rule.id}_{int(time.time())}"
            
            alert = Alert(
                id=alert_id,
                rule_id=rule.id,
                rule_name=rule.name,
                description=f"{rule.description}: {metric_name} = {value:.2f} (threshold: {rule.threshold})",
                severity=rule.severity,
                status=AlertStatus.OPEN,
                metric_name=metric_name,
                metric_value=value,
                threshold=rule.threshold,
                timestamp=datetime.now(timezone.utc),
                context={
                    'rule_tags': rule.tags,
                    'condition': rule.condition,
                    'comparison': rule.comparison
                }
            )
            
            # Check for rate limiting
            if not await self._check_rate_limit(alert):
                return
            
            self.active_alerts[alert_id] = alert
            
            # Queue for notification
            await self._queue_notification(alert)
            
            logger.warning(f"Rule alert created: {alert.description}")
            
        except Exception as e:
            logger.error(f"Failed to create rule alert: {e}")
    
    async def _check_cooldown(self, key: str, cooldown_minutes: int) -> bool:
        """Check if alert is in cooldown period"""
        current_time = time.time()
        
        if key in self.alert_rate_limiter:
            last_alert_time = self.alert_rate_limiter[key].get('last_alert_time', 0)
            if current_time - last_alert_time < cooldown_minutes * 60:
                return False
        
        self.alert_rate_limiter[key]['last_alert_time'] = current_time
        return True
    
    async def _check_rate_limit(self, alert: Alert) -> bool:
        """Check alert rate limiting"""
        current_time = time.time()
        key = f"{alert.severity.value}_{alert.metric_name}"
        
        rate_limit_info = self.alert_rate_limiter[key]
        
        # Reset counter if hour has passed
        if current_time - rate_limit_info['reset_time'] > 3600:
            rate_limit_info['count'] = 0
            rate_limit_info['reset_time'] = current_time
        
        # Check rate limit (severity-based)
        max_per_hour = {
            AlertSeverity.CRITICAL: 10,
            AlertSeverity.HIGH: 20,
            AlertSeverity.MEDIUM: 50,
            AlertSeverity.LOW: 100
        }.get(alert.severity, 100)
        
        if rate_limit_info['count'] >= max_per_hour:
            logger.warning(f"Rate limit exceeded for {key}")
            return False
        
        rate_limit_info['count'] += 1
        return True
    
    async def _queue_notification(self, alert: Alert) -> None:
        """Queue alert for notification"""
        try:
            # Determine channels based on severity
            channels_to_notify = []
            
            if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
                channels_to_notify.extend([AlertChannel.EMAIL, AlertChannel.SLACK])
                
                if alert.severity == AlertSeverity.CRITICAL:
                    channels_to_notify.append(AlertChannel.PAGERDUTY)
            
            elif alert.severity == AlertSeverity.MEDIUM:
                channels_to_notify.append(AlertChannel.SLACK)
            
            else:  # LOW severity
                channels_to_notify.append(AlertChannel.EMAIL)
            
            # Send notifications
            for channel in channels_to_notify:
                if channel in self.notification_channels:
                    await self._send_notification(alert, channel)
            
        except Exception as e:
            logger.error(f"Failed to queue notification for alert {alert.id}: {e}")
    
    async def _send_notification(self, alert: Alert, channel: AlertChannel) -> None:
        """Send notification to specific channel"""
        try:
            notification_channel = self.notification_channels[channel]
            
            if not notification_channel.enabled:
                return
            
            # Check channel rate limit
            current_time = time.time()
            if current_time - notification_channel.last_reset_time > 3600:
                notification_channel.last_sent_count = 0
                notification_channel.last_reset_time = current_time
            
            if notification_channel.last_sent_count >= notification_channel.rate_limit_per_hour:
                logger.warning(f"Rate limit exceeded for {channel.value} notifications")
                return
            
            # Format message
            message = self._format_alert_message(alert, channel)
            
            # Send based on channel type
            if channel == AlertChannel.EMAIL:
                await self._send_email_notification(message, notification_channel.config)
            elif channel == AlertChannel.SLACK:
                await self._send_slack_notification(message, notification_channel.config)
            elif channel == AlertChannel.WEBHOOK:
                await self._send_webhook_notification(alert, notification_channel.config)
            # Add other channel types as needed
            
            notification_channel.last_sent_count += 1
            
            logger.info(f"Notification sent via {channel.value} for alert {alert.id}")
            
        except Exception as e:
            logger.error(f"Failed to send notification via {channel.value}: {e}")
    
    def _format_alert_message(self, alert: Alert, channel: AlertChannel) -> str:
        """Format alert message for specific channel"""
        if channel == AlertChannel.SLACK:
            return f"""🚨 *{alert.severity.value.upper()} Alert*
            
*Rule:* {alert.rule_name}
*Description:* {alert.description}
*Metric:* {alert.metric_name} = {alert.metric_value:.2f}
*Time:* {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
*Status:* {alert.status.value}

Alert ID: {alert.id}
"""
        
        else:  # Default format
            return f"""ALERT: {alert.severity.value.upper()}
            
Rule: {alert.rule_name}
Description: {alert.description}
Metric: {alert.metric_name} = {alert.metric_value:.2f}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
Status: {alert.status.value}
Alert ID: {alert.id}
"""
    
    async def _send_email_notification(self, message: str, config: Dict[str, Any]) -> None:
        """Send email notification"""
        # Placeholder - implement with actual email service
        logger.info(f"Email notification: {message[:100]}...")
    
    async def _send_slack_notification(self, message: str, config: Dict[str, Any]) -> None:
        """Send Slack notification"""
        # Placeholder - implement with actual Slack API
        logger.info(f"Slack notification: {message[:100]}...")
    
    async def _send_webhook_notification(self, alert: Alert, config: Dict[str, Any]) -> None:
        """Send webhook notification"""
        # Placeholder - implement with HTTP client
        webhook_payload = {
            'alert': {
                'id': alert.id,
                'rule_name': alert.rule_name,
                'severity': alert.severity.value,
                'description': alert.description,
                'metric_name': alert.metric_name,
                'metric_value': alert.metric_value,
                'timestamp': alert.timestamp.isoformat(),
                'status': alert.status.value
            }
        }
        logger.info(f"Webhook notification: {json.dumps(webhook_payload)}")
    
    async def _initialize_notification_channels(self) -> None:
        """Initialize notification channels"""
        try:
            # Default channels configuration
            default_channels = {
                AlertChannel.EMAIL: {
                    'smtp_server': self.config.get('email', {}).get('smtp_server', 'localhost'),
                    'smtp_port': self.config.get('email', {}).get('smtp_port', 587),
                    'username': self.config.get('email', {}).get('username', ''),
                    'password': self.config.get('email', {}).get('password', ''),
                    'recipients': self.config.get('email', {}).get('recipients', [])
                },
                AlertChannel.SLACK: {
                    'webhook_url': self.config.get('slack', {}).get('webhook_url', ''),
                    'channel': self.config.get('slack', {}).get('channel', '#alerts')
                },
                AlertChannel.WEBHOOK: {
                    'url': self.config.get('webhook', {}).get('url', ''),
                    'headers': self.config.get('webhook', {}).get('headers', {})
                }
            }
            
            for channel_type, config in default_channels.items():
                self.notification_channels[channel_type] = NotificationChannel(
                    type=channel_type,
                    config=config,
                    enabled=bool(config.get('webhook_url') or config.get('smtp_server') or config.get('url')),
                    rate_limit_per_hour=self.config.get('rate_limits', {}).get(channel_type.value, 100)
                )
            
            logger.info(f"Initialized {len(self.notification_channels)} notification channels")
            
        except Exception as e:
            logger.error(f"Failed to initialize notification channels: {e}")
    
    async def _load_default_alert_rules(self) -> None:
        """Load default alert rules"""
        try:
            default_rules = [
                AlertRule(
                    id="high_error_rate",
                    name="High Error Rate",
                    description="Error rate exceeds threshold",
                    condition="error_rate",
                    severity=AlertSeverity.HIGH,
                    threshold=0.05,  # 5%
                    comparison="gt",
                    window_minutes=5,
                    cooldown_minutes=15,
                    channels=[AlertChannel.EMAIL, AlertChannel.SLACK]
                ),
                AlertRule(
                    id="high_response_time",
                    name="High Response Time",
                    description="Average response time exceeds threshold",
                    condition="avg_response_time",
                    severity=AlertSeverity.MEDIUM,
                    threshold=2.0,  # 2 seconds
                    comparison="gt",
                    window_minutes=10,
                    cooldown_minutes=30,
                    channels=[AlertChannel.SLACK]
                ),
                AlertRule(
                    id="low_cache_hit_rate",
                    name="Low Cache Hit Rate",
                    description="Cache hit rate below threshold",
                    condition="cache_hit_rate",
                    severity=AlertSeverity.LOW,
                    threshold=0.7,  # 70%
                    comparison="lt",
                    window_minutes=15,
                    cooldown_minutes=60,
                    channels=[AlertChannel.EMAIL]
                ),
                AlertRule(
                    id="high_threat_score",
                    name="High Threat Detection",
                    description="Threat score exceeds critical threshold",
                    condition="threat_score",
                    severity=AlertSeverity.CRITICAL,
                    threshold=0.9,  # 90%
                    comparison="gt",
                    window_minutes=1,
                    cooldown_minutes=5,
                    channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.PAGERDUTY]
                )
            ]
            
            for rule in default_rules:
                await self.add_alert_rule(rule)
            
            logger.info(f"Loaded {len(default_rules)} default alert rules")
            
        except Exception as e:
            logger.error(f"Failed to load default alert rules: {e}")
    
    async def _alert_evaluation_loop(self) -> None:
        """Background alert evaluation loop"""
        while True:
            try:
                await asyncio.sleep(30)  # Evaluate every 30 seconds
                await self._evaluate_alert_rules()
                
            except Exception as e:
                logger.error(f"Alert evaluation loop error: {e}")
                await asyncio.sleep(60)  # Back off on error
    
    async def _notification_loop(self) -> None:
        """Background notification processing loop"""
        while True:
            try:
                await asyncio.sleep(10)  # Process every 10 seconds
                
                # Process any pending notifications
                # In a full implementation, this would process a notification queue
                
            except Exception as e:
                logger.error(f"Notification loop error: {e}")
                await asyncio.sleep(30)  # Back off on error
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop for old alerts and metrics"""
        while True:
            try:
                await asyncio.sleep(3600)  # Cleanup every hour
                
                current_time = time.time()
                cleanup_threshold = current_time - (24 * 3600)  # 24 hours ago
                
                # Clean up old metric data
                for metric_name, metric_data in self.metrics.items():
                    with self.metric_locks[metric_name]:
                        # Keep only recent data
                        while metric_data and metric_data[0]['timestamp'] < cleanup_threshold:
                            metric_data.popleft()
                
                # Clean up rate limiter data
                for key in list(self.alert_rate_limiter.keys()):
                    if current_time - self.alert_rate_limiter[key]['reset_time'] > 7200:  # 2 hours
                        del self.alert_rate_limiter[key]
                
                logger.debug("Completed alerting system cleanup")
                
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(1800)  # Back off on error

# Global alerting system instance
alerting_system = None

async def get_alerting_system(config: Dict[str, Any] = None) -> RealTimeAlertingSystem:
    """Get or create global alerting system"""
    global alerting_system
    
    if alerting_system is None:
        alerting_system = RealTimeAlertingSystem(config or {})
        await alerting_system.start_alerting_system()
    
    return alerting_system