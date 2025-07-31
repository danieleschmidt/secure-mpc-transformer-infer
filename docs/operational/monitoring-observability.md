# Monitoring and Observability Framework

This document outlines the comprehensive monitoring and observability setup for the secure MPC transformer infrastructure.

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Applications  │───▶│   Prometheus    │───▶│     Grafana     │
│                 │    │   (Metrics)     │    │   (Dashboard)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Jaeger        │    │   Alertmanager  │    │   PagerDuty     │
│   (Tracing)     │    │   (Alerts)      │    │  (Incidents)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│   Fluentd       │    │   ElasticSearch │
│   (Logs)        │───▶│   (Log Storage) │
└─────────────────┘    └─────────────────┘
```

## Metrics Collection

### Core Metrics
Track essential system and application metrics:

#### MPC Protocol Metrics
```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# MPC operation counters
mpc_operations_total = Counter(
    'mpc_operations_total',
    'Total MPC operations performed',
    ['protocol', 'operation', 'party_id']
)

# Computation time histograms
mpc_computation_duration_seconds = Histogram(
    'mpc_computation_duration_seconds',
    'Time spent on MPC computations',
    ['model', 'protocol', 'batch_size'],
    buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0]
)

# GPU utilization gauge
gpu_utilization_percent = Gauge(
    'gpu_utilization_percent',
    'GPU utilization percentage',
    ['gpu_id', 'node']
)

# Memory usage metrics
memory_usage_bytes = Gauge(
    'memory_usage_bytes',
    'Memory usage in bytes',
    ['type', 'node']  # type: gpu_memory, system_memory
)

# Network communication metrics
network_bytes_total = Counter(
    'network_bytes_total',
    'Total network bytes transferred',
    ['direction', 'party_id', 'protocol']
)

# Cryptographic operation metrics
crypto_operations_total = Counter(
    'crypto_operations_total',
    'Total cryptographic operations',
    ['operation', 'key_size']  # operation: encrypt, decrypt, sign, verify
)
```

#### Application Performance Metrics
```python
# Application-specific metrics
inference_requests_total = Counter(
    'inference_requests_total',
    'Total inference requests',
    ['model', 'status']  # status: success, error, timeout
)

model_accuracy_ratio = Gauge(
    'model_accuracy_ratio',
    'Model accuracy on test data',
    ['model', 'dataset']
)

privacy_budget_remaining = Gauge(
    'privacy_budget_remaining',
    'Remaining differential privacy budget',
    ['session_id', 'epsilon_type']
)
```

### Custom Metrics Exporter
```python
# mpc_exporter.py
#!/usr/bin/env python3
"""Custom Prometheus exporter for MPC metrics"""

import time
import psutil
import nvidia_ml_py3 as nvml
from prometheus_client import start_http_server, Gauge, Counter
from secure_mpc_transformer.monitoring import MPCMetricsCollector

class MPCExporter:
    def __init__(self, port=9090):
        self.port = port
        self.collector = MPCMetricsCollector()
        
        # Initialize GPU monitoring
        nvml.nvmlInit()
        self.gpu_count = nvml.nvmlDeviceGetCount()
        
        # Define metrics
        self.gpu_memory_used = Gauge(
            'gpu_memory_used_bytes',
            'GPU memory used in bytes',
            ['gpu_id']
        )
        
        self.mpc_active_sessions = Gauge(
            'mpc_active_sessions',
            'Number of active MPC sessions',
            ['protocol']
        )
        
    def collect_metrics(self):
        """Collect all metrics"""
        while True:
            try:
                # GPU metrics
                for i in range(self.gpu_count):
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)
                    mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                    self.gpu_memory_used.labels(gpu_id=i).set(mem_info.used)
                
                # MPC session metrics
                sessions = self.collector.get_active_sessions()
                for protocol, count in sessions.items():
                    self.mpc_active_sessions.labels(protocol=protocol).set(count)
                    
            except Exception as e:
                print(f"Error collecting metrics: {e}")
                
            time.sleep(15)  # Collect every 15 seconds
            
    def start(self):
        """Start the metrics server"""
        start_http_server(self.port)
        print(f"MPC metrics exporter started on port {self.port}")
        self.collect_metrics()

if __name__ == '__main__':
    exporter = MPCExporter()
    exporter.start()
```

## Alerting Rules

### Critical Alerts
```yaml
# prometheus-alerts.yml
groups:
  - name: mpc-critical
    rules:
      - alert: MPCComputationTimeout
        expr: increase(mpc_computation_duration_seconds[5m]) > 300
        for: 2m
        labels:
          severity: critical
          component: mpc-protocol
        annotations:
          summary: "MPC computation taking too long"
          description: "MPC computation has exceeded 5 minutes for {{ $labels.protocol }}"
          
      - alert: GPUMemoryExhaustion
        expr: gpu_memory_used_bytes / gpu_memory_total_bytes > 0.95
        for: 1m
        labels:
          severity: critical
          component: gpu
        annotations:
          summary: "GPU memory almost exhausted"
          description: "GPU {{ $labels.gpu_id }} memory usage is at {{ $value }}%"
          
      - alert: MPCProtocolFailure
        expr: increase(mpc_operations_total{status="error"}[5m]) > 5
        for: 1m
        labels:
          severity: critical
          component: mpc-protocol
        annotations:
          summary: "High MPC protocol failure rate"
          description: "{{ $value }} MPC operations failed in the last 5 minutes"

  - name: mpc-warning
    rules:
      - alert: PrivacyBudgetLow
        expr: privacy_budget_remaining < 0.1
        for: 5m
        labels:
          severity: warning
          component: privacy
        annotations:
          summary: "Privacy budget running low"
          description: "Session {{ $labels.session_id }} has only {{ $value }} privacy budget remaining"
          
      - alert: NetworkLatencyHigh
        expr: histogram_quantile(0.95, mpc_network_latency_seconds) > 0.5
        for: 5m
        labels:
          severity: warning
          component: network
        annotations:
          summary: "High network latency between MPC parties"
          description: "95th percentile network latency is {{ $value }}s"
```

### Performance Alerts
```yaml
  - name: mpc-performance
    rules:
      - alert: GPUUtilizationLow
        expr: gpu_utilization_percent < 20
        for: 10m
        labels:
          severity: info
          component: performance
        annotations:
          summary: "Low GPU utilization"
          description: "GPU {{ $labels.gpu_id }} utilization is only {{ $value }}%"
          
      - alert: ModelAccuracyDrop
        expr: model_accuracy_ratio < 0.85
        for: 5m
        labels:
          severity: warning
          component: model-quality
        annotations:
          summary: "Model accuracy has dropped"
          description: "Model {{ $labels.model }} accuracy is {{ $value }}"
```

## Distributed Tracing

### Jaeger Configuration
```yaml
# jaeger-config.yml
apiVersion: v1
kind: ConfigMap
metadata:
  name: jaeger-config
data:
  jaeger.yaml: |
    es.server-urls: http://elasticsearch:9200
    es.username: jaeger
    es.password: ${ELASTICSEARCH_PASSWORD}
    collector:
      grpc-server:
        host-port: :14250
      http-server:
        host-port: :14268
    query:
      base-path: /jaeger
```

### Application Tracing
```python
# tracing.py
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

# Configure tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger-agent",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Instrument HTTP requests
RequestsInstrumentor().instrument()

class MPCTracing:
    def __init__(self):
        self.tracer = trace.get_tracer("mpc-transformer")
    
    def trace_mpc_operation(self, operation_name, protocol, party_id):
        """Context manager for tracing MPC operations"""
        return self.tracer.start_as_current_span(
            operation_name,
            attributes={
                "mpc.protocol": protocol,
                "mpc.party_id": party_id,
                "component": "mpc-engine"
            }
        )
    
    def trace_inference(self, model_name, batch_size):
        """Trace model inference operations"""
        return self.tracer.start_as_current_span(
            "model_inference",
            attributes={
                "model.name": model_name,
                "model.batch_size": batch_size,
                "component": "inference-engine"
            }
        )

# Usage example
tracing = MPCTracing()

def secure_inference(text, model, protocol):
    with tracing.trace_inference(model.name, len(text)):
        with tracing.trace_mpc_operation("secret_sharing", protocol, party_id=0):
            shares = secret_share(text)
            
        with tracing.trace_mpc_operation("secure_computation", protocol, party_id=0):
            result = mpc_compute(shares, model)
            
        return result
```

## Logging Strategy

### Centralized Logging with Fluentd
```yaml
# fluentd-config.yml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentd-config
data:
  fluent.conf: |
    <source>
      @type tail
      path /var/log/containers/*.log
      pos_file /var/log/fluentd-containers.log.pos
      tag kubernetes.*
      format json
      time_format %Y-%m-%dT%H:%M:%S.%NZ
    </source>
    
    <filter kubernetes.**>
      @type kubernetes_metadata
    </filter>
    
    # Parse MPC-specific logs
    <filter kubernetes.var.log.containers.mpc-**>
      @type parser
      key_name log
      format json
      reserve_data true
    </filter>
    
    # Security log filtering
    <filter kubernetes.**>
      @type grep
      <regexp>
        key log
        pattern (ERROR|SECURITY|CRYPTO|MPC_FAILURE)
      </regexp>
    </filter>
    
    <match kubernetes.**>
      @type elasticsearch
      host elasticsearch
      port 9200
      index_name mpc-logs
      type_name _doc
      include_tag_key true
      tag_key @log_name
    </match>
```

### Structured Logging
```python
# logging_config.py
import logging
import json
from datetime import datetime

class MPCFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add MPC-specific context
        if hasattr(record, 'party_id'):
            log_entry['mpc.party_id'] = record.party_id
        if hasattr(record, 'protocol'):
            log_entry['mpc.protocol'] = record.protocol
        if hasattr(record, 'operation'):
            log_entry['mpc.operation'] = record.operation
            
        # Security context
        if hasattr(record, 'security_level'):
            log_entry['security.level'] = record.security_level
            
        return json.dumps(log_entry)

# Configure logger
logger = logging.getLogger('mpc-transformer')
handler = logging.StreamHandler()
handler.setFormatter(MPCFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Usage
logger.info(
    "MPC operation completed",
    extra={
        'party_id': 1,
        'protocol': 'aby3',
        'operation': 'matmul',
        'duration_ms': 1250
    }
)
```

## Dashboard Configuration

### Grafana Dashboards
Main MPC overview dashboard:

```json
{
  "dashboard": {
    "title": "MPC Transformer Overview",
    "tags": ["mpc", "transformer", "security"],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "panels": [
      {
        "title": "MPC Operations Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(mpc_operations_total[5m])",
            "legendFormat": "{{protocol}} - {{operation}}"
          }
        ],
        "yAxes": [
          {
            "label": "Operations/sec",
            "min": 0
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "type": "singlestat",
        "targets": [
          {
            "expr": "avg(gpu_utilization_percent)",
            "legendFormat": "Average GPU Usage"
          }
        ],
        "valueName": "current",
        "format": "percent",
        "thresholds": "70,85"
      },
      {
        "title": "Inference Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, mpc_computation_duration_seconds)",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, mpc_computation_duration_seconds)",
            "legendFormat": "Median"
          }
        ]
      },
      {
        "title": "Network Traffic",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(network_bytes_total[5m])",
            "legendFormat": "{{direction}} - Party {{party_id}}"
          }
        ]
      }
    ]
  }
}
```

### Security Dashboard
```json
{
  "dashboard": {
    "title": "MPC Security Monitoring",
    "panels": [
      {
        "title": "Privacy Budget Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "privacy_budget_remaining",
            "legendFormat": "Session {{session_id}}"
          }
        ]
      },
      {
        "title": "Cryptographic Operations",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(crypto_operations_total[5m])",
            "legendFormat": "{{operation}}"
          }
        ]
      },
      {
        "title": "Protocol Failures",
        "type": "table",
        "targets": [
          {
            "expr": "increase(mpc_operations_total{status=\"error\"}[1h])",
            "format": "table"
          }
        ]
      }
    ]
  }
}
```

## Performance Monitoring

### SLI/SLO Definitions
```yaml
# slo-config.yml
slos:
  - name: mpc-computation-latency
    description: "MPC computation should complete within reasonable time"
    sli: histogram_quantile(0.95, mpc_computation_duration_seconds{model="bert-base"})
    target: 60  # 60 seconds for 95th percentile
    
  - name: inference-availability
    description: "Inference API should be available"
    sli: rate(inference_requests_total{status="success"}[5m]) / rate(inference_requests_total[5m])
    target: 0.99  # 99% success rate
    
  - name: gpu-utilization-efficiency
    description: "GPU resources should be efficiently utilized"
    sli: avg_over_time(gpu_utilization_percent[1h])
    target: 70  # 70% average utilization
```

### Capacity Planning
```python
# capacity_monitor.py
import numpy as np
from datetime import datetime, timedelta

class CapacityMonitor:
    def __init__(self, prometheus_client):
        self.prometheus = prometheus_client
        
    def predict_gpu_usage(self, days_ahead=7):
        """Predict GPU usage for capacity planning"""
        # Get historical data
        query = 'avg_over_time(gpu_utilization_percent[24h])'
        data = self.prometheus.query_range(
            query,
            start_time=datetime.now() - timedelta(days=30),
            end_time=datetime.now(),
            step='1h'
        )
        
        # Simple linear regression for trend
        x = np.arange(len(data))
        y = [float(d['value'][1]) for d in data]
        
        trend = np.polyfit(x, y, 1)[0]
        current_usage = y[-1]
        
        predicted_usage = current_usage + (trend * 24 * days_ahead)
        
        return {
            'current_usage': current_usage,
            'predicted_usage': predicted_usage,
            'trend_per_day': trend * 24,
            'capacity_warning': predicted_usage > 80
        }
```

## Incident Response Integration

### PagerDuty Integration
```yaml
# alertmanager-config.yml
global:
  pagerduty_url: 'https://events.pagerduty.com/v2/enqueue'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'
  routes:
  - match:
      severity: critical
    receiver: pagerduty-critical
  - match:
      component: mpc-protocol
    receiver: mpc-team

receivers:
- name: 'pagerduty-critical'
  pagerduty_configs:
  - service_key: '{{ .PAGERDUTY_SERVICE_KEY }}'
    description: 'Critical MPC Infrastructure Alert'
    details:
      summary: '{{ .GroupLabels.alertname }}'
      timestamp: '{{ .CommonAnnotations.timestamp }}'
      
- name: 'mpc-team'
  slack_configs:
  - api_url: '{{ .SLACK_WEBHOOK_URL }}'
    channel: '#mpc-alerts'
    text: 'MPC Alert: {{ .CommonAnnotations.summary }}'
```

### Runbook Automation
```python
# runbook_automation.py
from kubernetes import client, config
import requests

class RunbookAutomation:
    def __init__(self):
        config.load_incluster_config()
        self.k8s = client.AppsV1Api()
        
    def handle_gpu_memory_exhaustion(self, alert):
        """Automated response to GPU memory alerts"""
        # Scale down non-critical workloads
        self.scale_deployment('mpc-benchmark', replicas=0)
        
        # Clear GPU memory cache
        self.exec_pod_command(
            'mpc-compute-0',
            ['python', '-c', 'import torch; torch.cuda.empty_cache()']
        )
        
        # Send notification
        self.notify_oncall("GPU memory exhausted, scaled down benchmarks")
        
    def handle_mpc_protocol_failure(self, alert):
        """Automated response to MPC protocol failures"""
        # Restart affected MPC nodes
        protocol = alert['labels']['protocol']
        self.restart_mpc_nodes(protocol)
        
        # Enable debug logging
        self.update_log_level('DEBUG')
```

This comprehensive monitoring and observability framework provides full visibility into the MPC transformer system's performance, security, and operational health.