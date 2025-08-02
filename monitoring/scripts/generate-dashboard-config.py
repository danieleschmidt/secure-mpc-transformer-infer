#!/usr/bin/env python3
"""
Generate dynamic Grafana dashboard configurations for MPC Transformer monitoring.
This script creates customized dashboards based on deployment configuration.
"""

import json
import os
import argparse
from typing import Dict, List, Any
from pathlib import Path


class DashboardGenerator:
    """Generate Grafana dashboard configurations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dashboard_id = config.get('dashboard_id', 'mpc-transformer')
        self.title = config.get('title', 'MPC Transformer Monitoring')
        
    def generate_base_dashboard(self) -> Dict[str, Any]:
        """Generate base dashboard structure."""
        return {
            "dashboard": {
                "id": None,
                "title": self.title,
                "tags": ["mpc", "transformer", "inference", "security"],
                "style": "dark",
                "timezone": "browser",
                "refresh": "30s",
                "schemaVersion": 30,
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "timepicker": {
                    "refresh_intervals": ["5s", "10s", "30s", "1m", "5m", "15m"],
                    "time_options": ["5m", "15m", "1h", "6h", "12h", "24h", "2d", "7d"]
                },
                "panels": []
            },
            "folderId": 0,
            "overwrite": True
        }
    
    def create_performance_panel(self, panel_id: int, x: int, y: int) -> Dict[str, Any]:
        """Create MPC performance monitoring panel."""
        return {
            "id": panel_id,
            "title": "MPC Computation Performance",
            "type": "stat",
            "gridPos": {"h": 8, "w": 12, "x": x, "y": y},
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "thresholds"},
                    "mappings": [],
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "green", "value": None},
                            {"color": "red", "value": 100}
                        ]
                    },
                    "unit": "s"
                }
            },
            "targets": [
                {
                    "expr": "rate(mpc_computation_duration_seconds_sum[5m]) / rate(mpc_computation_duration_seconds_count[5m])",
                    "legendFormat": "Avg Inference Time",
                    "refId": "A"
                },
                {
                    "expr": "rate(mpc_communication_rounds_total[5m])",
                    "legendFormat": "Communication Rounds/s",
                    "refId": "B"
                }
            ]
        }
    
    def create_security_panel(self, panel_id: int, x: int, y: int) -> Dict[str, Any]:
        """Create security monitoring panel."""
        return {
            "id": panel_id,
            "title": "Security Metrics",
            "type": "table",
            "gridPos": {"h": 8, "w": 12, "x": x, "y": y},
            "fieldConfig": {
                "defaults": {
                    "custom": {"align": "auto", "displayMode": "auto"},
                    "mappings": [],
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "green", "value": None},
                            {"color": "red", "value": 1}
                        ]
                    }
                }
            },
            "targets": [
                {
                    "expr": "mpc_protocol_security_level",
                    "legendFormat": "Security Level (bits)",
                    "refId": "A"
                },
                {
                    "expr": "mpc_privacy_budget_remaining",
                    "legendFormat": "Privacy Budget Remaining",
                    "refId": "B"
                },
                {
                    "expr": "increase(mpc_security_violations_total[1h])",
                    "legendFormat": "Security Violations (1h)",
                    "refId": "C"
                }
            ]
        }
    
    def create_gpu_panel(self, panel_id: int, x: int, y: int) -> Dict[str, Any]:
        """Create GPU monitoring panel."""
        return {
            "id": panel_id,
            "title": "GPU Utilization",
            "type": "timeseries",
            "gridPos": {"h": 8, "w": 12, "x": x, "y": y},
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "palette-classic"},
                    "custom": {
                        "axisLabel": "",
                        "axisPlacement": "auto",
                        "barAlignment": 0,
                        "drawStyle": "line",
                        "fillOpacity": 0,
                        "gradientMode": "none",
                        "hideFrom": {"legend": False, "tooltip": False, "vis": False},
                        "lineInterpolation": "linear",
                        "lineWidth": 1,
                        "pointSize": 5,
                        "scaleDistribution": {"type": "linear"},
                        "showPoints": "auto",
                        "spanNulls": False,
                        "stacking": {"group": "A", "mode": "none"},
                        "thresholdsStyle": {"mode": "off"}
                    },
                    "mappings": [],
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "green", "value": None},
                            {"color": "red", "value": 80}
                        ]
                    },
                    "unit": "percent"
                }
            },
            "targets": [
                {
                    "expr": "gpu_utilization_percent",
                    "legendFormat": "GPU {{device}}",
                    "refId": "A"
                },
                {
                    "expr": "gpu_memory_utilization_percent",
                    "legendFormat": "GPU Memory {{device}}",
                    "refId": "B"
                }
            ]
        }
    
    def create_network_panel(self, panel_id: int, x: int, y: int) -> Dict[str, Any]:
        """Create network monitoring panel."""
        return {
            "id": panel_id,
            "title": "Network Communication",
            "type": "graph",
            "gridPos": {"h": 8, "w": 12, "x": x, "y": y},
            "yAxes": [
                {"label": "Bytes/sec", "show": True},
                {"label": "", "show": True}
            ],
            "targets": [
                {
                    "expr": "rate(mpc_network_bytes_sent_total[5m])",
                    "legendFormat": "Bytes Sent/s - Party {{party_id}}",
                    "refId": "A"
                },
                {
                    "expr": "rate(mpc_network_bytes_received_total[5m])",
                    "legendFormat": "Bytes Received/s - Party {{party_id}}",
                    "refId": "B"
                },
                {
                    "expr": "mpc_network_latency_milliseconds",
                    "legendFormat": "Latency (ms) - Party {{party_id}}",
                    "refId": "C",
                    "yAxis": 2
                }
            ]
        }
    
    def create_error_panel(self, panel_id: int, x: int, y: int) -> Dict[str, Any]:
        """Create error monitoring panel."""
        return {
            "id": panel_id,
            "title": "Error Rates",
            "type": "singlestat",
            "gridPos": {"h": 4, "w": 6, "x": x, "y": y},
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "thresholds"},
                    "mappings": [],
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "green", "value": None},
                            {"color": "yellow", "value": 0.01},
                            {"color": "red", "value": 0.05}
                        ]
                    },
                    "unit": "percentunit"
                }
            },
            "targets": [
                {
                    "expr": "rate(mpc_computation_errors_total[5m]) / rate(mpc_computation_total[5m])",
                    "legendFormat": "Error Rate",
                    "refId": "A"
                }
            ]
        }
    
    def generate_full_dashboard(self) -> Dict[str, Any]:
        """Generate complete dashboard with all panels."""
        dashboard = self.generate_base_dashboard()
        
        panels = [
            self.create_performance_panel(1, 0, 0),
            self.create_security_panel(2, 12, 0),
            self.create_gpu_panel(3, 0, 8),
            self.create_network_panel(4, 12, 8),
            self.create_error_panel(5, 0, 16),
        ]
        
        dashboard["dashboard"]["panels"] = panels
        return dashboard
    
    def save_dashboard(self, output_path: str):
        """Save dashboard configuration to file."""
        dashboard = self.generate_full_dashboard()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(dashboard, f, indent=2)
        
        print(f"Dashboard configuration saved to: {output_path}")


def generate_alerting_rules() -> Dict[str, Any]:
    """Generate Prometheus alerting rules for MPC monitoring."""
    return {
        "groups": [
            {
                "name": "mpc_transformer_alerts",
                "rules": [
                    {
                        "alert": "MPCInferenceLatencyHigh",
                        "expr": "mpc_computation_duration_seconds > 300",
                        "for": "5m",
                        "labels": {"severity": "warning"},
                        "annotations": {
                            "summary": "MPC inference latency is high",
                            "description": "MPC computation taking longer than 5 minutes on {{ $labels.instance }}"
                        }
                    },
                    {
                        "alert": "MPCSecurityViolation",
                        "expr": "increase(mpc_security_violations_total[5m]) > 0",
                        "for": "0m",
                        "labels": {"severity": "critical"},
                        "annotations": {
                            "summary": "MPC security violation detected",
                            "description": "Security violation detected in MPC protocol on {{ $labels.instance }}"
                        }
                    },
                    {
                        "alert": "GPUUtilizationLow",
                        "expr": "gpu_utilization_percent < 10",
                        "for": "10m",
                        "labels": {"severity": "info"},
                        "annotations": {
                            "summary": "GPU utilization is low",
                            "description": "GPU {{ $labels.device }} utilization below 10% for 10 minutes"
                        }
                    },
                    {
                        "alert": "NetworkLatencyHigh",
                        "expr": "mpc_network_latency_milliseconds > 1000",
                        "for": "5m",
                        "labels": {"severity": "warning"},
                        "annotations": {
                            "summary": "High network latency between MPC parties",
                            "description": "Network latency > 1s between parties on {{ $labels.instance }}"
                        }
                    },
                    {
                        "alert": "PrivacyBudgetLow",
                        "expr": "mpc_privacy_budget_remaining < 0.1",
                        "for": "1m",
                        "labels": {"severity": "warning"},
                        "annotations": {
                            "summary": "Privacy budget running low",
                            "description": "Privacy budget below 10% on {{ $labels.instance }}"
                        }
                    }
                ]
            }
        ]
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate Grafana dashboard configurations")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--output-dir", default="./dashboards", help="Output directory")
    parser.add_argument("--generate-alerts", action="store_true", help="Generate alerting rules")
    
    args = parser.parse_args()
    
    # Default configuration
    config = {
        "dashboard_id": "mpc-transformer-main",
        "title": "MPC Transformer - Main Dashboard"
    }
    
    # Load custom configuration if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    
    # Generate dashboard
    generator = DashboardGenerator(config)
    dashboard_path = os.path.join(args.output_dir, "mpc_transformer_dashboard.json")
    generator.save_dashboard(dashboard_path)
    
    # Generate alerting rules if requested
    if args.generate_alerts:
        alerts = generate_alerting_rules()
        alerts_path = os.path.join(args.output_dir, "mpc_transformer_alerts.yml")
        
        os.makedirs(os.path.dirname(alerts_path), exist_ok=True)
        with open(alerts_path, 'w') as f:
            import yaml
            yaml.dump(alerts, f, default_flow_style=False)
        
        print(f"Alerting rules saved to: {alerts_path}")
    
    print("Dashboard generation complete!")


if __name__ == "__main__":
    main()