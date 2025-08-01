#!/usr/bin/env python3
"""
Terragon Value Metrics Dashboard
Real-time tracking and reporting of autonomous SDLC value delivery
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

class ValueMetricsDashboard:
    """Dashboard for tracking and visualizing value delivery metrics"""
    
    def __init__(self, metrics_path: str = ".terragon/value-metrics.json"):
        self.metrics_path = Path(metrics_path)
        self.metrics = self._load_metrics()
    
    def _load_metrics(self) -> Dict:
        """Load current metrics data"""
        if not self.metrics_path.exists():
            return self._create_empty_metrics()
        return json.loads(self.metrics_path.read_text())
    
    def _create_empty_metrics(self) -> Dict:
        """Create empty metrics structure"""
        return {
            "repository": "secure-mpc-transformer",
            "version": "1.0",
            "lastUpdated": datetime.now().isoformat(),
            "maturityAssessment": {
                "currentLevel": "maturing",
                "scoreBreakdown": {
                    "testing": 85,
                    "security": 75,
                    "documentation": 90,
                    "automation": 65,
                    "monitoring": 80,
                    "deployment": 75
                },
                "overallScore": 78
            },
            "executionHistory": [],
            "backlogMetrics": {},
            "discoveryStats": {},
            "learningMetrics": {},
            "operationalMetrics": {}
        }
    
    def update_maturity_assessment(self, improvements: Dict) -> None:
        """Update repository maturity based on completed work"""
        current = self.metrics["maturityAssessment"]["scoreBreakdown"]
        
        for category, improvement in improvements.items():
            if category in current:
                current[category] = min(100, current[category] + improvement)
        
        # Recalculate overall score
        scores = list(current.values())
        self.metrics["maturityAssessment"]["overallScore"] = sum(scores) // len(scores)
        
        # Update maturity level
        overall = self.metrics["maturityAssessment"]["overallScore"]
        if overall >= 85:
            self.metrics["maturityAssessment"]["currentLevel"] = "advanced"
        elif overall >= 50:
            self.metrics["maturityAssessment"]["currentLevel"] = "maturing"
        elif overall >= 25:
            self.metrics["maturityAssessment"]["currentLevel"] = "developing"
        else:
            self.metrics["maturityAssessment"]["currentLevel"] = "nascent"
    
    def record_execution(self, item: Dict, outcome: Dict) -> None:
        """Record the execution of a value item"""
        execution_record = {
            "timestamp": datetime.now().isoformat(),
            "itemId": item["id"],
            "title": item["title"],
            "category": item["category"],
            "predictedScores": item.get("scores", {}),
            "predictedEffort": item.get("effort_estimate", 0),
            "actualEffort": outcome.get("actual_effort", 0),
            "actualImpact": outcome.get("actual_impact", {}),
            "success": outcome.get("success", True),
            "issues": outcome.get("issues", []),
            "learnings": outcome.get("learnings", "")
        }
        
        self.metrics["executionHistory"].append(execution_record)
        
        # Update operational metrics
        self._update_operational_metrics(execution_record)
        
        # Update learning metrics
        self._update_learning_metrics(execution_record)
    
    def _update_operational_metrics(self, execution: Dict) -> None:
        """Update operational performance metrics"""
        ops = self.metrics.setdefault("operationalMetrics", {})
        
        # Calculate success rate
        successful = sum(1 for e in self.metrics["executionHistory"] if e.get("success", True))
        total = len(self.metrics["executionHistory"])
        ops["autonomousPrSuccessRate"] = successful / total if total > 0 else 1.0
        
        # Calculate human intervention rate
        interventions = sum(1 for e in self.metrics["executionHistory"] if e.get("issues"))
        ops["humanInterventionRequired"] = interventions / total if total > 0 else 0.0
        
        # Calculate rollback rate
        rollbacks = sum(1 for e in self.metrics["executionHistory"] if not e.get("success", True))
        ops["rollbackRate"] = rollbacks / total if total > 0 else 0.0
        
        # Calculate mean time to value
        efforts = [e.get("actualEffort", 0) for e in self.metrics["executionHistory"]]
        ops["meanTimeToValue"] = sum(efforts) / len(efforts) if efforts else 0.0
        
        # Calculate total value delivered
        ops["totalValueDelivered"] = ops.get("totalValueDelivered", 0) + execution.get("actualImpact", {}).get("value_score", 0)
    
    def _update_learning_metrics(self, execution: Dict) -> None:
        """Update learning and adaptation metrics"""
        learning = self.metrics.setdefault("learningMetrics", {})
        
        # Calculate estimation accuracy
        predicted_effort = execution.get("predictedEffort", 0)
        actual_effort = execution.get("actualEffort", 0)
        
        if predicted_effort > 0 and actual_effort > 0:
            accuracy = min(predicted_effort, actual_effort) / max(predicted_effort, actual_effort)
            
            current_accuracy = learning.get("estimationAccuracy")
            if current_accuracy is None:
                learning["estimationAccuracy"] = accuracy
            else:
                # Exponential moving average
                learning["estimationAccuracy"] = 0.8 * current_accuracy + 0.2 * accuracy
        
        # Track adaptation cycles
        learning["adaptationCycles"] = learning.get("adaptationCycles", 0) + 1
    
    def generate_dashboard_report(self) -> str:
        """Generate comprehensive dashboard report"""
        report = []
        report.append("# 📊 Terragon Value Metrics Dashboard\n")
        report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report.append(f"**Repository**: {self.metrics['repository']}")
        report.append(f"**Maturity Level**: {self.metrics['maturityAssessment']['currentLevel'].upper()}")
        report.append(f"**Overall Score**: {self.metrics['maturityAssessment']['overallScore']}/100\n")
        
        # Maturity Assessment Section
        report.append("## 🎯 Repository Maturity Assessment\n")
        scores = self.metrics["maturityAssessment"]["scoreBreakdown"]
        for category, score in scores.items():
            emoji = "✅" if score >= 85 else "⚡" if score >= 70 else "⚠️"
            report.append(f"- **{category.title()}**: {score}/100 {emoji}")
        report.append("")
        
        # Execution History Section
        history = self.metrics.get("executionHistory", [])
        if history:
            report.append("## 🚀 Recent Execution History\n")
            report.append("| Date | Item | Category | Effort | Success | Impact |")
            report.append("|------|------|----------|--------|---------|---------|")
            
            for execution in history[-10:]:  # Last 10 executions
                date = execution["timestamp"][:10]
                title = execution["title"][:30] + "..." if len(execution["title"]) > 30 else execution["title"]
                category = execution["category"]
                effort = f"{execution.get('actualEffort', 0):.1f}h"
                success = "✅" if execution.get("success", True) else "❌"
                impact = execution.get("actualImpact", {}).get("value_score", 0)
                
                report.append(f"| {date} | {title} | {category} | {effort} | {success} | {impact} |")
            report.append("")
        
        # Operational Metrics Section
        ops = self.metrics.get("operationalMetrics", {})
        if ops:
            report.append("## 📈 Operational Performance\n")
            success_rate = ops.get('autonomousPrSuccessRate', 0) or 0
            intervention_rate = ops.get('humanInterventionRequired', 0) or 0
            rollback_rate = ops.get('rollbackRate', 0) or 0
            mean_time = ops.get('meanTimeToValue', 0) or 0
            total_value = ops.get('totalValueDelivered', 0) or 0
            
            report.append(f"- **Success Rate**: {success_rate:.1%}")
            report.append(f"- **Human Intervention**: {intervention_rate:.1%}")
            report.append(f"- **Rollback Rate**: {rollback_rate:.1%}")
            report.append(f"- **Mean Time to Value**: {mean_time:.1f} hours")
            report.append(f"- **Total Value Delivered**: ${total_value:,}")
            report.append("")
        
        # Learning Metrics Section
        learning = self.metrics.get("learningMetrics", {})
        if learning:
            report.append("## 🧠 Learning & Adaptation\n")
            estimation_accuracy = learning.get('estimationAccuracy', 0) or 0
            adaptation_cycles = learning.get('adaptationCycles', 0) or 0
            
            report.append(f"- **Estimation Accuracy**: {estimation_accuracy:.1%}")
            report.append(f"- **Adaptation Cycles**: {adaptation_cycles}")
            report.append("")
        
        # Value Trends Section
        if len(history) >= 5:
            report.append("## 📊 Value Delivery Trends\n")
            
            # Calculate weekly value delivery
            now = datetime.now()
            week_ago = now - timedelta(days=7)
            recent_executions = [
                e for e in history 
                if datetime.fromisoformat(e["timestamp"]) > week_ago
            ]
            
            weekly_value = sum(
                e.get("actualImpact", {}).get("value_score", 0) 
                for e in recent_executions
            )
            
            report.append(f"- **This Week**: {len(recent_executions)} items, ${weekly_value:,} value")
            
            # Calculate velocity trend
            if len(history) >= 10:
                recent_efforts = [e.get("actualEffort", 0) for e in history[-5:]]
                earlier_efforts = [e.get("actualEffort", 0) for e in history[-10:-5]]
                
                recent_avg = sum(recent_efforts) / len(recent_efforts) if recent_efforts else 0
                earlier_avg = sum(earlier_efforts) / len(earlier_efforts) if earlier_efforts else 0
                
                if earlier_avg > 0:
                    velocity_change = ((recent_avg - earlier_avg) / earlier_avg) * 100
                    trend = "📈 Improving" if velocity_change < -10 else "📉 Declining" if velocity_change > 10 else "➡️ Stable"
                    report.append(f"- **Velocity Trend**: {trend} ({velocity_change:+.1f}%)")
            
            report.append("")
        
        # Recommendations Section
        report.append("## 💡 Recommendations\n")
        
        # Maturity-based recommendations
        current_score = self.metrics["maturityAssessment"]["overallScore"]
        if current_score < 85:
            report.append("### 🎯 Maturity Improvements")
            scores = self.metrics["maturityAssessment"]["scoreBreakdown"]
            lowest_categories = sorted(scores.items(), key=lambda x: x[1])[:2]
            
            for category, score in lowest_categories:
                report.append(f"- **{category.title()}** ({score}/100): Focus area for next sprint")
            
            report.append("")
        
        # Operational recommendations
        rollback_rate = ops.get("rollbackRate", 0) or 0
        intervention_rate = ops.get("humanInterventionRequired", 0) or 0
        estimation_accuracy = learning.get("estimationAccuracy", 1.0) or 1.0
        
        if rollback_rate > 0.1:
            report.append("- ⚠️ **High Rollback Rate**: Review quality gates and testing")
        
        if intervention_rate > 0.2:
            report.append("- ⚠️ **High Intervention Rate**: Improve autonomous decision making")
        
        if estimation_accuracy < 0.7:
            report.append("- ⚠️ **Low Estimation Accuracy**: Recalibrate effort estimation model")
        
        report.append("")
        
        # Next Actions Section
        report.append("## 🎯 Suggested Next Actions\n")
        report.append("1. Review BACKLOG.md for highest-priority items")
        report.append("2. Execute top-scored autonomous value delivery")
        report.append("3. Monitor execution outcomes and update metrics")
        report.append("4. Adapt scoring models based on learnings")
        
        return "\n".join(report)
    
    def save_metrics(self) -> None:
        """Save updated metrics to file"""
        self.metrics["lastUpdated"] = datetime.now().isoformat()
        self.metrics_path.write_text(json.dumps(self.metrics, indent=2))
    
    def export_csv_report(self, output_path: str = ".terragon/metrics-export.csv") -> None:
        """Export execution history to CSV for analysis"""
        if not self.metrics.get("executionHistory"):
            return
        
        import csv
        
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = [
                'timestamp', 'itemId', 'title', 'category', 
                'predictedEffort', 'actualEffort', 'success',
                'valueScore', 'learnings'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for execution in self.metrics["executionHistory"]:
                writer.writerow({
                    'timestamp': execution['timestamp'],
                    'itemId': execution['itemId'],
                    'title': execution['title'],
                    'category': execution['category'],
                    'predictedEffort': execution.get('predictedEffort', 0),
                    'actualEffort': execution.get('actualEffort', 0),
                    'success': execution.get('success', True),
                    'valueScore': execution.get('actualImpact', {}).get('value_score', 0),
                    'learnings': execution.get('learnings', '')
                })

def main():
    """Generate and display metrics dashboard"""
    dashboard = ValueMetricsDashboard()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--export":
        dashboard.export_csv_report()
        print("📊 Metrics exported to .terragon/metrics-export.csv")
        return
    
    # Generate dashboard report
    report = dashboard.generate_dashboard_report()
    print(report)
    
    # Save updated metrics
    dashboard.save_metrics()

if __name__ == "__main__":
    main()