#!/usr/bin/env python3
"""
Generate project health dashboard for Secure MPC Transformer.
Creates comprehensive project health visualization and reports.
"""

import json
import os
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
import subprocess


class HealthDashboardGenerator:
    """Generate project health dashboard and reports."""
    
    def __init__(self, metrics_path: str = ".github/project-metrics.json"):
        self.metrics_path = Path(metrics_path)
        self.metrics = self.load_metrics()
        self.repo_root = Path(".")
        
    def load_metrics(self) -> Dict[str, Any]:
        """Load project metrics."""
        if self.metrics_path.exists():
            with open(self.metrics_path, 'r') as f:
                return json.load(f)
        return {}
    
    def collect_real_time_metrics(self) -> Dict[str, Any]:
        """Collect real-time metrics from repository."""
        metrics = {}
        
        # Git metrics
        try:
            # Count commits
            result = subprocess.run(['git', 'rev-list', '--count', 'HEAD'], 
                                  capture_output=True, text=True)
            metrics['total_commits'] = int(result.stdout.strip()) if result.returncode == 0 else 0
            
            # Count contributors
            result = subprocess.run(['git', 'shortlog', '-sn'], 
                                  capture_output=True, text=True)
            contributors = len(result.stdout.strip().split('\n')) if result.returncode == 0 else 0
            metrics['contributors'] = contributors
            
            # Last commit date
            result = subprocess.run(['git', 'log', '-1', '--format=%ci'], 
                                  capture_output=True, text=True)
            metrics['last_commit'] = result.stdout.strip() if result.returncode == 0 else None
            
        except Exception as e:
            print(f"Warning: Could not collect git metrics: {e}")
        
        # Code metrics
        try:
            # Count Python files
            python_files = list(Path("src").rglob("*.py")) if Path("src").exists() else []
            metrics['python_files'] = len(python_files)
            
            # Count lines of code
            total_lines = 0
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        total_lines += len(f.readlines())
                except Exception:
                    pass
            metrics['lines_of_code'] = total_lines
            
            # Count test files
            test_files = list(Path("tests").rglob("*.py")) if Path("tests").exists() else []
            metrics['test_files'] = len(test_files)
            
        except Exception as e:
            print(f"Warning: Could not collect code metrics: {e}")
        
        # Documentation metrics
        try:
            doc_files = list(Path("docs").rglob("*.md")) if Path("docs").exists() else []
            metrics['documentation_files'] = len(doc_files)
            
            # Check README exists and is substantial
            readme_path = Path("README.md")
            if readme_path.exists():
                with open(readme_path, 'r', encoding='utf-8') as f:
                    readme_lines = len(f.readlines())
                metrics['readme_lines'] = readme_lines
                metrics['has_substantial_readme'] = readme_lines > 50
            else:
                metrics['readme_lines'] = 0
                metrics['has_substantial_readme'] = False
                
        except Exception as e:
            print(f"Warning: Could not collect documentation metrics: {e}")
        
        return metrics
    
    def calculate_health_score(self, real_time_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall project health score."""
        score = 0
        max_score = 100
        breakdown = {}
        
        # Documentation score (20 points)
        doc_score = 0
        if real_time_metrics.get('has_substantial_readme', False):
            doc_score += 10
        if real_time_metrics.get('documentation_files', 0) > 5:
            doc_score += 10
        breakdown['documentation'] = {'score': doc_score, 'max': 20}
        score += doc_score
        
        # Code quality score (30 points)
        quality_score = 0
        if real_time_metrics.get('test_files', 0) > 0:
            quality_score += 10
        if real_time_metrics.get('lines_of_code', 0) > 1000:
            quality_score += 10
        # Check for CI/CD setup
        if Path(".github/workflows").exists():
            quality_score += 10
        breakdown['code_quality'] = {'score': quality_score, 'max': 30}
        score += quality_score
        
        # Development activity score (20 points)
        activity_score = 0
        if real_time_metrics.get('total_commits', 0) > 10:
            activity_score += 10
        if real_time_metrics.get('contributors', 0) > 1:
            activity_score += 10
        breakdown['development_activity'] = {'score': activity_score, 'max': 20}
        score += activity_score
        
        # Project setup score (30 points)
        setup_score = 0
        essential_files = ['LICENSE', 'CODE_OF_CONDUCT.md', 'CONTRIBUTING.md', 'SECURITY.md']
        for file in essential_files:
            if Path(file).exists():
                setup_score += 5
        if Path("pyproject.toml").exists() or Path("setup.py").exists():
            setup_score += 10
        breakdown['project_setup'] = {'score': setup_score, 'max': 30}
        score += setup_score
        
        # Calculate percentage
        health_percentage = (score / max_score) * 100
        
        return {
            'total_score': score,
            'max_score': max_score,
            'percentage': health_percentage,
            'grade': self.get_health_grade(health_percentage),
            'breakdown': breakdown
        }
    
    def get_health_grade(self, percentage: float) -> str:
        """Get health grade based on percentage."""
        if percentage >= 90:
            return 'A+'
        elif percentage >= 80:
            return 'A'
        elif percentage >= 70:
            return 'B'
        elif percentage >= 60:
            return 'C'
        elif percentage >= 50:
            return 'D'
        else:
            return 'F'
    
    def generate_html_dashboard(self, health_data: Dict[str, Any], 
                              real_time_metrics: Dict[str, Any]) -> str:
        """Generate HTML dashboard."""
        
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Project Health Dashboard - {project_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f7fa;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5rem;
            font-weight: 300;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1rem;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            padding: 40px;
        }}
        .metric-card {{
            background: #f8f9fc;
            border-radius: 8px;
            padding: 20px;
            border-left: 4px solid #667eea;
        }}
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
            margin: 10px 0;
        }}
        .metric-label {{
            color: #666;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .health-score {{
            text-align: center;
            padding: 40px;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }}
        .score-circle {{
            width: 120px;
            height: 120px;
            border-radius: 50%;
            background: rgba(255,255,255,0.2);
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 20px auto;
            font-size: 2rem;
            font-weight: bold;
        }}
        .grade-{grade_class} {{
            background: {grade_color};
        }}
        .breakdown {{
            padding: 40px;
        }}
        .breakdown-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 15px 0;
            padding: 10px;
            background: #f8f9fc;
            border-radius: 4px;
        }}
        .progress-bar {{
            width: 200px;
            height: 10px;
            background: #e0e0e0;
            border-radius: 5px;
            overflow: hidden;
        }}
        .progress-fill {{
            height: 100%;
            background: #667eea;
            transition: width 0.3s ease;
        }}
        .timestamp {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9rem;
            border-top: 1px solid #eee;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{project_name}</h1>
            <p>{project_description}</p>
        </div>
        
        <div class="health-score">
            <div class="score-circle grade-{grade_class}">
                {health_grade}
            </div>
            <h2>Overall Health Score: {health_percentage:.1f}%</h2>
            <p>{health_score}/{max_score} points</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Total Commits</div>
                <div class="metric-value">{total_commits}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Contributors</div>
                <div class="metric-value">{contributors}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Lines of Code</div>
                <div class="metric-value">{lines_of_code:,}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Test Files</div>
                <div class="metric-value">{test_files}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Documentation Files</div>
                <div class="metric-value">{documentation_files}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Python Files</div>
                <div class="metric-value">{python_files}</div>
            </div>
        </div>
        
        <div class="breakdown">
            <h3>Health Score Breakdown</h3>
            {breakdown_html}
        </div>
        
        <div class="timestamp">
            Generated on {timestamp}
        </div>
    </div>
</body>
</html>
        """
        
        # Generate breakdown HTML
        breakdown_html = ""
        for category, data in health_data['breakdown'].items():
            percentage = (data['score'] / data['max']) * 100
            breakdown_html += f"""
            <div class="breakdown-item">
                <span>{category.replace('_', ' ').title()}</span>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {percentage}%"></div>
                </div>
                <span>{data['score']}/{data['max']}</span>
            </div>
            """
        
        # Determine grade colors
        grade_colors = {
            'A+': '#4CAF50', 'A': '#8BC34A', 'B': '#FFC107',
            'C': '#FF9800', 'D': '#F44336', 'F': '#D32F2F'
        }
        
        grade_class = health_data['grade'].replace('+', 'plus').lower()
        grade_color = grade_colors.get(health_data['grade'], '#666')
        
        # Format the template
        return html_template.format(
            project_name=self.metrics.get('project', {}).get('name', 'Unknown Project'),
            project_description=self.metrics.get('project', {}).get('description', ''),
            health_grade=health_data['grade'],
            health_percentage=health_data['percentage'],
            health_score=health_data['total_score'],
            max_score=health_data['max_score'],
            grade_class=grade_class,
            grade_color=grade_color,
            total_commits=real_time_metrics.get('total_commits', 0),
            contributors=real_time_metrics.get('contributors', 0),
            lines_of_code=real_time_metrics.get('lines_of_code', 0),
            test_files=real_time_metrics.get('test_files', 0),
            documentation_files=real_time_metrics.get('documentation_files', 0),
            python_files=real_time_metrics.get('python_files', 0),
            breakdown_html=breakdown_html,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        )
    
    def generate_json_report(self, health_data: Dict[str, Any], 
                           real_time_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate JSON report."""
        return {
            'generated_at': datetime.now().isoformat(),
            'project': self.metrics.get('project', {}),
            'health_score': health_data,
            'metrics': real_time_metrics,
            'recommendations': self.generate_recommendations(health_data, real_time_metrics)
        }
    
    def generate_recommendations(self, health_data: Dict[str, Any], 
                               real_time_metrics: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Documentation recommendations
        if health_data['breakdown']['documentation']['score'] < 15:
            if not real_time_metrics.get('has_substantial_readme', False):
                recommendations.append("Improve README.md with comprehensive project description")
            if real_time_metrics.get('documentation_files', 0) < 5:
                recommendations.append("Add more documentation files (guides, API docs, etc.)")
        
        # Code quality recommendations  
        if health_data['breakdown']['code_quality']['score'] < 20:
            if real_time_metrics.get('test_files', 0) == 0:
                recommendations.append("Add unit tests to improve code quality")
            if not Path(".github/workflows").exists():
                recommendations.append("Set up CI/CD workflows for automated testing")
        
        # Development activity recommendations
        if health_data['breakdown']['development_activity']['score'] < 15:
            if real_time_metrics.get('total_commits', 0) < 10:
                recommendations.append("Increase development activity with more regular commits")
            if real_time_metrics.get('contributors', 0) <= 1:
                recommendations.append("Encourage community contributions")
        
        # Project setup recommendations
        if health_data['breakdown']['project_setup']['score'] < 20:
            essential_files = ['LICENSE', 'CODE_OF_CONDUCT.md', 'CONTRIBUTING.md', 'SECURITY.md']
            missing = [f for f in essential_files if not Path(f).exists()]
            if missing:
                recommendations.append(f"Add missing project files: {', '.join(missing)}")
        
        return recommendations
    
    def save_dashboard(self, output_dir: str = "dashboard"):
        """Generate and save dashboard files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Collect metrics
        real_time_metrics = self.collect_real_time_metrics()
        health_data = self.calculate_health_score(real_time_metrics)
        
        # Generate HTML dashboard
        html_content = self.generate_html_dashboard(health_data, real_time_metrics)
        with open(output_path / "index.html", 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Generate JSON report
        json_report = self.generate_json_report(health_data, real_time_metrics)
        with open(output_path / "health-report.json", 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=2)
        
        print(f"✓ Dashboard generated in {output_path}/")
        print(f"✓ Health Score: {health_data['percentage']:.1f}% ({health_data['grade']})")
        print(f"✓ Open dashboard: file://{output_path.absolute()}/index.html")
        
        return health_data


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate project health dashboard")
    parser.add_argument("--metrics", default=".github/project-metrics.json", 
                       help="Path to metrics configuration file")
    parser.add_argument("--output", default="dashboard", 
                       help="Output directory for dashboard files")
    parser.add_argument("--json-only", action="store_true", 
                       help="Generate only JSON report")
    
    args = parser.parse_args()
    
    try:
        generator = HealthDashboardGenerator(args.metrics)
        
        if args.json_only:
            real_time_metrics = generator.collect_real_time_metrics()
            health_data = generator.calculate_health_score(real_time_metrics)
            json_report = generator.generate_json_report(health_data, real_time_metrics)
            print(json.dumps(json_report, indent=2))
        else:
            health_data = generator.save_dashboard(args.output)
            
            # Print recommendations
            if health_data['percentage'] < 80:
                real_time_metrics = generator.collect_real_time_metrics()
                recommendations = generator.generate_recommendations(health_data, real_time_metrics)
                if recommendations:
                    print("\n📋 Recommendations for improvement:")
                    for i, rec in enumerate(recommendations, 1):
                        print(f"  {i}. {rec}")
    
    except Exception as e:
        print(f"Error generating dashboard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()