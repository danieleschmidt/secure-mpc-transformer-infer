#!/usr/bin/env python3
"""
Generate performance reports from benchmark results.

This script creates comprehensive HTML and markdown reports from benchmark data.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import statistics
import sys

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib, pandas, or seaborn not available. Plots will be disabled.")

class BenchmarkReporter:
    """Generate reports from benchmark results."""
    
    def __init__(self, results_dir: str, output_file: str):
        """
        Initialize benchmark reporter.
        
        Args:
            results_dir: Directory containing benchmark result files
            output_file: Output file path for the report
        """
        self.results_dir = Path(results_dir)
        self.output_file = Path(output_file)
        self.results = {}
        
    def load_results(self):
        """Load all benchmark results from the results directory."""
        print(f"Loading results from {self.results_dir}")
        
        json_files = list(self.results_dir.glob("*.json"))
        if not json_files:
            print(f"No JSON files found in {self.results_dir}")
            return
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Use filename as key (without extension)
                key = json_file.stem
                self.results[key] = data
                print(f"Loaded {key}")
                
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
    
    def generate_html_report(self) -> str:
        """Generate HTML performance report."""
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Secure MPC Transformer - Performance Report</title>
    <style>
        body {{ 
            font-family: Arial, sans-serif; 
            margin: 40px; 
            background-color: #f5f5f5;
        }}
        .container {{ 
            max-width: 1200px; 
            margin: 0 auto; 
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{ 
            color: #333; 
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .metric-card {{
            background: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
            display: inline-block;
            min-width: 200px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }}
        .metric-label {{
            color: #666;
            font-size: 14px;
        }}
        table {{ 
            border-collapse: collapse; 
            width: 100%; 
            margin: 20px 0;
        }}
        th, td {{ 
            border: 1px solid #ddd; 
            padding: 12px; 
            text-align: left; 
        }}
        th {{ 
            background-color: #4CAF50; 
            color: white; 
        }}
        tr:nth-child(even) {{ 
            background-color: #f2f2f2; 
        }}
        .status-success {{ color: #4CAF50; font-weight: bold; }}
        .status-failed {{ color: #f44336; font-weight: bold; }}
        .status-timeout {{ color: #ff9800; font-weight: bold; }}
        .plot-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .summary-stats {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Secure MPC Transformer Performance Report</h1>
        <p><strong>Generated:</strong> {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}</p>
        
        {self._generate_executive_summary()}
        {self._generate_detailed_results()}
        {self._generate_comparison_tables()}
        {self._generate_plots_section()}
        {self._generate_recommendations()}
        
    </div>
</body>
</html>
"""
        return html
    
    def _generate_executive_summary(self) -> str:
        """Generate executive summary section."""
        if not self.results:
            return "<h2>Executive Summary</h2><p>No benchmark results available.</p>"
        
        # Calculate summary statistics
        total_benchmarks = len(self.results)
        successful_benchmarks = sum(1 for r in self.results.values() 
                                  if r.get('status') == 'success')
        
        # Find best and worst performers
        latency_results = []
        for name, result in self.results.items():
            if result.get('status') == 'success' and 'latency_ms' in result:
                latency_results.append((name, result['latency_ms']['mean']))
        
        html = f"""
        <h2>📊 Executive Summary</h2>
        <div class="summary-stats">
            <div class="metric-card">
                <div class="metric-value">{total_benchmarks}</div>
                <div class="metric-label">Total Benchmarks</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{successful_benchmarks}</div>
                <div class="metric-label">Successful</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{successful_benchmarks/total_benchmarks:.1%}</div>
                <div class="metric-label">Success Rate</div>
            </div>
        """
        
        if latency_results:
            best_performer = min(latency_results, key=lambda x: x[1])
            worst_performer = max(latency_results, key=lambda x: x[1])
            
            html += f"""
            <div class="metric-card">
                <div class="metric-value">{best_performer[1]:.1f}ms</div>
                <div class="metric-label">Best Latency<br>({best_performer[0]})</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{worst_performer[1]:.1f}ms</div>
                <div class="metric-label">Worst Latency<br>({worst_performer[0]})</div>
            </div>
            """
        
        html += "</div>"
        return html
    
    def _generate_detailed_results(self) -> str:
        """Generate detailed results section."""
        html = "<h2>📈 Detailed Results</h2>"
        
        for name, result in self.results.items():
            status = result.get('status', 'unknown')
            status_class = f"status-{status}"
            
            html += f"""
            <h3>{name.replace('_', ' ').title()}</h3>
            <p><strong>Status:</strong> <span class="{status_class}">{status.upper()}</span></p>
            """
            
            if status == 'success':
                # Add configuration info
                config = result.get('config', {})
                if config:
                    html += "<h4>Configuration</h4><ul>"
                    for key, value in config.items():
                        html += f"<li><strong>{key.replace('_', ' ').title()}:</strong> {value}</li>"
                    html += "</ul>"
                
                # Add performance metrics
                html += "<h4>Performance Metrics</h4>"
                html += "<table><tr><th>Metric</th><th>Mean</th><th>Median</th><th>Std</th><th>Min</th><th>Max</th><th>P95</th><th>P99</th></tr>"
                
                for metric_name, stats in result.items():
                    if isinstance(stats, dict) and 'mean' in stats:
                        html += f"""
                        <tr>
                            <td>{metric_name.replace('_', ' ').title()}</td>
                            <td>{stats['mean']:.2f}</td>
                            <td>{stats['median']:.2f}</td>
                            <td>{stats['std']:.2f}</td>
                            <td>{stats['min']:.2f}</td>
                            <td>{stats['max']:.2f}</td>
                            <td>{stats['p95']:.2f}</td>
                            <td>{stats['p99']:.2f}</td>
                        </tr>
                        """
                
                html += "</table>"
            
            else:
                # Show error information
                error = result.get('error', 'Unknown error')
                html += f"<p><strong>Error:</strong> {error}</p>"
        
        return html
    
    def _generate_comparison_tables(self) -> str:
        """Generate comparison tables section."""
        html = "<h2>🔍 Performance Comparisons</h2>"
        
        # Group results by categories
        categories = {}
        for name, result in self.results.items():
            if result.get('status') != 'success':
                continue
            
            # Determine category based on name
            if 'bert' in name and 'protocol' not in name:
                category = 'BERT Models'
            elif 'protocol' in name:
                category = 'MPC Protocols'
            elif 'scalability' in name:
                category = 'Scalability'
            else:
                category = 'Other'
            
            if category not in categories:
                categories[category] = []
            categories[category].append((name, result))
        
        for category, results in categories.items():
            html += f"<h3>{category}</h3>"
            html += "<table><tr><th>Benchmark</th><th>Latency (ms)</th><th>Throughput (QPS)</th><th>GPU Memory (MB)</th></tr>"
            
            for name, result in results:
                latency = result.get('latency_ms', {}).get('mean', 'N/A')
                throughput = result.get('throughput_qps', {}).get('mean', 'N/A')
                gpu_memory = result.get('gpu_memory_mb', {}).get('mean', 'N/A')
                
                html += f"""
                <tr>
                    <td>{name.replace('_', ' ').title()}</td>
                    <td>{latency:.2f if isinstance(latency, (int, float)) else latency}</td>
                    <td>{throughput:.2f if isinstance(throughput, (int, float)) else throughput}</td>
                    <td>{gpu_memory:.2f if isinstance(gpu_memory, (int, float)) else gpu_memory}</td>
                </tr>
                """
            
            html += "</table>"
        
        return html
    
    def _generate_plots_section(self) -> str:
        """Generate plots section."""
        if not PLOTTING_AVAILABLE:
            return "<h2>📊 Performance Plots</h2><p>Plotting libraries not available. Install matplotlib, pandas, and seaborn to generate plots.</p>"
        
        html = "<h2>📊 Performance Plots</h2>"
        
        try:
            # Create plots
            plots_created = []
            
            # Latency comparison plot
            latency_plot = self._create_latency_plot()
            if latency_plot:
                plots_created.append(("Latency Comparison", latency_plot))
            
            # Throughput comparison plot
            throughput_plot = self._create_throughput_plot()
            if throughput_plot:
                plots_created.append(("Throughput Comparison", throughput_plot))
            
            # Memory usage plot
            memory_plot = self._create_memory_plot()
            if memory_plot:
                plots_created.append(("Memory Usage", memory_plot))
            
            # Add plots to HTML
            for title, plot_path in plots_created:
                html += f"""
                <div class="plot-container">
                    <h3>{title}</h3>
                    <img src="{plot_path}" alt="{title}">
                </div>
                """
            
            if not plots_created:
                html += "<p>No plots could be generated from the available data.</p>"
        
        except Exception as e:
            html += f"<p>Error generating plots: {e}</p>"
        
        return html
    
    def _create_latency_plot(self) -> Optional[str]:
        """Create latency comparison plot."""
        data = []
        for name, result in self.results.items():
            if result.get('status') == 'success' and 'latency_ms' in result:
                data.append({
                    'benchmark': name.replace('_', ' ').title(),
                    'latency': result['latency_ms']['mean']
                })
        
        if not data:
            return None
        
        df = pd.DataFrame(data)
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='benchmark', y='latency')
        plt.title('Latency Comparison Across Benchmarks')
        plt.xlabel('Benchmark')
        plt.ylabel('Latency (ms)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plot_path = self.output_file.parent / "latency_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path.name
    
    def _create_throughput_plot(self) -> Optional[str]:
        """Create throughput comparison plot."""
        data = []
        for name, result in self.results.items():
            if result.get('status') == 'success' and 'throughput_qps' in result:
                data.append({
                    'benchmark': name.replace('_', ' ').title(),
                    'throughput': result['throughput_qps']['mean']
                })
        
        if not data:
            return None
        
        df = pd.DataFrame(data)
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='benchmark', y='throughput')
        plt.title('Throughput Comparison Across Benchmarks')
        plt.xlabel('Benchmark')
        plt.ylabel('Throughput (QPS)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plot_path = self.output_file.parent / "throughput_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path.name
    
    def _create_memory_plot(self) -> Optional[str]:
        """Create memory usage plot."""
        data = []
        for name, result in self.results.items():
            if result.get('status') == 'success' and 'gpu_memory_mb' in result:
                data.append({
                    'benchmark': name.replace('_', ' ').title(),
                    'memory': result['gpu_memory_mb']['mean']
                })
        
        if not data:
            return None
        
        df = pd.DataFrame(data)
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='benchmark', y='memory')
        plt.title('GPU Memory Usage Across Benchmarks')
        plt.xlabel('Benchmark')
        plt.ylabel('GPU Memory (MB)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plot_path = self.output_file.parent / "memory_usage.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path.name
    
    def _generate_recommendations(self) -> str:
        """Generate performance recommendations."""
        html = "<h2>💡 Performance Recommendations</h2>"
        
        recommendations = []
        
        # Analyze results for recommendations
        gpu_results = []
        cpu_results = []
        
        for name, result in self.results.items():
            if result.get('status') != 'success':
                continue
            
            if 'gpu' in name.lower():
                gpu_results.append(result)
            elif 'cpu' in name.lower():
                cpu_results.append(result)
        
        # GPU vs CPU recommendation
        if gpu_results and cpu_results:
            gpu_avg = statistics.mean(r['latency_ms']['mean'] for r in gpu_results 
                                    if 'latency_ms' in r)
            cpu_avg = statistics.mean(r['latency_ms']['mean'] for r in cpu_results 
                                    if 'latency_ms' in r)
            
            if gpu_avg < cpu_avg * 0.8:
                recommendations.append(
                    "✅ <strong>GPU Acceleration:</strong> GPU shows significant performance improvement "
                    f"({gpu_avg:.1f}ms vs {cpu_avg:.1f}ms average latency). Recommend GPU deployment."
                )
            else:
                recommendations.append(
                    "⚠️ <strong>GPU Performance:</strong> GPU performance benefit is minimal. "
                    "Consider CPU deployment for cost efficiency."
                )
        
        # Protocol recommendations
        protocol_results = [(name, result) for name, result in self.results.items() 
                           if 'protocol' in name.lower() and result.get('status') == 'success']
        
        if protocol_results:
            best_protocol = min(protocol_results, 
                              key=lambda x: x[1].get('latency_ms', {}).get('mean', float('inf')))
            recommendations.append(
                f"🔒 <strong>Protocol Selection:</strong> {best_protocol[0]} shows best performance "
                f"({best_protocol[1]['latency_ms']['mean']:.1f}ms latency)."
            )
        
        # Memory recommendations
        high_memory_benchmarks = [(name, result) for name, result in self.results.items()
                                if (result.get('status') == 'success' and 
                                    result.get('gpu_memory_mb', {}).get('mean', 0) > 2000)]
        
        if high_memory_benchmarks:
            recommendations.append(
                "💾 <strong>Memory Usage:</strong> Some configurations require >2GB GPU memory. "
                "Consider memory optimization for deployment."
            )
        
        # Batch size recommendations
        scalability_results = [(name, result) for name, result in self.results.items()
                             if 'scalability' in name.lower() and result.get('status') == 'success']
        
        if scalability_results:
            recommendations.append(
                "📈 <strong>Scalability:</strong> Review scalability results to optimize batch size "
                "for your throughput requirements."
            )
        
        # Failed benchmarks
        failed_benchmarks = [name for name, result in self.results.items() 
                           if result.get('status') != 'success']
        
        if failed_benchmarks:
            recommendations.append(
                f"❌ <strong>Failed Benchmarks:</strong> {len(failed_benchmarks)} benchmarks failed. "
                "Review error logs and system requirements."
            )
        
        if not recommendations:
            recommendations.append("ℹ️ No specific recommendations available based on current results.")
        
        html += "<ul>"
        for rec in recommendations:
            html += f"<li>{rec}</li>"
        html += "</ul>"
        
        return html
    
    def generate_markdown_report(self) -> str:
        """Generate markdown performance report."""
        md = f"""# Secure MPC Transformer - Performance Report

**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}

## Executive Summary

"""
        
        if not self.results:
            md += "No benchmark results available.\n"
            return md
        
        total_benchmarks = len(self.results)
        successful_benchmarks = sum(1 for r in self.results.values() 
                                  if r.get('status') == 'success')
        
        md += f"""- **Total Benchmarks:** {total_benchmarks}
- **Successful:** {successful_benchmarks}
- **Success Rate:** {successful_benchmarks/total_benchmarks:.1%}

## Detailed Results

"""
        
        for name, result in self.results.items():
            status = result.get('status', 'unknown')
            md += f"### {name.replace('_', ' ').title()}\n\n"
            md += f"**Status:** {status.upper()}\n\n"
            
            if status == 'success':
                if 'latency_ms' in result:
                    stats = result['latency_ms']
                    md += f"**Latency:** {stats['mean']:.2f}ms (±{stats['std']:.2f})\n\n"
                
                if 'throughput_qps' in result:
                    stats = result['throughput_qps']
                    md += f"**Throughput:** {stats['mean']:.2f} QPS\n\n"
            else:
                error = result.get('error', 'Unknown error')
                md += f"**Error:** {error}\n\n"
        
        return md

def main():
    """Main report generator."""
    parser = argparse.ArgumentParser(description='Generate benchmark performance report')
    
    parser.add_argument('--input', required=True,
                       help='Directory containing benchmark results')
    parser.add_argument('--output', required=True,
                       help='Output file path for the report')
    parser.add_argument('--format', choices=['html', 'markdown'], default='html',
                       help='Report format')
    
    args = parser.parse_args()
    
    # Create reporter
    reporter = BenchmarkReporter(args.input, args.output)
    
    # Load results
    reporter.load_results()
    
    # Generate report
    if args.format == 'html':
        report_content = reporter.generate_html_report()
    else:
        report_content = reporter.generate_markdown_report()
    
    # Save report
    reporter.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(reporter.output_file, 'w') as f:
        f.write(report_content)
    
    print(f"Report generated: {reporter.output_file}")

if __name__ == '__main__':
    main()