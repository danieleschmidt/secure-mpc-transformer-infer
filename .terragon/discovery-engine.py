#!/usr/bin/env python3
"""
Terragon Autonomous SDLC Discovery Engine
Continuous value discovery and prioritization system
"""

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml

class ValueDiscoveryEngine:
    """Main discovery engine for finding and scoring work items"""
    
    def __init__(self, config_path: str = ".terragon/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.metrics_path = Path(".terragon/value-metrics.json") 
        self.discovered_items = []
        
    def _load_config(self) -> Dict:
        """Load Terragon configuration"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        return yaml.safe_load(self.config_path.read_text())
    
    def discover_all_signals(self) -> List[Dict]:
        """Run all discovery sources and collect signals"""
        print("🔍 Starting comprehensive value discovery...")
        
        all_items = []
        
        # Git history analysis
        if self._is_source_enabled("gitHistory"):
            print("📝 Analyzing git history for TODOs and technical debt...")
            all_items.extend(self._discover_git_history())
            
        # Static analysis
        if self._is_source_enabled("staticAnalysis"):
            print("🔬 Running static analysis tools...")
            all_items.extend(self._discover_static_analysis())
            
        # Security scanning  
        if self._is_source_enabled("securityScanning"):
            print("🛡️ Performing security vulnerability scanning...")
            all_items.extend(self._discover_security_issues())
            
        # Issue tracker integration
        if self._is_source_enabled("issueTrackers"):
            print("📋 Checking GitHub issues and discussions...")
            all_items.extend(self._discover_github_issues())
            
        # Performance monitoring
        if self._is_source_enabled("performanceMonitoring"):
            print("⚡ Analyzing performance bottlenecks...")
            all_items.extend(self._discover_performance_issues())
            
        # Compliance tracking
        if self._is_source_enabled("complianceTracking"):
            print("📊 Checking compliance and documentation gaps...")
            all_items.extend(self._discover_compliance_gaps())
        
        print(f"✅ Discovery complete! Found {len(all_items)} potential work items")
        return all_items
    
    def _is_source_enabled(self, source_name: str) -> bool:
        """Check if discovery source is enabled"""
        sources = self.config.get("discovery", {}).get("sources", [])
        for source in sources:
            if isinstance(source, dict) and source.get("name") == source_name:
                return source.get("enabled", True)
        return source_name in sources
    
    def _discover_git_history(self) -> List[Dict]:
        """Analyze git history for TODO/FIXME comments and technical debt indicators"""
        items = []
        
        patterns = self.config.get("discovery", {}).get("sources", [{}])[0].get("patterns", [
            "TODO", "FIXME", "HACK", "XXX", "DEPRECATED"
        ])
        
        for pattern in patterns:
            try:
                # Search for pattern in code files
                result = subprocess.run([
                    "git", "grep", "-n", "-i", pattern,
                    "--", "*.py", "*.yaml", "*.yml", "*.md", "*.toml"
                ], capture_output=True, text=True, cwd=".")
                
                if result.returncode == 0:
                    for line in result.stdout.strip().split("\n"):
                        if ":" in line:
                            file_path, line_num, content = line.split(":", 2)
                            items.append({
                                "id": f"git-{pattern.lower()}-{len(items)}",
                                "title": f"Address {pattern} in {file_path}",
                                "description": content.strip(),
                                "category": "technical_debt",
                                "source": "gitHistory",
                                "priority": "medium",
                                "file": file_path,
                                "line": int(line_num),
                                "effort_estimate": self._estimate_effort(content, pattern),
                                "created_at": datetime.now().isoformat()
                            })
            except subprocess.CalledProcessError:
                continue
                
        return items
    
    def _discover_static_analysis(self) -> List[Dict]:
        """Run static analysis tools to find code quality issues"""
        items = []
        
        # Run ruff for Python linting issues
        try:
            result = subprocess.run([
                "ruff", "check", "src/", "--output-format=json"
            ], capture_output=True, text=True)
            
            if result.stdout:
                issues = json.loads(result.stdout)
                for issue in issues:
                    items.append({
                        "id": f"ruff-{issue.get('code', 'unknown')}-{len(items)}",
                        "title": f"Fix {issue.get('code')}: {issue.get('message')}",
                        "description": issue.get("message", ""),
                        "category": "code_quality",
                        "source": "staticAnalysis",
                        "priority": self._map_ruff_priority(issue.get("code", "")),
                        "file": issue.get("filename", ""),
                        "line": issue.get("location", {}).get("row", 0),
                        "effort_estimate": 1,
                        "created_at": datetime.now().isoformat()
                    })
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            pass
            
        # Run mypy for type checking issues
        try:
            result = subprocess.run([
                "mypy", "src/", "--no-error-summary"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                for line in result.stdout.split("\n"):
                    if "error:" in line and ":" in line:
                        parts = line.split(":", 3)
                        if len(parts) >= 4:
                            items.append({
                                "id": f"mypy-{len(items)}",
                                "title": f"Fix type error in {parts[0]}",
                                "description": parts[3].strip(),
                                "category": "type_safety",
                                "source": "staticAnalysis",
                                "priority": "medium",
                                "file": parts[0],
                                "line": int(parts[1]) if parts[1].isdigit() else 0,
                                "effort_estimate": 2,
                                "created_at": datetime.now().isoformat()
                            })
        except (subprocess.CalledProcessError, ValueError):
            pass
            
        return items
    
    def _discover_security_issues(self) -> List[Dict]:
        """Scan for security vulnerabilities and compliance issues"""
        items = []
        
        # Run bandit for Python security issues
        try:
            result = subprocess.run([
                "bandit", "-r", "src/", "-f", "json"
            ], capture_output=True, text=True)
            
            if result.stdout:
                report = json.loads(result.stdout)
                for issue in report.get("results", []):
                    items.append({
                        "id": f"security-{issue.get('test_id')}-{len(items)}",
                        "title": f"Security: {issue.get('issue_text')}",
                        "description": issue.get("issue_text", ""),
                        "category": "security",
                        "source": "securityScanning",
                        "priority": self._map_security_priority(issue.get("issue_severity")),
                        "file": issue.get("filename", ""),
                        "line": issue.get("line_number", 0),
                        "effort_estimate": self._estimate_security_effort(issue.get("issue_severity")),
                        "created_at": datetime.now().isoformat()
                    })
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            pass
            
        # Check for dependency vulnerabilities
        try:
            result = subprocess.run([
                "safety", "check", "--json"
            ], capture_output=True, text=True)
            
            if result.stdout:
                vulnerabilities = json.loads(result.stdout)
                for vuln in vulnerabilities:
                    items.append({
                        "id": f"dependency-vuln-{len(items)}",
                        "title": f"Update vulnerable dependency: {vuln.get('package')}",
                        "description": vuln.get("advisory", ""),
                        "category": "security",
                        "source": "securityScanning", 
                        "priority": "high",
                        "effort_estimate": 1,
                        "created_at": datetime.now().isoformat()
                    })
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            pass
            
        return items
    
    def _discover_github_issues(self) -> List[Dict]:
        """Fetch open GitHub issues and convert to work items"""
        items = []
        
        try:
            # Use gh CLI to fetch issues
            result = subprocess.run([
                "gh", "issue", "list", "--json", 
                "number,title,body,labels,createdAt", "--limit", "50"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                issues = json.loads(result.stdout)
                for issue in issues:
                    category = self._categorize_github_issue(issue.get("labels", []))
                    items.append({
                        "id": f"github-issue-{issue['number']}",
                        "title": issue["title"],
                        "description": issue.get("body", "")[:200] + "..." if issue.get("body") else "",
                        "category": category,
                        "source": "issueTrackers",
                        "priority": self._prioritize_github_issue(issue.get("labels", [])),
                        "effort_estimate": self._estimate_github_effort(issue.get("labels", [])),
                        "external_id": issue["number"],
                        "created_at": issue["createdAt"]
                    })
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            pass
            
        return items
    
    def _discover_performance_issues(self) -> List[Dict]:
        """Analyze performance bottlenecks and optimization opportunities"""
        items = []
        
        # Check for large files that might impact performance
        try:
            result = subprocess.run([
                "find", "src/", "-name", "*.py", "-size", "+10k"
            ], capture_output=True, text=True)
            
            for file_path in result.stdout.strip().split("\n"):
                if file_path:
                    items.append({
                        "id": f"perf-large-file-{len(items)}",
                        "title": f"Review large file for optimization: {file_path}",
                        "description": f"File size exceeds 10KB, may need refactoring",
                        "category": "performance",
                        "source": "performanceMonitoring",
                        "priority": "low",
                        "file": file_path,
                        "effort_estimate": 4,
                        "created_at": datetime.now().isoformat()
                    })
        except subprocess.CalledProcessError:
            pass
            
        return items
    
    def _discover_compliance_gaps(self) -> List[Dict]:
        """Check for documentation and compliance gaps"""
        items = []
        
        # Check for missing docstrings in Python files
        try:
            result = subprocess.run([
                "grep", "-r", "-L", '"""', "src/", "--include=*.py"
            ], capture_output=True, text=True)
            
            for file_path in result.stdout.strip().split("\n"):
                if file_path and Path(file_path).stat().st_size > 100:  # Only non-trivial files
                    items.append({
                        "id": f"docs-missing-docstring-{len(items)}",
                        "title": f"Add documentation to {file_path}",
                        "description": "Missing module or class docstrings",
                        "category": "documentation",
                        "source": "complianceTracking",
                        "priority": "low",
                        "file": file_path,
                        "effort_estimate": 2,
                        "created_at": datetime.now().isoformat()
                    })
        except subprocess.CalledProcessError:
            pass
            
        return items
    
    def calculate_composite_scores(self, items: List[Dict]) -> List[Dict]:
        """Calculate WSJF + ICE + Technical Debt composite scores"""
        print("📊 Calculating composite scores for discovered items...")
        
        for item in items:
            # WSJF calculation
            wsjf_score = self._calculate_wsjf(item)
            
            # ICE calculation
            ice_score = self._calculate_ice(item)
            
            # Technical Debt calculation
            debt_score = self._calculate_technical_debt_score(item)
            
            # Composite score with adaptive weights
            weights = self.config["scoring"]["weights"]
            composite_score = (
                weights["wsjf"] * wsjf_score +
                weights["ice"] * ice_score + 
                weights["technicalDebt"] * debt_score
            )
            
            # Apply category boosts
            composite_score = self._apply_category_boosts(composite_score, item)
            
            item.update({
                "scores": {
                    "wsjf": wsjf_score,
                    "ice": ice_score,
                    "technicalDebt": debt_score,
                    "composite": round(composite_score, 2)
                }
            })
        
        # Sort by composite score descending
        items.sort(key=lambda x: x["scores"]["composite"], reverse=True)
        return items
    
    def _calculate_wsjf(self, item: Dict) -> float:
        """Calculate Weighted Shortest Job First score"""
        # User/Business Value (1-10)
        business_value = self._score_business_value(item)
        
        # Time Criticality (1-10)  
        time_criticality = self._score_time_criticality(item)
        
        # Risk Reduction (1-10)
        risk_reduction = self._score_risk_reduction(item)
        
        # Opportunity Enablement (1-10)
        opportunity = self._score_opportunity(item)
        
        cost_of_delay = business_value + time_criticality + risk_reduction + opportunity
        job_size = item.get("effort_estimate", 1)
        
        return cost_of_delay / max(job_size, 0.5)  # Avoid division by zero
    
    def _calculate_ice(self, item: Dict) -> float:
        """Calculate Impact, Confidence, Ease score"""
        impact = self._score_impact(item)      # 1-10
        confidence = self._score_confidence(item)  # 1-10
        ease = self._score_ease(item)         # 1-10
        
        return impact * confidence * ease
    
    def _calculate_technical_debt_score(self, item: Dict) -> float:
        """Calculate technical debt reduction value"""
        if item.get("category") == "technical_debt":
            debt_impact = self._score_debt_impact(item)
            debt_interest = self._score_debt_interest(item)
            hotspot_multiplier = self._get_hotspot_multiplier(item.get("file", ""))
            
            return (debt_impact + debt_interest) * hotspot_multiplier
        
        return 0.0
    
    def _apply_category_boosts(self, score: float, item: Dict) -> float:
        """Apply category-specific score boosts"""
        category = item.get("category", "")
        
        if category == "security":
            return score * self.config["scoring"]["thresholds"]["securityBoost"]
        elif "compliance" in category:
            return score * self.config["scoring"]["thresholds"]["complianceBoost"]
        elif category == "performance":
            return score * self.config["scoring"]["thresholds"].get("performanceBoost", 1.0)
        elif category == "documentation":
            return score * self.config["scoring"]["thresholds"]["documentationPenalty"]
        
        return score
    
    # Helper scoring methods (simplified for MVP)
    def _score_business_value(self, item: Dict) -> int:
        category_scores = {
            "security": 9, "performance": 7, "reliability": 6,
            "technical_debt": 5, "documentation": 3
        }
        return category_scores.get(item.get("category"), 5)
    
    def _score_time_criticality(self, item: Dict) -> int:
        if item.get("category") == "security":
            return 8
        return 5
    
    def _score_risk_reduction(self, item: Dict) -> int:
        if item.get("category") in ["security", "reliability"]:
            return 7
        return 4
    
    def _score_opportunity(self, item: Dict) -> int:
        if item.get("category") == "performance":
            return 6
        return 3
    
    def _score_impact(self, item: Dict) -> int:
        return self._score_business_value(item)
    
    def _score_confidence(self, item: Dict) -> int:
        # Higher confidence for well-defined technical issues
        if item.get("source") == "staticAnalysis":
            return 8
        return 6
    
    def _score_ease(self, item: Dict) -> int:
        effort = item.get("effort_estimate", 3)
        return max(1, 11 - effort)  # Inverse relationship
    
    def _score_debt_impact(self, item: Dict) -> float:
        return 5.0  # Simplified
    
    def _score_debt_interest(self, item: Dict) -> float:
        return 3.0  # Simplified
    
    def _get_hotspot_multiplier(self, file_path: str) -> float:
        """Get code churn/complexity multiplier"""
        if any(pattern in file_path for pattern in ["core", "main", "security", "crypto"]):
            return 2.0
        return 1.0
    
    # Helper methods for categorization and prioritization
    def _estimate_effort(self, content: str, pattern: str) -> int:
        """Estimate effort based on TODO/FIXME content"""
        if any(word in content.lower() for word in ["simple", "easy", "quick"]):
            return 1
        elif any(word in content.lower() for word in ["refactor", "redesign", "major"]):
            return 5
        return 3
    
    def _map_ruff_priority(self, code: str) -> str:
        """Map ruff error codes to priorities"""
        if code.startswith("E9") or code.startswith("F"):
            return "high"
        elif code.startswith("W"):
            return "low"
        return "medium"
    
    def _map_security_priority(self, severity: str) -> str:
        """Map security severity to priority"""
        severity_map = {
            "HIGH": "high",
            "MEDIUM": "medium", 
            "LOW": "low"
        }
        return severity_map.get(severity, "medium")
    
    def _estimate_security_effort(self, severity: str) -> int:
        """Estimate effort for security fixes"""
        effort_map = {
            "HIGH": 4,
            "MEDIUM": 2,
            "LOW": 1
        }
        return effort_map.get(severity, 2)
    
    def _categorize_github_issue(self, labels: List[Dict]) -> str:
        """Categorize GitHub issue based on labels"""
        label_names = [label.get("name", "").lower() for label in labels]
        
        if any("security" in name for name in label_names):
            return "security"
        elif any(name in ["performance", "optimization"] for name in label_names):
            return "performance"
        elif any(name in ["bug", "error"] for name in label_names):
            return "reliability"
        elif any(name in ["enhancement", "feature"] for name in label_names):
            return "feature"
        elif any(name in ["documentation", "docs"] for name in label_names):
            return "documentation"
        
        return "general"
    
    def _prioritize_github_issue(self, labels: List[Dict]) -> str:
        """Determine priority from GitHub labels"""
        label_names = [label.get("name", "").lower() for label in labels]
        
        if any(name in ["critical", "urgent", "high"] for name in label_names):
            return "high"
        elif any(name in ["low", "minor"] for name in label_names):
            return "low"
        
        return "medium"
    
    def _estimate_github_effort(self, labels: List[Dict]) -> int:
        """Estimate effort from GitHub labels"""
        label_names = [label.get("name", "").lower() for label in labels]
        
        if any("small" in name for name in label_names):
            return 1
        elif any("large" in name for name in label_names):
            return 8
        elif any("medium" in name for name in label_names):
            return 3
        
        return 5  # Default effort

def main():
    """Main discovery engine execution"""
    engine = ValueDiscoveryEngine()
    
    # Discover all signals
    discovered_items = engine.discover_all_signals()
    
    # Calculate composite scores
    scored_items = engine.calculate_composite_scores(discovered_items)
    
    # Filter by minimum score threshold
    min_score = engine.config["scoring"]["thresholds"]["minScore"]
    qualified_items = [item for item in scored_items if item["scores"]["composite"] >= min_score]
    
    print(f"\n🎯 {len(qualified_items)} items qualify for execution (score >= {min_score})")
    
    # Display top 10 items
    print("\n📋 Top Value Items:")
    print("-" * 80)
    for i, item in enumerate(qualified_items[:10], 1):
        print(f"{i:2d}. [{item['scores']['composite']:5.1f}] {item['title']}")
        print(f"     Category: {item['category']} | Effort: {item['effort_estimate']}h | Source: {item['source']}")
    
    # Save results
    output_file = Path(".terragon/discovered-backlog.json")
    output_file.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(),
        "total_discovered": len(discovered_items),
        "qualified_items": len(qualified_items),
        "items": qualified_items
    }, indent=2))
    
    print(f"\n💾 Results saved to {output_file}")
    
    return qualified_items

if __name__ == "__main__":
    main()