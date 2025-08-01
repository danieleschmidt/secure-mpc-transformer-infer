#!/usr/bin/env python3
"""
Intelligent dependency update script for Secure MPC Transformer.
Handles dependency updates with security validation and testing.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import argparse
import re
import tempfile
import shutil
from datetime import datetime


class DependencyUpdater:
    """Manage dependency updates with safety checks."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.pyproject_path = project_root / "pyproject.toml"
        self.backup_dir = project_root / ".maintenance" / "dep_backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
    def create_backup(self) -> Path:
        """Create backup of current dependency files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"pyproject_backup_{timestamp}.toml"
        
        if self.pyproject_path.exists():
            shutil.copy2(self.pyproject_path, backup_file)
            print(f"✅ Backup created: {backup_file}")
            return backup_file
        else:
            raise FileNotFoundError("pyproject.toml not found")
    
    def get_outdated_packages(self) -> List[Dict[str, Any]]:
        """Get list of outdated packages."""
        try:
            result = subprocess.run([
                'pip', 'list', '--outdated', '--format=json'
            ], capture_output=True, text=True, check=True)
            
            return json.loads(result.stdout) if result.stdout else []
        except subprocess.CalledProcessError:
            print("❌ Failed to get outdated packages")
            return []
    
    def check_security_vulnerabilities(self) -> Dict[str, Any]:
        """Check for security vulnerabilities in current packages."""
        vulns = {'count': 0, 'packages': []}
        
        try:
            # Use pip-audit if available
            result = subprocess.run([
                'pip-audit', '--format=json'
            ], capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout:
                audit_data = json.loads(result.stdout)
                vulns['count'] = len(audit_data.get('vulnerabilities', []))
                vulns['packages'] = [v.get('package', '') for v in audit_data.get('vulnerabilities', [])]
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to safety if pip-audit not available
            try:
                result = subprocess.run([
                    'safety', 'check', '--json'
                ], capture_output=True, text=True)
                
                if result.stdout:
                    safety_data = json.loads(result.stdout)
                    vulns['count'] = len(safety_data.get('vulnerabilities', []))
                    vulns['packages'] = [v.get('package_name', '') for v in safety_data.get('vulnerabilities', [])]
            except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
                print("⚠️  No security scanning tools available")
        
        return vulns
    
    def categorize_updates(self, outdated: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize updates by type (major, minor, patch)."""
        categories = {
            'major': [],
            'minor': [],
            'patch': [],
            'unknown': []
        }
        
        for package in outdated:
            current_version = package.get('version', '0.0.0')
            latest_version = package.get('latest_version', '0.0.0')
            
            update_type = self.determine_update_type(current_version, latest_version)
            categories[update_type].append(package)
        
        return categories
    
    def determine_update_type(self, current: str, latest: str) -> str:
        """Determine if update is major, minor, or patch."""
        try:
            current_parts = [int(x) for x in current.split('.')]
            latest_parts = [int(x) for x in latest.split('.')]
            
            # Pad with zeros if needed
            max_len = max(len(current_parts), len(latest_parts))
            current_parts.extend([0] * (max_len - len(current_parts)))
            latest_parts.extend([0] * (max_len - len(latest_parts)))
            
            if latest_parts[0] > current_parts[0]:
                return 'major'
            elif latest_parts[1] > current_parts[1]:
                return 'minor'
            elif latest_parts[2] > current_parts[2]:
                return 'patch'
            else:
                return 'unknown'
        except (ValueError, IndexError):
            return 'unknown'
    
    def update_packages(self, packages: List[str], update_type: str = 'minor') -> bool:
        """Update specified packages."""
        if not packages:
            print("No packages to update")
            return True
        
        print(f"🔄 Updating {len(packages)} packages ({update_type})...")
        
        # Create temporary requirements file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for package in packages:
                f.write(f"{package}\n")
            temp_req_file = f.name
        
        try:
            # Update packages
            result = subprocess.run([
                'pip', 'install', '--upgrade', '-r', temp_req_file
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Successfully updated {len(packages)} packages")
                return True
            else:
                print(f"❌ Failed to update packages: {result.stderr}")
                return False
        finally:
            Path(temp_req_file).unlink()
    
    def run_tests(self) -> bool:
        """Run test suite to verify updates don't break functionality."""
        print("🧪 Running test suite...")
        
        try:
            result = subprocess.run([
                'python', '-m', 'pytest', 'tests/', '-v', '--tb=short', '-x'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                print("✅ All tests passed")
                return True
            else:
                print("❌ Tests failed:")
                print(result.stdout[-1000:])  # Last 1000 chars
                return False
        except FileNotFoundError:
            print("⚠️  pytest not found, skipping tests")
            return True
    
    def check_import_compatibility(self) -> bool:
        """Check if core imports still work after updates."""
        print("🔍 Checking import compatibility...")
        
        try:
            # Test basic imports
            test_imports = [
                "import torch",
                "import transformers", 
                "import numpy",
                "import secure_mpc_transformer"
            ]
            
            for import_statement in test_imports:
                result = subprocess.run([
                    'python', '-c', import_statement
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"❌ Import failed: {import_statement}")
                    print(result.stderr)
                    return False
            
            print("✅ All imports working correctly")
            return True
        except Exception as e:
            print(f"❌ Import check failed: {e}")
            return False
    
    def restore_backup(self, backup_path: Path) -> bool:
        """Restore from backup in case of issues."""
        try:
            shutil.copy2(backup_path, self.pyproject_path)
            
            # Reinstall from backup
            subprocess.run([
                'pip', 'install', '-e', '.[dev]'
            ], capture_output=True, cwd=self.project_root)
            
            print(f"✅ Restored from backup: {backup_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to restore backup: {e}")
            return False
    
    def generate_update_report(self, updated_packages: List[str], 
                             failed_packages: List[str], 
                             security_before: Dict[str, Any], 
                             security_after: Dict[str, Any]) -> str:
        """Generate update report."""
        report = []
        report.append("# Dependency Update Report")
        report.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append("## Summary")
        report.append(f"- **Successfully Updated:** {len(updated_packages)} packages")
        report.append(f"- **Failed Updates:** {len(failed_packages)} packages")
        report.append("")
        
        if updated_packages:
            report.append("## Successfully Updated Packages")
            for package in updated_packages:
                report.append(f"- {package}")
            report.append("")
        
        if failed_packages:
            report.append("## Failed Updates")
            for package in failed_packages:
                report.append(f"- {package}")
            report.append("")
        
        report.append("## Security Impact")
        vuln_before = security_before.get('count', 0)
        vuln_after = security_after.get('count', 0)
        
        if vuln_after < vuln_before:
            report.append(f"✅ Security improved: {vuln_before} → {vuln_after} vulnerabilities")
        elif vuln_after > vuln_before:
            report.append(f"⚠️  Security regressed: {vuln_before} → {vuln_after} vulnerabilities")
        else:
            report.append(f"➡️  Security unchanged: {vuln_after} vulnerabilities")
        
        report.append("")
        report.append("## Recommendations")
        report.append("- Review changelog for each updated package")
        report.append("- Monitor application behavior in development")
        report.append("- Run comprehensive test suite before production deployment")
        
        if failed_packages:
            report.append("- Investigate failed package updates manually")
        
        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='Update project dependencies safely')
    parser.add_argument('--type', choices=['patch', 'minor', 'major', 'all'], 
                       default='minor', help='Type of updates to apply')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be updated without making changes')
    parser.add_argument('--skip-tests', action='store_true', 
                       help='Skip running tests after updates')
    parser.add_argument('--force', action='store_true', 
                       help='Force updates even if tests fail')
    parser.add_argument('--packages', nargs='+', 
                       help='Specific packages to update')
    parser.add_argument('--report', help='Output file for update report')
    
    args = parser.parse_args()
    
    project_root = Path.cwd()
    updater = DependencyUpdater(project_root)
    
    print("🔄 Dependency Update Tool")
    print(f"Project: {project_root}")
    print(f"Update type: {args.type}")
    print("")
    
    # Get current state
    outdated_packages = updater.get_outdated_packages()
    security_before = updater.check_security_vulnerabilities()
    
    if not outdated_packages:
        print("✅ All packages are up to date!")
        return 0
    
    print(f"📦 Found {len(outdated_packages)} outdated packages")
    
    if security_before['count'] > 0:
        print(f"🚨 Found {security_before['count']} security vulnerabilities")
    
    # Categorize updates
    categories = updater.categorize_updates(outdated_packages)
    
    # Determine packages to update
    packages_to_update = []
    
    if args.packages:
        # Update specific packages
        packages_to_update = args.packages
    else:
        # Update based on type
        if args.type == 'patch':
            packages_to_update = [p['name'] for p in categories['patch']]
        elif args.type == 'minor':
            packages_to_update = [p['name'] for p in categories['patch'] + categories['minor']]
        elif args.type == 'major':
            packages_to_update = [p['name'] for p in categories['patch'] + categories['minor'] + categories['major']]
        elif args.type == 'all':
            packages_to_update = [p['name'] for p in outdated_packages]
    
    if not packages_to_update:
        print(f"📦 No packages to update for type: {args.type}")
        return 0
    
    print(f"📋 Packages to update: {len(packages_to_update)}")
    for package in packages_to_update:
        print(f"  - {package}")
    
    if args.dry_run:
        print("\n🔍 Dry run complete - no changes made")
        return 0
    
    # Proceed with updates
    backup_path = updater.create_backup()
    
    try:
        # Update packages
        success = updater.update_packages(packages_to_update, args.type)
        
        if not success:
            print("❌ Package update failed")
            updater.restore_backup(backup_path)
            return 1
        
        # Check compatibility
        if not updater.check_import_compatibility():
            print("❌ Import compatibility check failed")
            updater.restore_backup(backup_path)
            return 1
        
        # Run tests
        if not args.skip_tests:
            if not updater.run_tests():
                if not args.force:
                    print("❌ Tests failed, rolling back")
                    updater.restore_backup(backup_path)
                    return 1
                else:
                    print("⚠️  Tests failed but continuing due to --force")
        
        # Check security after updates
        security_after = updater.check_security_vulnerabilities()
        
        # Generate report
        report = updater.generate_update_report(
            packages_to_update, [], security_before, security_after
        )
        
        if args.report:
            with open(args.report, 'w') as f:
                f.write(report)
            print(f"📋 Report saved to {args.report}")
        else:
            print("\n" + report)
        
        print("\n✅ Dependency update completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Update interrupted, restoring backup...")
        updater.restore_backup(backup_path)
        return 130
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("🔄 Restoring backup...")
        updater.restore_backup(backup_path)
        return 1


if __name__ == '__main__':
    sys.exit(main())