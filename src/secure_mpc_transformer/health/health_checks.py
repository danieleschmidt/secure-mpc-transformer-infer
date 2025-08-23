"""
Health check implementation for production monitoring.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status values."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str = ""
    response_time_ms: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class HealthChecker:
    """
    Comprehensive health checker for production monitoring.
    """
    
    def __init__(self):
        self.checks: List[callable] = []
        
    def add_check(self, check_func: callable) -> None:
        """Add a health check function."""
        self.checks.append(check_func)
    
    async def check_basic_health(self) -> HealthCheckResult:
        """Basic application health check."""
        start_time = time.time()
        
        try:
            # Basic system checks
            import os
            import sys
            
            # Check Python version
            if sys.version_info < (3, 10):
                return HealthCheckResult(
                    name="basic_health",
                    status=HealthStatus.UNHEALTHY,
                    message="Python version too old",
                    response_time_ms=(time.time() - start_time) * 1000
                )
            
            # Check disk space
            disk_usage = os.statvfs('/')
            free_space_percent = (disk_usage.f_bavail * disk_usage.f_frsize) / (disk_usage.f_blocks * disk_usage.f_frsize) * 100
            
            if free_space_percent < 10:
                return HealthCheckResult(
                    name="basic_health",
                    status=HealthStatus.DEGRADED,
                    message=f"Low disk space: {free_space_percent:.1f}% free",
                    response_time_ms=(time.time() - start_time) * 1000
                )
            
            return HealthCheckResult(
                name="basic_health",
                status=HealthStatus.HEALTHY,
                message="All basic checks passed",
                response_time_ms=(time.time() - start_time) * 1000
            )
        
        except Exception as e:
            logger.error(f"Basic health check failed: {e}")
            return HealthCheckResult(
                name="basic_health",
                status=HealthStatus.UNHEALTHY,
                message=f"Health check error: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000
            )
    
    async def check_dependencies(self) -> HealthCheckResult:
        """Check external dependencies."""
        start_time = time.time()
        
        try:
            # Try to import key dependencies
            dependencies = [
                "json",
                "asyncio", 
                "logging",
                "pathlib",
                "datetime"
            ]
            
            missing_deps = []
            for dep in dependencies:
                try:
                    __import__(dep)
                except ImportError:
                    missing_deps.append(dep)
            
            if missing_deps:
                return HealthCheckResult(
                    name="dependencies",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Missing dependencies: {', '.join(missing_deps)}",
                    response_time_ms=(time.time() - start_time) * 1000
                )
            
            return HealthCheckResult(
                name="dependencies",
                status=HealthStatus.HEALTHY,
                message="All dependencies available",
                response_time_ms=(time.time() - start_time) * 1000
            )
        
        except Exception as e:
            logger.error(f"Dependency check failed: {e}")
            return HealthCheckResult(
                name="dependencies",
                status=HealthStatus.UNHEALTHY,
                message=f"Dependency check error: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000
            )
    
    async def get_comprehensive_health(self) -> Dict[str, any]:
        """Get comprehensive health status."""
        start_time = time.time()
        
        # Run all health checks
        health_results = []
        
        basic_health = await self.check_basic_health()
        health_results.append(basic_health)
        
        deps_health = await self.check_dependencies()
        health_results.append(deps_health)
        
        # Run custom checks
        for check_func in self.checks:
            try:
                result = await check_func()
                health_results.append(result)
            except Exception as e:
                logger.error(f"Custom health check failed: {e}")
                health_results.append(HealthCheckResult(
                    name="custom_check",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Custom check error: {str(e)}"
                ))
        
        # Determine overall status
        statuses = [result.status for result in health_results]
        
        if HealthStatus.UNHEALTHY in statuses:
            overall_status = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        total_response_time = (time.time() - start_time) * 1000
        
        return {
            "status": overall_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "response_time_ms": total_response_time,
            "checks": [
                {
                    "name": result.name,
                    "status": result.status.value,
                    "message": result.message,
                    "response_time_ms": result.response_time_ms,
                    "timestamp": result.timestamp.isoformat()
                }
                for result in health_results
            ],
            "summary": {
                "total_checks": len(health_results),
                "healthy_checks": len([r for r in health_results if r.status == HealthStatus.HEALTHY]),
                "degraded_checks": len([r for r in health_results if r.status == HealthStatus.DEGRADED]),
                "unhealthy_checks": len([r for r in health_results if r.status == HealthStatus.UNHEALTHY])
            }
        }


# Global health checker instance
health_checker = HealthChecker()
