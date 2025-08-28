"""
Global Deployment Orchestrator

Intelligent global-first deployment system with multi-region coordination,
compliance management, and adaptive traffic routing for the secure MPC transformer.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict, deque
import hashlib
import threading

logger = logging.getLogger(__name__)


class Region(Enum):
    """Supported global regions."""
    US_EAST_1 = "us-east-1"
    US_WEST_2 = "us-west-2"
    EU_WEST_1 = "eu-west-1"
    EU_CENTRAL_1 = "eu-central-1"
    ASIA_PACIFIC_1 = "ap-southeast-1"
    ASIA_PACIFIC_2 = "ap-northeast-1"
    CANADA_CENTRAL = "ca-central-1"
    AUSTRALIA_EAST = "au-east-1"


class ComplianceFramework(Enum):
    """Compliance frameworks."""
    GDPR = "gdpr"  # General Data Protection Regulation
    CCPA = "ccpa"  # California Consumer Privacy Act
    PDPA = "pdpa"  # Personal Data Protection Act (Singapore)
    LGPD = "lgpd"  # Lei Geral de Proteção de Dados (Brazil)
    PIPEDA = "pipeda"  # Personal Information Protection and Electronic Documents Act (Canada)
    PRIVACY_ACT = "privacy_act"  # Privacy Act (Australia)


class DeploymentStrategy(Enum):
    """Deployment strategies."""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    RECREATE = "recreate"
    A_B_TEST = "a_b_test"


class TrafficRoutingStrategy(Enum):
    """Traffic routing strategies."""
    LATENCY_BASED = "latency_based"
    GEO_PROXIMITY = "geo_proximity"
    LOAD_BASED = "load_based"
    COMPLIANCE_BASED = "compliance_based"
    QUANTUM_OPTIMIZED = "quantum_optimized"


@dataclass
class RegionConfig:
    """Configuration for a deployment region."""
    region: Region
    enabled: bool = True
    capacity_units: int = 10
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)
    data_residency_required: bool = False
    quantum_capabilities: bool = True
    supported_languages: List[str] = field(default_factory=lambda: ["en", "es", "fr", "de", "ja", "zh"])
    deployment_strategy: DeploymentStrategy = DeploymentStrategy.ROLLING
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'region': self.region.value,
            'enabled': self.enabled,
            'capacity_units': self.capacity_units,
            'compliance_frameworks': [f.value for f in self.compliance_frameworks],
            'data_residency_required': self.data_residency_required,
            'quantum_capabilities': self.quantum_capabilities,
            'supported_languages': self.supported_languages,
            'deployment_strategy': self.deployment_strategy.value
        }


@dataclass
class DeploymentTarget:
    """Represents a deployment target."""
    region: Region
    version: str
    config_hash: str
    status: str = "pending"  # pending, deploying, active, failed
    health_score: float = 1.0
    traffic_weight: int = 0  # Percentage of traffic
    deployment_time: float = field(default_factory=time.time)
    last_health_check: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'region': self.region.value,
            'version': self.version,
            'config_hash': self.config_hash,
            'status': self.status,
            'health_score': self.health_score,
            'traffic_weight': self.traffic_weight,
            'deployment_time': self.deployment_time,
            'last_health_check': self.last_health_check
        }


class ComplianceManager:
    """Manages compliance requirements across regions."""
    
    def __init__(self):
        # Define compliance requirements by framework
        self.compliance_rules = {
            ComplianceFramework.GDPR: {
                'data_residency': True,
                'right_to_erasure': True,
                'data_portability': True,
                'consent_management': True,
                'encryption_at_rest': True,
                'encryption_in_transit': True,
                'audit_logging': True,
                'breach_notification': True,
                'dpo_required': True
            },
            ComplianceFramework.CCPA: {
                'data_residency': False,
                'right_to_delete': True,
                'right_to_know': True,
                'opt_out_sale': True,
                'non_discrimination': True,
                'encryption_at_rest': True,
                'audit_logging': True
            },
            ComplianceFramework.PDPA: {
                'data_residency': True,
                'consent_management': True,
                'data_breach_notification': True,
                'dpo_appointment': True,
                'encryption_at_rest': True,
                'access_controls': True
            }
        }
        
        # Region compliance mappings
        self.region_compliance = {
            Region.EU_WEST_1: [ComplianceFramework.GDPR],
            Region.EU_CENTRAL_1: [ComplianceFramework.GDPR],
            Region.US_EAST_1: [ComplianceFramework.CCPA],
            Region.US_WEST_2: [ComplianceFramework.CCPA],
            Region.CANADA_CENTRAL: [ComplianceFramework.PIPEDA],
            Region.ASIA_PACIFIC_1: [ComplianceFramework.PDPA],
            Region.AUSTRALIA_EAST: [ComplianceFramework.PRIVACY_ACT]
        }
    
    def get_compliance_requirements(self, region: Region) -> Dict[str, Any]:
        """Get compliance requirements for a region."""
        frameworks = self.region_compliance.get(region, [])
        
        combined_requirements = {}
        for framework in frameworks:
            rules = self.compliance_rules.get(framework, {})
            combined_requirements.update(rules)
        
        return combined_requirements
    
    def validate_deployment_compliance(
        self, 
        region: Region, 
        config: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate deployment configuration against compliance requirements."""
        requirements = self.get_compliance_requirements(region)
        violations = []
        
        for requirement, required in requirements.items():
            if required and not config.get(requirement, False):
                violations.append(f"Missing {requirement} for {region.value}")
        
        return len(violations) == 0, violations
    
    def get_data_residency_regions(self) -> List[Region]:
        """Get regions that require data residency."""
        residency_regions = []
        
        for region, frameworks in self.region_compliance.items():
            for framework in frameworks:
                rules = self.compliance_rules.get(framework, {})
                if rules.get('data_residency', False):
                    residency_regions.append(region)
                    break
        
        return residency_regions


class TrafficRouter:
    """Intelligent traffic routing system."""
    
    def __init__(self, strategy: TrafficRoutingStrategy = TrafficRoutingStrategy.LATENCY_BASED):
        self.strategy = strategy
        self.region_latencies = {}  # Cache of region latencies
        self.region_loads = defaultdict(float)
        self.geo_mappings = self._initialize_geo_mappings()
        
    def _initialize_geo_mappings(self) -> Dict[str, Region]:
        """Initialize geographical region mappings."""
        return {
            'US': Region.US_EAST_1,
            'CA': Region.CANADA_CENTRAL,
            'GB': Region.EU_WEST_1,
            'DE': Region.EU_CENTRAL_1,
            'FR': Region.EU_WEST_1,
            'IT': Region.EU_WEST_1,
            'ES': Region.EU_WEST_1,
            'SG': Region.ASIA_PACIFIC_1,
            'JP': Region.ASIA_PACIFIC_2,
            'AU': Region.AUSTRALIA_EAST,
            'BR': Region.US_EAST_1  # Closest available
        }
    
    def route_request(
        self, 
        user_location: Optional[str], 
        available_regions: List[Region],
        compliance_requirements: Optional[List[ComplianceFramework]] = None
    ) -> Region:
        """Route request to optimal region."""
        
        if not available_regions:
            raise ValueError("No available regions")
        
        if len(available_regions) == 1:
            return available_regions[0]
        
        if self.strategy == TrafficRoutingStrategy.GEO_PROXIMITY:
            return self._route_by_proximity(user_location, available_regions)
        
        elif self.strategy == TrafficRoutingStrategy.LATENCY_BASED:
            return self._route_by_latency(user_location, available_regions)
        
        elif self.strategy == TrafficRoutingStrategy.LOAD_BASED:
            return self._route_by_load(available_regions)
        
        elif self.strategy == TrafficRoutingStrategy.COMPLIANCE_BASED:
            return self._route_by_compliance(user_location, available_regions, compliance_requirements)
        
        elif self.strategy == TrafficRoutingStrategy.QUANTUM_OPTIMIZED:
            return self._route_quantum_optimized(available_regions)
        
        else:
            return available_regions[0]  # Fallback
    
    def _route_by_proximity(self, user_location: Optional[str], regions: List[Region]) -> Region:
        """Route based on geographical proximity."""
        if not user_location or user_location not in self.geo_mappings:
            return regions[0]
        
        preferred_region = self.geo_mappings[user_location]
        return preferred_region if preferred_region in regions else regions[0]
    
    def _route_by_latency(self, user_location: Optional[str], regions: List[Region]) -> Region:
        """Route based on network latency."""
        # Start with proximity and adjust for known latencies
        base_region = self._route_by_proximity(user_location, regions)
        
        # Check if we have better latency information
        if user_location in self.region_latencies:
            latencies = self.region_latencies[user_location]
            best_region = min(
                regions, 
                key=lambda r: latencies.get(r, float('inf'))
            )
            return best_region
        
        return base_region
    
    def _route_by_load(self, regions: List[Region]) -> Region:
        """Route based on current load."""
        return min(regions, key=lambda r: self.region_loads.get(r, 0))
    
    def _route_by_compliance(
        self, 
        user_location: Optional[str], 
        regions: List[Region],
        requirements: Optional[List[ComplianceFramework]]
    ) -> Region:
        """Route based on compliance requirements."""
        if not requirements:
            return self._route_by_proximity(user_location, regions)
        
        # Filter regions that meet compliance requirements
        compliance_manager = ComplianceManager()
        compliant_regions = []
        
        for region in regions:
            region_frameworks = compliance_manager.region_compliance.get(region, [])
            if any(req in region_frameworks for req in requirements):
                compliant_regions.append(region)
        
        if compliant_regions:
            return self._route_by_proximity(user_location, compliant_regions)
        else:
            # No compliant regions, fallback to proximity
            return self._route_by_proximity(user_location, regions)
    
    def _route_quantum_optimized(self, regions: List[Region]) -> Region:
        """Route using quantum optimization algorithms."""
        # This would implement quantum optimization for routing decisions
        # For now, use a weighted combination of factors
        
        scores = {}
        for region in regions:
            # Combine multiple factors with quantum-inspired weights
            load_score = 1.0 - (self.region_loads.get(region, 0) / 100.0)
            proximity_score = 0.8  # Would calculate actual proximity
            quantum_coherence = 0.9  # Would get from quantum system
            
            # Quantum-weighted score
            scores[region] = (
                load_score * 0.4 + 
                proximity_score * 0.3 + 
                quantum_coherence * 0.3
            )
        
        return max(regions, key=lambda r: scores.get(r, 0))
    
    def update_region_load(self, region: Region, load: float):
        """Update region load information."""
        self.region_loads[region] = load
    
    def update_region_latency(self, user_location: str, region: Region, latency: float):
        """Update latency information for a user location."""
        if user_location not in self.region_latencies:
            self.region_latencies[user_location] = {}
        self.region_latencies[user_location][region] = latency


class GlobalDeploymentOrchestrator:
    """Main global deployment orchestrator."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Core components
        self.compliance_manager = ComplianceManager()
        self.traffic_router = TrafficRouter(
            TrafficRoutingStrategy(self.config.get('routing_strategy', 'latency_based'))
        )
        
        # Region configurations
        self.region_configs: Dict[Region, RegionConfig] = {}
        self._initialize_default_regions()
        
        # Deployment tracking
        self.deployments: Dict[str, List[DeploymentTarget]] = {}  # version -> targets
        self.active_deployments: Dict[Region, DeploymentTarget] = {}
        
        # Health monitoring
        self.region_health: Dict[Region, float] = defaultdict(lambda: 1.0)
        self.health_check_interval = self.config.get('health_check_interval', 60.0)
        
        # Background tasks
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._traffic_optimizer_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info("Global Deployment Orchestrator initialized")
    
    def _initialize_default_regions(self):
        """Initialize default region configurations."""
        
        # North America
        self.region_configs[Region.US_EAST_1] = RegionConfig(
            region=Region.US_EAST_1,
            capacity_units=20,
            compliance_frameworks=[ComplianceFramework.CCPA],
            quantum_capabilities=True
        )
        
        self.region_configs[Region.US_WEST_2] = RegionConfig(
            region=Region.US_WEST_2,
            capacity_units=15,
            compliance_frameworks=[ComplianceFramework.CCPA],
            quantum_capabilities=True
        )
        
        self.region_configs[Region.CANADA_CENTRAL] = RegionConfig(
            region=Region.CANADA_CENTRAL,
            capacity_units=10,
            compliance_frameworks=[ComplianceFramework.PIPEDA],
            quantum_capabilities=True
        )
        
        # Europe
        self.region_configs[Region.EU_WEST_1] = RegionConfig(
            region=Region.EU_WEST_1,
            capacity_units=18,
            compliance_frameworks=[ComplianceFramework.GDPR],
            data_residency_required=True,
            quantum_capabilities=True
        )
        
        self.region_configs[Region.EU_CENTRAL_1] = RegionConfig(
            region=Region.EU_CENTRAL_1,
            capacity_units=12,
            compliance_frameworks=[ComplianceFramework.GDPR],
            data_residency_required=True,
            quantum_capabilities=True
        )
        
        # Asia Pacific
        self.region_configs[Region.ASIA_PACIFIC_1] = RegionConfig(
            region=Region.ASIA_PACIFIC_1,
            capacity_units=15,
            compliance_frameworks=[ComplianceFramework.PDPA],
            data_residency_required=True,
            quantum_capabilities=True
        )
        
        self.region_configs[Region.ASIA_PACIFIC_2] = RegionConfig(
            region=Region.ASIA_PACIFIC_2,
            capacity_units=12,
            compliance_frameworks=[],
            quantum_capabilities=True
        )
        
        self.region_configs[Region.AUSTRALIA_EAST] = RegionConfig(
            region=Region.AUSTRALIA_EAST,
            capacity_units=8,
            compliance_frameworks=[ComplianceFramework.PRIVACY_ACT],
            quantum_capabilities=False  # Limited quantum capabilities
        )
    
    async def start(self):
        """Start the global deployment orchestrator."""
        if self._running:
            return
            
        self._running = True
        
        # Start background tasks
        self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())
        self._traffic_optimizer_task = asyncio.create_task(self._traffic_optimization_loop())
        
        logger.info("Global Deployment Orchestrator started")
    
    async def stop(self):
        """Stop the orchestrator."""
        self._running = False
        
        # Stop background tasks
        for task in [self._health_monitor_task, self._traffic_optimizer_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Global Deployment Orchestrator stopped")
    
    async def deploy_globally(
        self, 
        version: str, 
        config: Dict[str, Any],
        target_regions: Optional[List[Region]] = None
    ) -> bool:
        """Deploy a version globally."""
        
        if target_regions is None:
            target_regions = [r for r, cfg in self.region_configs.items() if cfg.enabled]
        
        config_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()
        
        # Create deployment targets
        targets = []
        for region in target_regions:
            # Validate compliance
            compliant, violations = self.compliance_manager.validate_deployment_compliance(
                region, config
            )
            
            if not compliant:
                logger.warning(f"Skipping {region.value} due to compliance violations: {violations}")
                continue
            
            target = DeploymentTarget(
                region=region,
                version=version,
                config_hash=config_hash
            )
            targets.append(target)
        
        if not targets:
            logger.error("No valid deployment targets found")
            return False
        
        # Store deployment
        self.deployments[version] = targets
        
        # Execute deployments
        deployment_tasks = []
        for target in targets:
            task = asyncio.create_task(self._deploy_to_region(target, config))
            deployment_tasks.append(task)
        
        # Wait for all deployments
        results = await asyncio.gather(*deployment_tasks, return_exceptions=True)
        
        # Check results
        successful_deployments = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                targets[i].status = "failed"
                logger.error(f"Deployment to {targets[i].region.value} failed: {result}")
            elif result:
                targets[i].status = "active"
                successful_deployments += 1
            else:
                targets[i].status = "failed"
        
        success_rate = successful_deployments / len(targets)
        
        if success_rate >= 0.8:  # 80% success threshold
            logger.info(f"Global deployment successful: {successful_deployments}/{len(targets)} regions")
            return True
        else:
            logger.error(f"Global deployment failed: {successful_deployments}/{len(targets)} regions")
            return False
    
    async def _deploy_to_region(self, target: DeploymentTarget, config: Dict[str, Any]) -> bool:
        """Deploy to a specific region."""
        try:
            target.status = "deploying"
            logger.info(f"Deploying {target.version} to {target.region.value}")
            
            # Simulate deployment process
            await asyncio.sleep(5)  # Simulate deployment time
            
            # Update active deployment for region
            self.active_deployments[target.region] = target
            
            logger.info(f"Successfully deployed {target.version} to {target.region.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy to {target.region.value}: {e}")
            return False
    
    async def _health_monitor_loop(self):
        """Monitor health of regional deployments."""
        while self._running:
            try:
                await self._check_regional_health()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _check_regional_health(self):
        """Check health of all regional deployments."""
        for region, deployment in self.active_deployments.items():
            try:
                # Simulate health check
                health_score = await self._simulate_health_check(region)
                
                deployment.health_score = health_score
                deployment.last_health_check = time.time()
                self.region_health[region] = health_score
                
                if health_score < 0.5:
                    logger.warning(f"Low health score for {region.value}: {health_score}")
                
            except Exception as e:
                logger.error(f"Health check failed for {region.value}: {e}")
                self.region_health[region] = 0.0
    
    async def _simulate_health_check(self, region: Region) -> float:
        """Simulate health check for a region."""
        # Simulate some variability in health scores
        import random
        base_score = 0.9
        variability = random.uniform(-0.1, 0.1)
        return max(0.0, min(1.0, base_score + variability))
    
    async def _traffic_optimization_loop(self):
        """Optimize traffic distribution across regions."""
        while self._running:
            try:
                await self._optimize_traffic_distribution()
                await asyncio.sleep(300)  # Optimize every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in traffic optimization: {e}")
                await asyncio.sleep(300)
    
    async def _optimize_traffic_distribution(self):
        """Optimize traffic distribution based on health and load."""
        healthy_regions = [
            region for region, health in self.region_health.items()
            if health > 0.7  # Healthy threshold
        ]
        
        if not healthy_regions:
            return
        
        # Calculate optimal traffic weights
        total_capacity = sum(
            self.region_configs[region].capacity_units 
            for region in healthy_regions
        )
        
        for region in healthy_regions:
            if region in self.active_deployments:
                capacity = self.region_configs[region].capacity_units
                health = self.region_health[region]
                
                # Weight by capacity and health
                optimal_weight = int((capacity / total_capacity) * health * 100)
                self.active_deployments[region].traffic_weight = optimal_weight
    
    def route_request(
        self, 
        user_location: Optional[str] = None,
        compliance_requirements: Optional[List[ComplianceFramework]] = None
    ) -> Optional[Region]:
        """Route a request to the optimal region."""
        
        available_regions = [
            region for region, health in self.region_health.items()
            if health > 0.5  # Available threshold
        ]
        
        if not available_regions:
            return None
        
        return self.traffic_router.route_request(
            user_location, available_regions, compliance_requirements
        )
    
    def get_global_status(self) -> Dict[str, Any]:
        """Get comprehensive global deployment status."""
        
        # Region status
        region_status = {}
        for region, config in self.region_configs.items():
            deployment = self.active_deployments.get(region)
            region_status[region.value] = {
                'config': config.to_dict(),
                'health': self.region_health.get(region, 0.0),
                'deployment': deployment.to_dict() if deployment else None
            }
        
        # Deployment summary
        deployment_summary = {}
        for version, targets in self.deployments.items():
            deployment_summary[version] = {
                'total_targets': len(targets),
                'successful': len([t for t in targets if t.status == 'active']),
                'failed': len([t for t in targets if t.status == 'failed']),
                'pending': len([t for t in targets if t.status == 'pending'])
            }
        
        # Traffic distribution
        traffic_distribution = {}
        for region, deployment in self.active_deployments.items():
            traffic_distribution[region.value] = deployment.traffic_weight
        
        return {
            'orchestrator_running': self._running,
            'total_regions': len(self.region_configs),
            'active_regions': len([r for r, h in self.region_health.items() if h > 0.5]),
            'region_status': region_status,
            'deployment_summary': deployment_summary,
            'traffic_distribution': traffic_distribution,
            'routing_strategy': self.traffic_router.strategy.value,
            'compliance_frameworks_supported': [f.value for f in ComplianceFramework],
            'last_updated': time.time()
        }


# Global instance  
_orchestrator: Optional[GlobalDeploymentOrchestrator] = None


def get_orchestrator() -> GlobalDeploymentOrchestrator:
    """Get the global deployment orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = GlobalDeploymentOrchestrator()
    return _orchestrator