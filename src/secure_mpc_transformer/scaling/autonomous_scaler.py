"""
Autonomous Scaling Manager - Generation 3 Implementation

Advanced auto-scaling, load balancing, and resource optimization
for autonomous SDLC execution with quantum-inspired algorithms.
"""

import asyncio
import logging
import time
import psutil
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import statistics
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Scaling direction indicators"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"


class ResourceType(Enum):
    """Types of resources to monitor and scale"""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    NETWORK = "network"
    STORAGE = "storage"
    WORKERS = "workers"


@dataclass
class ResourceMetrics:
    """Resource utilization metrics"""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_available_gb: float = 0.0
    disk_io_percent: float = 0.0
    network_io_mbps: float = 0.0
    gpu_utilization: Optional[float] = None
    active_workers: int = 0
    queue_length: int = 0
    response_time_ms: float = 0.0
    throughput_rps: float = 0.0


@dataclass
class ScalingConfig:
    """Configuration for autonomous scaling"""
    min_workers: int = 2
    max_workers: int = 20
    target_cpu_percent: float = 70.0
    target_memory_percent: float = 80.0
    scale_up_threshold: float = 85.0
    scale_down_threshold: float = 50.0
    cooldown_period: float = 300.0  # 5 minutes
    monitoring_interval: float = 30.0  # 30 seconds
    prediction_window: int = 10  # samples for trend analysis
    enable_predictive_scaling: bool = True
    enable_quantum_optimization: bool = True


@dataclass
class ScalingEvent:
    """Record of a scaling event"""
    timestamp: float
    direction: ScalingDirection
    resource_type: ResourceType
    before_value: int
    after_value: int
    trigger_metric: str
    trigger_value: float
    success: bool
    duration_ms: float


class AutonomousScaler:
    """
    Advanced autonomous scaling manager with predictive algorithms.
    
    Implements quantum-inspired optimization for resource allocation
    and adaptive scaling based on performance metrics.
    """
    
    def __init__(self, config: Optional[ScalingConfig] = None):
        self.config = config or ScalingConfig()
        self.current_workers = self.config.min_workers
        self.metrics_history: List[ResourceMetrics] = []
        self.scaling_events: List[ScalingEvent] = []
        self.last_scaling_time = 0.0
        
        # Thread and process pools
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.process_pool: Optional[ProcessPoolExecutor] = None
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.performance_metrics = {
            "scaling_decisions": 0,
            "successful_scale_ups": 0,
            "successful_scale_downs": 0,
            "failed_scalings": 0,
            "avg_response_time": 0.0,
            "peak_throughput": 0.0
        }
        
        # Quantum state for optimization
        self.quantum_state = self._initialize_quantum_state()
        
        logger.info(f"AutonomousScaler initialized with {self.current_workers} workers")
    
    def _initialize_quantum_state(self) -> Dict[str, Any]:
        """Initialize quantum state for scaling optimization"""
        import numpy as np
        
        return {
            "superposition_weights": np.random.random(4) + 1j * np.random.random(4),
            "entanglement_matrix": np.random.random((4, 4)),
            "coherence_factor": 1.0,
            "optimization_history": []
        }
    
    async def start_monitoring(self) -> None:
        """Start autonomous monitoring and scaling"""
        if self.is_monitoring:
            logger.warning("Monitoring already started")
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # Initialize worker pools
        await self._initialize_worker_pools()
        
        logger.info("Autonomous scaling monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop monitoring and cleanup resources"""
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        await self._cleanup_worker_pools()
        
        logger.info("Autonomous scaling monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop for autonomous scaling"""
        while self.is_monitoring:
            try:
                # Collect metrics
                metrics = await self.collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only recent history
                if len(self.metrics_history) > self.config.prediction_window * 2:
                    self.metrics_history = self.metrics_history[-self.config.prediction_window:]
                
                # Make scaling decision
                decision = await self.make_scaling_decision(metrics)
                
                if decision != ScalingDirection.MAINTAIN:
                    await self.execute_scaling_decision(decision, metrics)
                
                # Update quantum state
                if self.config.enable_quantum_optimization:
                    self._update_quantum_state(metrics, decision)
                
                await asyncio.sleep(self.config.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.config.monitoring_interval)
    
    async def collect_system_metrics(self) -> ResourceMetrics:
        """Collect comprehensive system metrics"""
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_gb = memory.available / (1024**3)
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_io_percent = 0.0  # Simplified for demo
        
        # Network I/O
        network_io = psutil.net_io_counters()
        network_io_mbps = 0.0  # Simplified for demo
        
        # GPU metrics (if available)
        gpu_utilization = await self._get_gpu_utilization()
        
        # Worker pool metrics
        active_workers = self.current_workers
        queue_length = await self._get_queue_length()
        
        # Performance metrics
        response_time_ms = await self._measure_response_time()
        throughput_rps = await self._measure_throughput()
        
        return ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_available_gb=memory_available_gb,
            disk_io_percent=disk_io_percent,
            network_io_mbps=network_io_mbps,
            gpu_utilization=gpu_utilization,
            active_workers=active_workers,
            queue_length=queue_length,
            response_time_ms=response_time_ms,
            throughput_rps=throughput_rps
        )
    
    async def make_scaling_decision(self, current_metrics: ResourceMetrics) -> ScalingDirection:
        """Make intelligent scaling decision based on metrics and predictions"""
        
        # Check cooldown period
        if time.time() - self.last_scaling_time < self.config.cooldown_period:
            return ScalingDirection.MAINTAIN
        
        # Current resource utilization analysis
        scale_up_indicators = 0
        scale_down_indicators = 0
        
        # CPU analysis
        if current_metrics.cpu_percent > self.config.scale_up_threshold:
            scale_up_indicators += 1
        elif current_metrics.cpu_percent < self.config.scale_down_threshold:
            scale_down_indicators += 1
        
        # Memory analysis
        if current_metrics.memory_percent > self.config.scale_up_threshold:
            scale_up_indicators += 1
        elif current_metrics.memory_percent < self.config.scale_down_threshold:
            scale_down_indicators += 1
        
        # Queue length analysis
        if current_metrics.queue_length > self.current_workers * 2:
            scale_up_indicators += 1
        elif current_metrics.queue_length == 0 and self.current_workers > self.config.min_workers:
            scale_down_indicators += 1
        
        # Response time analysis
        if current_metrics.response_time_ms > 1000:  # > 1 second
            scale_up_indicators += 1
        elif current_metrics.response_time_ms < 100:  # < 100ms
            scale_down_indicators += 1
        
        # Predictive analysis
        if self.config.enable_predictive_scaling and len(self.metrics_history) >= 3:
            trend_decision = self._analyze_trends()
            if trend_decision == ScalingDirection.SCALE_UP:
                scale_up_indicators += 2  # Weight predictions higher
            elif trend_decision == ScalingDirection.SCALE_DOWN:
                scale_down_indicators += 2
        
        # Quantum optimization decision
        if self.config.enable_quantum_optimization:
            quantum_decision = self._quantum_scaling_decision(current_metrics)
            if quantum_decision == ScalingDirection.SCALE_UP:
                scale_up_indicators += 1
            elif quantum_decision == ScalingDirection.SCALE_DOWN:
                scale_down_indicators += 1
        
        # Make final decision
        if scale_up_indicators > scale_down_indicators and self.current_workers < self.config.max_workers:
            decision = ScalingDirection.SCALE_UP
        elif scale_down_indicators > scale_up_indicators and self.current_workers > self.config.min_workers:
            decision = ScalingDirection.SCALE_DOWN
        else:
            decision = ScalingDirection.MAINTAIN
        
        logger.debug(f"Scaling decision: {decision.value} (up:{scale_up_indicators}, down:{scale_down_indicators})")
        return decision
    
    def _analyze_trends(self) -> ScalingDirection:
        """Analyze metric trends for predictive scaling"""
        
        if len(self.metrics_history) < 3:
            return ScalingDirection.MAINTAIN
        
        # Analyze CPU trend
        recent_cpu = [m.cpu_percent for m in self.metrics_history[-3:]]
        cpu_trend = statistics.linear_regression(range(len(recent_cpu)), recent_cpu)[0]
        
        # Analyze memory trend
        recent_memory = [m.memory_percent for m in self.metrics_history[-3:]]
        memory_trend = statistics.linear_regression(range(len(recent_memory)), recent_memory)[0]
        
        # Analyze response time trend
        recent_response = [m.response_time_ms for m in self.metrics_history[-3:]]
        response_trend = statistics.linear_regression(range(len(recent_response)), recent_response)[0]
        
        # Decision based on trends
        if cpu_trend > 5 or memory_trend > 5 or response_trend > 50:
            return ScalingDirection.SCALE_UP
        elif cpu_trend < -5 and memory_trend < -5 and response_trend < -20:
            return ScalingDirection.SCALE_DOWN
        else:
            return ScalingDirection.MAINTAIN
    
    def _quantum_scaling_decision(self, metrics: ResourceMetrics) -> ScalingDirection:
        """Use quantum-inspired algorithms for scaling decisions"""
        import numpy as np
        
        # Create quantum state vector representing current system state
        state_vector = np.array([
            metrics.cpu_percent / 100,
            metrics.memory_percent / 100,
            min(metrics.response_time_ms / 1000, 1.0),
            min(metrics.queue_length / 10, 1.0)
        ])
        
        # Apply quantum superposition weights
        weights = self.quantum_state["superposition_weights"]
        weighted_state = state_vector * np.real(weights)
        
        # Apply entanglement effects
        entanglement = self.quantum_state["entanglement_matrix"]
        entangled_state = np.dot(entanglement, weighted_state)
        
        # Quantum measurement - collapse to scaling decision
        measurement = np.sum(np.abs(entangled_state)) * self.quantum_state["coherence_factor"]
        
        if measurement > 1.5:
            return ScalingDirection.SCALE_UP
        elif measurement < 0.5:
            return ScalingDirection.SCALE_DOWN
        else:
            return ScalingDirection.MAINTAIN
    
    def _update_quantum_state(self, metrics: ResourceMetrics, decision: ScalingDirection) -> None:
        """Update quantum state based on system performance"""
        import numpy as np
        
        # Update coherence based on decision effectiveness
        if decision != ScalingDirection.MAINTAIN:
            # Measure decision effectiveness (simplified)
            effectiveness = 1.0 - (metrics.cpu_percent / 100 - self.config.target_cpu_percent / 100) ** 2
            self.quantum_state["coherence_factor"] = 0.9 * self.quantum_state["coherence_factor"] + 0.1 * effectiveness
        
        # Evolve quantum state
        evolution_factor = 0.95
        noise = np.random.random(4) * 0.1
        self.quantum_state["superposition_weights"] *= evolution_factor
        self.quantum_state["superposition_weights"] += noise
        
        # Normalize
        norm = np.linalg.norm(self.quantum_state["superposition_weights"])
        if norm > 0:
            self.quantum_state["superposition_weights"] /= norm
    
    async def execute_scaling_decision(self, decision: ScalingDirection, 
                                     metrics: ResourceMetrics) -> bool:
        """Execute the scaling decision"""
        
        start_time = time.time()
        old_workers = self.current_workers
        new_workers = old_workers
        success = False
        
        try:
            if decision == ScalingDirection.SCALE_UP:
                new_workers = min(old_workers + 1, self.config.max_workers)
                if new_workers > old_workers:
                    await self._scale_up_workers(new_workers - old_workers)
                    self.current_workers = new_workers
                    self.performance_metrics["successful_scale_ups"] += 1
                    success = True
                    logger.info(f"Scaled up: {old_workers} -> {new_workers} workers")
            
            elif decision == ScalingDirection.SCALE_DOWN:
                new_workers = max(old_workers - 1, self.config.min_workers)
                if new_workers < old_workers:
                    await self._scale_down_workers(old_workers - new_workers)
                    self.current_workers = new_workers
                    self.performance_metrics["successful_scale_downs"] += 1
                    success = True
                    logger.info(f"Scaled down: {old_workers} -> {new_workers} workers")
            
            self.last_scaling_time = time.time()
            
        except Exception as e:
            logger.error(f"Scaling failed: {e}")
            self.performance_metrics["failed_scalings"] += 1
            success = False
        
        # Record scaling event
        duration_ms = (time.time() - start_time) * 1000
        
        event = ScalingEvent(
            timestamp=start_time,
            direction=decision,
            resource_type=ResourceType.WORKERS,
            before_value=old_workers,
            after_value=new_workers,
            trigger_metric="combined",
            trigger_value=metrics.cpu_percent,
            success=success,
            duration_ms=duration_ms
        )
        
        self.scaling_events.append(event)
        self.performance_metrics["scaling_decisions"] += 1
        
        return success
    
    async def _initialize_worker_pools(self) -> None:
        """Initialize thread and process pools"""
        
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.current_workers,
            thread_name_prefix="autonomous_worker"
        )
        
        # Process pool for CPU-intensive tasks
        self.process_pool = ProcessPoolExecutor(
            max_workers=min(self.current_workers, psutil.cpu_count() or 4)
        )
        
        logger.info(f"Initialized worker pools: {self.current_workers} threads")
    
    async def _cleanup_worker_pools(self) -> None:
        """Cleanup worker pools"""
        
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
            self.thread_pool = None
        
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
            self.process_pool = None
        
        logger.info("Worker pools cleaned up")
    
    async def _scale_up_workers(self, additional_workers: int) -> None:
        """Scale up worker pools"""
        
        if self.thread_pool:
            # Create new thread pool with more workers
            old_pool = self.thread_pool
            self.thread_pool = ThreadPoolExecutor(
                max_workers=self.current_workers + additional_workers,
                thread_name_prefix="autonomous_worker"
            )
            
            # Gracefully shutdown old pool
            old_pool.shutdown(wait=False)
    
    async def _scale_down_workers(self, fewer_workers: int) -> None:
        """Scale down worker pools"""
        
        if self.thread_pool:
            # Create new thread pool with fewer workers
            old_pool = self.thread_pool
            self.thread_pool = ThreadPoolExecutor(
                max_workers=max(self.current_workers - fewer_workers, 1),
                thread_name_prefix="autonomous_worker"
            )
            
            # Gracefully shutdown old pool
            old_pool.shutdown(wait=True)
    
    async def _get_gpu_utilization(self) -> Optional[float]:
        """Get GPU utilization if available"""
        try:
            # This would use nvidia-ml-py or similar in production
            # For demo, return mock value
            return None
        except Exception:
            return None
    
    async def _get_queue_length(self) -> int:
        """Get current task queue length"""
        # In production, this would check actual queue length
        # For demo, return mock value
        return 0
    
    async def _measure_response_time(self) -> float:
        """Measure average response time"""
        # In production, this would measure actual response times
        # For demo, return mock value based on CPU load
        cpu_load = psutil.cpu_percent()
        return max(50, cpu_load * 10)  # Mock response time in ms
    
    async def _measure_throughput(self) -> float:
        """Measure system throughput"""
        # In production, this would measure actual throughput
        # For demo, return mock value
        return max(1.0, 100 - psutil.cpu_percent())  # Mock requests per second
    
    async def submit_task(self, func: Callable, *args, **kwargs) -> Any:
        """Submit task to worker pool with auto-scaling"""
        
        if not self.thread_pool:
            await self._initialize_worker_pools()
        
        # Submit to appropriate pool based on task type
        if kwargs.get("cpu_intensive", False):
            if self.process_pool:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(self.process_pool, func, *args)
            else:
                raise RuntimeError("Process pool not available")
        else:
            if self.thread_pool:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(self.thread_pool, func, *args)
            else:
                raise RuntimeError("Thread pool not available")
    
    def get_scaling_metrics(self) -> Dict[str, Any]:
        """Get comprehensive scaling metrics"""
        
        recent_events = [
            e for e in self.scaling_events
            if time.time() - e.timestamp < 3600  # Last hour
        ]
        
        avg_cpu = 0.0
        avg_memory = 0.0
        avg_response_time = 0.0
        
        if self.metrics_history:
            recent_metrics = self.metrics_history[-10:]  # Last 10 samples
            avg_cpu = statistics.mean(m.cpu_percent for m in recent_metrics)
            avg_memory = statistics.mean(m.memory_percent for m in recent_metrics)
            avg_response_time = statistics.mean(m.response_time_ms for m in recent_metrics)
        
        return {
            "current_workers": self.current_workers,
            "min_workers": self.config.min_workers,
            "max_workers": self.config.max_workers,
            "avg_cpu_percent": avg_cpu,
            "avg_memory_percent": avg_memory,
            "avg_response_time_ms": avg_response_time,
            "recent_scaling_events": len(recent_events),
            "performance_metrics": self.performance_metrics.copy(),
            "quantum_coherence": self.quantum_state["coherence_factor"],
            "monitoring_active": self.is_monitoring,
            "last_scaling": time.time() - self.last_scaling_time if self.last_scaling_time else None
        }
    
    def get_scaling_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get scaling event history"""
        
        cutoff_time = time.time() - (hours * 3600)
        
        relevant_events = [
            {
                "timestamp": event.timestamp,
                "direction": event.direction.value,
                "resource_type": event.resource_type.value,
                "before_value": event.before_value,
                "after_value": event.after_value,
                "trigger_metric": event.trigger_metric,
                "trigger_value": event.trigger_value,
                "success": event.success,
                "duration_ms": event.duration_ms
            }
            for event in self.scaling_events
            if event.timestamp > cutoff_time
        ]
        
        return relevant_events
    
    async def force_scale(self, target_workers: int) -> bool:
        """Force scaling to specific number of workers"""
        
        if target_workers < self.config.min_workers or target_workers > self.config.max_workers:
            logger.error(f"Target workers {target_workers} outside bounds [{self.config.min_workers}, {self.config.max_workers}]")
            return False
        
        old_workers = self.current_workers
        
        try:
            if target_workers > old_workers:
                await self._scale_up_workers(target_workers - old_workers)
            elif target_workers < old_workers:
                await self._scale_down_workers(old_workers - target_workers)
            
            self.current_workers = target_workers
            self.last_scaling_time = time.time()
            
            logger.info(f"Force scaled: {old_workers} -> {target_workers} workers")
            return True
            
        except Exception as e:
            logger.error(f"Force scaling failed: {e}")
            return False