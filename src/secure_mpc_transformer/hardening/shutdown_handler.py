"""Graceful shutdown handling and cleanup for production deployment."""

import signal
import time
import asyncio
import threading
import logging
import atexit
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass
from enum import Enum
import sys
import os

logger = logging.getLogger(__name__)


class ShutdownPhase(Enum):
    """Phases of shutdown process."""
    NORMAL = "normal"
    GRACEFUL = "graceful"
    FORCED = "forced"
    EMERGENCY = "emergency"


class ComponentPriority(Enum):
    """Component shutdown priority levels."""
    HIGH = 1      # Critical components (databases, external connections)
    MEDIUM = 2    # Application components
    LOW = 3       # Monitoring, logging, cleanup tasks


@dataclass
class ShutdownComponent:
    """Represents a component that needs graceful shutdown."""
    
    name: str
    shutdown_func: Callable[[], None]
    async_shutdown_func: Optional[Callable[[], None]] = None
    priority: ComponentPriority = ComponentPriority.MEDIUM
    timeout: float = 30.0
    description: str = ""
    
    def __post_init__(self):
        if not self.description:
            self.description = f"Component: {self.name}"


class ShutdownManager:
    """Manages graceful application shutdown."""
    
    def __init__(self):
        self.components: List[ShutdownComponent] = []
        self.shutdown_in_progress = False
        self.shutdown_completed = False
        self.shutdown_start_time: Optional[float] = None
        
        # Signal handlers
        self._original_handlers = {}
        self._shutdown_signals = [signal.SIGTERM, signal.SIGINT]
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Hooks
        self.pre_shutdown_hooks: List[Callable[[], None]] = []
        self.post_shutdown_hooks: List[Callable[[], None]] = []
        
        # Configuration
        self.graceful_timeout = 60.0  # Total time for graceful shutdown
        self.force_timeout = 10.0     # Additional time before forced shutdown
        
        # Install signal handlers and exit handlers
        self._install_signal_handlers()
        self._install_exit_handlers()
        
        logger.info("Shutdown manager initialized")
    
    def register_component(self, component: ShutdownComponent):
        """Register a component for graceful shutdown."""
        with self._lock:
            self.components.append(component)
            # Sort by priority (highest priority first)
            self.components.sort(key=lambda c: c.priority.value)
        
        logger.debug(f"Registered shutdown component: {component.name}")
    
    def register_simple(self, name: str, shutdown_func: Callable,
                       priority: ComponentPriority = ComponentPriority.MEDIUM,
                       timeout: float = 30.0, description: str = ""):
        """Register a simple shutdown function."""
        component = ShutdownComponent(
            name=name,
            shutdown_func=shutdown_func,
            priority=priority,
            timeout=timeout,
            description=description or f"Component: {name}"
        )
        self.register_component(component)
    
    def register_async(self, name: str, async_shutdown_func: Callable,
                      priority: ComponentPriority = ComponentPriority.MEDIUM,
                      timeout: float = 30.0, description: str = ""):
        """Register an async shutdown function."""
        component = ShutdownComponent(
            name=name,
            shutdown_func=lambda: None,  # Dummy sync function
            async_shutdown_func=async_shutdown_func,
            priority=priority,
            timeout=timeout,
            description=description or f"Async component: {name}"
        )
        self.register_component(component)
    
    def add_pre_shutdown_hook(self, hook: Callable[[], None]):
        """Add a hook to run before shutdown starts."""
        self.pre_shutdown_hooks.append(hook)
    
    def add_post_shutdown_hook(self, hook: Callable[[], None]):
        """Add a hook to run after shutdown completes."""
        self.post_shutdown_hooks.append(hook)
    
    def _install_signal_handlers(self):
        """Install signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            signal_name = signal.Signals(signum).name
            logger.info(f"Received signal {signal_name} ({signum}), initiating graceful shutdown")
            self.shutdown()
        
        for sig in self._shutdown_signals:
            try:
                self._original_handlers[sig] = signal.signal(sig, signal_handler)
            except (OSError, ValueError) as e:
                # Some signals might not be available on all platforms
                logger.warning(f"Cannot install handler for signal {sig}: {e}")
    
    def _install_exit_handlers(self):
        """Install exit handlers for cleanup."""
        def exit_handler():
            if not self.shutdown_completed:
                logger.warning("Application exiting without graceful shutdown")
                self._emergency_shutdown()
        
        atexit.register(exit_handler)
    
    def shutdown(self, phase: ShutdownPhase = ShutdownPhase.GRACEFUL):
        """Initiate application shutdown."""
        with self._lock:
            if self.shutdown_in_progress:
                logger.info("Shutdown already in progress")
                return
            
            self.shutdown_in_progress = True
            self.shutdown_start_time = time.time()
        
        logger.info(f"Starting {phase.value} shutdown")
        
        try:
            # Run pre-shutdown hooks
            self._run_pre_shutdown_hooks()
            
            if phase == ShutdownPhase.GRACEFUL:
                self._graceful_shutdown()
            elif phase == ShutdownPhase.FORCED:
                self._forced_shutdown()
            else:
                self._emergency_shutdown()
        
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            self._emergency_shutdown()
        
        finally:
            self._finalize_shutdown()
    
    def _graceful_shutdown(self):
        """Perform graceful shutdown of all components."""
        logger.info("Performing graceful shutdown")
        
        start_time = time.time()
        failed_components = []
        
        # Shutdown components in priority order
        for component in self.components:
            if time.time() - start_time > self.graceful_timeout:
                logger.warning(f"Graceful shutdown timeout exceeded, switching to forced shutdown")
                self._forced_shutdown()
                return
            
            try:
                logger.info(f"Shutting down component: {component.name}")
                
                if component.async_shutdown_func:
                    # Handle async shutdown
                    asyncio.run(self._shutdown_component_async(component))
                else:
                    # Handle sync shutdown
                    self._shutdown_component_sync(component)
                
                logger.info(f"Successfully shut down component: {component.name}")
                
            except Exception as e:
                logger.error(f"Failed to shutdown component {component.name}: {e}")
                failed_components.append(component.name)
        
        if failed_components:
            logger.warning(f"Failed to shutdown components: {failed_components}")
        else:
            logger.info("All components shut down successfully")
    
    def _shutdown_component_sync(self, component: ShutdownComponent):
        """Shutdown a synchronous component with timeout."""
        def target():
            try:
                component.shutdown_func()
            except Exception as e:
                logger.error(f"Error in shutdown function for {component.name}: {e}")
                raise
        
        # Use threading for timeout support
        shutdown_thread = threading.Thread(target=target)
        shutdown_thread.daemon = True
        shutdown_thread.start()
        shutdown_thread.join(timeout=component.timeout)
        
        if shutdown_thread.is_alive():
            logger.warning(f"Component {component.name} shutdown timed out after {component.timeout}s")
            raise TimeoutError(f"Component {component.name} shutdown timeout")
    
    async def _shutdown_component_async(self, component: ShutdownComponent):
        """Shutdown an asynchronous component with timeout."""
        try:
            await asyncio.wait_for(
                component.async_shutdown_func(),
                timeout=component.timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Async component {component.name} shutdown timed out after {component.timeout}s")
            raise TimeoutError(f"Async component {component.name} shutdown timeout")
    
    def _forced_shutdown(self):
        """Perform forced shutdown (faster, less graceful)."""
        logger.warning("Performing forced shutdown")
        
        for component in self.components:
            try:
                logger.info(f"Force shutting down component: {component.name}")
                
                # Give each component a shorter timeout for forced shutdown
                timeout = min(component.timeout, self.force_timeout)
                
                if component.async_shutdown_func:
                    # Try to run async shutdown with short timeout
                    try:
                        asyncio.run(asyncio.wait_for(
                            component.async_shutdown_func(),
                            timeout=timeout
                        ))
                    except asyncio.TimeoutError:
                        logger.warning(f"Forced shutdown timeout for {component.name}")
                else:
                    # Run sync shutdown with timeout
                    shutdown_thread = threading.Thread(target=component.shutdown_func)
                    shutdown_thread.daemon = True
                    shutdown_thread.start()
                    shutdown_thread.join(timeout=timeout)
                    
                    if shutdown_thread.is_alive():
                        logger.warning(f"Forced shutdown timeout for {component.name}")
                
            except Exception as e:
                logger.error(f"Error during forced shutdown of {component.name}: {e}")
    
    def _emergency_shutdown(self):
        """Emergency shutdown - minimal cleanup."""
        logger.critical("Performing emergency shutdown")
        
        # Try to run critical cleanup only
        critical_components = [c for c in self.components if c.priority == ComponentPriority.HIGH]
        
        for component in critical_components:
            try:
                logger.critical(f"Emergency shutdown: {component.name}")
                component.shutdown_func()
            except Exception as e:
                logger.critical(f"Emergency shutdown failed for {component.name}: {e}")
    
    def _run_pre_shutdown_hooks(self):
        """Run pre-shutdown hooks."""
        logger.debug("Running pre-shutdown hooks")
        
        for hook in self.pre_shutdown_hooks:
            try:
                hook()
            except Exception as e:
                logger.error(f"Pre-shutdown hook failed: {e}")
    
    def _run_post_shutdown_hooks(self):
        """Run post-shutdown hooks."""
        logger.debug("Running post-shutdown hooks")
        
        for hook in self.post_shutdown_hooks:
            try:
                hook()
            except Exception as e:
                logger.error(f"Post-shutdown hook failed: {e}")
    
    def _finalize_shutdown(self):
        """Finalize shutdown process."""
        try:
            # Run post-shutdown hooks
            self._run_post_shutdown_hooks()
            
            # Calculate shutdown duration
            if self.shutdown_start_time:
                shutdown_duration = time.time() - self.shutdown_start_time
                logger.info(f"Shutdown completed in {shutdown_duration:.2f} seconds")
            
            # Mark shutdown as completed
            with self._lock:
                self.shutdown_completed = True
            
            # Restore original signal handlers
            for sig, handler in self._original_handlers.items():
                try:
                    signal.signal(sig, handler)
                except (OSError, ValueError):
                    pass
            
            logger.info("Shutdown process completed successfully")
            
        except Exception as e:
            logger.error(f"Error during shutdown finalization: {e}")
        
        finally:
            # Exit the application
            sys.exit(0)
    
    def is_shutdown_in_progress(self) -> bool:
        """Check if shutdown is in progress."""
        return self.shutdown_in_progress
    
    def get_shutdown_status(self) -> Dict[str, Any]:
        """Get current shutdown status."""
        with self._lock:
            status = {
                'shutdown_in_progress': self.shutdown_in_progress,
                'shutdown_completed': self.shutdown_completed,
                'components_registered': len(self.components),
                'graceful_timeout': self.graceful_timeout,
                'force_timeout': self.force_timeout
            }
            
            if self.shutdown_start_time:
                status['shutdown_duration'] = time.time() - self.shutdown_start_time
            
            return status


# Global shutdown manager instance
shutdown_manager = ShutdownManager()


def register_shutdown_component(name: str, shutdown_func: Callable,
                               priority: ComponentPriority = ComponentPriority.MEDIUM,
                               timeout: float = 30.0, description: str = ""):
    """Convenience function to register a shutdown component."""
    shutdown_manager.register_simple(name, shutdown_func, priority, timeout, description)


def register_async_shutdown_component(name: str, async_shutdown_func: Callable,
                                     priority: ComponentPriority = ComponentPriority.MEDIUM,
                                     timeout: float = 30.0, description: str = ""):
    """Convenience function to register an async shutdown component."""
    shutdown_manager.register_async(name, async_shutdown_func, priority, timeout, description)


def shutdown_handler(priority: ComponentPriority = ComponentPriority.MEDIUM,
                    timeout: float = 30.0, description: str = ""):
    """Decorator to register a function for graceful shutdown."""
    def decorator(func):
        component_name = func.__name__
        shutdown_manager.register_simple(component_name, func, priority, timeout, description)
        return func
    
    return decorator


def async_shutdown_handler(priority: ComponentPriority = ComponentPriority.MEDIUM,
                          timeout: float = 30.0, description: str = ""):
    """Decorator to register an async function for graceful shutdown."""
    def decorator(func):
        component_name = func.__name__
        shutdown_manager.register_async(component_name, func, priority, timeout, description)
        return func
    
    return decorator


# Example usage and built-in shutdown handlers

@shutdown_handler(priority=ComponentPriority.HIGH, timeout=10.0, 
                 description="Close database connections")
def shutdown_database_connections():
    """Shutdown database connections."""
    logger.info("Closing database connections")
    # This would close actual database connections
    time.sleep(0.5)  # Simulate cleanup time


@shutdown_handler(priority=ComponentPriority.MEDIUM, timeout=15.0,
                 description="Stop background tasks")
def shutdown_background_tasks():
    """Stop background tasks and workers."""
    logger.info("Stopping background tasks")
    # This would stop actual background tasks
    time.sleep(1.0)  # Simulate cleanup time


@shutdown_handler(priority=ComponentPriority.LOW, timeout=5.0,
                 description="Flush logs and cleanup temp files")
def shutdown_cleanup():
    """Final cleanup tasks."""
    logger.info("Performing final cleanup")
    # This would do final cleanup
    time.sleep(0.2)  # Simulate cleanup time


class HealthyShutdownMiddleware:
    """Middleware to handle graceful shutdown for web applications."""
    
    def __init__(self, app):
        self.app = app
        self.shutdown_manager = shutdown_manager
    
    def __call__(self, environ, start_response):
        # Check if shutdown is in progress
        if self.shutdown_manager.is_shutdown_in_progress():
            # Return 503 Service Unavailable
            status = '503 Service Unavailable'
            headers = [
                ('Content-Type', 'application/json'),
                ('Connection', 'close')
            ]
            
            response_body = json.dumps({
                'error': 'Service shutting down',
                'message': 'Server is performing graceful shutdown'
            }).encode()
            
            start_response(status, headers)
            return [response_body]
        
        # Normal request processing
        return self.app(environ, start_response)


# Additional utility functions

def wait_for_shutdown(timeout: Optional[float] = None):
    """Wait for shutdown to complete."""
    start_time = time.time()
    
    while not shutdown_manager.shutdown_completed:
        if timeout and (time.time() - start_time) > timeout:
            logger.warning(f"Shutdown wait timed out after {timeout} seconds")
            break
        
        time.sleep(0.1)


def trigger_graceful_shutdown():
    """Manually trigger graceful shutdown."""
    shutdown_manager.shutdown(ShutdownPhase.GRACEFUL)


def trigger_forced_shutdown():
    """Manually trigger forced shutdown."""
    shutdown_manager.shutdown(ShutdownPhase.FORCED)


# Import json for middleware
import json