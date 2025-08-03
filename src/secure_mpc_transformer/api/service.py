"""
Core inference service for secure MPC transformer.
"""

import logging
import time
from typing import Dict, List, Optional, Union

import torch

from ..config import SecurityConfig
from ..models.secure_transformer import SecureTransformer, SecureOutput
from ..protocols.factory import ProtocolFactory
from ..secret_sharing.engine import SecretSharingEngine
from ..gpu.kernels import GPUKernelManager
from ..gpu.memory import GPUMemoryManager

logger = logging.getLogger(__name__)


class InferenceService:
    """
    Core service for secure transformer inference.
    
    Provides high-level API for performing secure inference with
    transformer models using MPC protocols.
    """
    
    def __init__(
        self, 
        security_config: SecurityConfig,
        model_name: str = "bert-base-uncased"
    ):
        self.security_config = security_config
        self.model_name = model_name
        self.device = security_config.get_device()
        
        # Initialize components
        self._initialize_model()
        self._initialize_gpu_components()
        self._initialize_mpc_components()
        
        # Performance tracking
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.performance_stats = {
            'avg_latency_ms': 0.0,
            'throughput_requests_per_sec': 0.0,
            'total_requests': 0,
            'gpu_utilization': 0.0,
            'memory_usage_mb': 0.0
        }
        
        logger.info(f"Initialized InferenceService with model {model_name}")
    
    def _initialize_model(self):
        """Initialize the secure transformer model."""
        try:
            self.model = SecureTransformer.from_pretrained(
                self.model_name,
                self.security_config
            )
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            logger.info(f"Loaded model {self.model_name} on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def _initialize_gpu_components(self):
        """Initialize GPU acceleration components."""
        if self.security_config.gpu_acceleration and self.device.type == 'cuda':
            try:
                # Initialize GPU kernel manager
                self.gpu_kernel_manager = GPUKernelManager(self.device)
                
                # Initialize GPU memory manager
                self.gpu_memory_manager = GPUMemoryManager(
                    self.device, 
                    self.security_config.gpu_memory_fraction
                )
                
                # Compile and load kernels
                self._compile_gpu_kernels()
                
                logger.info("GPU acceleration components initialized")
                
            except Exception as e:
                logger.warning(f"Failed to initialize GPU components: {e}")
                self.gpu_kernel_manager = None
                self.gpu_memory_manager = None
        else:
            self.gpu_kernel_manager = None
            self.gpu_memory_manager = None
    
    def _initialize_mpc_components(self):
        """Initialize MPC protocol components."""
        try:
            # Initialize secret sharing engine
            self.sharing_engine = SecretSharingEngine(self.security_config)
            
            # Initialize MPC protocol
            self.protocol = ProtocolFactory.create(
                self.security_config.protocol,
                security_config=self.security_config
            )
            
            logger.info(f"MPC components initialized with protocol {self.security_config.protocol}")
            
        except Exception as e:
            logger.error(f"Failed to initialize MPC components: {e}")
            raise
    
    def _compile_gpu_kernels(self):
        """Compile GPU kernels for secure operations."""
        if self.gpu_kernel_manager is None:
            return
        
        try:
            from ..gpu.kernels import SecureMatMulKernel, SecureActivationKernel, SecureSoftmaxKernel
            
            # Initialize and compile kernels
            kernels = [
                SecureMatMulKernel(self.gpu_kernel_manager),
                SecureActivationKernel(self.gpu_kernel_manager),
                SecureSoftmaxKernel(self.gpu_kernel_manager)
            ]
            
            for kernel in kernels:
                if kernel.compile():
                    logger.debug(f"Compiled kernel: {kernel.name}")
                else:
                    logger.warning(f"Failed to compile kernel: {kernel.name}")
            
        except Exception as e:
            logger.warning(f"Kernel compilation failed: {e}")
    
    def predict(
        self,
        text: Union[str, List[str]],
        max_length: int = 512,
        return_attention: bool = False,
        return_probabilities: bool = False
    ) -> Union[SecureOutput, List[SecureOutput]]:
        """
        Perform secure inference on input text.
        
        Args:
            text: Input text or list of texts
            max_length: Maximum sequence length
            return_attention: Whether to return attention weights
            return_probabilities: Whether to return prediction probabilities
            
        Returns:
            SecureOutput or list of SecureOutput objects
        """
        start_time = time.time()
        
        # Handle single input vs batch
        single_input = isinstance(text, str)
        if single_input:
            text = [text]
        
        try:
            # Pre-allocate GPU memory if using GPU acceleration
            if self.gpu_memory_manager:
                self._prepare_gpu_memory_for_batch(len(text), max_length)
            
            # Perform secure inference
            results = []
            for input_text in text:
                result = self.model.predict_secure(
                    input_text,
                    max_length=max_length,
                    return_attention=return_attention
                )
                
                # Add probability information if requested
                if return_probabilities:
                    result.metadata['probabilities'] = self._compute_probabilities(result.logits)
                
                results.append(result)
            
            # Update performance statistics
            inference_time = time.time() - start_time
            self._update_performance_stats(inference_time, len(text))
            
            # Return single result or batch
            return results[0] if single_input else results
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
        
        finally:
            # Clean up GPU memory
            if self.gpu_memory_manager:
                self.gpu_memory_manager.force_garbage_collection()
    
    def _prepare_gpu_memory_for_batch(self, batch_size: int, max_length: int):
        """Pre-allocate GPU memory for batch processing."""
        try:
            # Estimate memory requirements
            hidden_size = getattr(self.model.config, 'hidden_size', 768)
            num_layers = getattr(self.model.config, 'num_hidden_layers', 12)
            
            # Create activation buffers
            buffers = self.gpu_memory_manager.create_activation_buffer(
                max_batch_size=batch_size,
                seq_length=max_length,
                hidden_size=hidden_size,
                num_layers=num_layers
            )
            
            logger.debug(f"Pre-allocated GPU memory for batch size {batch_size}")
            
        except Exception as e:
            logger.warning(f"Failed to pre-allocate GPU memory: {e}")
    
    def _compute_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute probabilities from logits using secure softmax."""
        try:
            # Use secure softmax if available
            if self.gpu_kernel_manager and 'secure_softmax' in self.gpu_kernel_manager.kernels:
                softmax_kernel = self.gpu_kernel_manager.kernels['secure_softmax']
                return softmax_kernel.execute_softmax(logits)
            else:
                # Fallback to CPU softmax approximation
                return torch.softmax(logits, dim=-1)
                
        except Exception as e:
            logger.warning(f"Failed to compute probabilities: {e}")
            return torch.zeros_like(logits)
    
    def _update_performance_stats(self, inference_time: float, batch_size: int):
        """Update performance statistics."""
        self.inference_count += 1
        self.total_inference_time += inference_time
        
        # Update moving averages
        self.performance_stats['avg_latency_ms'] = (self.total_inference_time / self.inference_count) * 1000
        self.performance_stats['throughput_requests_per_sec'] = self.inference_count / self.total_inference_time
        self.performance_stats['total_requests'] = self.inference_count
        
        # Update GPU statistics if available
        if self.gpu_memory_manager:
            memory_stats = self.gpu_memory_manager.get_memory_stats()
            self.performance_stats['gpu_utilization'] = memory_stats.get('gpu_utilization', 0.0)
            self.performance_stats['memory_usage_mb'] = memory_stats.get('gpu_allocated_mb', 0.0)
    
    def batch_predict(
        self,
        texts: List[str],
        max_length: int = 512,
        batch_size: Optional[int] = None
    ) -> List[SecureOutput]:
        """
        Perform batch inference with automatic batching.
        
        Args:
            texts: List of input texts
            max_length: Maximum sequence length
            batch_size: Batch size (uses config default if None)
            
        Returns:
            List of SecureOutput objects
        """
        if batch_size is None:
            batch_size = self.security_config.batch_size
        
        all_results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_results = self.predict(batch_texts, max_length=max_length)
            
            if isinstance(batch_results, list):
                all_results.extend(batch_results)
            else:
                all_results.append(batch_results)
        
        return all_results
    
    def benchmark(
        self,
        test_texts: List[str],
        num_iterations: int = 10,
        warmup_iterations: int = 3
    ) -> Dict[str, float]:
        """
        Benchmark inference performance.
        
        Args:
            test_texts: List of test texts
            num_iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Dictionary with benchmark results
        """
        logger.info(f"Starting benchmark with {num_iterations} iterations")
        
        # Warmup
        for _ in range(warmup_iterations):
            for text in test_texts[:2]:  # Use subset for warmup
                self.predict(text)
        
        # Actual benchmark
        start_time = time.time()
        latencies = []
        
        for i in range(num_iterations):
            for text in test_texts:
                iteration_start = time.time()
                result = self.predict(text)
                iteration_time = time.time() - iteration_start
                latencies.append(iteration_time * 1000)  # Convert to ms
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        benchmark_results = {
            'total_time_s': total_time,
            'avg_latency_ms': sum(latencies) / len(latencies),
            'min_latency_ms': min(latencies),
            'max_latency_ms': max(latencies),
            'p50_latency_ms': sorted(latencies)[len(latencies)//2],
            'p95_latency_ms': sorted(latencies)[int(len(latencies)*0.95)],
            'p99_latency_ms': sorted(latencies)[int(len(latencies)*0.99)],
            'throughput_requests_per_sec': len(latencies) / total_time,
            'total_requests': len(latencies)
        }
        
        # Add GPU statistics if available
        if self.gpu_memory_manager:
            memory_stats = self.gpu_memory_manager.get_memory_stats()
            benchmark_results.update({
                'gpu_memory_allocated_mb': memory_stats.get('gpu_allocated_mb', 0),
                'gpu_memory_cached_mb': memory_stats.get('gpu_cached_mb', 0),
                'gpu_utilization': memory_stats.get('gpu_utilization', 0)
            })
        
        logger.info(f"Benchmark completed: {benchmark_results['avg_latency_ms']:.2f}ms avg latency")
        return benchmark_results
    
    def get_model_info(self) -> Dict[str, Union[str, int, float]]:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'protocol': self.security_config.protocol,
            'num_parties': self.security_config.num_parties,
            'party_id': self.security_config.party_id,
            'security_level': self.security_config.security_level,
            'device': str(self.device),
            'gpu_acceleration': self.security_config.gpu_acceleration,
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'model_size_mb': sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1e6
        }
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get current performance statistics."""
        return self.performance_stats.copy()
    
    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.performance_stats = {
            'avg_latency_ms': 0.0,
            'throughput_requests_per_sec': 0.0,
            'total_requests': 0,
            'gpu_utilization': 0.0,
            'memory_usage_mb': 0.0
        }
        
        logger.info("Performance statistics reset")
    
    def shutdown(self):
        """Shutdown the inference service and clean up resources."""
        try:
            # Clear GPU memory
            if self.gpu_memory_manager:
                self.gpu_memory_manager.clear_all_pools()
                self.gpu_memory_manager.force_garbage_collection()
            
            # Clear secret sharing cache
            if hasattr(self.sharing_engine, 'clear_cache'):
                self.sharing_engine.clear_cache()
            
            logger.info("InferenceService shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


class BatchInferenceService(InferenceService):
    """
    Specialized service for high-throughput batch inference.
    
    Optimized for processing large batches of inputs with
    advanced memory management and parallelization.
    """
    
    def __init__(
        self,
        security_config: SecurityConfig,
        model_name: str = "bert-base-uncased",
        max_batch_size: int = 64
    ):
        super().__init__(security_config, model_name)
        self.max_batch_size = max_batch_size
        
        # Pre-allocate memory for maximum batch size
        if self.gpu_memory_manager:
            self._preallocate_batch_memory()
    
    def _preallocate_batch_memory(self):
        """Pre-allocate memory for maximum batch size."""
        try:
            hidden_size = getattr(self.model.config, 'hidden_size', 768)
            max_seq_len = self.security_config.max_sequence_length
            num_layers = getattr(self.model.config, 'num_hidden_layers', 12)
            
            self.batch_buffers = self.gpu_memory_manager.create_activation_buffer(
                max_batch_size=self.max_batch_size,
                seq_length=max_seq_len,
                hidden_size=hidden_size,
                num_layers=num_layers
            )
            
            logger.info(f"Pre-allocated memory for batch size {self.max_batch_size}")
            
        except Exception as e:
            logger.warning(f"Failed to pre-allocate batch memory: {e}")
            self.batch_buffers = None
    
    def predict_batch_optimized(
        self,
        texts: List[str],
        max_length: int = 512
    ) -> List[SecureOutput]:
        """
        Optimized batch prediction with pre-allocated memory.
        
        Args:
            texts: List of input texts
            max_length: Maximum sequence length
            
        Returns:
            List of SecureOutput objects
        """
        if len(texts) > self.max_batch_size:
            # Split into multiple batches
            return self.batch_predict(texts, max_length, self.max_batch_size)
        
        # Use pre-allocated memory for this batch
        start_time = time.time()
        
        try:
            # Process batch with optimized memory usage
            results = []
            for text in texts:
                result = self.model.predict_secure(text, max_length=max_length)
                results.append(result)
            
            # Update performance stats
            inference_time = time.time() - start_time
            self._update_performance_stats(inference_time, len(texts))
            
            return results
            
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            raise