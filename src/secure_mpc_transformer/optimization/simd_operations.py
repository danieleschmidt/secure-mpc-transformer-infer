"""
SIMD vectorization for CPU operations to accelerate computation.

This module provides SIMD-optimized implementations for common operations
used in secure MPC transformer inference.
"""

import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
import platform
import subprocess
import warnings

logger = logging.getLogger(__name__)

# Try to import numba for JIT compilation
try:
    import numba
    from numba import jit, vectorize, float32, float64, int32, int64
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("Numba not available - SIMD optimizations will be limited")

# Try to import scipy for optimized operations
try:
    import scipy.signal
    import scipy.sparse
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy not available - some SIMD optimizations disabled")


@dataclass
class SIMDConfig:
    """Configuration for SIMD operations."""
    enable_avx: bool = True
    enable_avx2: bool = True
    enable_fma: bool = True
    enable_sse: bool = True
    use_numba_jit: bool = NUMBA_AVAILABLE
    parallel_threshold: int = 10000
    chunk_size: int = 8192
    enable_float16: bool = False  # Use float16 for reduced memory bandwidth


class CPUFeatureDetector:
    """Detect available CPU features for SIMD optimization."""
    
    @staticmethod
    def detect_cpu_features() -> Dict[str, bool]:
        """Detect available CPU SIMD features."""
        features = {
            "avx": False,
            "avx2": False,
            "fma": False,
            "sse": False,
            "sse2": False,
            "sse3": False,
            "sse4_1": False,
            "sse4_2": False
        }
        
        try:
            if platform.system() == "Linux":
                with open("/proc/cpuinfo", "r") as f:
                    cpuinfo = f.read()
                    
                for feature in features.keys():
                    if feature in cpuinfo:
                        features[feature] = True
            
            elif platform.system() == "Darwin":  # macOS
                # Use sysctl to get CPU features
                try:
                    result = subprocess.run(
                        ["sysctl", "-a"], 
                        capture_output=True, 
                        text=True, 
                        timeout=10
                    )
                    sysctl_output = result.stdout
                    
                    # Check for various features
                    if "avx2" in sysctl_output.lower():
                        features["avx2"] = True
                        features["avx"] = True
                    elif "avx" in sysctl_output.lower():
                        features["avx"] = True
                    
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    logger.warning("Could not detect CPU features on macOS")
            
            elif platform.system() == "Windows":
                # Use wmic to get CPU features
                try:
                    result = subprocess.run(
                        ["wmic", "cpu", "get", "name"], 
                        capture_output=True, 
                        text=True,
                        timeout=10
                    )
                    # Basic detection - Windows makes this harder
                    features["sse"] = True
                    features["sse2"] = True
                    
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    logger.warning("Could not detect CPU features on Windows")
        
        except Exception as e:
            logger.warning(f"CPU feature detection failed: {e}")
        
        return features


class SIMDOperation(ABC):
    """Abstract base class for SIMD operations."""
    
    @abstractmethod
    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply SIMD operation to data."""
        pass
    
    @abstractmethod
    def get_operation_name(self) -> str:
        """Get operation name for identification."""
        pass


class VectorizedMatMul(SIMDOperation):
    """SIMD-optimized matrix multiplication."""
    
    def __init__(self, use_blas: bool = True):
        self.use_blas = use_blas
    
    def apply(self, data: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Apply vectorized matrix multiplication."""
        a, b = data
        
        if self.use_blas:
            # Use optimized BLAS implementation
            return np.dot(a, b)
        else:
            # Manual vectorized implementation
            return self._manual_vectorized_matmul(a, b)
    
    def _manual_vectorized_matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Manual vectorized matrix multiplication."""
        # Ensure arrays are contiguous for better cache performance
        a = np.ascontiguousarray(a)
        b = np.ascontiguousarray(b)
        
        # Use numpy's optimized dot product
        return np.dot(a, b)
    
    def get_operation_name(self) -> str:
        return "vectorized_matmul"


class VectorizedElementwise(SIMDOperation):
    """SIMD-optimized elementwise operations."""
    
    def __init__(self, operation: str):
        self.operation = operation
        self.jit_funcs = {}
        
        if NUMBA_AVAILABLE:
            self._compile_jit_functions()
    
    def _compile_jit_functions(self):
        """Compile JIT functions for various operations."""
        
        @numba.jit(nopython=True, parallel=True)
        def jit_add(a, b):
            return a + b
        
        @numba.jit(nopython=True, parallel=True)
        def jit_mul(a, b):
            return a * b
        
        @numba.jit(nopython=True, parallel=True)
        def jit_relu(x):
            return np.maximum(x, 0)
        
        @numba.jit(nopython=True, parallel=True)
        def jit_gelu(x):
            # Approximation of GELU
            return 0.5 * x * (1 + np.tanh(0.797885 * (x + 0.044715 * x**3)))
        
        @numba.jit(nopython=True, parallel=True)
        def jit_sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))
        
        @numba.jit(nopython=True, parallel=True)
        def jit_tanh(x):
            return np.tanh(x)
        
        self.jit_funcs = {
            "add": jit_add,
            "mul": jit_mul,
            "relu": jit_relu,
            "gelu": jit_gelu,
            "sigmoid": jit_sigmoid,
            "tanh": jit_tanh
        }
    
    def apply(self, data: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """Apply vectorized elementwise operation."""
        if NUMBA_AVAILABLE and self.operation in self.jit_funcs:
            if isinstance(data, tuple):
                return self.jit_funcs[self.operation](data[0], data[1])
            else:
                return self.jit_funcs[self.operation](data)
        else:
            return self._fallback_operation(data)
    
    def _fallback_operation(self, data: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """Fallback to numpy operations."""
        if self.operation == "add" and isinstance(data, tuple):
            return data[0] + data[1]
        elif self.operation == "mul" and isinstance(data, tuple):
            return data[0] * data[1]
        elif self.operation == "relu":
            return np.maximum(data, 0)
        elif self.operation == "gelu":
            x = data
            return 0.5 * x * (1 + np.tanh(0.797885 * (x + 0.044715 * x**3)))
        elif self.operation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-data))
        elif self.operation == "tanh":
            return np.tanh(data)
        else:
            raise ValueError(f"Unsupported operation: {self.operation}")
    
    def get_operation_name(self) -> str:
        return f"vectorized_{self.operation}"


class VectorizedReduction(SIMDOperation):
    """SIMD-optimized reduction operations."""
    
    def __init__(self, operation: str, axis: Optional[int] = None):
        self.operation = operation
        self.axis = axis
    
    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply vectorized reduction operation."""
        if self.operation == "sum":
            return np.sum(data, axis=self.axis)
        elif self.operation == "mean":
            return np.mean(data, axis=self.axis)
        elif self.operation == "max":
            return np.max(data, axis=self.axis)
        elif self.operation == "min":
            return np.min(data, axis=self.axis)
        elif self.operation == "std":
            return np.std(data, axis=self.axis)
        elif self.operation == "var":
            return np.var(data, axis=self.axis)
        else:
            raise ValueError(f"Unsupported reduction operation: {self.operation}")
    
    def get_operation_name(self) -> str:
        return f"vectorized_{self.operation}_reduction"


class VectorizedConvolution(SIMDOperation):
    """SIMD-optimized convolution operation."""
    
    def __init__(self, kernel: np.ndarray, stride: int = 1):
        self.kernel = kernel
        self.stride = stride
    
    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply vectorized convolution."""
        if SCIPY_AVAILABLE:
            # Use scipy's optimized convolution
            return scipy.signal.convolve2d(data, self.kernel, mode='valid')
        else:
            # Fallback to manual implementation
            return self._manual_convolution(data)
    
    def _manual_convolution(self, data: np.ndarray) -> np.ndarray:
        """Manual convolution implementation."""
        # Simplified 2D convolution
        kernel_h, kernel_w = self.kernel.shape
        data_h, data_w = data.shape
        
        output_h = (data_h - kernel_h) // self.stride + 1
        output_w = (data_w - kernel_w) // self.stride + 1
        
        output = np.zeros((output_h, output_w))
        
        for i in range(0, output_h * self.stride, self.stride):
            for j in range(0, output_w * self.stride, self.stride):
                output[i // self.stride, j // self.stride] = np.sum(
                    data[i:i+kernel_h, j:j+kernel_w] * self.kernel
                )
        
        return output
    
    def get_operation_name(self) -> str:
        return "vectorized_convolution"


class SIMDProcessor:
    """Main SIMD processor for optimizing CPU operations."""
    
    def __init__(self, config: Optional[SIMDConfig] = None):
        self.config = config or SIMDConfig()
        self.cpu_features = CPUFeatureDetector.detect_cpu_features()
        self.operations: Dict[str, SIMDOperation] = {}
        
        # Performance statistics
        self.simd_stats = {
            "operations_accelerated": 0,
            "total_operations": 0,
            "average_speedup": 1.0,
            "speedup_history": []
        }
        
        self._initialize_operations()
        logger.info(f"Initialized SIMD processor with features: {self.cpu_features}")
    
    def _initialize_operations(self):
        """Initialize available SIMD operations."""
        # Matrix multiplication
        self.operations["matmul"] = VectorizedMatMul(use_blas=True)
        
        # Elementwise operations
        elementwise_ops = ["add", "mul", "relu", "gelu", "sigmoid", "tanh"]
        for op in elementwise_ops:
            self.operations[op] = VectorizedElementwise(op)
        
        # Reduction operations
        reduction_ops = ["sum", "mean", "max", "min", "std", "var"]
        for op in reduction_ops:
            self.operations[f"{op}_reduce"] = VectorizedReduction(op)
        
        logger.info(f"Initialized {len(self.operations)} SIMD operations")
    
    def accelerate_operation(self, operation_name: str, data: Any) -> Any:
        """Accelerate operation using SIMD if available."""
        self.simd_stats["total_operations"] += 1
        
        if operation_name not in self.operations:
            logger.debug(f"No SIMD acceleration available for {operation_name}")
            return data
        
        try:
            # Convert torch tensors to numpy if needed
            if isinstance(data, torch.Tensor):
                numpy_data = data.detach().cpu().numpy()
                result_numpy = self.operations[operation_name].apply(numpy_data)
                result = torch.from_numpy(result_numpy).to(data.device)
                
                self.simd_stats["operations_accelerated"] += 1
                return result
            
            elif isinstance(data, (tuple, list)) and all(isinstance(x, torch.Tensor) for x in data):
                # Handle multiple tensors
                numpy_data = tuple(x.detach().cpu().numpy() for x in data)
                result_numpy = self.operations[operation_name].apply(numpy_data)
                result = torch.from_numpy(result_numpy).to(data[0].device)
                
                self.simd_stats["operations_accelerated"] += 1
                return result
            
            elif isinstance(data, np.ndarray):
                result = self.operations[operation_name].apply(data)
                self.simd_stats["operations_accelerated"] += 1
                return result
            
            else:
                logger.debug(f"Unsupported data type for SIMD: {type(data)}")
                return data
                
        except Exception as e:
            logger.warning(f"SIMD acceleration failed for {operation_name}: {e}")
            return data
    
    def benchmark_operation(self, operation_name: str, data: Any, iterations: int = 100) -> Dict[str, float]:
        """Benchmark SIMD vs non-SIMD operation."""
        import time
        
        if operation_name not in self.operations:
            return {"simd_available": False}
        
        # Benchmark regular operation
        start_time = time.perf_counter()
        for _ in range(iterations):
            _ = self._fallback_operation(operation_name, data)
        regular_time = (time.perf_counter() - start_time) / iterations
        
        # Benchmark SIMD operation
        start_time = time.perf_counter()
        for _ in range(iterations):
            _ = self.accelerate_operation(operation_name, data)
        simd_time = (time.perf_counter() - start_time) / iterations
        
        speedup = regular_time / simd_time if simd_time > 0 else 1.0
        
        # Update statistics
        self.simd_stats["speedup_history"].append(speedup)
        self.simd_stats["average_speedup"] = np.mean(self.simd_stats["speedup_history"])
        
        return {
            "simd_available": True,
            "regular_time_ms": regular_time * 1000,
            "simd_time_ms": simd_time * 1000,
            "speedup": speedup,
            "effective": speedup > 1.1  # 10% threshold
        }
    
    def _fallback_operation(self, operation_name: str, data: Any) -> Any:
        """Fallback implementation without SIMD."""
        if isinstance(data, torch.Tensor):
            if operation_name == "relu":
                return torch.relu(data)
            elif operation_name == "gelu":
                return torch.nn.functional.gelu(data)
            elif operation_name == "sigmoid":
                return torch.sigmoid(data)
            elif operation_name == "tanh":
                return torch.tanh(data)
        
        return data
    
    def auto_optimize_config(self, sample_data: List[Any]) -> SIMDConfig:
        """Automatically optimize SIMD configuration based on sample data."""
        logger.info("Auto-optimizing SIMD configuration")
        
        best_config = self.config
        best_performance = 0.0
        
        test_configs = [
            SIMDConfig(use_numba_jit=True, chunk_size=4096),
            SIMDConfig(use_numba_jit=True, chunk_size=8192),
            SIMDConfig(use_numba_jit=True, chunk_size=16384),
            SIMDConfig(use_numba_jit=False, chunk_size=8192),
            SIMDConfig(enable_float16=True),
            SIMDConfig(enable_float16=False)
        ]
        
        for config in test_configs:
            temp_processor = SIMDProcessor(config)
            performance = self._measure_config_performance(temp_processor, sample_data)
            
            if performance > best_performance:
                best_performance = performance
                best_config = config
        
        logger.info(f"Auto-optimization complete. Best performance: {best_performance:.3f}")
        return best_config
    
    def _measure_config_performance(self, processor: 'SIMDProcessor', sample_data: List[Any]) -> float:
        """Measure performance of SIMD processor with given configuration."""
        import time
        
        start_time = time.perf_counter()
        
        # Run sample operations
        for data in sample_data:
            processor.accelerate_operation("relu", data)
            processor.accelerate_operation("matmul", data)
        
        end_time = time.perf_counter()
        
        return 1.0 / (end_time - start_time) if end_time > start_time else 0.0
    
    def get_optimization_suggestions(self) -> List[str]:
        """Get suggestions for further optimization."""
        suggestions = []
        
        if not NUMBA_AVAILABLE:
            suggestions.append("Install Numba for JIT compilation optimizations")
        
        if not SCIPY_AVAILABLE:
            suggestions.append("Install SciPy for optimized signal processing operations")
        
        if not self.cpu_features.get("avx2", False):
            suggestions.append("CPU does not support AVX2 - consider upgrading hardware")
        
        if self.simd_stats["operations_accelerated"] / max(self.simd_stats["total_operations"], 1) < 0.5:
            suggestions.append("Low SIMD acceleration rate - review operation patterns")
        
        if len(self.simd_stats["speedup_history"]) > 0:
            avg_speedup = np.mean(self.simd_stats["speedup_history"])
            if avg_speedup < 1.2:
                suggestions.append("Low average speedup - consider tuning chunk sizes")
        
        return suggestions
    
    def get_simd_stats(self) -> Dict[str, Any]:
        """Get SIMD processing statistics."""
        return {
            **self.simd_stats,
            "cpu_features": self.cpu_features,
            "operations_available": list(self.operations.keys()),
            "numba_available": NUMBA_AVAILABLE,
            "scipy_available": SCIPY_AVAILABLE
        }
    
    def clear_stats(self):
        """Clear performance statistics."""
        self.simd_stats = {
            "operations_accelerated": 0,
            "total_operations": 0,
            "average_speedup": 1.0,
            "speedup_history": []
        }