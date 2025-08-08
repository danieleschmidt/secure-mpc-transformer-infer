#!/usr/bin/env python3
"""
Generation 3 Scaling Features Demo

This example demonstrates the comprehensive scaling features implemented
for the secure MPC transformer system, including:

1. Performance Optimization (quantization, pruning, GPU optimization, tensor fusion, SIMD, adaptive batching, memory mapping)
2. Advanced Caching (L1/L2/distributed caching with warming and coherence)
3. Concurrent Processing (worker pools, async optimization)
4. Auto-scaling & Load Management 
5. High-Performance Features (profiling, optimization hints)

This showcases enterprise-grade performance and scalability capabilities.
"""

import asyncio
import time
import torch
import logging
from typing import List, Dict, Any
import numpy as np

# Import Generation 3 scaling components
from secure_mpc_transformer import (
    # Core
    SecureTransformer, TransformerConfig, InferenceService,
    
    # Performance Optimization
    ModelOptimizer, OptimizationConfig,
    GPUMemoryManager, CUDAStreamManager,
    TensorFusionEngine, FusionConfig,
    SIMDProcessor,
    AdaptiveBatchManager,
    MemoryMappedModelLoader,
    
    # Caching System
    CacheManager, CacheConfig,
    CacheWarmer,
    
    # Concurrent Processing  
    WorkerPool, DynamicWorkerPool, WorkerConfig,
    AsyncOptimizer, AsyncContext,
    
    # Profiling
    PerformanceProfiler, ProfilingConfig
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Generation3ScalingDemo:
    """Comprehensive demo of Generation 3 scaling features."""
    
    def __init__(self):
        self.setup_components()
    
    def setup_components(self):
        """Initialize all Generation 3 scaling components."""
        logger.info("🚀 Initializing Generation 3 Scaling Components...")
        
        # 1. Performance Optimization Components
        logger.info("📊 Setting up Performance Optimization...")
        
        self.optimization_config = OptimizationConfig(
            enable_quantization=True,
            quantization_bits=8,
            enable_pruning=True,
            pruning_ratio=0.2,
            enable_mixed_precision=True
        )
        
        self.model_optimizer = ModelOptimizer()
        
        # GPU Memory Management
        self.gpu_manager = GPUMemoryManager(enable_memory_pool=True)
        self.gpu_manager.start_memory_monitoring()
        
        # Tensor Fusion Engine
        fusion_config = FusionConfig(
            enable_elementwise_fusion=True,
            enable_matmul_fusion=True,
            enable_attention_fusion=True
        )
        self.tensor_fusion = TensorFusionEngine(fusion_config)
        
        # SIMD Processor
        self.simd_processor = SIMDProcessor()
        
        # Adaptive Batch Manager
        self.batch_manager = AdaptiveBatchManager()
        self.batch_manager.start()
        
        # Memory-Mapped Model Loader
        self.mmap_loader = MemoryMappedModelLoader()
        
        # 2. Advanced Caching System
        logger.info("🗄️ Setting up Advanced Caching System...")
        
        cache_config = CacheConfig(
            l1_max_memory_mb=2048,  # 2GB L1 cache
            l2_max_memory_mb=8192,  # 8GB L2 cache
            distributed_enable=False,  # Would need Redis for full demo
            enable_cache_warming=True,
            enable_coherence=True
        )
        
        self.cache_manager = CacheManager(cache_config)
        self.cache_warmer = CacheWarmer(config=None, cache_manager=self.cache_manager)
        
        # 3. Concurrent Processing
        logger.info("⚡ Setting up Concurrent Processing...")
        
        worker_config = WorkerConfig(
            min_workers=2,
            max_workers=8,
            scaling_strategy="adaptive"
        )
        
        self.worker_pool = DynamicWorkerPool(worker_config)
        
        # Async Optimization
        async_config = None  # Use defaults
        self.async_optimizer = AsyncOptimizer(async_config) if async_config else None
        
        # 4. Performance Profiling
        logger.info("📈 Setting up Performance Profiling...")
        
        profiling_config = ProfilingConfig(
            level="detailed",
            enable_cpu_profiling=True,
            enable_memory_profiling=True,
            enable_gpu_profiling=True,
            sampling_interval_ms=100
        )
        
        self.profiler = PerformanceProfiler(profiling_config)
        
        logger.info("✅ Generation 3 Scaling Components Initialized!")
    
    async def run_comprehensive_demo(self):
        """Run comprehensive demo of all scaling features."""
        logger.info("🎯 Starting Comprehensive Generation 3 Scaling Demo...")
        
        # Start profiling
        self.profiler.start_profiling()
        
        try:
            # Demo 1: Performance Optimization Pipeline
            await self.demo_performance_optimization()
            
            # Demo 2: Advanced Caching System
            await self.demo_advanced_caching()
            
            # Demo 3: Concurrent Processing
            await self.demo_concurrent_processing()
            
            # Demo 4: Integrated High-Performance Inference
            await self.demo_integrated_inference()
            
            # Demo 5: Auto-scaling Simulation
            await self.demo_auto_scaling()
            
        finally:
            # Generate profiling report
            report = self.profiler.stop_profiling()
            self.print_performance_report(report)
    
    async def demo_performance_optimization(self):
        """Demonstrate performance optimization features."""
        logger.info("🔧 Demo 1: Performance Optimization Pipeline")
        
        # Create a sample model for optimization
        config = TransformerConfig(
            model_name="bert-base-uncased",
            hidden_size=768,
            num_hidden_layers=12
        )
        
        # Simulate model creation and optimization
        logger.info("  📊 Creating sample transformer model...")
        model = self._create_sample_model(config)
        
        # Model Optimization
        logger.info("  ⚙️ Applying model optimizations (quantization, pruning)...")
        with self.profiler.profile_context("model_optimization"):
            optimized_model = self.model_optimizer.optimize_model(model, self.optimization_config)
            opt_stats = self.model_optimizer.get_optimization_stats()
            
            logger.info(f"    ✨ Compression ratio: {opt_stats.get('total_compression_ratio', 1.0):.2f}x")
            logger.info(f"    💾 Model size reduced: {opt_stats.get('original_size_mb', 0):.1f}MB -> {opt_stats.get('optimized_size_mb', 0):.1f}MB")
        
        # GPU Memory Optimization
        logger.info("  🎮 Demonstrating GPU memory optimization...")
        with self.gpu_manager.managed_memory_context(0):  # GPU 0
            sample_tensor = torch.randn(1024, 768, device='cuda' if torch.cuda.is_available() else 'cpu')
            
            # Memory stats
            gpu_stats = self.gpu_manager.get_comprehensive_stats()
            logger.info(f"    📊 GPU memory stats: {gpu_stats.get('memory_stats', {})}")
        
        # Tensor Fusion Demo
        logger.info("  🔗 Demonstrating tensor fusion...")
        sample_ops = [torch.nn.ReLU(), torch.nn.Linear(768, 768)]
        fused_op = self.tensor_fusion.fuse_operations(sample_ops)
        if fused_op:
            logger.info("    ✅ Successfully fused tensor operations")
        
        # SIMD Processing Demo
        logger.info("  🏎️ Demonstrating SIMD acceleration...")
        sample_data = torch.randn(1000, 768)
        accelerated_result = self.simd_processor.accelerate_operation("relu", sample_data)
        simd_stats = self.simd_processor.get_simd_stats()
        logger.info(f"    ⚡ SIMD operations accelerated: {simd_stats.get('operations_accelerated', 0)}")
        
        # Adaptive Batching Demo
        logger.info("  📦 Demonstrating adaptive batch sizing...")
        optimal_batch_size = self.batch_manager.get_next_batch_size(current_workload_size=100)
        logger.info(f"    🎯 Optimal batch size: {optimal_batch_size}")
        
        logger.info("✅ Performance Optimization Demo Complete!\n")
    
    async def demo_advanced_caching(self):
        """Demonstrate advanced caching system."""
        logger.info("🗄️ Demo 2: Advanced Caching System")
        
        # Cache some sample data
        logger.info("  💾 Populating multi-level cache...")
        sample_data = {
            "model_weights_layer_0": torch.randn(768, 768),
            "model_weights_layer_1": torch.randn(768, 768),
            "intermediate_results": torch.randn(32, 768),
            "attention_cache": torch.randn(32, 12, 64, 64)
        }
        
        # Store in cache with different strategies
        for key, data in sample_data.items():
            self.cache_manager.put(key, data, namespace="demo")
        
        # Demonstrate cache hits
        logger.info("  🎯 Testing cache retrieval...")
        hit_count = 0
        for key in sample_data.keys():
            cached_data = self.cache_manager.get(key, namespace="demo")
            if cached_data is not None:
                hit_count += 1
        
        logger.info(f"    ✅ Cache hits: {hit_count}/{len(sample_data)}")
        
        # Cache warming demonstration
        logger.info("  🔥 Demonstrating cache warming...")
        
        # Register a warming source
        def sample_loader(key: str):
            return torch.randn(768, 768)  # Simulate loading
        
        self.cache_warmer.register_warming_source("sample_models", sample_loader)
        
        # Warm cache with predicted keys
        warm_keys = ["future_model_1", "future_model_2", "future_model_3"]
        warm_results = self.cache_warmer.warm_cache_immediate(warm_keys, source="sample_models")
        
        successful_warming = sum(warm_results.values())
        logger.info(f"    🔥 Successfully warmed {successful_warming}/{len(warm_keys)} cache entries")
        
        # Get cache statistics
        cache_stats = self.cache_manager.get_cache_stats()
        overall_stats = cache_stats.get('overall', {})
        logger.info(f"    📊 Cache hit rate: {overall_stats.get('hit_rate', 0):.2%}")
        logger.info(f"    📊 L1 cache entries: {cache_stats.get('levels', {}).get('l1', {}).get('total_entries', 0)}")
        
        logger.info("✅ Advanced Caching Demo Complete!\n")
    
    async def demo_concurrent_processing(self):
        """Demonstrate concurrent processing capabilities."""
        logger.info("⚡ Demo 3: Concurrent Processing")
        
        # Worker Pool Demo
        logger.info("  👷 Demonstrating dynamic worker pools...")
        
        # Submit some sample tasks
        def sample_computation(x: int) -> int:
            time.sleep(0.1)  # Simulate work
            return x * x
        
        tasks = []
        for i in range(20):
            future = self.worker_pool.submit(sample_computation, i, task_type="cpu_intensive")
            tasks.append(future)
        
        # Wait for completion
        results = []
        for task in tasks:
            result = task.result()
            results.append(result)
        
        logger.info(f"    ✅ Completed {len(results)} concurrent tasks")
        
        # Get worker pool metrics
        pool_metrics = self.worker_pool.get_metrics()
        for pool_name, metrics in pool_metrics.items():
            logger.info(f"    📊 {pool_name} pool: {metrics.completed_tasks} completed, "
                       f"{metrics.throughput:.1f} tasks/sec")
        
        # Async Processing Demo (if available)
        if self.async_optimizer:
            logger.info("  🔄 Demonstrating async optimization...")
            
            async def sample_async_task(x: int) -> int:
                await asyncio.sleep(0.01)  # Simulate async work
                return x * 2
            
            # Execute async tasks with optimization
            async_tasks = []
            for i in range(10):
                task = self.async_optimizer.execute_optimized(
                    sample_async_task, i, 
                    task_id=f"async_task_{i}",
                    cache_key=f"result_{i}"
                )
                async_tasks.append(task)
            
            async_results = await asyncio.gather(*async_tasks)
            logger.info(f"    ✅ Completed {len(async_results)} async optimized tasks")
            
            # Get async metrics
            async_metrics = self.async_optimizer.get_metrics()
            logger.info(f"    📊 Async hit rate: {async_metrics.cache_hit_rate():.2%}")
        
        logger.info("✅ Concurrent Processing Demo Complete!\n")
    
    async def demo_integrated_inference(self):
        """Demonstrate integrated high-performance inference."""
        logger.info("🧠 Demo 4: Integrated High-Performance Inference")
        
        logger.info("  🔄 Simulating optimized inference pipeline...")
        
        # Simulate end-to-end optimized inference
        batch_sizes = [1, 4, 8, 16, 32]
        inference_times = []
        
        for batch_size in batch_sizes:
            # Get optimal batch size from adaptive manager
            optimal_batch = self.batch_manager.get_next_batch_size(batch_size)
            
            # Simulate inference with all optimizations
            start_time = time.perf_counter()
            
            with self.profiler.profile_context(f"inference_batch_{optimal_batch}"):
                # Simulate model inference with optimizations
                await self._simulate_optimized_inference(optimal_batch)
            
            inference_time = time.perf_counter() - start_time
            inference_times.append(inference_time)
            
            # Record batch execution for adaptive learning
            self.batch_manager.record_batch_execution(
                optimal_batch, 
                inference_time, 
                memory_usage_mb=100.0
            )
            
            logger.info(f"    ⚡ Batch size {optimal_batch}: {inference_time:.3f}s")
        
        # Calculate throughput improvements
        avg_inference_time = sum(inference_times) / len(inference_times)
        estimated_throughput = sum(batch_sizes) / sum(inference_times)
        
        logger.info(f"    📊 Average inference time: {avg_inference_time:.3f}s")
        logger.info(f"    📊 Estimated throughput: {estimated_throughput:.1f} samples/sec")
        
        # Get comprehensive performance summary
        batch_summary = self.batch_manager.get_performance_summary()
        logger.info(f"    🎯 Batch manager efficiency: {batch_summary.get('average_efficiency', 0):.2f}")
        
        logger.info("✅ Integrated High-Performance Inference Demo Complete!\n")
    
    async def demo_auto_scaling(self):
        """Demonstrate auto-scaling capabilities."""
        logger.info("📈 Demo 5: Auto-scaling Simulation")
        
        logger.info("  🎢 Simulating variable load patterns...")
        
        # Simulate increasing load
        load_pattern = [10, 25, 50, 100, 75, 50, 25, 10]  # Requests per iteration
        
        for i, load in enumerate(load_pattern):
            logger.info(f"  📊 Load simulation {i+1}: {load} concurrent requests")
            
            # Get current optimal batch size
            optimal_batch = self.batch_manager.get_next_batch_size(load)
            
            # Simulate processing the load
            processing_time = await self._simulate_load_processing(load, optimal_batch)
            
            # Record metrics for adaptive learning
            self.batch_manager.record_batch_execution(
                optimal_batch,
                processing_time,
                memory_usage_mb=load * 5.0,  # Simulate memory scaling
                cpu_utilization=min(load * 0.8, 100)  # Simulate CPU scaling
            )
            
            logger.info(f"    ⚡ Processed {load} requests in {processing_time:.3f}s with batch size {optimal_batch}")
            
            # Brief pause between load changes
            await asyncio.sleep(0.1)
        
        # Get final performance summary
        final_summary = self.batch_manager.get_performance_summary()
        logger.info(f"    🏆 Final system efficiency: {final_summary.get('average_efficiency', 0):.2f}")
        logger.info(f"    📊 Total batches processed: {final_summary.get('total_batches', 0)}")
        logger.info(f"    📊 Average throughput: {final_summary.get('average_throughput', 0):.1f} req/sec")
        
        logger.info("✅ Auto-scaling Demo Complete!\n")
    
    async def _simulate_optimized_inference(self, batch_size: int):
        """Simulate optimized inference process."""
        # Simulate GPU processing with memory management
        with self.gpu_manager.managed_memory_context(0):
            # Simulate tensor operations
            input_tensor = torch.randn(batch_size, 512, device='cuda' if torch.cuda.is_available() else 'cpu')
            
            # Apply SIMD acceleration (CPU fallback)
            if input_tensor.device.type == 'cpu':
                processed = self.simd_processor.accelerate_operation("relu", input_tensor)
            else:
                processed = torch.relu(input_tensor)
            
            # Simulate model computation
            await asyncio.sleep(0.01 * batch_size)  # Scale with batch size
            
            return processed
    
    async def _simulate_load_processing(self, load: int, batch_size: int) -> float:
        """Simulate processing variable load."""
        start_time = time.perf_counter()
        
        # Process in batches
        num_batches = (load + batch_size - 1) // batch_size  # Ceiling division
        
        for _ in range(num_batches):
            await self._simulate_optimized_inference(batch_size)
        
        return time.perf_counter() - start_time
    
    def _create_sample_model(self, config: TransformerConfig) -> torch.nn.Module:
        """Create a sample model for demonstration."""
        # Create a simple model structure for optimization demo
        model = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(config.hidden_size, config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(config.hidden_size, config.hidden_size)
        )
        return model
    
    def print_performance_report(self, report):
        """Print comprehensive performance report."""
        logger.info("📋 COMPREHENSIVE PERFORMANCE REPORT")
        logger.info("=" * 60)
        
        logger.info(f"📊 Profiling Duration: {report.profiling_duration:.2f} seconds")
        logger.info(f"📊 Total Samples: {report.total_samples}")
        logger.info(f"📊 Functions Profiled: {len(report.function_profiles)}")
        
        # System metrics
        if report.system_metrics:
            cpu_metrics = report.system_metrics.get('cpu', {})
            memory_metrics = report.system_metrics.get('memory', {})
            gpu_metrics = report.system_metrics.get('gpu', {})
            
            logger.info(f"🖥️  Average CPU Usage: {cpu_metrics.get('average', 0):.1f}%")
            logger.info(f"💾 Peak Memory Usage: {memory_metrics.get('max_mb', 0):.1f}MB")
            logger.info(f"🎮 Average GPU Usage: {gpu_metrics.get('average_utilization', 0):.1f}%")
        
        # Performance issues
        if report.performance_issues:
            logger.info(f"⚠️  Performance Issues Found: {len(report.performance_issues)}")
            for issue in report.performance_issues[:3]:  # Show top 3
                logger.info(f"   - {issue.get('type', 'Unknown')}: {issue}")
        
        # Optimization hints
        if report.optimization_hints:
            logger.info(f"💡 Optimization Hints: {len(report.optimization_hints)}")
            for hint in report.optimization_hints[:3]:  # Show top 3
                logger.info(f"   - {hint}")
        
        # Top functions by execution time
        if report.function_profiles:
            logger.info("🔥 Top Functions by Execution Time:")
            sorted_functions = sorted(
                report.function_profiles.items(),
                key=lambda x: x[1].total_time,
                reverse=True
            )
            
            for name, profile in sorted_functions[:5]:  # Top 5
                logger.info(f"   - {name}: {profile.total_time*1000:.1f}ms total, "
                           f"{profile.call_count} calls, {profile.average_time*1000:.1f}ms avg")
        
        logger.info("=" * 60)
        logger.info("🎉 Generation 3 Scaling Demo Complete!")
    
    def shutdown(self):
        """Clean shutdown of all components."""
        logger.info("🛑 Shutting down Generation 3 components...")
        
        # Shutdown in reverse order of initialization
        try:
            self.batch_manager.stop()
            self.gpu_manager.stop_memory_monitoring()
            self.worker_pool.shutdown()
            if self.async_optimizer:
                asyncio.create_task(self.async_optimizer.shutdown())
            self.cache_manager.shutdown()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        logger.info("✅ Shutdown complete")


async def main():
    """Main demo execution."""
    print("🚀 Secure MPC Transformer - Generation 3 Scaling Demo")
    print("=" * 60)
    print("This demo showcases enterprise-grade performance and scaling features:")
    print("• Performance Optimization (quantization, pruning, GPU, tensor fusion, SIMD)")
    print("• Advanced Multi-level Caching with intelligent warming")
    print("• High-performance Concurrent Processing")
    print("• Auto-scaling and Load Management")  
    print("• Comprehensive Performance Profiling")
    print("=" * 60)
    
    demo = Generation3ScalingDemo()
    
    try:
        await demo.run_comprehensive_demo()
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
    finally:
        demo.shutdown()


if __name__ == "__main__":
    asyncio.run(main())