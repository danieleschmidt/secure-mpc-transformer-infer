"""
Research Module for Novel Algorithmic Contributions

This module contains cutting-edge research implementations including:
- Post-quantum resistant MPC protocols with quantum-inspired optimization  
- Hybrid quantum-classical algorithms for secure computation
- Novel comparative studies and experimental frameworks
- Academic-quality benchmarking and validation systems

All implementations are designed for defensive security applications and
follow rigorous academic standards for reproducibility and peer review.
"""

from .post_quantum_mpc import PostQuantumMPCProtocol, QuantumResistantOptimizer
from .hybrid_algorithms import HybridQuantumClassicalScheduler, QuantumEnhancedMPC
from .experimental_framework import ResearchExperimentRunner, ComparativeStudyFramework
from .novel_algorithms import AdaptiveMPCOrchestrator, QuantumInspiredSecurityOptimizer
from .validation_framework import StatisticalValidationEngine, ReproducibilityFramework

__all__ = [
    "PostQuantumMPCProtocol",
    "QuantumResistantOptimizer", 
    "HybridQuantumClassicalScheduler",
    "QuantumEnhancedMPC",
    "ResearchExperimentRunner",
    "ComparativeStudyFramework",
    "AdaptiveMPCOrchestrator",
    "QuantumInspiredSecurityOptimizer",
    "StatisticalValidationEngine",
    "ReproducibilityFramework"
]

# Research module version for citation tracking
__version__ = "0.1.0-research"
__authors__ = ["Daniel Schmidt", "Terragon Labs Research Team"]
__institution__ = "Terragon Labs"
__research_area__ = "Quantum-Enhanced Secure Multi-Party Computation"