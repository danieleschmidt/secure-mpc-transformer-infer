"""
Autonomous Bio-Evolution Executor for Secure MPC Transformer System.

This module implements bio-inspired autonomous execution patterns
that evolve and adapt the system capabilities dynamically.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import numpy as np
from pathlib import Path


class BioEvolutionPhase(Enum):
    """Bio-evolution phases for autonomous system adaptation."""
    GENESIS = "genesis"          # Initial system birth
    ADAPTATION = "adaptation"    # Learning and adjusting
    MUTATION = "mutation"        # Introducing variations
    SELECTION = "selection"      # Keeping best adaptations
    REPRODUCTION = "reproduction" # Spreading successful patterns
    SYMBIOSIS = "symbiosis"      # Collaborative evolution


@dataclass
class BioGene:
    """Represents a genetic pattern in the system."""
    gene_id: str
    pattern_type: str
    effectiveness: float
    adaptation_count: int = 0
    birth_time: datetime = field(default_factory=datetime.now)
    parent_genes: List[str] = field(default_factory=list)
    traits: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BioPhenotype:
    """Observable characteristics resulting from gene expression."""
    phenotype_id: str
    genes: List[BioGene]
    performance_metrics: Dict[str, float]
    fitness_score: float
    environment_adaptations: List[str]


class AutonomousBioExecutor:
    """
    Bio-inspired autonomous executor that evolves system capabilities.
    
    Uses genetic algorithms and evolutionary patterns to continuously
    improve system performance and adapt to changing conditions.
    """
    
    def __init__(
        self,
        population_size: int = 50,
        mutation_rate: float = 0.1,
        selection_pressure: float = 0.3,
        max_generations: int = 1000
    ):
        self.logger = logging.getLogger(__name__)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.selection_pressure = selection_pressure
        self.max_generations = max_generations
        
        # Evolution tracking
        self.gene_pool: Dict[str, BioGene] = {}
        self.phenotype_population: List[BioPhenotype] = []
        self.generation_history: List[Dict[str, Any]] = []
        self.current_generation = 0
        
        # Performance tracking
        self.fitness_history: List[float] = []
        self.adaptation_metrics: Dict[str, List[float]] = {}
        
        self.logger.info("Autonomous Bio Executor initialized")
        
    async def initialize_genesis_population(self) -> None:
        """Initialize the first generation of bio-patterns."""
        self.logger.info("🧬 Initializing Genesis Population")
        
        # Create foundational genes for core system capabilities
        foundational_genes = [
            BioGene(
                gene_id="core_security",
                pattern_type="security_protocol",
                effectiveness=0.85,
                traits={"encryption_level": 128, "threat_detection": True}
            ),
            BioGene(
                gene_id="quantum_optimization",
                pattern_type="performance_optimization",
                effectiveness=0.78,
                traits={"quantum_enabled": True, "optimization_depth": 4}
            ),
            BioGene(
                gene_id="adaptive_caching",
                pattern_type="resource_management",
                effectiveness=0.72,
                traits={"cache_hit_rate": 0.85, "adaptive_sizing": True}
            ),
            BioGene(
                gene_id="error_resilience",
                pattern_type="reliability",
                effectiveness=0.80,
                traits={"recovery_time": 0.5, "failure_tolerance": 0.95}
            ),
            BioGene(
                gene_id="load_balancing",
                pattern_type="scalability",
                effectiveness=0.75,
                traits={"auto_scaling": True, "resource_efficiency": 0.88}
            )
        ]
        
        # Add genes to pool
        for gene in foundational_genes:
            self.gene_pool[gene.gene_id] = gene
            
        # Create initial phenotypes through random gene combinations
        for i in range(self.population_size):
            selected_genes = np.random.choice(
                foundational_genes, 
                size=np.random.randint(2, 4), 
                replace=False
            ).tolist()
            
            phenotype = BioPhenotype(
                phenotype_id=f"genesis_{i:03d}",
                genes=selected_genes,
                performance_metrics={},
                fitness_score=0.0,
                environment_adaptations=[]
            )
            
            self.phenotype_population.append(phenotype)
            
        self.logger.info(f"Genesis population created: {len(self.phenotype_population)} phenotypes")
        
    async def evaluate_fitness(self, phenotype: BioPhenotype) -> float:
        """Evaluate the fitness of a phenotype based on performance metrics."""
        fitness_components = []
        
        # Security effectiveness (30% weight)
        security_genes = [g for g in phenotype.genes if g.pattern_type == "security_protocol"]
        security_score = np.mean([g.effectiveness for g in security_genes]) if security_genes else 0.5
        fitness_components.append(security_score * 0.3)
        
        # Performance optimization (25% weight)
        perf_genes = [g for g in phenotype.genes if g.pattern_type == "performance_optimization"]
        perf_score = np.mean([g.effectiveness for g in perf_genes]) if perf_genes else 0.5
        fitness_components.append(perf_score * 0.25)
        
        # Reliability (20% weight)
        reliability_genes = [g for g in phenotype.genes if g.pattern_type == "reliability"]
        reliability_score = np.mean([g.effectiveness for g in reliability_genes]) if reliability_genes else 0.5
        fitness_components.append(reliability_score * 0.2)
        
        # Resource efficiency (15% weight)
        resource_genes = [g for g in phenotype.genes if g.pattern_type == "resource_management"]
        resource_score = np.mean([g.effectiveness for g in resource_genes]) if resource_genes else 0.5
        fitness_components.append(resource_score * 0.15)
        
        # Scalability (10% weight)
        scale_genes = [g for g in phenotype.genes if g.pattern_type == "scalability"]
        scale_score = np.mean([g.effectiveness for g in scale_genes]) if scale_genes else 0.5
        fitness_components.append(scale_score * 0.1)
        
        return sum(fitness_components)
        
    async def mutate_gene(self, gene: BioGene) -> BioGene:
        """Create a mutated version of a gene."""
        mutated_traits = gene.traits.copy()
        
        # Randomly mutate traits
        for trait_name, trait_value in mutated_traits.items():
            if np.random.random() < self.mutation_rate:
                if isinstance(trait_value, (int, float)):
                    # Add random variation
                    variation = np.random.normal(0, 0.1) * trait_value
                    mutated_traits[trait_name] = max(0, trait_value + variation)
                elif isinstance(trait_value, bool):
                    # Flip boolean with low probability
                    if np.random.random() < 0.1:
                        mutated_traits[trait_name] = not trait_value
                        
        # Create mutated gene
        mutated_gene = BioGene(
            gene_id=f"{gene.gene_id}_mut_{gene.adaptation_count + 1}",
            pattern_type=gene.pattern_type,
            effectiveness=min(1.0, gene.effectiveness + np.random.normal(0, 0.05)),
            adaptation_count=gene.adaptation_count + 1,
            parent_genes=[gene.gene_id],
            traits=mutated_traits
        )
        
        return mutated_gene
        
    async def crossover_genes(self, parent1: BioGene, parent2: BioGene) -> BioGene:
        """Create offspring gene through crossover of two parent genes."""
        # Combine traits from both parents
        combined_traits = {}
        all_trait_names = set(parent1.traits.keys()) | set(parent2.traits.keys())
        
        for trait_name in all_trait_names:
            # Randomly choose trait from one parent or average if both have it
            if trait_name in parent1.traits and trait_name in parent2.traits:
                if isinstance(parent1.traits[trait_name], (int, float)):
                    combined_traits[trait_name] = (
                        parent1.traits[trait_name] + parent2.traits[trait_name]
                    ) / 2
                else:
                    combined_traits[trait_name] = np.random.choice([
                        parent1.traits[trait_name], 
                        parent2.traits[trait_name]
                    ])
            elif trait_name in parent1.traits:
                combined_traits[trait_name] = parent1.traits[trait_name]
            else:
                combined_traits[trait_name] = parent2.traits[trait_name]
                
        # Create offspring gene
        offspring_gene = BioGene(
            gene_id=f"cross_{parent1.gene_id}_{parent2.gene_id}_{datetime.now().timestamp()}",
            pattern_type=np.random.choice([parent1.pattern_type, parent2.pattern_type]),
            effectiveness=(parent1.effectiveness + parent2.effectiveness) / 2,
            parent_genes=[parent1.gene_id, parent2.gene_id],
            traits=combined_traits
        )
        
        return offspring_gene
        
    async def selection_phase(self) -> List[BioPhenotype]:
        """Select the fittest phenotypes for the next generation."""
        # Evaluate fitness for all phenotypes
        for phenotype in self.phenotype_population:
            phenotype.fitness_score = await self.evaluate_fitness(phenotype)
            
        # Sort by fitness (descending)
        self.phenotype_population.sort(key=lambda p: p.fitness_score, reverse=True)
        
        # Select top performers
        selection_count = int(len(self.phenotype_population) * self.selection_pressure)
        selected = self.phenotype_population[:selection_count]
        
        # Record best fitness
        if selected:
            self.fitness_history.append(selected[0].fitness_score)
            
        self.logger.info(f"Selection phase: {len(selected)} phenotypes selected")
        return selected
        
    async def reproduction_phase(self, selected_phenotypes: List[BioPhenotype]) -> None:
        """Create new generation through reproduction and mutation."""
        new_population = []
        
        # Keep elite performers
        elite_count = max(1, len(selected_phenotypes) // 4)
        new_population.extend(selected_phenotypes[:elite_count])
        
        # Create offspring through crossover and mutation
        while len(new_population) < self.population_size:
            # Select two parents
            parent1 = np.random.choice(selected_phenotypes)
            parent2 = np.random.choice(selected_phenotypes)
            
            # Create offspring genes
            offspring_genes = []
            for i in range(min(len(parent1.genes), len(parent2.genes))):
                if np.random.random() < 0.7:  # Crossover probability
                    offspring_gene = await self.crossover_genes(
                        parent1.genes[i], parent2.genes[i]
                    )
                else:
                    # Mutation only
                    parent_gene = np.random.choice([parent1.genes[i], parent2.genes[i]])
                    offspring_gene = await self.mutate_gene(parent_gene)
                    
                offspring_genes.append(offspring_gene)
                
            # Create offspring phenotype
            offspring = BioPhenotype(
                phenotype_id=f"gen{self.current_generation + 1}_{len(new_population):03d}",
                genes=offspring_genes,
                performance_metrics={},
                fitness_score=0.0,
                environment_adaptations=[]
            )
            
            new_population.append(offspring)
            
        self.phenotype_population = new_population
        self.current_generation += 1
        
        self.logger.info(f"Reproduction complete: Generation {self.current_generation}")
        
    async def evolve_generation(self) -> Dict[str, Any]:
        """Evolve one generation of the bio-system."""
        start_time = datetime.now()
        
        # Selection phase
        selected = await self.selection_phase()
        
        # Reproduction phase
        await self.reproduction_phase(selected)
        
        # Record generation statistics
        generation_stats = {
            "generation": self.current_generation,
            "population_size": len(self.phenotype_population),
            "best_fitness": max(p.fitness_score for p in self.phenotype_population),
            "avg_fitness": np.mean([p.fitness_score for p in self.phenotype_population]),
            "evolution_time": (datetime.now() - start_time).total_seconds(),
            "gene_diversity": len(set(g.gene_id for p in self.phenotype_population for g in p.genes))
        }
        
        self.generation_history.append(generation_stats)
        return generation_stats
        
    async def run_autonomous_evolution(self, target_fitness: float = 0.95) -> None:
        """Run continuous autonomous evolution until target fitness is achieved."""
        self.logger.info("🌱 Starting Autonomous Bio-Evolution")
        
        # Initialize genesis population
        await self.initialize_genesis_population()
        
        # Evolution loop
        while self.current_generation < self.max_generations:
            stats = await self.evolve_generation()
            
            self.logger.info(
                f"Generation {stats['generation']}: "
                f"Best={stats['best_fitness']:.3f}, "
                f"Avg={stats['avg_fitness']:.3f}, "
                f"Diversity={stats['gene_diversity']}"
            )
            
            # Check convergence
            if stats["best_fitness"] >= target_fitness:
                self.logger.info(f"🎯 Target fitness achieved: {stats['best_fitness']:.3f}")
                break
                
            # Adaptive mutation rate
            if len(self.fitness_history) > 10:
                recent_improvement = (
                    self.fitness_history[-1] - self.fitness_history[-10]
                )
                if recent_improvement < 0.01:  # Stagnation
                    self.mutation_rate = min(0.3, self.mutation_rate * 1.1)
                else:
                    self.mutation_rate = max(0.05, self.mutation_rate * 0.95)
                    
            await asyncio.sleep(0.1)  # Allow other tasks to run
            
        self.logger.info("🧬 Autonomous Evolution Complete")
        
    def get_best_phenotype(self) -> Optional[BioPhenotype]:
        """Get the current best performing phenotype."""
        if not self.phenotype_population:
            return None
            
        return max(self.phenotype_population, key=lambda p: p.fitness_score)
        
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get comprehensive evolution summary."""
        best_phenotype = self.get_best_phenotype()
        
        return {
            "total_generations": self.current_generation,
            "best_fitness": best_phenotype.fitness_score if best_phenotype else 0.0,
            "gene_pool_size": len(self.gene_pool),
            "fitness_improvement": (
                self.fitness_history[-1] - self.fitness_history[0] 
                if len(self.fitness_history) > 1 else 0.0
            ),
            "convergence_rate": np.mean(np.diff(self.fitness_history)) if len(self.fitness_history) > 1 else 0.0,
            "best_phenotype_traits": {
                g.gene_id: g.traits for g in best_phenotype.genes
            } if best_phenotype else {},
            "evolution_timeline": self.generation_history
        }


async def main():
    """Demo the autonomous bio-evolution system."""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize bio executor
    bio_executor = AutonomousBioExecutor(
        population_size=30,
        mutation_rate=0.15,
        selection_pressure=0.4,
        max_generations=50
    )
    
    # Run evolution
    await bio_executor.run_autonomous_evolution(target_fitness=0.92)
    
    # Print summary
    summary = bio_executor.get_evolution_summary()
    print("\n🧬 EVOLUTION SUMMARY:")
    print(f"Generations: {summary['total_generations']}")
    print(f"Best Fitness: {summary['best_fitness']:.3f}")
    print(f"Fitness Improvement: {summary['fitness_improvement']:.3f}")
    print(f"Gene Pool Size: {summary['gene_pool_size']}")


if __name__ == "__main__":
    asyncio.run(main())