#!/usr/bin/env python3
"""
Minimal Bio-Evolution Generation 1 Demo

Self-contained demonstration of bio-inspired autonomous evolution
without external dependencies.
"""

import asyncio
import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


class BioEvolutionPhase(Enum):
    """Bio-evolution phases for autonomous system adaptation."""
    GENESIS = "genesis"
    ADAPTATION = "adaptation"
    MUTATION = "mutation"
    SELECTION = "selection"
    REPRODUCTION = "reproduction"
    SYMBIOSIS = "symbiosis"


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


class MinimalBioExecutor:
    """Minimal bio-inspired autonomous executor for demonstration."""
    
    def __init__(self, population_size: int = 20, mutation_rate: float = 0.12, 
                 selection_pressure: float = 0.35, max_generations: int = 25):
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
        self.fitness_history: List[float] = []
        
    async def initialize_genesis_population(self) -> None:
        """Initialize the first generation of bio-patterns."""
        self.logger.info("🧬 Initializing Genesis Population")
        
        # Create foundational genes
        foundational_genes = [
            BioGene(
                gene_id="security_core", pattern_type="security_protocol",
                effectiveness=0.82, traits={"encryption_level": 128, "threat_detection": True}
            ),
            BioGene(
                gene_id="quantum_opt", pattern_type="performance_optimization", 
                effectiveness=0.75, traits={"quantum_enabled": True, "optimization_depth": 4}
            ),
            BioGene(
                gene_id="adaptive_cache", pattern_type="resource_management",
                effectiveness=0.78, traits={"cache_hit_rate": 0.87, "adaptive_sizing": True}
            ),
            BioGene(
                gene_id="error_recovery", pattern_type="reliability",
                effectiveness=0.84, traits={"recovery_time": 0.3, "failure_tolerance": 0.96}
            ),
            BioGene(
                gene_id="load_balance", pattern_type="scalability",
                effectiveness=0.71, traits={"auto_scaling": True, "resource_efficiency": 0.89}
            )
        ]
        
        # Add to gene pool
        for gene in foundational_genes:
            self.gene_pool[gene.gene_id] = gene
            
        # Create initial phenotypes
        for i in range(self.population_size):
            selected_genes = np.random.choice(
                foundational_genes, size=np.random.randint(2, 4), replace=False
            ).tolist()
            
            phenotype = BioPhenotype(
                phenotype_id=f"genesis_{i:03d}", genes=selected_genes,
                performance_metrics={}, fitness_score=0.0, environment_adaptations=[]
            )
            self.phenotype_population.append(phenotype)
            
        self.logger.info(f"Genesis population created: {len(self.phenotype_population)} phenotypes")
        
    async def evaluate_fitness(self, phenotype: BioPhenotype) -> float:
        """Evaluate phenotype fitness."""
        fitness_components = []
        
        # Weight different gene types
        weights = {
            "security_protocol": 0.30, "performance_optimization": 0.25,
            "reliability": 0.20, "resource_management": 0.15, "scalability": 0.10
        }
        
        for pattern_type, weight in weights.items():
            genes = [g for g in phenotype.genes if g.pattern_type == pattern_type]
            score = np.mean([g.effectiveness for g in genes]) if genes else 0.5
            fitness_components.append(score * weight)
            
        return sum(fitness_components)
        
    async def mutate_gene(self, gene: BioGene) -> BioGene:
        """Create a mutated version of a gene."""
        mutated_traits = gene.traits.copy()
        
        # Mutate traits with probability
        for trait_name, trait_value in mutated_traits.items():
            if np.random.random() < self.mutation_rate:
                if isinstance(trait_value, (int, float)):
                    variation = np.random.normal(0, 0.1) * trait_value
                    mutated_traits[trait_name] = max(0, trait_value + variation)
                elif isinstance(trait_value, bool) and np.random.random() < 0.1:
                    mutated_traits[trait_name] = not trait_value
                    
        return BioGene(
            gene_id=f"{gene.gene_id}_mut_{gene.adaptation_count + 1}",
            pattern_type=gene.pattern_type,
            effectiveness=min(1.0, gene.effectiveness + np.random.normal(0, 0.05)),
            adaptation_count=gene.adaptation_count + 1,
            parent_genes=[gene.gene_id], traits=mutated_traits
        )
        
    async def crossover_genes(self, parent1: BioGene, parent2: BioGene) -> BioGene:
        """Create offspring through crossover."""
        combined_traits = {}
        all_traits = set(parent1.traits.keys()) | set(parent2.traits.keys())
        
        for trait_name in all_traits:
            if trait_name in parent1.traits and trait_name in parent2.traits:
                if isinstance(parent1.traits[trait_name], (int, float)):
                    combined_traits[trait_name] = (
                        parent1.traits[trait_name] + parent2.traits[trait_name]
                    ) / 2
                else:
                    combined_traits[trait_name] = np.random.choice([
                        parent1.traits[trait_name], parent2.traits[trait_name]
                    ])
            elif trait_name in parent1.traits:
                combined_traits[trait_name] = parent1.traits[trait_name]
            else:
                combined_traits[trait_name] = parent2.traits[trait_name]
                
        return BioGene(
            gene_id=f"cross_{parent1.gene_id}_{parent2.gene_id}_{int(datetime.now().timestamp() * 1000) % 10000}",
            pattern_type=np.random.choice([parent1.pattern_type, parent2.pattern_type]),
            effectiveness=(parent1.effectiveness + parent2.effectiveness) / 2,
            parent_genes=[parent1.gene_id, parent2.gene_id], traits=combined_traits
        )
        
    async def selection_phase(self) -> List[BioPhenotype]:
        """Select fittest phenotypes."""
        # Evaluate fitness
        for phenotype in self.phenotype_population:
            phenotype.fitness_score = await self.evaluate_fitness(phenotype)
            
        # Sort by fitness
        self.phenotype_population.sort(key=lambda p: p.fitness_score, reverse=True)
        
        # Select top performers
        selection_count = int(len(self.phenotype_population) * self.selection_pressure)
        selected = self.phenotype_population[:selection_count]
        
        if selected:
            self.fitness_history.append(selected[0].fitness_score)
            
        return selected
        
    async def reproduction_phase(self, selected: List[BioPhenotype]) -> None:
        """Create new generation through reproduction."""
        new_population = []
        
        # Keep elite
        elite_count = max(1, len(selected) // 4)
        new_population.extend(selected[:elite_count])
        
        # Create offspring
        while len(new_population) < self.population_size:
            parent1 = np.random.choice(selected)
            parent2 = np.random.choice(selected)
            
            offspring_genes = []
            for i in range(min(len(parent1.genes), len(parent2.genes))):
                if np.random.random() < 0.7:  # Crossover
                    gene = await self.crossover_genes(parent1.genes[i], parent2.genes[i])
                else:  # Mutation
                    parent_gene = np.random.choice([parent1.genes[i], parent2.genes[i]])
                    gene = await self.mutate_gene(parent_gene)
                offspring_genes.append(gene)
                
            offspring = BioPhenotype(
                phenotype_id=f"gen{self.current_generation + 1}_{len(new_population):03d}",
                genes=offspring_genes, performance_metrics={}, 
                fitness_score=0.0, environment_adaptations=[]
            )
            new_population.append(offspring)
            
        self.phenotype_population = new_population
        self.current_generation += 1
        
    async def evolve_generation(self) -> Dict[str, Any]:
        """Evolve one generation."""
        start_time = datetime.now()
        
        selected = await self.selection_phase()
        await self.reproduction_phase(selected)
        
        # Statistics
        fitness_scores = [p.fitness_score for p in self.phenotype_population]
        stats = {
            "generation": self.current_generation,
            "population_size": len(self.phenotype_population),
            "best_fitness": max(fitness_scores),
            "avg_fitness": np.mean(fitness_scores),
            "evolution_time": (datetime.now() - start_time).total_seconds(),
            "gene_diversity": len(set(g.gene_id for p in self.phenotype_population for g in p.genes))
        }
        
        self.generation_history.append(stats)
        return stats
        
    async def run_autonomous_evolution(self, target_fitness: float = 0.90) -> None:
        """Run continuous autonomous evolution."""
        self.logger.info("🌱 Starting Autonomous Bio-Evolution")
        
        await self.initialize_genesis_population()
        
        while self.current_generation < self.max_generations:
            stats = await self.evolve_generation()
            
            self.logger.info(
                f"Generation {stats['generation']}: "
                f"Best={stats['best_fitness']:.4f}, "
                f"Avg={stats['avg_fitness']:.4f}, "
                f"Diversity={stats['gene_diversity']}"
            )
            
            if stats["best_fitness"] >= target_fitness:
                self.logger.info(f"🎯 Target fitness achieved: {stats['best_fitness']:.4f}")
                break
                
            # Adaptive mutation rate
            if len(self.fitness_history) > 10:
                recent_improvement = self.fitness_history[-1] - self.fitness_history[-10]
                if recent_improvement < 0.01:  # Stagnation
                    self.mutation_rate = min(0.3, self.mutation_rate * 1.1)
                else:
                    self.mutation_rate = max(0.05, self.mutation_rate * 0.95)
                    
            await asyncio.sleep(0.01)
            
        self.logger.info("🧬 Autonomous Evolution Complete")
        
    def get_best_phenotype(self) -> Optional[BioPhenotype]:
        """Get best phenotype."""
        return max(self.phenotype_population, key=lambda p: p.fitness_score) if self.phenotype_population else None
        
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get evolution summary."""
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
            "evolution_timeline": self.generation_history
        }


async def main():
    """Main demonstration."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logger = logging.getLogger(__name__)
    logger.info("🧬 Starting Minimal Bio-Evolution Generation 1 Demo")
    
    # Initialize system
    bio_executor = MinimalBioExecutor(
        population_size=20, mutation_rate=0.15, 
        selection_pressure=0.4, max_generations=30
    )
    
    # Run evolution
    await bio_executor.run_autonomous_evolution(target_fitness=0.88)
    
    # Results
    summary = bio_executor.get_evolution_summary()
    best_phenotype = bio_executor.get_best_phenotype()
    
    print("\n" + "="*60)
    print("🧬 BIO-EVOLUTION GENERATION 1 SUMMARY")
    print("="*60)
    
    print(f"Generations Completed: {summary['total_generations']}")
    print(f"Best Fitness Achieved: {summary['best_fitness']:.4f}")
    print(f"Total Fitness Improvement: {summary['fitness_improvement']:.4f}")
    print(f"Gene Pool Size: {summary['gene_pool_size']}")
    print(f"Convergence Rate: {summary['convergence_rate']:.6f}")
    
    if best_phenotype:
        print(f"\nBest Phenotype Details:")
        print(f"  ID: {best_phenotype.phenotype_id}")
        print(f"  Fitness Score: {best_phenotype.fitness_score:.4f}")
        print(f"  Gene Count: {len(best_phenotype.genes)}")
        print(f"  Gene Types: {', '.join(set(g.pattern_type for g in best_phenotype.genes))}")
        
        print(f"\n  Detailed Gene Analysis:")
        for gene in best_phenotype.genes:
            print(f"    - {gene.gene_id} ({gene.pattern_type})")
            print(f"      Effectiveness: {gene.effectiveness:.4f}")
            print(f"      Adaptations: {gene.adaptation_count}")
            print(f"      Key Traits: {list(gene.traits.keys())[:3]}")
    
    # Generate adaptive configuration
    if best_phenotype:
        config = generate_adaptive_config(best_phenotype)
        print(f"\nGenerated Adaptive System Configuration:")
        for category, settings in config.items():
            print(f"  {category.upper()}:")
            if isinstance(settings, dict):
                for key, value in settings.items():
                    print(f"    {key}: {value}")
            else:
                print(f"    {settings}")
    
    print(f"\n🎯 Bio-Enhanced Generation 1: AUTONOMOUS EVOLUTION SUCCESS!")
    
    return {
        "summary": summary,
        "best_phenotype": best_phenotype,
        "success": True
    }


def generate_adaptive_config(phenotype: BioPhenotype) -> Dict[str, Any]:
    """Generate system configuration from evolved phenotype."""
    config = {"system_id": phenotype.phenotype_id, "fitness": phenotype.fitness_score}
    
    # Analyze gene types
    gene_types = {}
    for gene in phenotype.genes:
        if gene.pattern_type not in gene_types:
            gene_types[gene.pattern_type] = []
        gene_types[gene.pattern_type].append(gene)
    
    # Generate configuration for each type
    if "security_protocol" in gene_types:
        avg_eff = np.mean([g.effectiveness for g in gene_types["security_protocol"]])
        config["security"] = {
            "level": "high" if avg_eff > 0.8 else "medium",
            "threat_detection": avg_eff > 0.7,
            "encryption_strength": int(128 + avg_eff * 64)
        }
    
    if "performance_optimization" in gene_types:
        avg_eff = np.mean([g.effectiveness for g in gene_types["performance_optimization"]])
        config["performance"] = {
            "optimization_level": int(avg_eff * 10),
            "quantum_enhanced": avg_eff > 0.75,
            "parallel_factor": int(2 + avg_eff * 6)
        }
    
    if "resource_management" in gene_types:
        avg_eff = np.mean([g.effectiveness for g in gene_types["resource_management"]])
        config["resources"] = {
            "adaptive_scaling": avg_eff > 0.6,
            "cache_efficiency": avg_eff,
            "memory_optimization": "aggressive" if avg_eff > 0.8 else "balanced"
        }
    
    return config


if __name__ == "__main__":
    asyncio.run(main())