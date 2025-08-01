"""
Sample data fixtures for testing secure MPC transformer inference.
"""

import torch
import numpy as np
from typing import Dict, List, Any
from pathlib import Path
import json


class SampleDataGenerator:
    """Generate sample data for testing."""
    
    @staticmethod
    def bert_tokenized_inputs(batch_size: int = 4, seq_len: int = 128) -> Dict[str, torch.Tensor]:
        """Generate sample BERT tokenized inputs."""
        return {
            "input_ids": torch.randint(1, 30000, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len),
            "token_type_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
        }
    
    @staticmethod
    def transformer_hidden_states(
        batch_size: int = 4, 
        seq_len: int = 128, 
        hidden_size: int = 768
    ) -> torch.Tensor:
        """Generate sample transformer hidden states."""
        return torch.randn(batch_size, seq_len, hidden_size)
    
    @staticmethod
    def attention_weights(
        batch_size: int = 4,
        num_heads: int = 12,
        seq_len: int = 128
    ) -> torch.Tensor:
        """Generate sample attention weights."""
        weights = torch.randn(batch_size, num_heads, seq_len, seq_len)
        return torch.softmax(weights, dim=-1)
    
    @staticmethod
    def model_weights(hidden_size: int = 768) -> Dict[str, torch.Tensor]:
        """Generate sample model weights."""
        return {
            "query_weight": torch.randn(hidden_size, hidden_size),
            "key_weight": torch.randn(hidden_size, hidden_size),
            "value_weight": torch.randn(hidden_size, hidden_size),
            "output_weight": torch.randn(hidden_size, hidden_size),
            "ffn_weight_1": torch.randn(hidden_size, hidden_size * 4),
            "ffn_weight_2": torch.randn(hidden_size * 4, hidden_size),
        }
    
    @staticmethod
    def secret_shares(
        secret: torch.Tensor, 
        num_parties: int = 3
    ) -> List[torch.Tensor]:
        """Generate additive secret shares."""
        shares = []
        cumsum = torch.zeros_like(secret)
        
        for i in range(num_parties - 1):
            share = torch.randn_like(secret)
            shares.append(share)
            cumsum += share
        
        # Last share ensures sum equals original secret
        shares.append(secret - cumsum)
        return shares
    
    @staticmethod
    def encryption_parameters() -> Dict[str, Any]:
        """Generate sample encryption parameters."""
        return {
            "poly_modulus_degree": 16384,
            "coeff_modulus_bits": [60, 40, 40, 60],
            "plain_modulus": 1024,
            "security_level": 128,
        }


class TestDatasets:
    """Predefined test datasets."""
    
    TEXT_SAMPLES = [
        "The quick brown fox jumps over the lazy dog.",
        "Secure multi-party computation enables privacy-preserving machine learning.",
        "BERT model inference with homomorphic encryption and GPU acceleration.",
        "Transformer architectures revolutionized natural language processing.",
        "Privacy-preserving AI systems protect sensitive data during computation.",
        "Multi-party protocols enable collaborative machine learning without data sharing.",
        "GPU kernels accelerate cryptographic operations for practical deployment.",
        "Differential privacy provides mathematical guarantees for data protection."
    ]
    
    CLASSIFICATION_LABELS = [
        "positive", "negative", "neutral", "unknown"
    ]
    
    NER_LABELS = [
        "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"
    ]
    
    @classmethod
    def get_classification_dataset(cls, num_samples: int = 100) -> List[Dict[str, Any]]:
        """Generate classification dataset."""
        np.random.seed(42)
        dataset = []
        
        for i in range(num_samples):
            text = np.random.choice(cls.TEXT_SAMPLES)
            label = np.random.choice(cls.CLASSIFICATION_LABELS)
            
            dataset.append({
                "id": i,
                "text": text,
                "label": label,
                "confidence": np.random.uniform(0.7, 1.0)
            })
        
        return dataset
    
    @classmethod
    def get_ner_dataset(cls, num_samples: int = 50) -> List[Dict[str, Any]]:
        """Generate NER dataset."""
        np.random.seed(42)
        dataset = []
        
        for i in range(num_samples):
            text = np.random.choice(cls.TEXT_SAMPLES)
            tokens = text.split()
            labels = [np.random.choice(cls.NER_LABELS) for _ in tokens]
            
            dataset.append({
                "id": i,
                "tokens": tokens,
                "labels": labels
            })
        
        return dataset


def save_test_fixtures(output_dir: Path):
    """Save test fixtures to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate and save sample data
    generator = SampleDataGenerator()
    
    # BERT inputs
    bert_inputs = generator.bert_tokenized_inputs(batch_size=8, seq_len=256)
    torch.save(bert_inputs, output_dir / "bert_inputs.pt")
    
    # Hidden states
    hidden_states = generator.transformer_hidden_states(batch_size=8, seq_len=256)
    torch.save(hidden_states, output_dir / "hidden_states.pt")
    
    # Model weights
    weights = generator.model_weights()
    torch.save(weights, output_dir / "model_weights.pt")
    
    # Datasets
    classification_data = TestDatasets.get_classification_dataset(200)
    with open(output_dir / "classification_dataset.json", "w") as f:
        json.dump(classification_data, f, indent=2)
    
    ner_data = TestDatasets.get_ner_dataset(100)
    with open(output_dir / "ner_dataset.json", "w") as f:
        json.dump(ner_data, f, indent=2)
    
    print(f"Test fixtures saved to {output_dir}")


if __name__ == "__main__":
    # Generate fixtures when run directly
    fixtures_dir = Path(__file__).parent
    save_test_fixtures(fixtures_dir)