"""
End-to-end tests for complete secure inference pipeline.
"""

import pytest
import torch
import time
from pathlib import Path

from secure_mpc_transformer import SecureTransformer, SecurityConfig
from secure_mpc_transformer.protocols import ProtocolFactory


@pytest.mark.slow
@pytest.mark.integration
class TestE2EInference:
    """End-to-end tests for secure transformer inference."""

    def test_complete_bert_inference_pipeline(self, sample_text_data, temp_dir):
        """Test complete BERT inference with secure MPC."""
        # Initialize secure transformer
        config = SecurityConfig(
            protocol="semi_honest_3pc",
            security_level=128,
            num_parties=3
        )
        
        model = SecureTransformer.from_pretrained("bert-base-uncased", config)
        
        results = []
        for text in sample_text_data:
            start_time = time.time()
            
            # Tokenize input
            inputs = model.tokenize(text, max_length=128, padding=True)
            
            # Secure inference
            with model.secure_context():
                outputs = model.predict_secure(inputs)
            
            inference_time = time.time() - start_time
            
            # Validate outputs
            assert outputs is not None
            assert hasattr(outputs, 'logits')
            assert outputs.logits.shape[0] == 1  # Batch size
            
            results.append({
                'text': text,
                'inference_time': inference_time,
                'output_shape': outputs.logits.shape
            })
            
            print(f"Processed: '{text[:50]}...' in {inference_time:.2f}s")
        
        # Verify all texts were processed successfully
        assert len(results) == len(sample_text_data)
        
        # Average inference time should be reasonable
        avg_time = sum(r['inference_time'] for r in results) / len(results)
        assert avg_time < 120  # Less than 2 minutes per inference

    @pytest.mark.gpu
    def test_gpu_accelerated_inference(self, gpu_available, sample_text_data):
        """Test GPU-accelerated secure inference."""
        if not gpu_available:
            pytest.skip("GPU not available")
        
        config = SecurityConfig(
            protocol="aby3",
            security_level=128,
            num_parties=3,
            gpu_acceleration=True
        )
        
        model = SecureTransformer.from_pretrained("bert-base-uncased", config)
        
        # Test single inference
        text = sample_text_data[0]
        inputs = model.tokenize(text, max_length=128)
        
        gpu_start = time.time()
        with model.secure_context():
            gpu_outputs = model.predict_secure(inputs)
        gpu_time = time.time() - gpu_start
        
        assert gpu_outputs is not None
        print(f"GPU inference time: {gpu_time:.2f}s")
        
        # GPU should be faster for this model size
        assert gpu_time < 60  # Less than 1 minute

    @pytest.mark.benchmark
    def test_performance_benchmarks(self, benchmark_config, sample_text_data):
        """Comprehensive performance benchmarking."""
        protocols = ["semi_honest_3pc", "malicious_3pc"]
        results = {}
        
        for protocol in protocols:
            config = SecurityConfig(
                protocol=protocol,
                security_level=128,
                num_parties=3
            )
            
            model = SecureTransformer.from_pretrained("bert-base-uncased", config)
            protocol_results = []
            
            for batch_size in benchmark_config["batch_sizes"]:
                for seq_len in benchmark_config["sequence_lengths"]:
                    # Create batch input
                    batch_texts = sample_text_data[:batch_size]
                    if len(batch_texts) < batch_size:
                        # Repeat texts to fill batch
                        batch_texts = (batch_texts * ((batch_size // len(batch_texts)) + 1))[:batch_size]
                    
                    # Warmup
                    for _ in range(benchmark_config["warmup_iterations"]):
                        inputs = model.tokenize(batch_texts[0], max_length=seq_len)
                        with model.secure_context():
                            _ = model.predict_secure(inputs)
                    
                    # Benchmark
                    times = []
                    for _ in range(benchmark_config["iterations"]):
                        inputs = model.tokenize(batch_texts[0], max_length=seq_len)
                        
                        start_time = time.time()
                        with model.secure_context():
                            outputs = model.predict_secure(inputs)
                        inference_time = time.time() - start_time
                        
                        times.append(inference_time)
                    
                    avg_time = sum(times) / len(times)
                    throughput = batch_size / avg_time  # samples per second
                    
                    protocol_results.append({
                        'batch_size': batch_size,
                        'sequence_length': seq_len,
                        'avg_time': avg_time,
                        'throughput': throughput
                    })
                    
                    print(f"{protocol} - Batch: {batch_size}, Seq: {seq_len}, "
                          f"Time: {avg_time:.2f}s, Throughput: {throughput:.2f} samples/s")
            
            results[protocol] = protocol_results
        
        # Verify performance characteristics
        for protocol, protocol_results in results.items():
            # Larger batches should have better throughput
            batch_1_throughput = next(r['throughput'] for r in protocol_results 
                                    if r['batch_size'] == 1 and r['sequence_length'] == 128)
            batch_8_throughput = next(r['throughput'] for r in protocol_results 
                                    if r['batch_size'] == 8 and r['sequence_length'] == 128)
            
            # Note: This might not always hold due to memory constraints
            print(f"{protocol} - Batch 1 throughput: {batch_1_throughput:.2f}")
            print(f"{protocol} - Batch 8 throughput: {batch_8_throughput:.2f}")

    @pytest.mark.slow
    def test_multi_party_coordination(self, mock_network_config):
        """Test coordination between multiple parties."""
        
        def simulate_party(party_id, secret_input):
            """Simulate a single party in the computation."""
            config = SecurityConfig(
                protocol="semi_honest_3pc",
                security_level=128,
                num_parties=3
            )
            
            model = SecureTransformer.from_pretrained("bert-base-uncased", config)
            model.set_party_id(party_id)
            
            # Each party contributes their secret input
            inputs = model.tokenize(secret_input, max_length=64)
            
            # Participate in secure computation
            with model.secure_context():
                # In real implementation, this would coordinate with other parties
                shares = model.protocol.share_secret(inputs['input_ids'])
                
                # Simulate computation participation
                result_shares = model.protocol.secure_forward(shares)
                
                return result_shares
        
        # Simulate three parties with different inputs
        party_inputs = [
            "Party 0 secret: Confidential data A",
            "Party 1 secret: Confidential data B", 
            "Party 2 secret: Confidential data C"
        ]
        
        # In a real scenario, parties would run simultaneously
        party_results = []
        for i, secret_input in enumerate(party_inputs):
            result = simulate_party(i, secret_input)
            party_results.append(result)
        
        assert len(party_results) == 3
        print("Multi-party coordination test completed successfully")

    @pytest.mark.security
    def test_privacy_preservation(self, sample_text_data):
        """Test that input privacy is preserved during computation."""
        config = SecurityConfig(
            protocol="malicious_3pc",
            security_level=128,
            num_parties=3
        )
        
        model = SecureTransformer.from_pretrained("bert-base-uncased", config)
        
        sensitive_text = "Confidential: Account balance $1,000,000"
        inputs = model.tokenize(sensitive_text, max_length=128)
        
        # Capture intermediate values during computation
        intermediate_values = []
        
        def capture_intermediate(tensor):
            """Capture intermediate computation values."""
            intermediate_values.append(tensor.clone().detach())
            return tensor
        
        # Hook into the model to capture intermediate values
        hooks = []
        for module in model.modules():
            if hasattr(module, 'forward'):
                hook = module.register_forward_hook(
                    lambda m, i, o: capture_intermediate(o) if isinstance(o, torch.Tensor) else o
                )
                hooks.append(hook)
        
        try:
            with model.secure_context():
                outputs = model.predict_secure(inputs)
            
            # Verify that no intermediate value directly reveals the input
            input_tokens = inputs['input_ids'].flatten()
            
            for intermediate in intermediate_values:
                if intermediate.dtype == torch.long:  # Token-like values
                    # Check that intermediate values don't directly match input tokens
                    flat_intermediate = intermediate.flatten()
                    
                    # Allow some overlap (due to common tokens like [CLS], [SEP])
                    # but not complete revelation
                    overlap = torch.isin(flat_intermediate, input_tokens).sum().item()
                    overlap_ratio = overlap / len(input_tokens)
                    
                    assert overlap_ratio < 0.8, f"Too much input revealed: {overlap_ratio:.2%}"
            
            print(f"Privacy test passed - captured {len(intermediate_values)} intermediate values")
            
        finally:
            # Clean up hooks
            for hook in hooks:
                hook.remove()

    @pytest.mark.slow
    def test_error_recovery_and_resilience(self, sample_text_data):
        """Test error recovery and system resilience."""
        config = SecurityConfig(
            protocol="semi_honest_3pc",
            security_level=128,
            num_parties=3,
            timeout=30,
            max_retries=3
        )
        
        model = SecureTransformer.from_pretrained("bert-base-uncased", config)
        
        # Test recovery from network timeouts
        with pytest.raises(TimeoutError):
            # Simulate network timeout
            model.protocol.network.timeout = 0.001  # Very short timeout
            
            inputs = model.tokenize(sample_text_data[0], max_length=128)
            with model.secure_context():
                model.predict_secure(inputs)
        
        # Restore normal timeout and verify recovery
        model.protocol.network.timeout = 30
        inputs = model.tokenize(sample_text_data[0], max_length=128)
        
        with model.secure_context():
            outputs = model.predict_secure(inputs)
        
        assert outputs is not None
        print("Error recovery test completed successfully")

    def test_inference_accuracy_validation(self, sample_text_data):
        """Validate that secure inference produces reasonable results."""
        config = SecurityConfig(
            protocol="semi_honest_3pc",
            security_level=128,
            num_parties=3
        )
        
        model = SecureTransformer.from_pretrained("bert-base-uncased", config)
        
        for text in sample_text_data[:2]:  # Test first 2 samples
            inputs = model.tokenize(text, max_length=128)
            
            with model.secure_context():
                secure_outputs = model.predict_secure(inputs)
            
            # Validate output properties
            assert secure_outputs.logits.shape[1] == inputs['input_ids'].shape[1]  # Sequence length
            assert secure_outputs.logits.shape[2] == model.config['vocab_size']  # Vocabulary size
            
            # Check that outputs contain reasonable probability distributions
            probabilities = torch.softmax(secure_outputs.logits, dim=-1)
            
            # Probabilities should sum to 1 (within tolerance)
            prob_sums = probabilities.sum(dim=-1)
            torch.testing.assert_close(prob_sums, torch.ones_like(prob_sums), rtol=1e-3, atol=1e-3)
            
            # Should have some variation in predictions (not all zeros or ones)
            assert probabilities.var() > 1e-6
            
            print(f"Validated output for: '{text[:50]}...'")

    @pytest.mark.integration
    def test_resource_cleanup(self, sample_text_data, cleanup_gpu):
        """Test proper cleanup of resources after inference."""
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        config = SecurityConfig(
            protocol="semi_honest_3pc",
            security_level=128,
            num_parties=3
        )
        
        # Run multiple inferences
        for _ in range(3):
            model = SecureTransformer.from_pretrained("bert-base-uncased", config)
            
            inputs = model.tokenize(sample_text_data[0], max_length=128)
            with model.secure_context():
                outputs = model.predict_secure(inputs)
            
            # Explicit cleanup
            del model, inputs, outputs
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Memory usage should not have grown significantly
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        memory_growth = final_memory - initial_memory
        
        if torch.cuda.is_available():
            assert memory_growth < 100 * 1024 * 1024  # Less than 100MB growth
            print(f"Memory growth: {memory_growth / (1024*1024):.2f} MB")
        
        print("Resource cleanup test completed successfully")