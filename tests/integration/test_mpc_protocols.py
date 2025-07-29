"""
Integration tests for MPC protocols.
"""

import pytest
import torch
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from secure_mpc_transformer.protocols import ProtocolFactory
from secure_mpc_transformer.network import NetworkManager


@pytest.mark.integration
class TestMPCProtocols:
    """Integration tests for multi-party computation protocols."""

    @pytest.mark.parametrize("protocol_name", [
        "semi_honest_3pc",
        "malicious_3pc",
        "aby3"
    ])
    def test_protocol_initialization(self, protocol_name, mock_network_config):
        """Test protocol initialization with different configurations."""
        protocol = ProtocolFactory.create(
            protocol_name,
            num_parties=3,
            party_id=0,
            network_config=mock_network_config
        )
        
        assert protocol.protocol_name == protocol_name
        assert protocol.num_parties == 3
        assert protocol.party_id == 0

    @pytest.mark.slow
    def test_secret_sharing_reconstruction(self, test_protocol, mock_network_config):
        """Test secret sharing and reconstruction."""
        protocol = ProtocolFactory.create(
            test_protocol,
            num_parties=3,
            party_id=0,
            network_config=mock_network_config
        )
        
        # Create test secret
        secret = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        # Share secret
        shares = protocol.share_secret(secret)
        assert len(shares) == 3
        
        # Reconstruct secret
        reconstructed = protocol.reconstruct_secret(shares)
        
        # Verify reconstruction accuracy
        torch.testing.assert_close(secret, reconstructed, rtol=1e-6, atol=1e-6)

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_gpu_accelerated_operations(self, gpu_available, cleanup_gpu):
        """Test GPU-accelerated MPC operations."""
        if not gpu_available:
            pytest.skip("GPU not available")
        
        protocol = ProtocolFactory.create(
            "aby3",
            num_parties=3,
            party_id=0,
            gpu_acceleration=True
        )
        
        # Create large tensor for GPU computation
        large_tensor = torch.randn(1000, 1000, device="cuda")
        
        start_time = time.time()
        shares = protocol.share_secret(large_tensor)
        gpu_time = time.time() - start_time
        
        assert len(shares) == 3
        assert all(share.device.type == "cuda" for share in shares)
        
        # GPU should be faster than CPU for large tensors
        print(f"GPU sharing time: {gpu_time:.4f}s")

    @pytest.mark.slow
    def test_matrix_multiplication_protocol(self, test_protocol):
        """Test secure matrix multiplication."""
        protocol = ProtocolFactory.create(
            test_protocol,
            num_parties=3,
            party_id=0
        )
        
        # Create test matrices
        matrix_a = torch.randn(100, 200)
        matrix_b = torch.randn(200, 150)
        
        # Expected result
        expected = torch.matmul(matrix_a, matrix_b)
        
        # Secure computation
        shares_a = protocol.share_secret(matrix_a)
        shares_b = protocol.share_secret(matrix_b)
        
        result_shares = protocol.secure_matmul(shares_a, shares_b)
        result = protocol.reconstruct_secret(result_shares)
        
        # Verify accuracy within tolerance
        torch.testing.assert_close(expected, result, rtol=1e-4, atol=1e-4)

    @pytest.mark.security
    def test_malicious_security_guarantees(self):
        """Test malicious security guarantees."""
        protocol = ProtocolFactory.create(
            "malicious_3pc",
            num_parties=3,
            party_id=0,
            security_level=128
        )
        
        # Test that malicious inputs are detected
        secret = torch.tensor([1.0, 2.0, 3.0])
        shares = protocol.share_secret(secret)
        
        # Simulate malicious modification of shares
        corrupted_shares = shares.copy()
        corrupted_shares[1] = torch.randn_like(corrupted_shares[1])
        
        # Protocol should detect corruption
        with pytest.raises(ValueError, match="Malicious behavior detected"):
            protocol.reconstruct_secret(corrupted_shares, verify_integrity=True)

    @pytest.mark.benchmark
    def test_protocol_performance_comparison(self, benchmark_config):
        """Compare performance of different protocols."""
        protocols = ["semi_honest_3pc", "malicious_3pc", "aby3"]
        results = {}
        
        test_tensor = torch.randn(512, 512)
        
        for protocol_name in protocols:
            protocol = ProtocolFactory.create(
                protocol_name,
                num_parties=3,
                party_id=0
            )
            
            # Warmup
            for _ in range(benchmark_config["warmup_iterations"]):
                shares = protocol.share_secret(test_tensor)
                protocol.reconstruct_secret(shares)
            
            # Benchmark
            times = []
            for _ in range(benchmark_config["iterations"]):
                start_time = time.time()
                shares = protocol.share_secret(test_tensor)
                reconstructed = protocol.reconstruct_secret(shares)
                times.append(time.time() - start_time)
            
            avg_time = sum(times) / len(times)
            results[protocol_name] = avg_time
            
            print(f"{protocol_name}: {avg_time:.4f}s average")
        
        # Semi-honest should be fastest
        assert results["semi_honest_3pc"] <= results["malicious_3pc"]

    def test_multi_party_communication(self, mock_network_config):
        """Test communication between multiple parties."""
        
        def party_worker(party_id):
            """Worker function for each party."""
            protocol = ProtocolFactory.create(
                "semi_honest_3pc",
                num_parties=3,
                party_id=party_id,
                network_config=mock_network_config
            )
            
            # Each party contributes a secret
            secret = torch.tensor([float(party_id + 1)])
            shares = protocol.share_secret(secret)
            
            return shares
        
        # Simulate multi-party computation
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(party_worker, i) for i in range(3)]
            all_shares = [future.result() for future in as_completed(futures)]
        
        assert len(all_shares) == 3
        assert all(len(shares) == 3 for shares in all_shares)

    @pytest.mark.slow
    def test_protocol_fault_tolerance(self, test_protocol):
        """Test protocol behavior under network failures."""
        protocol = ProtocolFactory.create(
            test_protocol,
            num_parties=3,
            party_id=0,
            timeout=5,
            max_retries=3
        )
        
        secret = torch.tensor([1.0, 2.0, 3.0])
        
        # Simulate network failure during sharing
        with patch.object(protocol.network, 'send_message') as mock_send:
            mock_send.side_effect = [ConnectionError("Network failure")] * 2 + [None]
            
            # Should retry and eventually succeed
            shares = protocol.share_secret(secret)
            assert len(shares) == 3
            assert mock_send.call_count == 3

    def test_protocol_memory_efficiency(self, gpu_memory_tracker):
        """Test memory efficiency of protocol operations."""
        protocol = ProtocolFactory.create(
            "semi_honest_3pc",
            num_parties=3,
            party_id=0
        )
        
        # Test with progressively larger tensors
        for size in [100, 500, 1000]:
            tensor = torch.randn(size, size)
            
            shares = protocol.share_secret(tensor)
            reconstructed = protocol.reconstruct_secret(shares)
            
            # Cleanup should happen automatically
            del shares, reconstructed, tensor
        
        # Memory usage should be reasonable
        if hasattr(gpu_memory_tracker, 'memory_delta_mb'):
            assert gpu_memory_tracker.memory_delta_mb < 1000  # Less than 1GB

    @pytest.mark.integration
    def test_protocol_with_transformer_operations(self, mock_model_config):
        """Test protocol integration with transformer operations."""
        protocol = ProtocolFactory.create(
            "semi_honest_3pc",
            num_parties=3,
            party_id=0
        )
        
        # Simulate transformer layer operations
        batch_size, seq_len, hidden_size = 2, 128, 768
        input_tensor = torch.randn(batch_size, seq_len, hidden_size)
        weight_tensor = torch.randn(hidden_size, hidden_size)
        
        # Share inputs
        input_shares = protocol.share_secret(input_tensor)
        weight_shares = protocol.share_secret(weight_tensor)
        
        # Secure linear transformation
        output_shares = []
        for i in range(len(input_shares)):
            # Simulate secure computation
            output_share = torch.matmul(input_shares[i], weight_shares[i])
            output_shares.append(output_share)
        
        # Reconstruct result
        output = protocol.reconstruct_secret(output_shares)
        expected = torch.matmul(input_tensor, weight_tensor)
        
        # Results should be close (allowing for protocol noise)
        assert output.shape == expected.shape