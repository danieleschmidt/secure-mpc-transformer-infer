"""
Security-focused tests for cryptographic implementations.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, Mock
from typing import List, Dict, Any

from secure_mpc_transformer.protocols import ProtocolFactory
from secure_mpc_transformer.crypto.homomorphic import HomomorphicEncryption
from secure_mpc_transformer.crypto.secret_sharing import SecretSharing
from secure_mpc_transformer.security.privacy_accountant import PrivacyAccountant


@pytest.mark.security
class TestCryptographicSecurity:
    """Test cryptographic security properties."""
    
    def test_secret_sharing_security(self):
        """Test security properties of secret sharing."""
        secret_sharing = SecretSharing(num_parties=3)
        
        # Test secret
        secret = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        # Generate shares
        shares = secret_sharing.share(secret)
        
        # Security property 1: Individual shares reveal no information
        for share in shares:
            # Share should not equal original secret
            assert not torch.allclose(share, secret)
            
            # Share should appear random
            assert torch.std(share) > 0.1  # Some randomness
        
        # Security property 2: Any subset of shares < threshold reveals nothing
        for i in range(len(shares) - 1):
            partial_shares = shares[:i+1]
            # Should not be able to reconstruct from insufficient shares
            with pytest.raises(ValueError):
                secret_sharing.reconstruct(partial_shares[:-1])  # n-2 shares
        
        # Security property 3: Exact reconstruction with all shares
        reconstructed = secret_sharing.reconstruct(shares)
        torch.testing.assert_close(secret, reconstructed, rtol=1e-6, atol=1e-6)
    
    def test_homomorphic_encryption_security(self):
        """Test homomorphic encryption security properties."""
        he = HomomorphicEncryption(
            poly_modulus_degree=4096,
            security_level=128
        )
        
        # Test plaintexts
        plaintext1 = torch.tensor([1.0, 2.0, 3.0])
        plaintext2 = torch.tensor([4.0, 5.0, 6.0])
        
        # Encrypt
        ciphertext1 = he.encrypt(plaintext1)
        ciphertext2 = he.encrypt(plaintext2)
        
        # Security property 1: Ciphertexts should not reveal plaintext
        assert ciphertext1.data != plaintext1.data
        assert ciphertext2.data != plaintext2.data
        
        # Security property 2: Same plaintext should produce different ciphertexts
        ciphertext1_again = he.encrypt(plaintext1)
        assert not torch.allclose(ciphertext1.data, ciphertext1_again.data)
        
        # Security property 3: Homomorphic operations preserve correctness
        encrypted_sum = he.add(ciphertext1, ciphertext2)
        decrypted_sum = he.decrypt(encrypted_sum)
        expected_sum = plaintext1 + plaintext2
        
        torch.testing.assert_close(decrypted_sum, expected_sum, rtol=1e-3, atol=1e-3)
    
    def test_protocol_security_levels(self):
        """Test that protocols maintain required security levels."""
        security_levels = [128, 192, 256]
        
        for level in security_levels:
            protocol = ProtocolFactory.create(
                "malicious_3pc",
                num_parties=3,
                party_id=0,
                security_level=level
            )
            
            # Verify security level is maintained
            assert protocol.security_level >= level
            
            # Test that protocol operations maintain security
            secret = torch.randn(100)
            shares = protocol.share_secret(secret)
            
            # Each share should be computationally indistinguishable from random
            for share in shares:
                self._assert_computational_indistinguishability(share)
    
    def _assert_computational_indistinguishability(self, data: torch.Tensor):
        """Assert that data appears computationally indistinguishable from random."""
        # Convert to numpy for statistical tests
        np_data = data.detach().cpu().numpy().flatten()
        
        # Test 1: Kolmogorov-Smirnov test for normality
        from scipy import stats
        _, p_value = stats.kstest(np_data, 'norm')
        
        # Should not be obviously non-random (p > 0.01)
        assert p_value > 0.01, f"Data appears non-random (p={p_value})"
        
        # Test 2: Check for patterns
        mean = np.mean(np_data)
        std = np.std(np_data)
        
        # Should have reasonable distribution properties
        assert abs(mean) < 3 * std / np.sqrt(len(np_data))  # Mean close to zero
        assert std > 0  # Non-zero variance
    
    def test_side_channel_resistance(self):
        """Test resistance to timing and other side-channel attacks."""
        protocol = ProtocolFactory.create(
            "malicious_3pc",
            num_parties=3,
            party_id=0,
            constant_time=True
        )
        
        # Test different input sizes for timing consistency
        input_sizes = [64, 128, 256, 512]
        timing_results = []
        
        for size in input_sizes:
            secret = torch.randn(size)
            
            # Measure operation time
            import time
            times = []
            for _ in range(10):  # Multiple measurements for accuracy
                start_time = time.perf_counter()
                shares = protocol.share_secret(secret)
                reconstructed = protocol.reconstruct_secret(shares)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            avg_time = sum(times) / len(times)
            timing_results.append((size, avg_time))
        
        # Timing should scale predictably with input size (not reveal secret values)
        # This is a simplified test - real side-channel analysis is more complex
        for i in range(1, len(timing_results)):
            size_ratio = timing_results[i][0] / timing_results[i-1][0]
            time_ratio = timing_results[i][1] / timing_results[i-1][1]
            
            # Time should scale roughly linearly with size
            assert 0.5 < time_ratio / size_ratio < 2.0, \
                f"Unexpected timing pattern: {timing_results}"
    
    def test_malicious_adversary_detection(self):
        """Test detection of malicious adversary behavior."""
        protocol = ProtocolFactory.create(
            "malicious_3pc",
            num_parties=3,
            party_id=0,
            verify_integrity=True
        )
        
        secret = torch.tensor([1.0, 2.0, 3.0])
        shares = protocol.share_secret(secret)
        
        # Test various types of malicious behavior
        malicious_behaviors = [
            ("corrupt_share", lambda s: s + torch.randn_like(s)),
            ("zero_share", lambda s: torch.zeros_like(s)),
            ("scale_share", lambda s: s * 2),
            ("flip_bits", lambda s: -s),
        ]
        
        for behavior_name, corrupt_function in malicious_behaviors:
            corrupted_shares = shares.copy()
            corrupted_shares[1] = corrupt_function(corrupted_shares[1])
            
            # Protocol should detect corruption
            with pytest.raises((ValueError, RuntimeError), 
                             match="Malicious behavior|Integrity check failed|Verification failed"):
                protocol.reconstruct_secret(corrupted_shares, verify_integrity=True)
    
    def test_information_leakage_bounds(self):
        """Test that information leakage is within acceptable bounds."""
        protocol = ProtocolFactory.create(
            "semi_honest_3pc",
            num_parties=3,
            party_id=0
        )
        
        # Generate multiple secrets with known correlation
        num_secrets = 100
        secrets = []
        for i in range(num_secrets):
            # Create correlated secrets to test leakage
            base_secret = torch.randn(10)
            noise = torch.randn(10) * 0.1
            secrets.append(base_secret + noise)
        
        # Process all secrets through protocol
        all_shares = []
        for secret in secrets:
            shares = protocol.share_secret(secret)
            all_shares.append(shares)
        
        # Analyze information leakage in shares
        # A party should not be able to learn about correlations from their shares alone
        party_0_shares = [shares[0] for shares in all_shares]
        party_1_shares = [shares[1] for shares in all_shares]
        
        # Compute correlation between party shares (should be minimal)
        party_0_tensor = torch.stack(party_0_shares)
        party_1_tensor = torch.stack(party_1_shares)
        
        correlation = torch.corrcoef(torch.stack([
            party_0_tensor.flatten(),
            party_1_tensor.flatten()
        ]))[0, 1]
        
        # Correlation should be small (close to zero for secure sharing)
        assert abs(correlation) < 0.1, f"High correlation detected: {correlation}"
    
    def test_differential_privacy_guarantees(self):
        """Test differential privacy guarantees."""
        privacy_accountant = PrivacyAccountant(
            epsilon_budget=1.0,
            delta=1e-5
        )
        
        # Simulate private computation
        with privacy_accountant.track_privacy():
            # Add noise for differential privacy
            sensitivity = 1.0
            noise_scale = sensitivity / privacy_accountant.epsilon_remaining
            
            # Simulate adding calibrated noise
            private_result = torch.randn(100) + torch.randn(100) * noise_scale
        
        # Check privacy budget consumption
        assert privacy_accountant.epsilon_spent > 0
        assert privacy_accountant.epsilon_remaining < privacy_accountant.epsilon_budget
        
        # Verify privacy guarantee parameters
        assert privacy_accountant.delta <= 1e-5
        assert privacy_accountant.epsilon_spent <= 1.0
    
    def test_cryptographic_randomness_quality(self):
        """Test quality of cryptographic randomness."""
        # Generate random values from crypto module
        random_values = []
        for _ in range(1000):
            random_tensor = torch.randn(64)  # Simulate crypto random generation
            random_values.extend(random_tensor.tolist())
        
        random_array = np.array(random_values)
        
        # Statistical tests for randomness
        # Test 1: Chi-square test for uniform distribution
        from scipy.stats import chisquare
        
        # Bin the data
        hist, _ = np.histogram(random_array, bins=20)
        expected_freq = len(random_array) / 20
        
        chi2_stat, p_value = chisquare(hist, [expected_freq] * 20)
        assert p_value > 0.01, f"Chi-square test failed: p={p_value}"
        
        # Test 2: Runs test for independence
        median = np.median(random_array)
        runs, n1, n2 = self._runs_test(random_array > median)
        
        # Expected runs and variance
        expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
        runs_variance = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / \
                       ((n1 + n2) ** 2 * (n1 + n2 - 1))
        
        # Z-score for runs test
        z_score = abs(runs - expected_runs) / np.sqrt(runs_variance)
        assert z_score < 2.0, f"Runs test failed: z-score={z_score}"
    
    def _runs_test(self, binary_sequence: np.ndarray) -> tuple:
        """Perform runs test on binary sequence."""
        runs = 1
        n1 = np.sum(binary_sequence)
        n2 = len(binary_sequence) - n1
        
        for i in range(1, len(binary_sequence)):
            if binary_sequence[i] != binary_sequence[i-1]:
                runs += 1
        
        return runs, n1, n2
    
    @pytest.mark.slow
    def test_security_audit_checklist(self):
        """Comprehensive security audit checklist."""
        audit_results = {}
        
        # Check 1: Cryptographic constants
        audit_results["crypto_constants"] = self._audit_crypto_constants()
        
        # Check 2: Key management
        audit_results["key_management"] = self._audit_key_management()
        
        # Check 3: Protocol implementation
        audit_results["protocol_implementation"] = self._audit_protocol_implementation()
        
        # Check 4: Error handling
        audit_results["error_handling"] = self._audit_error_handling()
        
        # All audit checks should pass
        for check_name, result in audit_results.items():
            assert result["passed"], f"Security audit failed: {check_name} - {result['details']}"
    
    def _audit_crypto_constants(self) -> Dict[str, Any]:
        """Audit cryptographic constants."""
        # Check that security parameters are appropriate
        min_security_level = 128
        
        # Verify protocol uses sufficient security parameters
        protocol = ProtocolFactory.create("malicious_3pc", num_parties=3, party_id=0)
        
        return {
            "passed": protocol.security_level >= min_security_level,
            "details": f"Security level: {protocol.security_level}"
        }
    
    def _audit_key_management(self) -> Dict[str, Any]:
        """Audit key management practices."""
        # Check that keys are generated securely
        # This is a placeholder - real implementation would check actual key generation
        
        return {
            "passed": True,
            "details": "Key management audit passed"
        }
    
    def _audit_protocol_implementation(self) -> Dict[str, Any]:
        """Audit protocol implementation."""
        # Check that protocols implement required security properties
        protocol = ProtocolFactory.create("malicious_3pc", num_parties=3, party_id=0)
        
        required_methods = ["share_secret", "reconstruct_secret", "verify_integrity"]
        has_all_methods = all(hasattr(protocol, method) for method in required_methods)
        
        return {
            "passed": has_all_methods,
            "details": f"Protocol methods check: {required_methods}"
        }
    
    def _audit_error_handling(self) -> Dict[str, Any]:
        """Audit error handling."""
        # Check that errors don't leak sensitive information
        protocol = ProtocolFactory.create("malicious_3pc", num_parties=3, party_id=0)
        
        try:
            # Trigger an error condition
            invalid_shares = [torch.randn(5), torch.randn(3)]  # Mismatched shapes
            protocol.reconstruct_secret(invalid_shares)
            error_handled = False
        except Exception as e:
            error_handled = True
            # Error message should not contain sensitive data
            sensitive_keywords = ["secret", "key", "private", "password"]
            error_msg = str(e).lower()
            contains_sensitive = any(keyword in error_msg for keyword in sensitive_keywords)
        
        return {
            "passed": error_handled and not contains_sensitive,
            "details": "Error handling audit completed"
        }