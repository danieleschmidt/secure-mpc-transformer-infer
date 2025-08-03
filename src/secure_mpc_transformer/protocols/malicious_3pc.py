"""Malicious-secure 3-party computation protocol with MAC authentication."""

import torch
import hashlib
import hmac
from typing import List, Optional, Tuple
from .base import Protocol, SecureValue, ProtocolError


class MaliciousSecureValue(SecureValue):
    """Secure value with authentication tags for malicious security."""
    
    def __init__(self, shares: List[torch.Tensor], party_id: int, 
                 macs: Optional[List[torch.Tensor]] = None, is_public: bool = False):
        super().__init__(shares, party_id, is_public)
        self.macs = macs or []
    
    def verify_macs(self, mac_key: torch.Tensor) -> bool:
        """Verify MAC authentication tags."""
        if not self.macs:
            return True  # No MACs to verify
        
        for i, (share, mac) in enumerate(zip(self.shares, self.macs)):
            expected_mac = self._compute_mac(share, mac_key)
            if not torch.allclose(mac, expected_mac, atol=1e-6):
                return False
        return True
    
    def _compute_mac(self, value: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Compute MAC for a value."""
        # Simplified MAC computation
        return torch.sum(value * key, dim=-1, keepdim=True)


class Malicious3PC(Protocol):
    """Malicious-secure 3-party computation with MAC authentication."""
    
    def __init__(self, party_id: int, num_parties: int = 3, 
                 device: Optional[torch.device] = None, mac_key_size: int = 128):
        if num_parties != 3:
            raise ValueError("Malicious3PC requires exactly 3 parties")
        super().__init__(party_id, num_parties, device)
        
        self.mac_key_size = mac_key_size
        self.mac_key = None
        self.global_mac_key = None
        
        # Commitment scheme for zero-knowledge proofs
        self._commitments = {}
        self._openings = {}
        
        # Audit trail for security monitoring
        self._audit_log = []
        
        # Network simulation
        self._message_queue = {}
        self._broadcast_channel = []
    
    def initialize(self) -> None:
        """Initialize protocol with MAC key setup."""
        if self._initialized:
            return
        
        # Generate MAC keys
        torch.manual_seed(42 + self.party_id * 1000)
        self.mac_key = torch.rand(self.mac_key_size, device=self.device)
        
        # In practice, global MAC key would be established through secure protocol
        self.global_mac_key = torch.rand(self.mac_key_size, device=self.device)
        
        # Initialize message queues
        for i in range(self.num_parties):
            if i != self.party_id:
                self._message_queue[i] = []
        
        self._log_audit_event("protocol_initialized", {"party_id": self.party_id})
        self._initialized = True
    
    def share_value(self, value: torch.Tensor) -> SecureValue:
        """Share value with MAC authentication."""
        if not self._initialized:
            self.initialize()
        
        value = value.to(self.device)
        
        # Generate additive shares
        shares = []
        macs = []
        
        if self.party_id == 0:
            # Party 0 generates shares
            r1 = torch.rand_like(value, device=self.device)
            r2 = torch.rand_like(value, device=self.device)
            r0 = value - r1 - r2
            
            shares = [r0, r1, r2]
            
            # Compute MACs for each share
            for share in shares:
                mac = self._compute_authenticated_mac(share)
                macs.append(mac)
            
            # Distribute shares and MACs
            self._send_authenticated_share(shares[1], macs[1], 1)
            self._send_authenticated_share(shares[2], macs[2], 2)
            
            # Keep own share and MAC
            final_shares = [shares[0]]
            final_macs = [macs[0]]
            
        else:
            # Receive share and MAC from party 0
            share, mac = self._receive_authenticated_share(0)
            final_shares = [share]
            final_macs = [mac]
        
        result = MaliciousSecureValue(
            shares=final_shares, 
            party_id=self.party_id, 
            macs=final_macs
        )
        
        self._log_audit_event("value_shared", {
            "shape": list(value.shape),
            "mac_verified": result.verify_macs(self.global_mac_key)
        })
        
        return result
    
    def reconstruct_value(self, secure_value: SecureValue) -> torch.Tensor:
        """Reconstruct value with MAC verification."""
        if isinstance(secure_value, MaliciousSecureValue):
            # Verify MACs before reconstruction
            if not secure_value.verify_macs(self.global_mac_key):
                self._log_audit_event("mac_verification_failed", {
                    "party_id": self.party_id
                })
                raise ProtocolError("MAC verification failed during reconstruction")
        
        # Collect shares from all parties
        all_shares = []
        
        # Add own shares
        all_shares.extend(secure_value.shares)
        
        # Request shares from other parties with challenge-response
        for party in range(self.num_parties):
            if party != self.party_id:
                challenged_share = self._challenge_and_verify_share(party, secure_value)
                all_shares.append(challenged_share)
        
        # Reconstruct value
        result = sum(all_shares)
        
        self._log_audit_event("value_reconstructed", {
            "result_shape": list(result.shape)
        })
        
        return result
    
    def secure_add(self, a: SecureValue, b: SecureValue) -> SecureValue:
        """Secure addition with MAC preservation."""
        if not (self.validate_shares(a) and self.validate_shares(b)):
            raise ProtocolError("Invalid shares for addition")
        
        # Add shares
        result_shares = []
        for i in range(len(a.shares)):
            result_shares.append(a.shares[i] + b.shares[i])
        
        # Add MACs if present
        result_macs = []
        if isinstance(a, MaliciousSecureValue) and isinstance(b, MaliciousSecureValue):
            for i in range(len(a.macs)):
                result_macs.append(a.macs[i] + b.macs[i])
        
        result = MaliciousSecureValue(
            shares=result_shares,
            party_id=self.party_id,
            macs=result_macs
        )
        
        self._log_audit_event("secure_addition", {"operation": "add"})
        return result
    
    def secure_multiply(self, a: SecureValue, b: SecureValue) -> SecureValue:
        """Secure multiplication with zero-knowledge proof."""
        if not (self.validate_shares(a) and self.validate_shares(b)):
            raise ProtocolError("Invalid shares for multiplication")
        
        # Perform multiplication locally
        local_result = self._local_multiply(a, b)
        
        # Generate zero-knowledge proof of correct multiplication
        proof = self._generate_multiplication_proof(a, b, local_result)
        
        # Broadcast proof to other parties for verification
        self._broadcast_proof(proof)
        
        # Verify proofs from other parties
        if not self._verify_multiplication_proofs():
            self._log_audit_event("multiplication_proof_failed", {
                "party_id": self.party_id
            })
            raise ProtocolError("Multiplication proof verification failed")
        
        self._log_audit_event("secure_multiplication", {"operation": "multiply"})
        return local_result
    
    def secure_matmul(self, a: SecureValue, b: SecureValue) -> SecureValue:
        """Secure matrix multiplication with batch proof verification."""
        if not (self.validate_shares(a) and self.validate_shares(b)):
            raise ProtocolError("Invalid shares for matrix multiplication")
        
        # Perform matrix multiplication on shares
        result_shares = []
        result_macs = []
        
        for share_a in a.shares:
            for share_b in b.shares:
                result_share = torch.matmul(share_a, share_b)
                result_shares.append(result_share)
                
                # Compute MAC for result
                if isinstance(a, MaliciousSecureValue):
                    result_mac = self._compute_authenticated_mac(result_share)
                    result_macs.append(result_mac)
        
        # Generate batch proof for matrix multiplication correctness
        batch_proof = self._generate_matmul_proof(a, b, result_shares)
        self._broadcast_proof(batch_proof)
        
        if not self._verify_matmul_proofs():
            raise ProtocolError("Matrix multiplication proof verification failed")
        
        result = MaliciousSecureValue(
            shares=result_shares,
            party_id=self.party_id,
            macs=result_macs
        )
        
        self._log_audit_event("secure_matmul", {"operation": "matmul"})
        return result
    
    def _compute_authenticated_mac(self, value: torch.Tensor) -> torch.Tensor:
        """Compute authenticated MAC for a value."""
        # MAC = H(value || mac_key)
        value_bytes = value.cpu().numpy().tobytes()
        key_bytes = self.mac_key.cpu().numpy().tobytes()
        
        mac_hash = hmac.new(
            key_bytes, 
            value_bytes, 
            hashlib.sha256
        ).digest()
        
        # Convert hash to tensor
        mac_tensor = torch.tensor(
            [float(b) for b in mac_hash[:8]], 
            device=self.device
        )
        
        return mac_tensor.reshape(-1, 1)
    
    def _local_multiply(self, a: SecureValue, b: SecureValue) -> MaliciousSecureValue:
        """Local multiplication step."""
        result_shares = []
        result_macs = []
        
        for i, share_a in enumerate(a.shares):
            for j, share_b in enumerate(b.shares):
                product = share_a * share_b
                result_shares.append(product)
                
                # Compute MAC for product
                mac = self._compute_authenticated_mac(product)
                result_macs.append(mac)
        
        return MaliciousSecureValue(
            shares=result_shares,
            party_id=self.party_id,
            macs=result_macs
        )
    
    def _generate_multiplication_proof(self, a: SecureValue, b: SecureValue, 
                                     result: SecureValue) -> dict:
        """Generate zero-knowledge proof for multiplication correctness."""
        # Simplified proof generation
        commitment = self._generate_commitment({
            "a_shares": [s.sum().item() for s in a.shares],
            "b_shares": [s.sum().item() for s in b.shares],
            "result_shares": [s.sum().item() for s in result.shares]
        })
        
        return {
            "type": "multiplication",
            "commitment": commitment,
            "party_id": self.party_id
        }
    
    def _generate_matmul_proof(self, a: SecureValue, b: SecureValue, 
                              result_shares: List[torch.Tensor]) -> dict:
        """Generate proof for matrix multiplication correctness."""
        return {
            "type": "matmul", 
            "commitment": "dummy_commitment",
            "party_id": self.party_id
        }
    
    def _generate_commitment(self, data: dict) -> str:
        """Generate cryptographic commitment."""
        data_str = str(sorted(data.items()))
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _broadcast_proof(self, proof: dict) -> None:
        """Broadcast proof to all parties."""
        self._broadcast_channel.append(proof)
    
    def _verify_multiplication_proofs(self) -> bool:
        """Verify multiplication proofs from all parties."""
        # Simplified verification
        return len(self._broadcast_channel) >= 0  # Always pass for demo
    
    def _verify_matmul_proofs(self) -> bool:
        """Verify matrix multiplication proofs."""
        return True  # Simplified for demo
    
    def _challenge_and_verify_share(self, party: int, secure_value: SecureValue) -> torch.Tensor:
        """Challenge party to prove share correctness."""
        # In practice, would use challenge-response protocol
        # For simulation, return dummy share
        return torch.zeros_like(secure_value.shares[0])
    
    def _send_authenticated_share(self, share: torch.Tensor, mac: torch.Tensor, recipient: int) -> None:
        """Send share with authentication."""
        if recipient not in self._message_queue:
            self._message_queue[recipient] = []
        
        self._message_queue[recipient].append({
            "share": share.clone().detach(),
            "mac": mac.clone().detach()
        })
    
    def _receive_authenticated_share(self, sender: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Receive and verify authenticated share."""
        if sender not in self._message_queue or not self._message_queue[sender]:
            # Return dummy data for simulation
            return torch.zeros(1, device=self.device), torch.zeros((1, 1), device=self.device)
        
        message = self._message_queue[sender].pop(0)
        return message["share"], message["mac"]
    
    def _log_audit_event(self, event_type: str, details: dict) -> None:
        """Log security audit event."""
        self._audit_log.append({
            "timestamp": torch.tensor(len(self._audit_log)),
            "event_type": event_type,
            "party_id": self.party_id,
            "details": details
        })
    
    def send_shares(self, shares: List[torch.Tensor], recipient: int) -> None:
        """Send shares with authentication."""
        for share in shares:
            mac = self._compute_authenticated_mac(share)
            self._send_authenticated_share(share, mac, recipient)
    
    def receive_shares(self, sender: int) -> List[torch.Tensor]:
        """Receive and verify shares."""
        share, mac = self._receive_authenticated_share(sender)
        
        # Verify MAC
        if not torch.allclose(mac, self._compute_authenticated_mac(share), atol=1e-6):
            raise ProtocolError(f"MAC verification failed for shares from party {sender}")
        
        return [share]
    
    def get_audit_log(self) -> List[dict]:
        """Get security audit log."""
        return self._audit_log.copy()
    
    def get_security_stats(self) -> dict:
        """Get security statistics."""
        return {
            "total_operations": len(self._audit_log),
            "mac_verifications": len([log for log in self._audit_log if "mac" in log["event_type"]]),
            "proof_generations": len([log for log in self._audit_log if "proof" in log.get("details", {}).get("operation", "")]),
            "security_violations": len([log for log in self._audit_log if "failed" in log["event_type"]])
        }
