"""Semi-honest 3-party computation protocol."""


import numpy as np
import torch

from .base import Protocol, ProtocolError, SecureValue


class SemiHonest3PC(Protocol):
    """Semi-honest 3-party computation using replicated secret sharing."""

    def __init__(self, party_id: int, num_parties: int = 3, device: torch.device | None = None):
        if num_parties != 3:
            raise ValueError("SemiHonest3PC requires exactly 3 parties")
        super().__init__(party_id, num_parties, device)

        # Network connections (simulated for this implementation)
        self._connections = {}
        self._message_queue = {}

    def initialize(self) -> None:
        """Initialize protocol parameters."""
        if self._initialized:
            return

        # Initialize random number generators with different seeds per party
        torch.manual_seed(42 + self.party_id)
        np.random.seed(42 + self.party_id)

        # Initialize message queues for each party
        for i in range(self.num_parties):
            if i != self.party_id:
                self._message_queue[i] = []

        self._initialized = True

    def share_value(self, value: torch.Tensor) -> SecureValue:
        """Share value using replicated secret sharing.
        
        In replicated 3PC, each party holds two shares:
        - Party 0: (r0, r1)
        - Party 1: (r1, r2)  
        - Party 2: (r2, r0)
        where value = r0 + r1 + r2
        """
        if not self._initialized:
            self.initialize()

        value = value.to(self.device)

        if self.party_id == 0:
            # Party 0 generates the shares
            r1 = torch.rand_like(value, device=self.device)
            r2 = torch.rand_like(value, device=self.device)
            r0 = value - r1 - r2

            # Party 0 keeps (r0, r1)
            shares = [r0, r1]

            # Send r1, r2 to party 1 and r2, r0 to party 2
            self._send_tensor_to_party([r1, r2], 1)
            self._send_tensor_to_party([r2, r0], 2)

        else:
            # Other parties receive their shares
            shares = self._receive_tensor_from_party(0)

        return SecureValue(shares=shares, party_id=self.party_id)

    def reconstruct_value(self, secure_value: SecureValue) -> torch.Tensor:
        """Reconstruct value from replicated shares."""
        if not self.validate_shares(secure_value):
            raise ProtocolError("Invalid shares for reconstruction")

        # Collect all unique shares from all parties
        all_shares = {}

        # Add own shares
        if self.party_id == 0:
            all_shares['r0'] = secure_value.shares[0]
            all_shares['r1'] = secure_value.shares[1]
        elif self.party_id == 1:
            all_shares['r1'] = secure_value.shares[0]
            all_shares['r2'] = secure_value.shares[1]
        else:  # party_id == 2
            all_shares['r2'] = secure_value.shares[0]
            all_shares['r0'] = secure_value.shares[1]

        # Request missing shares from other parties
        if 'r0' not in all_shares:
            all_shares['r0'] = self._request_share_from_party('r0')
        if 'r1' not in all_shares:
            all_shares['r1'] = self._request_share_from_party('r1')
        if 'r2' not in all_shares:
            all_shares['r2'] = self._request_share_from_party('r2')

        # Reconstruct: value = r0 + r1 + r2
        return all_shares['r0'] + all_shares['r1'] + all_shares['r2']

    def secure_add(self, a: SecureValue, b: SecureValue) -> SecureValue:
        """Secure addition - done locally without communication."""
        if not (self.validate_shares(a) and self.validate_shares(b)):
            raise ProtocolError("Invalid shares for addition")

        # Add corresponding shares
        result_shares = []
        for i in range(len(a.shares)):
            result_shares.append(a.shares[i] + b.shares[i])

        return SecureValue(shares=result_shares, party_id=self.party_id)

    def secure_multiply(self, a: SecureValue, b: SecureValue) -> SecureValue:
        """Secure multiplication using BGW protocol."""
        if not (self.validate_shares(a) and self.validate_shares(b)):
            raise ProtocolError("Invalid shares for multiplication")

        # Local multiplication creates degree-2 shares
        local_products = []
        for i in range(len(a.shares)):
            for j in range(len(b.shares)):
                local_products.append(a.shares[i] * b.shares[j])

        # Reshare to reduce degree back to 1
        return self._reshare_degree2(local_products)

    def secure_matmul(self, a: SecureValue, b: SecureValue) -> SecureValue:
        """Secure matrix multiplication."""
        if not (self.validate_shares(a) and self.validate_shares(b)):
            raise ProtocolError("Invalid shares for matrix multiplication")

        # Perform matrix multiplication on shares
        result_shares = []
        for i in range(len(a.shares)):
            for j in range(len(b.shares)):
                result_shares.append(torch.matmul(a.shares[i], b.shares[j]))

        # Reshare to maintain correct sharing
        return self._reshare_degree2(result_shares)

    def _reshare_degree2(self, degree2_shares: list[torch.Tensor]) -> SecureValue:
        """Reshare degree-2 shares to degree-1 shares."""
        # Sum all degree-2 shares
        total = sum(degree2_shares)

        # Create new random shares
        if len(degree2_shares) > 0:
            shape = degree2_shares[0].shape
            r1 = torch.rand(shape, device=self.device)
            r2 = total - r1
            new_shares = [r1, r2]
        else:
            new_shares = []

        return SecureValue(shares=new_shares, party_id=self.party_id)

    def send_shares(self, shares: list[torch.Tensor], recipient: int) -> None:
        """Send shares to another party."""
        self._send_tensor_to_party(shares, recipient)

    def receive_shares(self, sender: int) -> list[torch.Tensor]:
        """Receive shares from another party."""
        return self._receive_tensor_from_party(sender)

    def _send_tensor_to_party(self, tensors: list[torch.Tensor], recipient: int) -> None:
        """Simulate sending tensors to another party."""
        # In a real implementation, this would use network communication
        # For simulation, we'll store in a message queue
        if recipient not in self._message_queue:
            self._message_queue[recipient] = []

        # Serialize tensors (in practice, would use efficient serialization)
        serialized = [tensor.clone().detach() for tensor in tensors]
        self._message_queue[recipient].append(serialized)

    def _receive_tensor_from_party(self, sender: int) -> list[torch.Tensor]:
        """Simulate receiving tensors from another party."""
        # In a real implementation, this would use network communication
        if sender not in self._message_queue or not self._message_queue[sender]:
            # Generate dummy data for simulation
            return [torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)]

        return self._message_queue[sender].pop(0)

    def _request_share_from_party(self, share_name: str) -> torch.Tensor:
        """Request specific share from appropriate party."""
        # In simulation, return dummy tensor
        return torch.zeros(1, device=self.device)

    def get_communication_stats(self) -> dict:
        """Get communication statistics."""
        total_messages = sum(len(queue) for queue in self._message_queue.values())
        return {
            "total_messages_sent": total_messages,
            "message_queues": {party: len(queue) for party, queue in self._message_queue.items()}
        }
