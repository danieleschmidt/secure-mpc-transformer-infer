"""Inference service for secure MPC transformer operations."""

import asyncio
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch

from ..models.secure_transformer import SecureTransformer, TransformerConfig
from ..protocols.base import SecureValue
from ..utils.metrics import MetricsCollector
from ..utils.validators import InputValidator

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """Request object for secure inference."""

    text: str
    model_name: str = "bert-base-uncased"
    max_length: int | None = None
    batch_size: int = 1
    return_attention: bool = False
    return_hidden_states: bool = False
    protocol_config: dict[str, Any] | None = None

    def __post_init__(self):
        if self.max_length is None:
            self.max_length = 512

        if self.protocol_config is None:
            self.protocol_config = {}


@dataclass
class InferenceResult:
    """Result object for secure inference."""

    request_id: str
    input_text: str
    model_name: str
    output_tensor: torch.Tensor
    secure_output: SecureValue
    latency_ms: float
    protocol_info: dict[str, Any]
    input_shape: tuple
    output_shape: tuple
    attention_weights: torch.Tensor | None = None
    hidden_states: list[torch.Tensor] | None = None
    security_metrics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary format."""
        result_dict = asdict(self)

        # Convert tensors to lists for JSON serialization
        if self.output_tensor is not None:
            result_dict['output_tensor'] = self.output_tensor.detach().cpu().numpy().tolist()

        if self.attention_weights is not None:
            result_dict['attention_weights'] = self.attention_weights.detach().cpu().numpy().tolist()

        if self.hidden_states is not None:
            result_dict['hidden_states'] = [
                h.detach().cpu().numpy().tolist() for h in self.hidden_states
            ]

        # Remove secure_output as it's not serializable
        result_dict.pop('secure_output', None)

        return result_dict


class InferenceService:
    """Service for managing secure transformer inference operations."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.models: dict[str, SecureTransformer] = {}
        self.validator = InputValidator()
        self.metrics = MetricsCollector()

        # Service configuration
        self.max_concurrent_requests = self.config.get('max_concurrent_requests', 10)
        self.request_timeout = self.config.get('request_timeout', 300)  # 5 minutes
        self.cache_enabled = self.config.get('cache_enabled', False)

        self._request_counter = 0
        self._active_requests: dict[str, asyncio.Task] = {}

        logger.info("InferenceService initialized")

    async def predict(self, request: InferenceRequest) -> InferenceResult:
        """Execute secure inference on input text."""
        request_id = self._generate_request_id()

        # Validate request
        self.validator.validate_inference_request(request)

        # Check concurrent request limit
        if len(self._active_requests) >= self.max_concurrent_requests:
            raise RuntimeError(f"Maximum concurrent requests ({self.max_concurrent_requests}) exceeded")

        # Start metrics tracking
        start_time = time.time()
        self.metrics.increment_counter("inference_requests_total")

        try:
            # Load or get cached model
            model = await self._get_model(request.model_name, request.protocol_config)

            # Execute secure inference
            result = await self._execute_inference(request_id, model, request)

            # Record success metrics
            latency = (time.time() - start_time) * 1000
            self.metrics.observe_histogram("inference_latency_ms", latency)
            self.metrics.increment_counter("inference_requests_success")

            logger.info(f"Inference completed: {request_id}, latency: {latency:.2f}ms")

            return result

        except Exception as e:
            self.metrics.increment_counter("inference_requests_failed")
            logger.error(f"Inference failed: {request_id}, error: {str(e)}")
            raise

        finally:
            # Clean up active request tracking
            self._active_requests.pop(request_id, None)

    async def predict_batch(self, requests: list[InferenceRequest]) -> list[InferenceResult]:
        """Execute batch inference on multiple requests."""
        if not requests:
            return []

        logger.info(f"Starting batch inference for {len(requests)} requests")

        # Execute requests concurrently
        tasks = [self.predict(request) for request in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions in results
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch request {i} failed: {str(result)}")
                # Create error result
                error_result = InferenceResult(
                    request_id=f"batch_{i}_error",
                    input_text=requests[i].text,
                    model_name=requests[i].model_name,
                    output_tensor=torch.tensor([]),
                    secure_output=None,
                    latency_ms=0,
                    protocol_info={"error": str(result)},
                    input_shape=(0,),
                    output_shape=(0,)
                )
                successful_results.append(error_result)
            else:
                successful_results.append(result)

        return successful_results

    async def _get_model(self, model_name: str, protocol_config: dict[str, Any]) -> SecureTransformer:
        """Load or retrieve cached model."""
        cache_key = f"{model_name}_{hash(str(protocol_config))}"

        if cache_key not in self.models:
            logger.info(f"Loading model: {model_name}")

            # Create configuration
            config = TransformerConfig.from_pretrained(model_name, **protocol_config)

            # Load model
            model = SecureTransformer.from_pretrained(model_name, **protocol_config)

            # Cache model if enabled
            if self.cache_enabled:
                self.models[cache_key] = model

            logger.info(f"Model loaded: {model_name}")
            return model

        return self.models[cache_key]

    async def _execute_inference(self, request_id: str, model: SecureTransformer,
                                request: InferenceRequest) -> InferenceResult:
        """Execute the actual secure inference."""

        # Create task for timeout handling
        task = asyncio.create_task(self._run_inference(model, request))
        self._active_requests[request_id] = task

        try:
            # Wait for completion with timeout
            result = await asyncio.wait_for(task, timeout=self.request_timeout)

            return InferenceResult(
                request_id=request_id,
                input_text=request.text,
                model_name=request.model_name,
                output_tensor=result['output_tensor'],
                secure_output=result['secure_output'],
                latency_ms=result['latency_ms'],
                protocol_info=result['protocol_info'],
                input_shape=result['input_shape'],
                output_shape=result['output_shape'],
                security_metrics=self._compute_security_metrics(result)
            )

        except asyncio.TimeoutError:
            task.cancel()
            raise RuntimeError(f"Inference timeout after {self.request_timeout}s")

    async def _run_inference(self, model: SecureTransformer, request: InferenceRequest) -> dict[str, Any]:
        """Run the actual model inference."""
        # Execute in thread pool to avoid blocking async loop
        loop = asyncio.get_event_loop()

        def _inference():
            return model.predict_secure(request.text)

        result = await loop.run_in_executor(None, _inference)
        return result

    def _compute_security_metrics(self, inference_result: dict[str, Any]) -> dict[str, Any]:
        """Compute security-related metrics for the inference."""
        return {
            "protocol_name": inference_result['protocol_info'].get('protocol_name'),
            "security_level": 128,  # Default security level
            "parties_involved": inference_result['protocol_info'].get('num_parties', 3),
            "computation_rounds": 1,  # Simplified for this implementation
            "communication_overhead_estimate": self._estimate_communication_overhead(inference_result)
        }

    def _estimate_communication_overhead(self, result: dict[str, Any]) -> float:
        """Estimate communication overhead for the inference."""
        # Simplified estimation based on tensor sizes
        input_size = torch.prod(torch.tensor(result['input_shape'])).item()
        output_size = torch.prod(torch.tensor(result['output_shape'])).item()

        # Estimate: 3 parties, 3 rounds of communication, float32 values
        estimated_bytes = (input_size + output_size) * 4 * 3 * 3

        return estimated_bytes / (1024 * 1024)  # Convert to MB

    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        self._request_counter += 1
        timestamp = int(time.time() * 1000)
        return f"req_{timestamp}_{self._request_counter}"

    def get_service_stats(self) -> dict[str, Any]:
        """Get service statistics and metrics."""
        return {
            "active_requests": len(self._active_requests),
            "loaded_models": len(self.models),
            "max_concurrent_requests": self.max_concurrent_requests,
            "cache_enabled": self.cache_enabled,
            "total_requests": self._request_counter,
            "metrics": self.metrics.get_all_metrics()
        }

    def clear_model_cache(self):
        """Clear the model cache."""
        self.models.clear()
        logger.info("Model cache cleared")

    async def health_check(self) -> dict[str, Any]:
        """Perform service health check."""
        try:
            # Test with a simple inference
            test_request = InferenceRequest(
                text="Test input for health check",
                model_name="bert-base-uncased"
            )

            start_time = time.time()
            result = await self.predict(test_request)
            health_check_latency = (time.time() - start_time) * 1000

            return {
                "status": "healthy",
                "health_check_latency_ms": health_check_latency,
                "service_stats": self.get_service_stats(),
                "timestamp": time.time()
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "service_stats": self.get_service_stats(),
                "timestamp": time.time()
            }


class MultiPartyInferenceService:
    """Service for coordinating multi-party secure inference."""

    def __init__(self, party_id: int, party_endpoints: list[str], config: dict[str, Any] | None = None):
        self.party_id = party_id
        self.party_endpoints = party_endpoints
        self.config = config or {}
        self.inference_service = InferenceService(config)

        # Multi-party specific configuration
        self.coordination_timeout = self.config.get('coordination_timeout', 600)  # 10 minutes
        self.retry_attempts = self.config.get('retry_attempts', 3)

        logger.info(f"MultiPartyInferenceService initialized for party {party_id}")

    async def coordinate_inference(self, request: InferenceRequest) -> InferenceResult:
        """Coordinate secure inference across multiple parties."""
        logger.info("Starting multi-party inference coordination")

        # Phase 1: Setup and synchronization
        await self._synchronize_parties(request)

        # Phase 2: Secret sharing
        shared_inputs = await self._distribute_secret_shares(request)

        # Phase 3: Secure computation
        computation_result = await self._execute_secure_computation(shared_inputs, request)

        # Phase 4: Result reconstruction
        final_result = await self._reconstruct_result(computation_result)

        return final_result

    async def _synchronize_parties(self, request: InferenceRequest):
        """Synchronize all parties before starting computation."""
        # Implementation would involve network communication
        # For now, simulate synchronization delay
        await asyncio.sleep(0.1)
        logger.info("Parties synchronized")

    async def _distribute_secret_shares(self, request: InferenceRequest) -> dict[str, Any]:
        """Distribute secret shares of input data."""
        # Implementation would involve actual secret sharing protocol
        # For now, return placeholder
        return {"shares_distributed": True, "request": request}

    async def _execute_secure_computation(self, shared_inputs: dict[str, Any],
                                         request: InferenceRequest) -> dict[str, Any]:
        """Execute secure computation on shared inputs."""
        # Delegate to local inference service
        result = await self.inference_service.predict(request)
        return {"computation_result": result}

    async def _reconstruct_result(self, computation_result: dict[str, Any]) -> InferenceResult:
        """Reconstruct final result from computation shares."""
        # Implementation would involve result reconstruction protocol
        # For now, return the local computation result
        return computation_result["computation_result"]
