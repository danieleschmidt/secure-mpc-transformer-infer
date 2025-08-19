"""Distributed tracing system for secure MPC operations."""

import functools
import inspect
import json
import logging
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SpanKind(Enum):
    """OpenTelemetry span kinds."""
    UNSPECIFIED = 0
    INTERNAL = 1
    SERVER = 2
    CLIENT = 3
    PRODUCER = 4
    CONSUMER = 5


class SpanStatus(Enum):
    """Span status codes."""
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class SpanContext:
    """Trace and span context information."""

    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    trace_flags: int = 0
    baggage: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)


@dataclass
class SpanEvent:
    """Span event data."""

    name: str
    timestamp: float
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)


@dataclass
class SpanLink:
    """Link between spans."""

    context: SpanContext
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'context': self.context.to_dict(),
            'attributes': self.attributes
        }


@dataclass
class Span:
    """Distributed tracing span."""

    name: str
    context: SpanContext
    kind: SpanKind
    start_time: float
    end_time: float | None = None
    status: SpanStatus = SpanStatus.UNSET
    status_message: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)
    links: list[SpanLink] = field(default_factory=list)
    resource: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float | None:
        """Get span duration in seconds."""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    @property
    def is_finished(self) -> bool:
        """Check if span is finished."""
        return self.end_time is not None

    def add_event(self, name: str, attributes: dict[str, Any] = None):
        """Add an event to the span."""
        event = SpanEvent(
            name=name,
            timestamp=time.time(),
            attributes=attributes or {}
        )
        self.events.append(event)

    def set_attribute(self, key: str, value: Any):
        """Set span attribute."""
        self.attributes[key] = value

    def set_attributes(self, attributes: dict[str, Any]):
        """Set multiple span attributes."""
        self.attributes.update(attributes)

    def set_status(self, status: SpanStatus, message: str | None = None):
        """Set span status."""
        self.status = status
        self.status_message = message

    def add_link(self, context: SpanContext, attributes: dict[str, Any] = None):
        """Add a link to another span."""
        link = SpanLink(context=context, attributes=attributes or {})
        self.links.append(link)

    def finish(self):
        """Finish the span."""
        if not self.is_finished:
            self.end_time = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        data = {
            'name': self.name,
            'context': self.context.to_dict(),
            'kind': self.kind.value,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'status': self.status.value,
            'status_message': self.status_message,
            'attributes': self.attributes,
            'events': [event.to_dict() for event in self.events],
            'links': [link.to_dict() for link in self.links],
            'resource': self.resource
        }
        return data

    def to_jaeger_format(self) -> dict[str, Any]:
        """Convert to Jaeger tracing format."""
        tags = []
        for key, value in self.attributes.items():
            tags.append({
                'key': key,
                'type': 'string',
                'value': str(value)
            })

        # Add status as tag
        tags.append({
            'key': 'span.status',
            'type': 'string',
            'value': self.status.value
        })

        if self.status_message:
            tags.append({
                'key': 'span.status.message',
                'type': 'string',
                'value': self.status_message
            })

        logs = []
        for event in self.events:
            fields = []
            for key, value in event.attributes.items():
                fields.append({
                    'key': key,
                    'value': str(value)
                })

            logs.append({
                'timestamp': int(event.timestamp * 1_000_000),  # microseconds
                'fields': fields
            })

        references = []
        if self.context.parent_span_id:
            references.append({
                'refType': 'CHILD_OF',
                'traceID': self.context.trace_id,
                'spanID': self.context.parent_span_id
            })

        for link in self.links:
            references.append({
                'refType': 'FOLLOWS_FROM',
                'traceID': link.context.trace_id,
                'spanID': link.context.span_id
            })

        return {
            'traceID': self.context.trace_id,
            'spanID': self.context.span_id,
            'operationName': self.name,
            'startTime': int(self.start_time * 1_000_000),  # microseconds
            'duration': int((self.duration or 0) * 1_000_000),  # microseconds
            'tags': tags,
            'logs': logs,
            'references': references,
            'process': {
                'serviceName': self.resource.get('service.name', 'secure-mpc-transformer'),
                'tags': [
                    {
                        'key': key,
                        'type': 'string',
                        'value': str(value)
                    }
                    for key, value in self.resource.items()
                ]
            }
        }


class TraceContext:
    """Thread-local trace context."""

    def __init__(self):
        self._storage = threading.local()

    def get_current_span(self) -> Span | None:
        """Get current active span."""
        return getattr(self._storage, 'current_span', None)

    def set_current_span(self, span: Span | None):
        """Set current active span."""
        self._storage.current_span = span

    def get_trace_id(self) -> str | None:
        """Get current trace ID."""
        span = self.get_current_span()
        return span.context.trace_id if span else None

    def get_span_id(self) -> str | None:
        """Get current span ID."""
        span = self.get_current_span()
        return span.context.span_id if span else None


class SpanProcessor:
    """Base span processor interface."""

    def on_start(self, span: Span):
        """Called when span starts."""
        pass

    def on_end(self, span: Span):
        """Called when span ends."""
        pass

    def shutdown(self):
        """Shutdown the processor."""
        pass


class BatchSpanProcessor(SpanProcessor):
    """Batch span processor for efficient export."""

    def __init__(self, exporter, max_queue_size: int = 1000,
                 max_batch_size: int = 100, batch_timeout: float = 5.0):
        self.exporter = exporter
        self.max_queue_size = max_queue_size
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout

        self.span_queue = deque()
        self.queue_lock = threading.Lock()
        self.shutdown_event = threading.Event()

        # Start worker thread
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def on_end(self, span: Span):
        """Add span to export queue."""
        with self.queue_lock:
            if len(self.span_queue) < self.max_queue_size:
                self.span_queue.append(span)
            else:
                logger.warning("Span queue full, dropping span")

    def _worker(self):
        """Background worker to export spans."""
        batch = []
        last_export = time.time()

        while not self.shutdown_event.is_set():
            try:
                # Collect spans for batch
                with self.queue_lock:
                    while self.span_queue and len(batch) < self.max_batch_size:
                        batch.append(self.span_queue.popleft())

                # Export batch if ready
                current_time = time.time()
                should_export = (
                    len(batch) >= self.max_batch_size or
                    (batch and current_time - last_export >= self.batch_timeout)
                )

                if should_export and batch:
                    try:
                        self.exporter.export(batch)
                        batch.clear()
                        last_export = current_time
                    except Exception as e:
                        logger.error(f"Failed to export spans: {e}")
                        # Clear batch to avoid infinite retries
                        batch.clear()

                time.sleep(0.1)  # Short sleep to avoid busy waiting

            except Exception as e:
                logger.error(f"Span processor worker error: {e}")

    def shutdown(self):
        """Shutdown the processor."""
        self.shutdown_event.set()

        # Export remaining spans
        if hasattr(self, 'worker_thread'):
            self.worker_thread.join(timeout=5.0)

        with self.queue_lock:
            if self.span_queue:
                try:
                    remaining_spans = list(self.span_queue)
                    self.exporter.export(remaining_spans)
                except Exception as e:
                    logger.error(f"Failed to export remaining spans: {e}")


class ConsoleSpanExporter:
    """Console span exporter for debugging."""

    def export(self, spans: list[Span]):
        """Export spans to console."""
        for span in spans:
            logger.info(f"Span: {span.name} [{span.context.trace_id}/{span.context.span_id}] "
                       f"Duration: {span.duration:.3f}s Status: {span.status.value}")


class JSONSpanExporter:
    """JSON file span exporter."""

    def __init__(self, filename: str):
        self.filename = filename
        self.file_lock = threading.Lock()

    def export(self, spans: list[Span]):
        """Export spans to JSON file."""
        with self.file_lock:
            try:
                with open(self.filename, 'a') as f:
                    for span in spans:
                        json.dump(span.to_dict(), f, separators=(',', ':'))
                        f.write('\n')
            except Exception as e:
                logger.error(f"Failed to export spans to file: {e}")


class JaegerSpanExporter:
    """Jaeger span exporter."""

    def __init__(self, endpoint: str, service_name: str = "secure-mpc-transformer"):
        self.endpoint = endpoint
        self.service_name = service_name

    def export(self, spans: list[Span]):
        """Export spans to Jaeger."""
        try:
            import requests

            jaeger_spans = []
            for span in spans:
                jaeger_span = span.to_jaeger_format()
                jaeger_spans.append(jaeger_span)

            payload = {
                'data': [{
                    'traceID': spans[0].context.trace_id if spans else '',
                    'spans': jaeger_spans,
                    'processes': {
                        'p1': {
                            'serviceName': self.service_name,
                            'tags': []
                        }
                    }
                }]
            }

            response = requests.post(
                f"{self.endpoint}/api/traces",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=5.0
            )

            if response.status_code != 200:
                logger.warning(f"Jaeger export failed: {response.status_code} {response.text}")

        except ImportError:
            logger.error("requests library required for Jaeger export")
        except Exception as e:
            logger.error(f"Failed to export spans to Jaeger: {e}")


class DistributedTracer:
    """Main distributed tracing system."""

    def __init__(self, service_name: str = "secure-mpc-transformer",
                 config: dict[str, Any] | None = None):
        self.service_name = service_name
        self.config = config or {}

        # Context management
        self.context = TraceContext()

        # Span storage and processing
        self.processors: list[SpanProcessor] = []
        self.active_spans: dict[str, Span] = {}
        self.finished_spans: deque = deque(maxlen=10000)

        # Resource attributes
        self.resource_attributes = {
            'service.name': service_name,
            'service.version': '0.1.0',
            'telemetry.sdk.name': 'custom-tracer',
            'telemetry.sdk.version': '1.0.0'
        }

        # Configuration
        self.enabled = self.config.get('enabled', True)
        self.sample_rate = self.config.get('sample_rate', 1.0)  # Sample 100% by default

        # Initialize default exporters
        self._initialize_exporters()

        logger.info(f"Distributed tracer initialized for service: {service_name}")

    def _initialize_exporters(self):
        """Initialize default span exporters."""
        exporters_config = self.config.get('exporters', {})

        # Console exporter for development
        if exporters_config.get('console', {}).get('enabled', False):
            console_exporter = ConsoleSpanExporter()
            processor = BatchSpanProcessor(console_exporter)
            self.add_span_processor(processor)

        # JSON file exporter
        json_config = exporters_config.get('json', {})
        if json_config.get('enabled', False):
            filename = json_config.get('filename', 'traces.jsonl')
            json_exporter = JSONSpanExporter(filename)
            processor = BatchSpanProcessor(json_exporter)
            self.add_span_processor(processor)

        # Jaeger exporter
        jaeger_config = exporters_config.get('jaeger', {})
        if jaeger_config.get('enabled', False):
            endpoint = jaeger_config.get('endpoint', 'http://localhost:14268')
            jaeger_exporter = JaegerSpanExporter(endpoint, self.service_name)
            processor = BatchSpanProcessor(jaeger_exporter)
            self.add_span_processor(processor)

    def add_span_processor(self, processor: SpanProcessor):
        """Add a span processor."""
        self.processors.append(processor)

    def start_span(self, name: str, kind: SpanKind = SpanKind.INTERNAL,
                   attributes: dict[str, Any] = None,
                   links: list[SpanLink] = None,
                   parent_context: SpanContext | None = None) -> Span:
        """Start a new span."""
        if not self.enabled:
            # Return a no-op span
            return self._create_noop_span(name)

        # Check sampling
        if not self._should_sample():
            return self._create_noop_span(name)

        # Determine parent
        current_span = self.context.get_current_span()
        parent_span_id = None
        trace_id = None

        if parent_context:
            parent_span_id = parent_context.span_id
            trace_id = parent_context.trace_id
        elif current_span:
            parent_span_id = current_span.context.span_id
            trace_id = current_span.context.trace_id

        if not trace_id:
            trace_id = self._generate_trace_id()

        # Create span context
        span_context = SpanContext(
            trace_id=trace_id,
            span_id=self._generate_span_id(),
            parent_span_id=parent_span_id
        )

        # Create span
        span = Span(
            name=name,
            context=span_context,
            kind=kind,
            start_time=time.time(),
            attributes=attributes or {},
            links=links or [],
            resource=self.resource_attributes.copy()
        )

        # Store active span
        self.active_spans[span.context.span_id] = span

        # Notify processors
        for processor in self.processors:
            try:
                processor.on_start(span)
            except Exception as e:
                logger.error(f"Span processor on_start error: {e}")

        return span

    def _create_noop_span(self, name: str) -> Span:
        """Create a no-op span that doesn't get processed."""
        context = SpanContext(
            trace_id="00000000000000000000000000000000",
            span_id="0000000000000000"
        )

        span = Span(
            name=name,
            context=context,
            kind=SpanKind.INTERNAL,
            start_time=time.time()
        )

        # Mark as no-op
        span._noop = True
        return span

    def finish_span(self, span: Span):
        """Finish a span."""
        if hasattr(span, '_noop'):
            return

        span.finish()

        # Remove from active spans
        self.active_spans.pop(span.context.span_id, None)

        # Add to finished spans
        self.finished_spans.append(span)

        # Notify processors
        for processor in self.processors:
            try:
                processor.on_end(span)
            except Exception as e:
                logger.error(f"Span processor on_end error: {e}")

    def _should_sample(self) -> bool:
        """Determine if this span should be sampled."""
        import random
        return random.random() < self.sample_rate

    def _generate_trace_id(self) -> str:
        """Generate a new trace ID."""
        return uuid.uuid4().hex

    def _generate_span_id(self) -> str:
        """Generate a new span ID."""
        return uuid.uuid4().hex[:16]

    @contextmanager
    def span(self, name: str, kind: SpanKind = SpanKind.INTERNAL,
             attributes: dict[str, Any] = None, set_current: bool = True):
        """Context manager for creating spans."""
        span = self.start_span(name, kind, attributes)

        # Set as current span if requested
        previous_span = None
        if set_current:
            previous_span = self.context.get_current_span()
            self.context.set_current_span(span)

        try:
            yield span
            span.set_status(SpanStatus.OK)
        except Exception as e:
            span.set_status(SpanStatus.ERROR, str(e))
            span.add_event("exception", {
                "exception.type": type(e).__name__,
                "exception.message": str(e)
            })
            raise
        finally:
            self.finish_span(span)

            # Restore previous span
            if set_current:
                self.context.set_current_span(previous_span)

    @asynccontextmanager
    async def async_span(self, name: str, kind: SpanKind = SpanKind.INTERNAL,
                        attributes: dict[str, Any] = None, set_current: bool = True):
        """Async context manager for creating spans."""
        span = self.start_span(name, kind, attributes)

        # Set as current span if requested
        previous_span = None
        if set_current:
            previous_span = self.context.get_current_span()
            self.context.set_current_span(span)

        try:
            yield span
            span.set_status(SpanStatus.OK)
        except Exception as e:
            span.set_status(SpanStatus.ERROR, str(e))
            span.add_event("exception", {
                "exception.type": type(e).__name__,
                "exception.message": str(e)
            })
            raise
        finally:
            self.finish_span(span)

            # Restore previous span
            if set_current:
                self.context.set_current_span(previous_span)

    def get_trace_stats(self) -> dict[str, Any]:
        """Get tracing statistics."""
        return {
            'enabled': self.enabled,
            'sample_rate': self.sample_rate,
            'active_spans': len(self.active_spans),
            'finished_spans': len(self.finished_spans),
            'processors': len(self.processors),
            'service_name': self.service_name
        }

    def shutdown(self):
        """Shutdown the tracer."""
        # Finish all active spans
        for span in list(self.active_spans.values()):
            span.set_status(SpanStatus.ERROR, "Tracer shutdown")
            self.finish_span(span)

        # Shutdown processors
        for processor in self.processors:
            try:
                processor.shutdown()
            except Exception as e:
                logger.error(f"Processor shutdown error: {e}")


def trace_function(name: str = None, kind: SpanKind = SpanKind.INTERNAL,
                  attributes: dict[str, Any] = None):
    """Decorator for tracing functions."""
    def decorator(func):
        span_name = name or f"{func.__module__}.{func.__name__}"

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                async with tracer.async_span(span_name, kind, attributes):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with tracer.span(span_name, kind, attributes):
                    return func(*args, **kwargs)
            return sync_wrapper

    return decorator


def trace_method(name: str = None, kind: SpanKind = SpanKind.INTERNAL):
    """Decorator for tracing class methods."""
    def decorator(func):
        def get_span_name(*args, **kwargs):
            if args and hasattr(args[0], '__class__'):
                class_name = args[0].__class__.__name__
                method_name = func.__name__
                return name or f"{class_name}.{method_name}"
            return name or func.__name__

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                span_name = get_span_name(*args, **kwargs)
                async with tracer.async_span(span_name, kind):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                span_name = get_span_name(*args, **kwargs)
                with tracer.span(span_name, kind):
                    return func(*args, **kwargs)
            return sync_wrapper

    return decorator


# Global tracer instance
tracer = DistributedTracer()
