"""API middleware for security and rate limiting."""

import time
import logging
from typing import Dict, Any, Optional
from collections import defaultdict, deque

try:
    from fastapi import Request, Response
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse
except ImportError:
    BaseHTTPMiddleware = object
    Request = None
    Response = None
    JSONResponse = None

from ..services.security_service import SecurityService

logger = logging.getLogger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for request validation and threat detection."""
    
    def __init__(self, app, security_service: SecurityService):
        super().__init__(app)
        self.security_service = security_service
        
    async def dispatch(self, request: Request, call_next):
        """Process request through security validation."""
        start_time = time.time()
        
        # Extract request information
        source_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        request_id = f"req_{int(time.time() * 1000)}"
        
        # Store request info in state
        request.state.request_id = request_id
        request.state.start_time = start_time
        
        try:
            # Skip security validation for health checks and docs
            if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json"]:
                response = await call_next(request)
                return response
            
            # Validate request based on method and content
            if request.method in ["POST", "PUT", "PATCH"]:
                # Get request body for validation
                body = await request.body()
                request_data = {}
                
                if body:
                    try:
                        import json
                        request_data = json.loads(body.decode())
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        # Non-JSON body, skip detailed validation
                        pass
                
                # Validate request
                is_valid, errors = self.security_service.validate_request(
                    request_data, source_ip, user_id=None
                )
                
                if not is_valid:
                    logger.warning(f"Security validation failed: {errors}")
                    return JSONResponse(
                        status_code=400,
                        content={
                            "error": "Security validation failed",
                            "details": errors,
                            "request_id": request_id
                        }
                    )
                
                # Reconstruct request with body
                from starlette.requests import Request as StarletteRequest
                scope = request.scope.copy()
                receive = self._create_receive_with_body(body)
                request = StarletteRequest(scope, receive)
            
            # Process request
            response = await call_next(request)
            
            # Log successful request
            processing_time = (time.time() - start_time) * 1000
            if processing_time > 1000:  # Log slow requests
                self.security_service.auditor.log_security_event(
                    event_type="PERFORMANCE_WARNING",
                    severity="MEDIUM",
                    source_ip=source_ip,
                    details={
                        "processing_time_ms": processing_time,
                        "endpoint": str(request.url),
                        "method": request.method
                    }
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Security middleware error: {e}")
            
            # Log security event
            self.security_service.auditor.log_security_event(
                event_type="MIDDLEWARE_ERROR",
                severity="HIGH",
                source_ip=source_ip,
                details={
                    "error": str(e),
                    "endpoint": str(request.url),
                    "user_agent": user_agent
                }
            )
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal security error",
                    "request_id": request_id
                }
            )
    
    def _create_receive_with_body(self, body: bytes):
        """Create a receive callable with the given body."""
        async def receive():
            return {
                "type": "http.request",
                "body": body,
                "more_body": False
            }
        return receive


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""
    
    def __init__(self, app, config: Optional[Dict[str, Any]] = None):
        super().__init__(app)
        self.config = config or {}
        
        # Rate limiting configuration
        self.requests_per_minute = self.config.get('requests_per_minute', 100)
        self.burst_limit = self.config.get('burst_limit', 20)
        self.window_size = 60  # seconds
        
        # Storage for rate limiting
        self.request_counts = defaultdict(deque)
        self.last_cleanup = time.time()
        
    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting to requests."""
        source_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        # Cleanup old entries periodically
        if current_time - self.last_cleanup > 60:
            self._cleanup_old_entries(current_time)
            self.last_cleanup = current_time
        
        # Check rate limits
        if self._is_rate_limited(source_ip, current_time):
            logger.warning(f"Rate limit exceeded for IP: {source_ip}")
            
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.requests_per_minute} requests per minute allowed",
                    "retry_after": 60
                },
                headers={"Retry-After": "60"}
            )
        
        # Record request
        self.request_counts[source_ip].append(current_time)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = self._get_remaining_requests(source_ip, current_time)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(current_time) + 60)
        
        return response
    
    def _is_rate_limited(self, source_ip: str, current_time: float) -> bool:
        """Check if the source IP is rate limited."""
        requests = self.request_counts[source_ip]
        
        # Remove old requests outside the window
        cutoff_time = current_time - self.window_size
        while requests and requests[0] < cutoff_time:
            requests.popleft()
        
        # Check if over the limit
        return len(requests) >= self.requests_per_minute
    
    def _get_remaining_requests(self, source_ip: str, current_time: float) -> int:
        """Get remaining requests for the source IP."""
        requests = self.request_counts[source_ip]
        
        # Remove old requests
        cutoff_time = current_time - self.window_size
        while requests and requests[0] < cutoff_time:
            requests.popleft()
        
        return max(0, self.requests_per_minute - len(requests))
    
    def _cleanup_old_entries(self, current_time: float):
        """Clean up old entries to prevent memory leaks."""
        cutoff_time = current_time - (self.window_size * 2)  # Keep extra buffer
        
        ips_to_remove = []
        for ip, requests in self.request_counts.items():
            # Remove old requests
            while requests and requests[0] < cutoff_time:
                requests.popleft()
            
            # Remove empty IP entries
            if not requests:
                ips_to_remove.append(ip)
        
        for ip in ips_to_remove:
            del self.request_counts[ip]


class LoggingMiddleware(BaseHTTPMiddleware):
    """Request/response logging middleware."""
    
    def __init__(self, app, config: Optional[Dict[str, Any]] = None):
        super().__init__(app)
        self.config = config or {}
        self.log_requests = self.config.get('log_requests', True)
        self.log_responses = self.config.get('log_responses', False)
        self.log_bodies = self.config.get('log_bodies', False)
        
    async def dispatch(self, request: Request, call_next):
        """Log request and response information."""
        start_time = time.time()
        
        # Log request
        if self.log_requests:
            logger.info(f"Request: {request.method} {request.url}")
            
            if self.log_bodies and request.method in ["POST", "PUT", "PATCH"]:
                try:
                    body = await request.body()
                    if body:
                        # Only log first 500 characters to avoid huge logs
                        body_str = body.decode()[:500]
                        logger.debug(f"Request body: {body_str}")
                    
                    # Reconstruct request
                    from starlette.requests import Request as StarletteRequest
                    scope = request.scope.copy()
                    receive = self._create_receive_with_body(body)
                    request = StarletteRequest(scope, receive)
                except Exception as e:
                    logger.warning(f"Failed to log request body: {e}")
        
        # Process request
        response = await call_next(request)
        
        # Log response
        processing_time = (time.time() - start_time) * 1000
        
        if self.log_responses:
            logger.info(
                f"Response: {response.status_code} "
                f"({processing_time:.2f}ms) "
                f"for {request.method} {request.url}"
            )
        
        # Add timing header
        response.headers["X-Process-Time"] = f"{processing_time:.2f}ms"
        
        return response
    
    def _create_receive_with_body(self, body: bytes):
        """Create a receive callable with the given body."""
        async def receive():
            return {
                "type": "http.request",
                "body": body,
                "more_body": False
            }
        return receive


class CacheMiddleware(BaseHTTPMiddleware):
    """Simple response caching middleware."""
    
    def __init__(self, app, config: Optional[Dict[str, Any]] = None):
        super().__init__(app)
        self.config = config or {}
        self.cache_ttl = self.config.get('cache_ttl', 300)  # 5 minutes
        self.cache = {}
        self.cache_times = {}
        
    async def dispatch(self, request: Request, call_next):
        """Cache GET responses."""
        # Only cache GET requests
        if request.method != "GET":
            return await call_next(request)
        
        # Skip caching for certain endpoints
        skip_paths = ["/health", "/metrics", "/docs", "/redoc", "/openapi.json"]
        if any(request.url.path.startswith(path) for path in skip_paths):
            return await call_next(request)
        
        # Generate cache key
        cache_key = f"{request.method}:{request.url}"
        current_time = time.time()
        
        # Check cache
        if cache_key in self.cache:
            cache_time = self.cache_times.get(cache_key, 0)
            if current_time - cache_time < self.cache_ttl:
                # Return cached response
                cached_response = self.cache[cache_key]
                
                # Create new response with cached data
                return JSONResponse(
                    content=cached_response["content"],
                    status_code=cached_response["status_code"],
                    headers={
                        **cached_response["headers"],
                        "X-Cache": "HIT",
                        "X-Cache-Age": str(int(current_time - cache_time))
                    }
                )
        
        # Process request
        response = await call_next(request)
        
        # Cache successful responses
        if response.status_code == 200:
            try:
                # Read response body
                body = b""
                async for chunk in response.body_iterator:
                    body += chunk
                
                # Parse JSON content
                import json
                content = json.loads(body.decode())
                
                # Store in cache
                self.cache[cache_key] = {
                    "content": content,
                    "status_code": response.status_code,
                    "headers": dict(response.headers)
                }
                self.cache_times[cache_key] = current_time
                
                # Create new response
                response = JSONResponse(
                    content=content,
                    status_code=response.status_code,
                    headers={
                        **dict(response.headers),
                        "X-Cache": "MISS"
                    }
                )
                
            except Exception as e:
                logger.warning(f"Failed to cache response: {e}")
        
        return response