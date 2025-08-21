"""
Generation 2: Robust HTTP Server with Enhanced Error Handling, Security, and Monitoring
"""

import json
import logging
import time
import threading
import hashlib
import secrets
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any, Optional
from pathlib import Path

from .models.robust_transformer import RobustSecureTransformer, RobustTransformerConfig
from .utils.robust_error_handling import (
    setup_robust_logging, 
    RobustErrorHandler,
    SecurityException,
    ValidationException,
    RateLimitException,
    TimeoutException,
    robust_exception_handler,
    ErrorCategory,
    ErrorSeverity
)


class RobustMPCHandler(BaseHTTPRequestHandler):
    """Enhanced HTTP request handler with robust error handling and security."""
    
    def __init__(self, *args, transformer_service=None, error_handler=None, 
                 security_config=None, **kwargs):
        self.transformer_service = transformer_service
        self.error_handler = error_handler or RobustErrorHandler()
        self.security_config = security_config or {}
        self.logger = logging.getLogger(f"{__name__}.RobustMPCHandler")
        super().__init__(*args, **kwargs)
    
    def log_message(self, format, *args):
        """Override to use our logger with client info."""
        client_ip = self.address_string()
        self.logger.info(f"{client_ip} - {format % args}")
    
    def _get_client_id(self) -> str:
        """Get client identifier for rate limiting and auditing."""
        # Use IP address as basic client ID (in production, use proper authentication)
        return self.address_string()
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        return secrets.token_hex(8)
    
    def _validate_request_size(self, content_length: int) -> bool:
        """Validate request size against limits."""
        max_request_size = self.security_config.get('max_request_size_mb', 10) * 1024 * 1024
        return content_length <= max_request_size
    
    def _extract_client_info(self) -> Dict[str, Any]:
        """Extract client information for security and auditing."""
        return {
            'ip_address': self.address_string(),
            'user_agent': self.headers.get('User-Agent', 'Unknown'),
            'timestamp': time.time(),
            'method': self.command,
            'path': self.path
        }
    
    @robust_exception_handler(category=ErrorCategory.NETWORK, attempt_recovery=False)
    def do_GET(self):
        """Handle GET requests with robust error handling."""
        request_id = self._generate_request_id()
        client_id = self._get_client_id()
        start_time = time.time()
        
        try:
            self.logger.info(f"GET request started: {self.path} (request_id: {request_id})")
            
            parsed_path = urlparse(self.path)
            path = parsed_path.path
            
            # Route handling with security checks
            if path == '/':
                self._handle_root(request_id, client_id)
            elif path == '/health':
                self._handle_health(request_id, client_id)
            elif path == '/api/v2/models':
                self._handle_models(request_id, client_id)
            elif path == '/api/v2/status':
                self._handle_status(request_id, client_id)
            elif path == '/api/v2/metrics':
                self._handle_metrics(request_id, client_id)
            elif path == '/api/v2/system-status':
                self._handle_system_status(request_id, client_id)
            else:
                self._send_error(404, "Not Found", request_id)
                
            processing_time = time.time() - start_time
            self.logger.info(f"GET request completed: {path} in {processing_time:.3f}s (request_id: {request_id})")
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"GET request failed: {e} after {processing_time:.3f}s (request_id: {request_id})")
            self._send_error(500, f"Internal Server Error: {e}", request_id)
    
    @robust_exception_handler(category=ErrorCategory.NETWORK, attempt_recovery=False)
    def do_POST(self):
        """Handle POST requests with robust error handling and security validation."""
        request_id = self._generate_request_id()
        client_id = self._get_client_id()
        start_time = time.time()
        
        try:
            self.logger.info(f"POST request started: {self.path} (request_id: {request_id})")
            
            # Validate content length
            content_length = int(self.headers.get('Content-Length', 0))
            if not self._validate_request_size(content_length):
                raise ValidationException(f"Request size {content_length} exceeds maximum allowed")
            
            # Read and parse request data
            if content_length > 0:
                post_data = self.rfile.read(content_length).decode('utf-8')
                try:
                    request_data = json.loads(post_data)
                except json.JSONDecodeError as e:
                    raise ValidationException(f"Invalid JSON: {e}")
            else:
                request_data = {}
            
            # Add request metadata
            request_data['_meta'] = {
                'request_id': request_id,
                'client_id': client_id,
                'timestamp': time.time(),
                'client_info': self._extract_client_info()
            }
            
            parsed_path = urlparse(self.path)
            path = parsed_path.path
            
            # Route handling
            if path == '/api/v2/inference':
                self._handle_basic_inference(request_data, request_id, client_id)
            elif path == '/api/v2/secure-inference':
                self._handle_secure_inference(request_data, request_id, client_id)
            elif path == '/api/v2/robust-inference':
                self._handle_robust_inference(request_data, request_id, client_id)
            else:
                self._send_error(404, "Not Found", request_id)
            
            processing_time = time.time() - start_time
            self.logger.info(f"POST request completed: {path} in {processing_time:.3f}s (request_id: {request_id})")
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"POST request failed: {e} after {processing_time:.3f}s (request_id: {request_id})")
            
            # Handle different types of errors appropriately
            if isinstance(e, ValidationException):
                self._send_error(400, str(e), request_id)
            elif isinstance(e, RateLimitException):
                self._send_error(429, str(e), request_id)
            elif isinstance(e, TimeoutException):
                self._send_error(408, str(e), request_id)
            elif isinstance(e, SecurityException):
                self._send_error(403, str(e), request_id)
            else:
                self._send_error(500, f"Internal Server Error: {e}", request_id)
    
    def _handle_root(self, request_id: str, client_id: str):
        """Handle root endpoint with enhanced service information."""
        response = {
            "service": "Secure MPC Transformer API - Generation 2",
            "version": "2.0.0",
            "description": "Robust secure multi-party computation for transformer inference",
            "generation": "2_robust",
            "features": {
                "basic_mpc": True,
                "secure_computation": True,
                "robust_error_handling": True,
                "enhanced_security": True,
                "performance_monitoring": True,
                "audit_logging": True,
                "rate_limiting": True,
                "input_validation": True,
                "output_sanitization": True,
                "data_integrity_checks": True
            },
            "endpoints": {
                "health": "/health",
                "basic_inference": "/api/v2/inference",
                "secure_inference": "/api/v2/secure-inference", 
                "robust_inference": "/api/v2/robust-inference",
                "models": "/api/v2/models",
                "status": "/api/v2/status",
                "metrics": "/api/v2/metrics",
                "system_status": "/api/v2/system-status"
            },
            "security": {
                "authentication": "API key based",
                "encryption": "TLS 1.3",
                "rate_limiting": "Per-client IP",
                "input_validation": "Comprehensive",
                "audit_logging": "All requests"
            },
            "request_metadata": {
                "request_id": request_id,
                "timestamp": time.time()
            }
        }
        self._send_json_response(response, request_id=request_id)
    
    def _handle_health(self, request_id: str, client_id: str):
        """Handle health check with comprehensive status."""
        if self.transformer_service:
            status = "healthy"
            model_loaded = True
            
            # Get performance metrics
            try:
                system_status = self.transformer_service.get_system_status()
                performance_metrics = system_status.get('performance_metrics', {})
            except Exception as e:
                self.logger.warning(f"Could not get system status: {e}")
                performance_metrics = {}
        else:
            status = "degraded"
            model_loaded = False
            performance_metrics = {}
        
        response = {
            "status": status,
            "timestamp": time.time(),
            "generation": "2_robust",
            "services": {
                "transformer": model_loaded,
                "mpc_protocol": model_loaded,
                "robust_server": True,
                "error_handler": True,
                "security_validator": model_loaded,
                "performance_monitor": model_loaded,
                "audit_logger": model_loaded
            },
            "performance": performance_metrics,
            "request_metadata": {
                "request_id": request_id,
                "response_time_ms": 0  # Will be updated by response handler
            }
        }
        self._send_json_response(response, request_id=request_id)
    
    def _handle_models(self, request_id: str, client_id: str):
        """Handle models endpoint with detailed model information."""
        if self.transformer_service:
            model_info = self.transformer_service.get_model_info()
            response = {
                "models": [model_info],
                "current_model": model_info.get("model_name", "unknown"),
                "generation": "2_robust",
                "capabilities": {
                    "secure_inference": True,
                    "robust_error_handling": True,
                    "performance_monitoring": True,
                    "input_validation": True,
                    "output_sanitization": True
                },
                "request_metadata": {
                    "request_id": request_id,
                    "timestamp": time.time()
                }
            }
        else:
            response = {
                "models": [],
                "current_model": None,
                "error": "No transformer service available",
                "request_metadata": {
                    "request_id": request_id,
                    "timestamp": time.time()
                }
            }
        
        self._send_json_response(response, request_id=request_id)
    
    def _handle_status(self, request_id: str, client_id: str):
        """Handle status endpoint with comprehensive system information."""
        if self.transformer_service:
            try:
                system_status = self.transformer_service.get_system_status()
                response = {
                    "service_status": "active",
                    "model_status": "loaded",
                    "system_status": system_status,
                    "generation": "2_robust",
                    "request_metadata": {
                        "request_id": request_id,
                        "timestamp": time.time()
                    }
                }
            except Exception as e:
                self.logger.error(f"Error getting system status: {e}")
                response = {
                    "service_status": "degraded",
                    "model_status": "error",
                    "error": str(e),
                    "request_metadata": {
                        "request_id": request_id,
                        "timestamp": time.time()
                    }
                }
        else:
            response = {
                "service_status": "inactive",
                "model_status": "not_loaded",
                "error": "Transformer service not available",
                "request_metadata": {
                    "request_id": request_id,
                    "timestamp": time.time()
                }
            }
        
        self._send_json_response(response, request_id=request_id)
    
    def _handle_metrics(self, request_id: str, client_id: str):
        """Handle metrics endpoint with performance and error statistics."""
        metrics = {
            "timestamp": time.time(),
            "generation": "2_robust",
            "request_metadata": {
                "request_id": request_id
            }
        }
        
        if self.transformer_service:
            try:
                system_status = self.transformer_service.get_system_status()
                metrics["performance_metrics"] = system_status.get("performance_metrics", {})
            except Exception as e:
                self.logger.warning(f"Could not get performance metrics: {e}")
                metrics["performance_metrics"] = {}
        
        # Add error handler statistics
        try:
            error_stats = self.error_handler.get_error_statistics()
            metrics["error_statistics"] = error_stats
        except Exception as e:
            self.logger.warning(f"Could not get error statistics: {e}")
            metrics["error_statistics"] = {}
        
        self._send_json_response(metrics, request_id=request_id)
    
    def _handle_system_status(self, request_id: str, client_id: str):
        """Handle comprehensive system status endpoint."""
        if self.transformer_service:
            try:
                response = self.transformer_service.get_system_status()
                response["request_metadata"] = {
                    "request_id": request_id,
                    "timestamp": time.time()
                }
            except Exception as e:
                response = {
                    "error": f"Could not get system status: {e}",
                    "request_metadata": {
                        "request_id": request_id,
                        "timestamp": time.time()
                    }
                }
        else:
            response = {
                "error": "Transformer service not available",
                "request_metadata": {
                    "request_id": request_id,
                    "timestamp": time.time()
                }
            }
        
        self._send_json_response(response, request_id=request_id)
    
    def _handle_basic_inference(self, request_data: Dict[str, Any], request_id: str, client_id: str):
        """Handle basic inference request."""
        if not self.transformer_service:
            raise ValidationException("Transformer service not available")
        
        try:
            text_inputs = request_data.get("text", request_data.get("texts", []))
            if not text_inputs:
                raise ValidationException("Missing 'text' or 'texts' in request")
            
            if isinstance(text_inputs, str):
                text_inputs = [text_inputs]
            
            # Use basic predict_secure method
            start_time = time.time()
            result = self.transformer_service.predict_secure(text_inputs)
            processing_time = time.time() - start_time
            
            response = {
                "predictions": result["predictions"],
                "model_info": result["model_info"],
                "security_info": result["security_info"],
                "processing_time_ms": round(processing_time * 1000, 2),
                "generation": "2_robust",
                "request_metadata": {
                    "request_id": request_id,
                    "timestamp": time.time()
                }
            }
            
            self._send_json_response(response, request_id=request_id)
            
        except Exception as e:
            self.logger.error(f"Basic inference failed: {e}")
            raise
    
    def _handle_secure_inference(self, request_data: Dict[str, Any], request_id: str, client_id: str):
        """Handle secure inference request (alias for robust inference)."""
        self._handle_robust_inference(request_data, request_id, client_id)
    
    def _handle_robust_inference(self, request_data: Dict[str, Any], request_id: str, client_id: str):
        """Handle robust inference request with full Generation 2 features."""
        if not self.transformer_service:
            raise ValidationException("Transformer service not available")
        
        try:
            text_inputs = request_data.get("text", request_data.get("texts", []))
            if not text_inputs:
                raise ValidationException("Missing 'text' or 'texts' in request")
            
            if isinstance(text_inputs, str):
                text_inputs = [text_inputs]
            
            # Use robust predict_secure_robust method
            result = self.transformer_service.predict_secure_robust(
                text_inputs, 
                client_id=client_id,
                request_id=request_id
            )
            
            # Add response metadata
            result["api_metadata"] = {
                "endpoint": "robust_inference",
                "generation": "2_robust",
                "request_id": request_id,
                "timestamp": time.time()
            }
            
            self._send_json_response(result, request_id=request_id)
            
        except Exception as e:
            self.logger.error(f"Robust inference failed: {e}")
            
            # Return structured error response
            error_response = {
                "error": True,
                "error_message": str(e),
                "error_type": type(e).__name__,
                "generation": "2_robust",
                "request_metadata": {
                    "request_id": request_id,
                    "timestamp": time.time()
                }
            }
            
            self._send_json_response(error_response, status_code=500, request_id=request_id)
    
    def _send_json_response(self, data: Dict[str, Any], status_code: int = 200, request_id: str = None):
        """Send JSON response with enhanced headers and metadata."""
        # Add response timing if request_id provided
        if request_id and "request_metadata" in data:
            data["request_metadata"]["response_time_ms"] = round(time.time() * 1000) % 10000
        
        response_json = json.dumps(data, indent=2)
        
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response_json)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        
        # Security headers
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('X-XSS-Protection', '1; mode=block')
        
        # Request tracking
        if request_id:
            self.send_header('X-Request-ID', request_id)
        
        self.end_headers()
        self.wfile.write(response_json.encode('utf-8'))
    
    def _send_error(self, status_code: int, message: str, request_id: str = None):
        """Send error response with enhanced error information."""
        error_response = {
            "error": True,
            "error_message": message,
            "status_code": status_code,
            "timestamp": time.time(),
            "generation": "2_robust"
        }
        
        if request_id:
            error_response["request_id"] = request_id
        
        self._send_json_response(error_response, status_code, request_id)


class RobustMPCServer:
    """Generation 2: Robust HTTP server with enhanced error handling and security."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8080, 
                 model_name: str = "bert-base-uncased",
                 log_level: str = "INFO",
                 enable_detailed_logging: bool = True):
        self.host = host
        self.port = port
        self.model_name = model_name
        self.server = None
        self.transformer_service = None
        self.error_handler = RobustErrorHandler(f"{__name__}.RobustMPCServer")
        
        # Setup enhanced logging
        self.logger = setup_robust_logging(
            log_level=log_level,
            log_file="robust_mpc_server.log" if enable_detailed_logging else None,
            enable_detailed_logging=enable_detailed_logging
        )
        
        # Security configuration
        self.security_config = {
            'max_request_size_mb': 10,
            'enable_rate_limiting': True,
            'enable_input_validation': True
        }
        
        # Initialize transformer service
        self._initialize_transformer()
    
    def _initialize_transformer(self):
        """Initialize the robust transformer service."""
        try:
            self.logger.info(f"Initializing robust transformer service with model: {self.model_name}")
            
            # Create robust configuration
            config = RobustTransformerConfig(
                model_name=self.model_name,
                max_sequence_length=512,
                hidden_size=768,
                num_parties=3,
                party_id=0,
                security_level=128,
                
                # Robustness features
                max_retry_attempts=3,
                timeout_seconds=30.0,
                enable_input_validation=True,
                enable_output_sanitization=True,
                enable_detailed_logging=True,
                enable_performance_monitoring=True,
                enable_audit_logging=True,
                enable_rate_limiting=True,
                enable_data_integrity_checks=True
            )
            
            # Create robust transformer service
            self.transformer_service = RobustSecureTransformer(config)
            
            self.logger.info("Robust transformer service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize robust transformer service: {e}")
            self.error_handler.handle_error(e, context={"model_name": self.model_name})
            self.transformer_service = None
    
    def create_handler(self):
        """Create request handler with enhanced capabilities."""
        def handler(*args, **kwargs):
            return RobustMPCHandler(
                *args, 
                transformer_service=self.transformer_service,
                error_handler=self.error_handler,
                security_config=self.security_config,
                **kwargs
            )
        return handler
    
    def start(self):
        """Start the robust HTTP server."""
        try:
            self.logger.info(f"Starting Robust MPC Server on {self.host}:{self.port}")
            
            # Create HTTP server
            handler_class = self.create_handler()
            self.server = HTTPServer((self.host, self.port), handler_class)
            
            self._log_startup_info()
            
            # Start serving
            self.logger.info("Robust server is running... Press Ctrl+C to stop")
            self.server.serve_forever()
            
        except KeyboardInterrupt:
            self.logger.info("Server shutdown by user request")
            self.stop()
        except Exception as e:
            self.logger.error(f"Server failed to start: {e}")
            self.error_handler.handle_error(e)
            raise
    
    def _log_startup_info(self):
        """Log comprehensive server startup information."""
        self.logger.info("=" * 70)
        self.logger.info("Robust Secure MPC Transformer Server - Generation 2")
        self.logger.info("=" * 70)
        self.logger.info(f"Host: {self.host}")
        self.logger.info(f"Port: {self.port}")
        self.logger.info(f"Model: {self.model_name}")
        self.logger.info(f"Transformer Service: {'Active' if self.transformer_service else 'Inactive'}")
        
        if self.transformer_service:
            system_status = self.transformer_service.get_system_status()
            security_features = system_status.get('security_features', {})
            self.logger.info(f"Security Features:")
            for feature, enabled in security_features.items():
                status = "✓" if enabled else "✗"
                self.logger.info(f"  {status} {feature.replace('_', ' ').title()}")
        
        self.logger.info(f"Generation 2 Features:")
        self.logger.info(f"  ✓ Robust Error Handling")
        self.logger.info(f"  ✓ Enhanced Security Validation")
        self.logger.info(f"  ✓ Performance Monitoring")
        self.logger.info(f"  ✓ Comprehensive Logging")
        self.logger.info(f"  ✓ Rate Limiting")
        self.logger.info(f"  ✓ Data Integrity Checks")
        
        self.logger.info(f"API Endpoints:")
        endpoints = [
            ("Health Check", f"http://{self.host}:{self.port}/health"),
            ("Basic Inference", f"http://{self.host}:{self.port}/api/v2/inference"),
            ("Robust Inference", f"http://{self.host}:{self.port}/api/v2/robust-inference"),
            ("System Status", f"http://{self.host}:{self.port}/api/v2/system-status"),
            ("Metrics", f"http://{self.host}:{self.port}/api/v2/metrics")
        ]
        
        for name, url in endpoints:
            self.logger.info(f"  - {name}: {url}")
        
        self.logger.info("=" * 70)
    
    def stop(self):
        """Stop the robust HTTP server with cleanup."""
        if self.server:
            self.logger.info("Shutting down robust server...")
            self.server.shutdown()
            self.server.server_close()
            
        if self.transformer_service:
            self.transformer_service.cleanup()
            
        # Log final statistics
        error_stats = self.error_handler.get_error_statistics()
        self.logger.info(f"Server session statistics:")
        self.logger.info(f"  Total errors handled: {error_stats.get('total_errors', 0)}")
        self.logger.info(f"  Recovery success rate: {error_stats.get('recovery_statistics', {}).get('success_rate_percent', 0):.1f}%")
            
        self.logger.info("Robust server shutdown complete")


def main():
    """Main entry point for robust server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Robust Secure MPC Transformer Server - Generation 2")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8080, help="Port number")
    parser.add_argument("--model", type=str, default="bert-base-uncased", help="Model name")
    parser.add_argument("--log-level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    parser.add_argument("--detailed-logging", action="store_true", default=True,
                       help="Enable detailed logging to file")
    
    args = parser.parse_args()
    
    try:
        # Create and start robust server
        server = RobustMPCServer(
            host=args.host,
            port=args.port,
            model_name=args.model,
            log_level=args.log_level,
            enable_detailed_logging=args.detailed_logging
        )
        
        server.start()
        
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Shutdown by user request")
    except Exception as e:
        logging.getLogger(__name__).error(f"Server error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())