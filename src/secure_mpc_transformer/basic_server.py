"""Basic HTTP server for Generation 1 - Works without FastAPI dependencies."""

import json
import logging
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any
import threading

from .models.basic_transformer import BasicSecureTransformer, BasicTransformerConfig
from .config import SecurityConfig
from .utils.error_handling import setup_logging

logger = logging.getLogger(__name__)


class BasicMPCHandler(BaseHTTPRequestHandler):
    """HTTP request handler for basic MPC transformer service."""
    
    def __init__(self, *args, transformer_service=None, **kwargs):
        self.transformer_service = transformer_service
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        try:
            if path == '/':
                self._handle_root()
            elif path == '/health':
                self._handle_health()
            elif path == '/api/v1/models':
                self._handle_models()
            elif path == '/api/v1/status':
                self._handle_status()
            else:
                self._send_error(404, "Not Found")
        except Exception as e:
            logger.error(f"GET request failed: {e}")
            self._send_error(500, f"Internal Server Error: {e}")
    
    def do_POST(self):
        """Handle POST requests."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                post_data = self.rfile.read(content_length).decode('utf-8')
                try:
                    request_data = json.loads(post_data)
                except json.JSONDecodeError:
                    self._send_error(400, "Invalid JSON")
                    return
            else:
                request_data = {}
            
            if path == '/api/v1/inference':
                self._handle_inference(request_data)
            elif path == '/api/v1/secure-inference':
                self._handle_secure_inference(request_data)
            else:
                self._send_error(404, "Not Found")
        except Exception as e:
            logger.error(f"POST request failed: {e}")
            self._send_error(500, f"Internal Server Error: {e}")
    
    def _handle_root(self):
        """Handle root endpoint."""
        response = {
            "service": "Secure MPC Transformer API - Generation 1",
            "version": "1.0.0",
            "description": "Basic secure multi-party computation for transformer inference",
            "endpoints": {
                "health": "/health",
                "inference": "/api/v1/inference",
                "secure_inference": "/api/v1/secure-inference",
                "models": "/api/v1/models",
                "status": "/api/v1/status"
            },
            "features": {
                "basic_mpc": True,
                "secure_computation": True,
                "minimal_dependencies": True,
                "generation": "1_basic"
            }
        }
        self._send_json_response(response)
    
    def _handle_health(self):
        """Handle health check endpoint."""
        if self.transformer_service:
            status = "healthy"
            model_loaded = True
        else:
            status = "degraded"
            model_loaded = False
        
        response = {
            "status": status,
            "timestamp": time.time(),
            "services": {
                "transformer": model_loaded,
                "mpc_protocol": model_loaded,
                "basic_server": True
            },
            "generation": "1_basic"
        }
        self._send_json_response(response)
    
    def _handle_models(self):
        """Handle models endpoint."""
        if self.transformer_service:
            model_info = self.transformer_service.get_model_info()
            response = {
                "models": [model_info],
                "current_model": model_info.get("model_name", "unknown"),
                "generation": "1_basic"
            }
        else:
            response = {
                "models": [],
                "current_model": None,
                "error": "No transformer service available"
            }
        
        self._send_json_response(response)
    
    def _handle_status(self):
        """Handle status endpoint."""
        if self.transformer_service:
            model_info = self.transformer_service.get_model_info()
            response = {
                "service_status": "active",
                "model_status": "loaded",
                "model_info": model_info,
                "generation": "1_basic",
                "features": {
                    "secure_computation": True,
                    "basic_mpc": True,
                    "minimal_dependencies": True
                }
            }
        else:
            response = {
                "service_status": "inactive",
                "model_status": "not_loaded",
                "error": "Transformer service not available"
            }
        
        self._send_json_response(response)
    
    def _handle_inference(self, request_data: Dict[str, Any]):
        """Handle basic inference request."""
        if not self.transformer_service:
            self._send_error(503, "Transformer service not available")
            return
        
        try:
            text_inputs = request_data.get("text", request_data.get("texts", []))
            if not text_inputs:
                self._send_error(400, "Missing 'text' or 'texts' in request")
                return
            
            if isinstance(text_inputs, str):
                text_inputs = [text_inputs]
            
            # For basic inference, use the transformer directly (non-secure)
            start_time = time.time()
            
            # Preprocess
            inputs = self.transformer_service.preprocess_text(text_inputs)
            
            # Basic forward pass (non-secure for comparison)
            outputs = self.transformer_service.transformer.forward(inputs["input_ids"])
            
            processing_time = time.time() - start_time
            
            # Format response
            predictions = []
            for i, text in enumerate(text_inputs):
                prediction = {
                    "text": text,
                    "processed": True,
                    "tokens": len(self.transformer_service.tokenizer.tokenize(text)),
                    "secure": False
                }
                predictions.append(prediction)
            
            response = {
                "predictions": predictions,
                "processing_time": processing_time,
                "model": self.transformer_service.config.model_name,
                "secure_computation": False,
                "generation": "1_basic"
            }
            
            self._send_json_response(response)
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            self._send_error(500, f"Inference error: {e}")
    
    def _handle_secure_inference(self, request_data: Dict[str, Any]):
        """Handle secure inference request."""
        if not self.transformer_service:
            self._send_error(503, "Transformer service not available")
            return
        
        try:
            text_inputs = request_data.get("text", request_data.get("texts", []))
            if not text_inputs:
                self._send_error(400, "Missing 'text' or 'texts' in request")
                return
            
            if isinstance(text_inputs, str):
                text_inputs = [text_inputs]
            
            # Perform secure inference
            result = self.transformer_service.predict_secure(text_inputs)
            
            response = {
                "secure_predictions": result["predictions"],
                "model_info": result["model_info"],
                "security_info": result["security_info"],
                "generation": "1_basic"
            }
            
            self._send_json_response(response)
            
        except Exception as e:
            logger.error(f"Secure inference failed: {e}")
            self._send_error(500, f"Secure inference error: {e}")
    
    def _send_json_response(self, data: Dict[str, Any], status_code: int = 200):
        """Send JSON response."""
        response_json = json.dumps(data, indent=2)
        
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response_json)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        self.wfile.write(response_json.encode('utf-8'))
    
    def _send_error(self, status_code: int, message: str):
        """Send error response."""
        error_response = {
            "error": message,
            "status_code": status_code,
            "timestamp": time.time()
        }
        self._send_json_response(error_response, status_code)
    
    def log_message(self, format, *args):
        """Override to use our logger."""
        logger.info(f"{self.address_string()} - {format % args}")


class BasicMPCServer:
    """Basic HTTP server for MPC transformer service."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8080, 
                 model_name: str = "bert-base-uncased"):
        self.host = host
        self.port = port
        self.model_name = model_name
        self.server = None
        self.transformer_service = None
        
        # Initialize transformer service
        self._initialize_transformer()
    
    def _initialize_transformer(self):
        """Initialize the transformer service."""
        try:
            logger.info(f"Initializing transformer service with model: {self.model_name}")
            
            # Create basic configuration
            config = BasicTransformerConfig(
                model_name=self.model_name,
                max_sequence_length=512,
                hidden_size=768,
                num_parties=3,
                party_id=0,
                security_level=128
            )
            
            # Create transformer service
            self.transformer_service = BasicSecureTransformer(config)
            
            logger.info("Transformer service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize transformer service: {e}")
            self.transformer_service = None
    
    def create_handler(self):
        """Create request handler with transformer service."""
        def handler(*args, **kwargs):
            return BasicMPCHandler(*args, transformer_service=self.transformer_service, **kwargs)
        return handler
    
    def start(self):
        """Start the HTTP server."""
        try:
            logger.info(f"Starting Basic MPC Server on {self.host}:{self.port}")
            
            # Create HTTP server
            handler_class = self.create_handler()
            self.server = HTTPServer((self.host, self.port), handler_class)
            
            logger.info("=" * 60)
            logger.info("Basic Secure MPC Transformer Server - Generation 1")
            logger.info("=" * 60)
            logger.info(f"Host: {self.host}")
            logger.info(f"Port: {self.port}")
            logger.info(f"Model: {self.model_name}")
            logger.info(f"Transformer Service: {'Active' if self.transformer_service else 'Inactive'}")
            logger.info(f"Endpoints:")
            logger.info(f"  - Health: http://{self.host}:{self.port}/health")
            logger.info(f"  - Basic Inference: http://{self.host}:{self.port}/api/v1/inference")
            logger.info(f"  - Secure Inference: http://{self.host}:{self.port}/api/v1/secure-inference")
            logger.info(f"  - Models: http://{self.host}:{self.port}/api/v1/models")
            logger.info(f"  - Status: http://{self.host}:{self.port}/api/v1/status")
            logger.info("=" * 60)
            
            # Start serving
            logger.info("Server is running... Press Ctrl+C to stop")
            self.server.serve_forever()
            
        except KeyboardInterrupt:
            logger.info("Server shutdown by user request")
            self.stop()
        except Exception as e:
            logger.error(f"Server failed to start: {e}")
            raise
    
    def stop(self):
        """Stop the HTTP server."""
        if self.server:
            logger.info("Shutting down server...")
            self.server.shutdown()
            self.server.server_close()
            
        if self.transformer_service:
            self.transformer_service.cleanup()
            
        logger.info("Server shutdown complete")


def main():
    """Main entry point for basic server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Basic Secure MPC Transformer Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8080, help="Port number")
    parser.add_argument("--model", type=str, default="bert-base-uncased", help="Model name")
    parser.add_argument("--log-level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    
    try:
        # Create and start server
        server = BasicMPCServer(
            host=args.host,
            port=args.port,
            model_name=args.model
        )
        
        server.start()
        
    except KeyboardInterrupt:
        logger.info("Shutdown by user request")
    except Exception as e:
        logger.error(f"Server error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())