#!/usr/bin/env python3
"""
Enhanced Main Entry Point for Secure MPC Transformer System
Generation 1 Enhancement: Improved Initialization and Error Handling
"""

import sys
import os
import asyncio
import logging
import signal
from pathlib import Path
from typing import Optional, Dict, Any
import time

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from secure_mpc_transformer.server import SecureMPCServer, create_app
    from secure_mpc_transformer.config import SecurityConfig, get_default_config
    from secure_mpc_transformer.utils.error_handling import setup_logging
    from secure_mpc_transformer.cli import main as cli_main
    SERVER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Server components not fully available: {e}")
    SERVER_AVAILABLE = False


class EnhancedMPCLauncher:
    """Enhanced launcher with improved initialization and error handling."""
    
    def __init__(self):
        self.shutdown_requested = False
        self.startup_time = time.time()
        self.logger = None
        
    def setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers."""
        def signal_handler(signum, frame):
            if self.logger:
                self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.shutdown_requested = True
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def initialize_logging(self, log_level: str = "INFO", log_file: Optional[str] = None):
        """Initialize enhanced logging system."""
        if SERVER_AVAILABLE:
            setup_logging(log_level=log_level, log_file=log_file)
        else:
            # Fallback logging setup
            logging.basicConfig(
                level=getattr(logging, log_level.upper(), logging.INFO),
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                filename=log_file if log_file else None
            )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Enhanced MPC Launcher initialized")
        return self.logger
    
    def check_system_requirements(self) -> Dict[str, bool]:
        """Check system requirements and dependencies."""
        requirements = {
            "python_version": sys.version_info >= (3, 10),
            "server_components": SERVER_AVAILABLE,
            "config_accessible": True,
            "write_permissions": True
        }
        
        # Check config directory
        try:
            config_dir = Path(__file__).parent / "config"
            config_dir.mkdir(exist_ok=True)
            test_file = config_dir / "test_write.tmp"
            test_file.write_text("test")
            test_file.unlink()
        except Exception:
            requirements["config_accessible"] = False
            requirements["write_permissions"] = False
        
        return requirements
    
    def create_default_config(self) -> Dict[str, Any]:
        """Create enhanced default configuration."""
        if SERVER_AVAILABLE:
            config = get_default_config()
        else:
            config = {
                "server": {"host": "0.0.0.0", "port": 8080},
                "security": {"protocol_name": "aby3", "security_level": 128},
                "fallback_mode": True
            }
        
        # Add enhanced defaults
        config.update({
            "enhanced_features": {
                "auto_recovery": True,
                "graceful_shutdown": True,
                "performance_monitoring": True,
                "health_checks": True
            },
            "startup": {
                "max_init_time": 300,  # 5 minutes
                "retry_attempts": 3,
                "checkpoint_logging": True
            }
        })
        
        return config
    
    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """Validate configuration with enhanced error reporting."""
        try:
            if SERVER_AVAILABLE and "security" in config:
                # Validate security config if available
                security_cfg = SecurityConfig(**config["security"])
                security_cfg.validate()
            
            # Validate basic structure
            required_sections = ["server"]
            for section in required_sections:
                if section not in config:
                    self.logger.error(f"Missing required config section: {section}")
                    return False
            
            self.logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    def start_server_mode(self, config: Dict[str, Any]) -> int:
        """Start server with enhanced initialization."""
        if not SERVER_AVAILABLE:
            self.logger.error("Server components not available. Please install required dependencies.")
            return 1
        
        try:
            self.logger.info("Initializing Secure MPC Transformer Server...")
            
            # Create server with enhanced configuration
            server = SecureMPCServer()
            server.config.update(config)
            
            # Pre-flight checks
            self.logger.info("Running pre-flight checks...")
            
            # Check GPU availability if enabled
            if config.get("security", {}).get("gpu_acceleration", False):
                try:
                    import torch
                    if torch.cuda.is_available():
                        self.logger.info(f"GPU detected: {torch.cuda.get_device_name()}")
                    else:
                        self.logger.warning("GPU acceleration requested but no CUDA GPU available")
                except ImportError:
                    self.logger.warning("PyTorch not available - GPU acceleration disabled")
            
            # Log system info
            startup_duration = time.time() - self.startup_time
            self.logger.info(f"Startup completed in {startup_duration:.2f}s")
            
            # Start server
            self.logger.info("Starting FastAPI server...")
            server.run()
            
            return 0
            
        except KeyboardInterrupt:
            self.logger.info("Server shutdown by user request")
            return 0
        except Exception as e:
            self.logger.error(f"Server startup failed: {e}", exc_info=True)
            return 1
    
    def start_cli_mode(self, args) -> int:
        """Start CLI mode with fallback handling."""
        try:
            if SERVER_AVAILABLE:
                return cli_main()
            else:
                self.logger.info("Running in basic CLI mode (limited functionality)")
                print("Secure MPC Transformer CLI")
                print("Limited mode - server components not available")
                print("Available commands: config, version, check")
                return self.handle_basic_cli(args)
        except Exception as e:
            if self.logger:
                self.logger.error(f"CLI mode failed: {e}")
            return 1
    
    def handle_basic_cli(self, args) -> int:
        """Handle basic CLI commands when full server not available."""
        if hasattr(args, 'command'):
            if args.command == "config":
                config = self.create_default_config()
                print("Default configuration:")
                import json
                print(json.dumps(config, indent=2))
            elif args.command == "version":
                print("Secure MPC Transformer v0.3.0 (Basic Mode)")
            elif args.command == "check":
                requirements = self.check_system_requirements()
                print("System Requirements Check:")
                for req, status in requirements.items():
                    status_str = "✓" if status else "✗"
                    print(f"  {status_str} {req}")
            else:
                print(f"Command '{args.command}' not available in basic mode")
                return 1
        else:
            print("No command specified. Use --help for available options.")
        return 0
    
    def create_health_check(self) -> Dict[str, Any]:
        """Create system health check."""
        requirements = self.check_system_requirements()
        
        health = {
            "status": "healthy" if all(requirements.values()) else "degraded",
            "timestamp": time.time(),
            "uptime": time.time() - self.startup_time,
            "requirements": requirements,
            "components": {
                "server": SERVER_AVAILABLE,
                "logging": self.logger is not None,
                "config": True
            }
        }
        
        return health


def main():
    """Enhanced main entry point with improved error handling."""
    launcher = EnhancedMPCLauncher()
    
    try:
        import argparse
        
        parser = argparse.ArgumentParser(
            description="Enhanced Secure MPC Transformer System",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python enhanced_main_entry_point.py server --port 8080
  python enhanced_main_entry_point.py cli
  python enhanced_main_entry_point.py check
  python enhanced_main_entry_point.py config --output config.json
            """
        )
        
        parser.add_argument("--log-level", type=str, default="INFO",
                           choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                           help="Set logging level")
        parser.add_argument("--log-file", type=str, help="Log to file")
        parser.add_argument("--config", type=str, help="Configuration file")
        
        subparsers = parser.add_subparsers(dest="mode", help="Operation mode")
        
        # Server mode
        server_parser = subparsers.add_parser("server", help="Start API server")
        server_parser.add_argument("--host", default="0.0.0.0", help="Host address")
        server_parser.add_argument("--port", type=int, default=8080, help="Port number")
        server_parser.add_argument("--workers", type=int, default=1, help="Worker processes")
        
        # CLI mode  
        cli_parser = subparsers.add_parser("cli", help="Start CLI interface")
        
        # Utility commands
        check_parser = subparsers.add_parser("check", help="System health check")
        config_parser = subparsers.add_parser("config", help="Configuration utilities")
        config_parser.add_argument("--output", help="Output config to file")
        
        args = parser.parse_args()
        
        # Setup signal handlers and logging
        launcher.setup_signal_handlers()
        logger = launcher.initialize_logging(args.log_level, args.log_file)
        
        # System requirements check
        requirements = launcher.check_system_requirements()
        if not requirements["python_version"]:
            logger.error("Python 3.10+ required")
            return 1
        
        # Load/create configuration
        if args.config and Path(args.config).exists():
            import json
            with open(args.config) as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {args.config}")
        else:
            config = launcher.create_default_config()
            logger.info("Using default configuration")
        
        # Apply CLI overrides
        if hasattr(args, 'host') and args.host:
            config.setdefault("server", {})["host"] = args.host
        if hasattr(args, 'port') and args.port:
            config.setdefault("server", {})["port"] = args.port
        if hasattr(args, 'workers') and args.workers:
            config.setdefault("server", {})["workers"] = args.workers
        
        # Validate configuration
        if not launcher.validate_configuration(config):
            logger.error("Configuration validation failed")
            return 1
        
        # Handle different modes
        if args.mode == "server":
            return launcher.start_server_mode(config)
        elif args.mode == "cli":
            return launcher.start_cli_mode(args)
        elif args.mode == "check":
            health = launcher.create_health_check()
            print("System Health Check:")
            import json
            print(json.dumps(health, indent=2))
            return 0 if health["status"] == "healthy" else 1
        elif args.mode == "config":
            if args.output:
                import json
                with open(args.output, 'w') as f:
                    json.dump(config, f, indent=2)
                print(f"Configuration saved to {args.output}")
            else:
                import json
                print(json.dumps(config, indent=2))
            return 0
        else:
            # Default to server mode
            logger.info("No mode specified, defaulting to server mode")
            return launcher.start_server_mode(config)
            
    except KeyboardInterrupt:
        if launcher.logger:
            launcher.logger.info("Shutdown by user request")
        return 0
    except Exception as e:
        if launcher.logger:
            launcher.logger.error(f"Launcher failed: {e}", exc_info=True)
        else:
            print(f"Critical error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())