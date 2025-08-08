#!/usr/bin/env python3
"""
Main entry point for the Secure MPC Transformer system.

This script provides a unified interface to start and manage the secure 
multi-party computation transformer service with quantum-inspired task planning.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from secure_mpc_transformer.server import main as server_main
from secure_mpc_transformer.utils.error_handling import setup_logging


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Secure MPC Transformer with Quantum-Inspired Task Planning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py server --host 0.0.0.0 --port 8080
  python main.py server --config config/production.json
  python main.py --log-level DEBUG server --port 8080
        """
    )
    
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Set the logging level")
    parser.add_argument("--log-file", type=str, help="Log to file instead of console")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Start the API server")
    server_parser.add_argument("--config", "-c", type=str, help="Configuration file path")
    server_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    server_parser.add_argument("--port", type=int, default=8080, help="Port number")
    server_parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    server_parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(
        log_level=args.log_level,
        log_file=args.log_file
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Secure MPC Transformer System")
    logger.info(f"Command: {args.command}")
    logger.info(f"Log level: {args.log_level}")
    
    try:
        if args.command == "server":
            # Update sys.argv to pass arguments to server_main
            server_args = ["server"]
            if args.config:
                server_args.extend(["--config", args.config])
            if args.host != "0.0.0.0":
                server_args.extend(["--host", args.host])
            if args.port != 8080:
                server_args.extend(["--port", str(args.port)])
            if args.log_level != "INFO":
                server_args.extend(["--log-level", args.log_level])
            
            # Replace sys.argv for server_main
            original_argv = sys.argv[:]
            sys.argv = ["server"] + server_args[1:]
            
            try:
                server_main()
            finally:
                sys.argv = original_argv
        
        else:
            parser.print_help()
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"System error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()