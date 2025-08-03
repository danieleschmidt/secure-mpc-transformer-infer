"""Command-line interface for secure MPC transformer."""

import asyncio
import argparse
import sys
import os
from typing import Optional
import logging

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from .config import SecurityConfig, ProtocolType, SecurityLevel
from .models.secure_transformer import SecureTransformer
from .protocols.factory import ProtocolFactory


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_security_config(args) -> SecurityConfig:
    """Create security configuration from CLI arguments."""
    return SecurityConfig(
        protocol=ProtocolType(args.protocol),
        security_level=SecurityLevel(args.security_level),
        gpu_acceleration=args.gpu,
        party_id=args.party_id,
        num_parties=args.num_parties,
        host=args.host,
        port=args.port,
        use_tls=args.tls,
        batch_size=args.batch_size,
        num_threads=args.threads
    )


def cmd_inference(args) -> None:
    """Run secure inference."""
    print(f"Running secure inference with model: {args.model}")
    print(f"Input text: {args.text}")
    
    # Create security configuration
    config = create_security_config(args)
    config.validate()
    
    # Initialize secure transformer
    model = SecureTransformer.from_pretrained(args.model, config)
    
    # Perform secure inference
    result = model.predict_secure(args.text)
    
    print(f"\nResults:")
    print(f"  Predicted text: {result.decoded_text}")
    print(f"  Latency: {result.latency_ms:.2f} ms")
    print(f"  Protocol: {config.protocol.value}")
    print(f"  Security level: {config.security_level.value} bits")
    
    if args.verbose:
        print(f"\nDetailed statistics:")
        for key, value in result.computation_stats.items():
            print(f"  {key}: {value}")


def cmd_benchmark(args) -> None:
    """Run performance benchmark."""
    print(f"Benchmarking model: {args.model}")
    
    config = create_security_config(args)
    config.validate()
    
    model = SecureTransformer.from_pretrained(args.model, config)
    
    print(f"Running {args.iterations} iterations...")
    results = model.benchmark(num_inferences=args.iterations, sequence_length=args.sequence_length)
    
    print(f"\nBenchmark Results:")
    print(f"  Average latency: {results['avg_latency_ms']:.2f} ms")
    print(f"  Min latency: {results['min_latency_ms']:.2f} ms")
    print(f"  Max latency: {results['max_latency_ms']:.2f} ms")
    print(f"  Throughput: {results['throughput_inferences_per_sec']:.2f} inferences/sec")


def cmd_protocols(args) -> None:
    """List available protocols."""
    print("Available MPC Protocols:")
    print("========================")
    
    protocols = ProtocolFactory.get_available_protocols()
    
    for protocol_name in protocols:
        try:
            info = ProtocolFactory.get_protocol_info(protocol_name)
            print(f"\n{protocol_name}:")
            print(f"  Description: {info.get('description', 'N/A')}")
            print(f"  Security Model: {info.get('security_model', 'N/A')}")
            print(f"  Performance: {info.get('performance', 'N/A')}")
            if 'features' in info:
                print(f"  Features: {', '.join(info['features'])}")
        except Exception as e:
            print(f"\n{protocol_name}: Error getting info - {e}")


def cmd_server(args) -> None:
    """Start MPC server (placeholder)."""
    print(f"Starting MPC server on {args.host}:{args.port}")
    print(f"Party ID: {args.party_id}/{args.num_parties}")
    print(f"Protocol: {args.protocol}")
    
    # This would start the actual server
    print("Server functionality not yet implemented.")
    print("This would start a gRPC server for multi-party computation.")


def cmd_test_protocol(args) -> None:
    """Test protocol functionality."""
    print(f"Testing protocol: {args.protocol}")
    
    config = create_security_config(args)
    protocol = ProtocolFactory.create_from_config(config)
    
    print(f"Protocol info: {protocol.get_protocol_info()}")
    
    # Run basic test if available
    if hasattr(protocol, 'benchmark_operations'):
        print("Running protocol benchmark...")
        results = protocol.benchmark_operations(num_ops=10)
        
        print("Benchmark results:")
        for operation, time_ms in results.items():
            print(f"  {operation}: {time_ms:.2f} ms")
    else:
        print("No benchmark available for this protocol.")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Secure MPC Transformer - Privacy-preserving transformer inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s inference --text "Hello world" --model bert-base-uncased
  %(prog)s benchmark --model bert-base-uncased --iterations 10
  %(prog)s protocols
  %(prog)s server --party-id 0 --num-parties 3
        """
    )
    
    # Global arguments
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--protocol', default='semi_honest_3pc', 
                       choices=['semi_honest_3pc', 'malicious_3pc', 'aby3', 'replicated_3pc'],
                       help='MPC protocol to use')
    parser.add_argument('--security-level', type=int, default=128, choices=[80, 128, 256],
                       help='Security level in bits')
    parser.add_argument('--gpu', action='store_true', help='Enable GPU acceleration')
    parser.add_argument('--party-id', type=int, default=0, help='Party ID (0-based)')
    parser.add_argument('--num-parties', type=int, default=3, help='Total number of parties')
    parser.add_argument('--host', default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=50051, help='Server port')
    parser.add_argument('--tls', action='store_true', help='Enable TLS')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--threads', type=int, default=8, help='Number of threads')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Inference command
    inference_parser = subparsers.add_parser('inference', help='Run secure inference')
    inference_parser.add_argument('--text', required=True, help='Input text for inference')
    inference_parser.add_argument('--model', default='bert-base-uncased', help='Model name')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run performance benchmark')
    benchmark_parser.add_argument('--model', default='bert-base-uncased', help='Model name')
    benchmark_parser.add_argument('--iterations', type=int, default=10, help='Number of iterations')
    benchmark_parser.add_argument('--sequence-length', type=int, default=128, help='Input sequence length')
    
    # Protocols command
    subparsers.add_parser('protocols', help='List available protocols')
    
    # Server command
    server_parser = subparsers.add_parser('server', help='Start MPC server')
    
    # Test protocol command
    test_parser = subparsers.add_parser('test-protocol', help='Test protocol functionality')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Execute command
    if args.command == 'inference':
        cmd_inference(args)
    elif args.command == 'benchmark':
        cmd_benchmark(args)
    elif args.command == 'protocols':
        cmd_protocols(args)
    elif args.command == 'server':
        cmd_server(args)
    elif args.command == 'test-protocol':
        cmd_test_protocol(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
