#!/usr/bin/env python3
"""
Benchmark: MPC inference accuracy vs plaintext, and communication cost.

Generates a synthetic sequence classification task, runs both MPC and
plaintext forward passes, and reports:
  1. Prediction agreement (MPC ≈ plaintext)
  2. Mean logit MSE (quantifies numerical error from poly approximations)
  3. Communication cost breakdown

Usage:
    python benchmark.py
    python benchmark.py --n_samples 200 --seq_len 8 --d_model 16
"""

import argparse
import time
import numpy as np
from secure_mpc import MPCTransformer, CommCostEstimator, SecretSharer


def main():
    parser = argparse.ArgumentParser(description="MPC transformer benchmark")
    parser.add_argument("--n_samples", type=int, default=50, help="Inference samples")
    parser.add_argument("--seq_len", type=int, default=8, help="Sequence length")
    parser.add_argument("--d_model", type=int, default=16, help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=2, help="Attention heads")
    parser.add_argument("--d_ff", type=int, default=32, help="FFN hidden dim")
    parser.add_argument("--n_layers", type=int, default=2, help="Transformer blocks")
    parser.add_argument("--n_classes", type=int, default=4, help="Output classes")
    parser.add_argument("--poly_deg", type=int, default=2, choices=[1, 2, 3],
                        help="Softmax polynomial degree")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"\n{'='*60}")
    print(f"  Secure MPC Transformer — Accuracy & Cost Benchmark")
    print(f"{'='*60}")
    print(f"  d_model={args.d_model}, n_heads={args.n_heads}, d_ff={args.d_ff}")
    print(f"  n_layers={args.n_layers}, seq_len={args.seq_len}, n_classes={args.n_classes}")
    print(f"  poly_softmax_degree={args.poly_deg}, n_samples={args.n_samples}")

    # Build model
    sharer = SecretSharer(n_parties=2, rng=np.random.default_rng(args.seed))
    model = MPCTransformer.random_init(
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        n_classes=args.n_classes,
        seed=args.seed,
        sharer=sharer,
    )

    # Generate random "embedded" inputs — in a real system these would
    # be secret-shared embeddings of the client's tokenized input.
    inputs = [
        rng.standard_normal((args.seq_len, args.d_model)).astype(float)
        for _ in range(args.n_samples)
    ]

    # --- Plaintext pass ---
    print(f"\n[1/3] Running plaintext inference ({args.n_samples} samples)...")
    pt_start = time.perf_counter()
    pt_preds = [model.predict(x, mpc=False) for x in inputs]
    pt_logits = [model.plaintext_forward(x) for x in inputs]
    pt_time = time.perf_counter() - pt_start

    # --- MPC pass ---
    print(f"[2/3] Running MPC inference ({args.n_samples} samples)...")
    mpc_start = time.perf_counter()
    mpc_preds = [model.predict(x, mpc=True) for x in inputs]
    mpc_logits = [model.forward(x) for x in inputs]
    mpc_time = time.perf_counter() - mpc_start

    # --- Accuracy comparison ---
    agreement = sum(p == m for p, m in zip(pt_preds, mpc_preds)) / args.n_samples
    logit_mse = float(np.mean([
        np.mean((pl - ml) ** 2)
        for pl, ml in zip(pt_logits, mpc_logits)
    ]))

    print(f"\n{'─'*60}")
    print(f"  ACCURACY RESULTS")
    print(f"{'─'*60}")
    print(f"  Prediction agreement (MPC == plaintext): {agreement*100:.1f}%")
    print(f"  Mean logit MSE (MPC vs plaintext):        {logit_mse:.6f}")
    print(f"  Plaintext inference time:  {pt_time*1000:.1f} ms total  "
          f"({pt_time/args.n_samples*1000:.2f} ms/sample)")
    print(f"  MPC inference time:        {mpc_time*1000:.1f} ms total  "
          f"({mpc_time/args.n_samples*1000:.2f} ms/sample)")

    # Note on overhead
    overhead = mpc_time / pt_time if pt_time > 0 else float("inf")
    print(f"  MPC overhead factor:       {overhead:.1f}×  (simulation; real MPC "
          f"cost is dominated by network round-trips)")

    # --- Communication cost ---
    print(f"\n{'─'*60}")
    print(f"  COMMUNICATION COST ESTIMATE (analytical)")
    print(f"{'─'*60}")
    estimator = CommCostEstimator(
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        seq_len=args.seq_len,
        n_classes=args.n_classes,
        poly_deg=args.poly_deg,
    )
    print(estimator.report())

    # Also report actual tracked rounds from last MPC forward pass
    print(f"  Actual tracked rounds (last sample):  {model.comm_rounds}")
    print(f"  Actual tracked elements (last sample): {model.comm_elements:,}")

    print(f"\n  Done.\n")


if __name__ == "__main__":
    main()
