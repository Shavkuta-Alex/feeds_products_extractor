"""
CLI for Best-Seller Prediction Pipeline
=========================================
Modes:
  demo           — run on synthetic data (no files needed)
  train-profile  — train one-class model on best-seller feeds (Phase 1)
  train-binary   — train binary classifier with best-seller + general feeds (Phase 2)
  score          — score new products with a saved model
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

from pipeline import (
    ProductScoringPipeline,
    load_feed,
    normalize_columns,
)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_and_tag_feed(spec: str) -> pd.DataFrame:
    """Parse 'path/to/file.csv:store_name' and load."""
    if ":" in spec and not spec.startswith("/"):
        # Could be ambiguous with Windows paths, but we're on Mac/Linux
        parts = spec.rsplit(":", 1)
        path, store = parts[0], parts[1]
    elif ":" in spec:
        # Absolute path — find last colon that isn't part of the path
        idx = spec.rfind(":")
        path, store = spec[:idx], spec[idx + 1:]
    else:
        path, store = spec, None

    print(f"  Loading {path}" + (f" (store: {store})" if store else "") + "...")
    df = load_feed(path, store_source=store)
    print(f"    → {len(df):,} rows, {len(df.columns)} columns")
    return df


def load_multi_store(feed_specs: list) -> pd.DataFrame:
    """Load multiple feeds and concatenate."""
    dfs = [load_and_tag_feed(spec) for spec in feed_specs]
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Combined: {len(combined):,} rows")
    return combined


def create_binary_dataset(
    best_sellers: pd.DataFrame,
    general: pd.DataFrame,
    negative_ratio: float = 3.0,
) -> tuple:
    """
    Anti-join general against best sellers to derive negatives.
    Match on product_id, or fallback to title+price.
    Returns (combined_df, labels).
    """
    # Try matching on product_id
    if "product_id" in best_sellers.columns and "product_id" in general.columns:
        bs_ids = set(best_sellers["product_id"].dropna().unique())
        neg_mask = ~general["product_id"].isin(bs_ids)
    else:
        # Fallback: match on title + price
        bs_keys = set(
            best_sellers["title"].fillna("").str.lower() + "|" +
            best_sellers["price"].astype(str)
        )
        gen_keys = general["title"].fillna("").str.lower() + "|" + general["price"].astype(str)
        neg_mask = ~gen_keys.isin(bs_keys)

    negatives = general[neg_mask]
    print(f"  Negatives found: {len(negatives):,} (from {len(general):,} general products)")

    # Downsample negatives
    max_neg = int(len(best_sellers) * negative_ratio)
    if len(negatives) > max_neg:
        negatives = negatives.sample(n=max_neg, random_state=42)
        print(f"  Downsampled to: {len(negatives):,}")

    # Combine
    best_sellers = best_sellers.copy()
    negatives = negatives.copy()
    labels = pd.concat([
        pd.Series(1, index=best_sellers.index),
        pd.Series(0, index=negatives.index),
    ])
    combined = pd.concat([best_sellers, negatives], ignore_index=True)
    labels = labels.reset_index(drop=True)
    print(f"  Final dataset: {len(combined):,} ({labels.sum():,} positive, {(~labels.astype(bool)).sum():,} negative)")
    return combined, labels


# ---------------------------------------------------------------------------
# Dashboard JSON builder
# ---------------------------------------------------------------------------

def score_distribution(scores, bins=20):
    counts, edges = np.histogram(scores, bins=bins, range=(0, 100))
    return [
        {"bin_start": round(edges[i], 1), "bin_end": round(edges[i + 1], 1), "count": int(counts[i])}
        for i in range(len(counts))
    ]


def store_stats(df):
    if "store_source" not in df.columns:
        return []
    stats = []
    for store in df["store_source"].unique():
        mask = df["store_source"] == store
        sub = df[mask]
        stats.append({
            "store": store,
            "total": int(mask.sum()),
            "predicted_selling": int(sub["predicted_selling"].sum()) if "predicted_selling" in sub else 0,
            "avg_score": round(sub["combined_score"].mean(), 1) if "combined_score" in sub else 0,
            "median_score": round(sub["combined_score"].median(), 1) if "combined_score" in sub else 0,
        })
    return stats


def sample_products(df, n=10):
    cols = ["product_id", "title", "price", "store_source",
            "heuristic_score", "combined_score", "predicted_selling"]
    if "ml_probability" in df.columns:
        cols.append("ml_probability")
    available = [c for c in cols if c in df.columns]
    top = df.nlargest(n, "combined_score")[available]
    return {
        "top_scored": top.to_dict(orient="records"),
    }


def build_dashboard_json(report, scored_df):
    output = {"training_report": report}

    output["scored_feed"] = {
        "total_products": len(scored_df),
        "predicted_selling": int(scored_df["predicted_selling"].sum()) if "predicted_selling" in scored_df else 0,
        "score_distribution": score_distribution(scored_df["combined_score"]),
        "store_stats": store_stats(scored_df),
        "sample_products": sample_products(scored_df),
    }

    return output


# ---------------------------------------------------------------------------
# CLI modes
# ---------------------------------------------------------------------------

def run_demo(args):
    """Run on synthetic data."""
    from data_generator import generate_ebay_feed, generate_aliexpress_feed

    print("Generating synthetic data...")
    ebay = generate_ebay_feed(2000)
    ali = generate_aliexpress_feed(1500)

    ebay_norm = normalize_columns(ebay, store_source="ebay")
    ali_norm = normalize_columns(ali, store_source="aliexpress")
    combined = pd.concat([ebay_norm, ali_norm], ignore_index=True)
    print(f"Combined synthetic feed: {len(combined):,} rows")

    pipeline = ProductScoringPipeline(mode="one_class")
    print("\nTraining one-class model...")
    report = pipeline.train(combined)
    print(f"  Train size: {report['train_size']:,}, Features: {report['feature_count']}")

    print("\nScoring products...")
    scored = pipeline.score(combined, threshold=args.threshold)

    output = build_dashboard_json(report, scored)
    out_path = args.output or "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n✅ Demo complete!")
    print(f"   Products scored: {len(scored):,}")
    print(f"   Predicted selling: {scored['predicted_selling'].sum():,}")
    print(f"   Results saved to {out_path}")


def run_train_profile(args):
    """Train one-class model on best-seller feeds."""
    if not args.best_sellers:
        print("Error: --best-sellers required", file=sys.stderr)
        sys.exit(1)

    print("Loading best-seller feeds...")
    combined = load_multi_store(args.best_sellers)

    pipeline = ProductScoringPipeline(mode="one_class")
    print("\nTraining one-class model...")
    report = pipeline.train(combined)
    print(f"  Train size: {report['train_size']:,}")
    print(f"  Features: {report['feature_count']}")
    print(f"  Anomaly score: mean={report['anomaly_score_mean']:.4f}, "
          f"std={report['anomaly_score_std']:.4f}")
    print(f"  Percentiles: {report['anomaly_score_percentiles']}")

    # Score all products
    print("\nScoring all products...")
    scored = pipeline.score(combined, threshold=args.threshold)
    selling = scored["predicted_selling"].sum()
    print(f"  Predicted selling: {selling:,} / {len(scored):,} "
          f"({selling / len(scored) * 100:.1f}%)")

    # Save model
    if args.save_model:
        pipeline.save(args.save_model)
        print(f"\n  Model saved to {args.save_model}/")

    # Save results
    output = build_dashboard_json(report, scored)
    out_path = args.output or "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Dashboard JSON saved to {out_path}")

    print("\n✅ Training complete!")


def run_train_binary(args):
    """Train binary classifier with best-seller + general feeds."""
    if not args.best_sellers or not args.general:
        print("Error: --best-sellers and --general both required", file=sys.stderr)
        sys.exit(1)

    print("Loading best-seller feeds...")
    best_sellers = load_multi_store(args.best_sellers)

    print("\nLoading general feeds...")
    general = load_multi_store(args.general)

    print("\nCreating binary dataset...")
    combined, labels = create_binary_dataset(best_sellers, general, args.negative_ratio)

    pipeline = ProductScoringPipeline(mode="binary")
    print("\nTraining binary classifier...")
    report = pipeline.train(combined, labels=labels)
    print(f"  AUC-ROC: {report['auc_roc']}")
    cr = report["classification_report"]
    print(f"  Precision: {cr['1']['precision']:.3f}")
    print(f"  Recall: {cr['1']['recall']:.3f}")
    print(f"  F1: {cr['1']['f1-score']:.3f}")

    # Feature importance
    if pipeline.classifier:
        fi = pipeline.classifier.get_feature_importance()
        if fi:
            print("\n  Top 10 features:")
            for feat, imp in list(fi.items())[:10]:
                print(f"    {feat}: {imp:.4f}")
            report["feature_importance"] = [
                {"feature": k, "importance": round(v, 4)}
                for k, v in list(fi.items())[:20]
            ]

    # Score all products
    print("\nScoring all products...")
    scored = pipeline.score(combined, threshold=args.threshold)

    if args.save_model:
        pipeline.save(args.save_model)
        print(f"\n  Model saved to {args.save_model}/")

    output = build_dashboard_json(report, scored)
    out_path = args.output or "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Dashboard JSON saved to {out_path}")

    print("\n✅ Training complete!")


def run_score(args):
    """Score new products with a saved model."""
    if not args.load_model:
        print("Error: --load-model required", file=sys.stderr)
        sys.exit(1)
    if not args.feed:
        print("Error: --feed required", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model from {args.load_model}/...")
    pipeline = ProductScoringPipeline.load(args.load_model)
    print(f"  Mode: {pipeline.mode}")

    print(f"\nLoading feed: {args.feed}")
    df = load_feed(args.feed)
    print(f"  → {len(df):,} rows")

    print("\nScoring...")
    scored = pipeline.score(df, threshold=args.threshold)
    selling = scored["predicted_selling"].sum()
    print(f"  Predicted selling: {selling:,} / {len(scored):,} "
          f"({selling / len(scored) * 100:.1f}%)")

    out_path = args.output or "scored_products.csv"
    if out_path.endswith(".json"):
        output = build_dashboard_json({"mode": pipeline.mode}, scored)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
    else:
        score_cols = ["product_id", "title", "price", "store_source",
                      "heuristic_score", "ml_probability", "combined_score",
                      "predicted_selling"]
        available = [c for c in score_cols if c in scored.columns]
        scored[available].to_csv(out_path, index=False)

    print(f"  Output saved to {out_path}")
    print("\n✅ Scoring complete!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Best-Seller Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # demo
    demo_p = subparsers.add_parser("demo", help="Run on synthetic data")
    demo_p.add_argument("--threshold", type=float, default=50.0)
    demo_p.add_argument("--output", type=str, default=None)

    # train-profile
    tp = subparsers.add_parser("train-profile", help="Train one-class model (Phase 1)")
    tp.add_argument("--best-sellers", nargs="+", required=True,
                    help="Best-seller feeds: 'path.csv:store_name'")
    tp.add_argument("--save-model", type=str, default=None)
    tp.add_argument("--threshold", type=float, default=50.0)
    tp.add_argument("--output", type=str, default=None)

    # train-binary
    tb = subparsers.add_parser("train-binary", help="Train binary classifier (Phase 2)")
    tb.add_argument("--best-sellers", nargs="+", required=True)
    tb.add_argument("--general", nargs="+", required=True,
                    help="General/all-products feeds: 'path.csv:store_name'")
    tb.add_argument("--negative-ratio", type=float, default=3.0)
    tb.add_argument("--save-model", type=str, default=None)
    tb.add_argument("--threshold", type=float, default=50.0)
    tb.add_argument("--output", type=str, default=None)

    # score
    sc = subparsers.add_parser("score", help="Score new products")
    sc.add_argument("--feed", type=str, required=True)
    sc.add_argument("--load-model", type=str, required=True)
    sc.add_argument("--threshold", type=float, default=50.0)
    sc.add_argument("--output", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "demo" or args.command is None:
        run_demo(args if args.command else argparse.Namespace(threshold=50.0, output=None))
    elif args.command == "train-profile":
        run_train_profile(args)
    elif args.command == "train-binary":
        run_train_binary(args)
    elif args.command == "score":
        run_score(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
