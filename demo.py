"""
Demo Runner – runs the full pipeline and exports results as JSON
for the interactive dashboard.
"""

import json
import sys
sys.path.insert(0, "/home/claude/product_scorer")

import numpy as np
import pandas as pd
from pipeline import ProductScoringPipeline, ScoringWeights
from data_generator import generate_labeled_feed, generate_unlabeled_feed

np.random.seed(42)

# === 1. Generate synthetic feeds ===
print("Generating synthetic data...")
labeled_feed = generate_labeled_feed(n=5000, selling_ratio=0.35)
unlabeled_feed = generate_unlabeled_feed(n=3000)

# === 2. Run pipeline on labeled feed (train) ===
print("Running pipeline on labeled feed (training)...")
pipeline = ProductScoringPipeline()
scored_labeled, features_labeled, training_report = pipeline.process_feed(
    labeled_feed.drop(columns=["is_selling"]),
    labels=labeled_feed["is_selling"],
    threshold=50.0,
)

# === 3. Predict on unlabeled feed ===
print("Predicting on unlabeled feed...")
scored_unlabeled = pipeline.predict_unlabeled(
    unlabeled_feed.drop(columns=["_hidden_is_selling"]),
    threshold=50.0,
)
# Attach hidden truth for evaluation
scored_unlabeled["_hidden_is_selling"] = unlabeled_feed["_hidden_is_selling"].values

# === 4. Feature importance ===
feature_importance = pipeline.classifier.get_feature_importance()

# === 5. Build output JSON ===
print("Building dashboard data...")

# Score distributions
def score_distribution(scores, bins=20):
    counts, edges = np.histogram(scores, bins=bins, range=(0, 100))
    return [{"bin_start": round(edges[i], 1), "bin_end": round(edges[i+1], 1),
             "count": int(counts[i])} for i in range(len(counts))]

# Category breakdown
def category_stats(df):
    stats = []
    for cat in df["category"].unique():
        mask = df["category"] == cat
        sub = df[mask]
        stats.append({
            "category": cat,
            "total": int(mask.sum()),
            "predicted_selling": int(sub["predicted_selling"].sum()),
            "avg_score": round(sub["combined_score"].mean(), 1),
            "median_score": round(sub["combined_score"].median(), 1),
        })
    return sorted(stats, key=lambda x: -x["avg_score"])

# Unlabeled accuracy (vs hidden truth)
unlabeled_accuracy = (
    scored_unlabeled["predicted_selling"] == scored_unlabeled["_hidden_is_selling"]
).mean()

# Top features
top_features = [
    {"feature": k, "importance": round(v, 4)}
    for k, v in list(feature_importance.items())[:15]
]

# Sample products (top and bottom scored)
def sample_products(df, n=10):
    cols = ["product_id", "title", "category", "price", "brand",
            "image_count", "variant_count", "heuristic_score", "combined_score",
            "predicted_selling"]
    available_cols = [c for c in cols if c in df.columns]
    if "ml_probability" in df.columns:
        available_cols.append("ml_probability")
    top = df.nlargest(n, "combined_score")[available_cols]
    bottom = df.nsmallest(n, "combined_score")[available_cols]
    return {
        "top_scored": top.to_dict(orient="records"),
        "bottom_scored": bottom.to_dict(orient="records"),
    }

# Build full output
output = {
    "training_report": {
        "auc_roc": training_report["auc_roc"],
        "precision_selling": round(training_report["classification_report"]["1"]["precision"], 3),
        "recall_selling": round(training_report["classification_report"]["1"]["recall"], 3),
        "f1_selling": round(training_report["classification_report"]["1"]["f1-score"], 3),
        "accuracy": round(training_report["classification_report"]["accuracy"], 3),
        "train_size": training_report["train_size"],
        "test_size": training_report["test_size"],
    },
    "labeled_feed": {
        "total_products": len(scored_labeled),
        "predicted_selling": int(scored_labeled["predicted_selling"].sum()),
        "actual_selling": int(labeled_feed["is_selling"].sum()),
        "score_distribution": score_distribution(scored_labeled["combined_score"]),
        "category_stats": category_stats(scored_labeled),
        "sample_products": sample_products(scored_labeled),
    },
    "unlabeled_feed": {
        "total_products": len(scored_unlabeled),
        "predicted_selling": int(scored_unlabeled["predicted_selling"].sum()),
        "hidden_actual_selling": int(scored_unlabeled["_hidden_is_selling"].sum()),
        "accuracy_vs_hidden": round(unlabeled_accuracy, 4),
        "score_distribution": score_distribution(scored_unlabeled["combined_score"]),
        "category_stats": category_stats(scored_unlabeled),
        "sample_products": sample_products(scored_unlabeled),
    },
    "feature_importance": top_features,
}

with open("/home/claude/product_scorer/results.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n✅ Pipeline complete!")
print(f"   Labeled feed:   {output['labeled_feed']['total_products']} products → {output['labeled_feed']['predicted_selling']} predicted selling")
print(f"   Unlabeled feed:  {output['unlabeled_feed']['total_products']} products → {output['unlabeled_feed']['predicted_selling']} predicted selling")
print(f"   ML AUC-ROC:      {output['training_report']['auc_roc']}")
print(f"   Unlabeled accuracy vs hidden truth: {output['unlabeled_feed']['accuracy_vs_hidden']:.1%}")
print(f"\nResults saved to results.json")
print(json.dumps(output, indent=2, default=str))
