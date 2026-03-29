"""
Product Selling Likelihood Pipeline
====================================
Scores products from store feeds on their likelihood of being active sellers,
using only basic product fields (no ratings, no sold-units required).

Supports two modes:
  1. Heuristic scoring  – rule-based, works on any feed immediately
  2. ML classification   – trained on feeds that DO have sales labels,
                           then applied to feeds that don't

Designed to process millions of products in batched, category-level chunks.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 1. Feature Engineering – extract signals from basic product fields
# ---------------------------------------------------------------------------

class FeatureEngineer:
    """
    Extracts predictive features from raw product fields that are commonly
    available across feeds: title, description, price, category, image_count,
    variant_count, brand, last_modified, etc.
    """

    # Power words that indicate optimized / active listings
    POWER_WORDS = {
        "premium", "bestseller", "best seller", "top rated", "new",
        "sale", "deal", "limited", "exclusive", "official", "genuine",
        "organic", "free shipping", "fast shipping", "warranty",
        "upgraded", "latest", "2024", "2025", "2026",
    }

    SPAM_SIGNALS = {
        "wholesale", "bulk lot", "dropship", "test listing", "do not buy",
        "placeholder", "sample", "draft",
    }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered feature columns to a products DataFrame."""
        features = pd.DataFrame(index=df.index)

        # --- Title signals ---
        if "title" in df.columns:
            features["title_length"] = df["title"].fillna("").str.len()
            features["title_word_count"] = df["title"].fillna("").str.split().str.len()
            features["title_has_brand"] = (
                df.apply(self._title_contains_brand, axis=1)
                if "brand" in df.columns else 0
            )
            features["title_power_words"] = df["title"].fillna("").apply(
                lambda t: sum(1 for w in self.POWER_WORDS if w in t.lower())
            )
            features["title_spam_signals"] = df["title"].fillna("").apply(
                lambda t: sum(1 for w in self.SPAM_SIGNALS if w in t.lower())
            )
            features["title_caps_ratio"] = df["title"].fillna("").apply(
                lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1)
            )
            features["title_has_special_chars"] = df["title"].fillna("").apply(
                lambda t: len(re.findall(r"[!★☆♥♦]", t))
            )

        # --- Description signals ---
        if "description" in df.columns:
            features["desc_length"] = df["description"].fillna("").str.len()
            features["desc_word_count"] = df["description"].fillna("").str.split().str.len()
            features["has_description"] = (df["description"].fillna("").str.len() > 20).astype(int)
            features["desc_has_html"] = df["description"].fillna("").str.contains(
                r"<[a-z]", case=False, regex=True
            ).astype(int)
            features["desc_bullet_count"] = df["description"].fillna("").apply(
                lambda d: d.count("•") + d.count("✓") + d.count("- ")
            )

        # --- Price signals ---
        if "price" in df.columns:
            features["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)
            features["price_log"] = np.log1p(features["price"])
            features["price_is_round"] = (features["price"] % 1 == 0).astype(int)
            features["price_ends_99"] = (
                (features["price"] * 100 % 100).round().isin([99, 95, 49])
            ).astype(int)  # Psychological pricing

            # Category-relative pricing (z-score within category)
            if "category" in df.columns:
                cat_stats = features.groupby(df["category"])["price"].transform
                cat_mean = cat_stats("mean")
                cat_std = cat_stats("std").replace(0, 1)
                features["price_category_zscore"] = (features["price"] - cat_mean) / cat_std
            else:
                features["price_category_zscore"] = 0

        # --- Listing completeness ---
        if "image_count" in df.columns:
            features["image_count"] = pd.to_numeric(df["image_count"], errors="coerce").fillna(0)
            features["has_multiple_images"] = (features["image_count"] > 1).astype(int)
        elif "images" in df.columns:
            features["image_count"] = df["images"].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
            features["has_multiple_images"] = (features["image_count"] > 1).astype(int)

        if "variant_count" in df.columns:
            features["variant_count"] = pd.to_numeric(df["variant_count"], errors="coerce").fillna(0)
            features["has_variants"] = (features["variant_count"] > 1).astype(int)

        # Completeness score: how many optional fields are filled
        optional_fields = ["description", "brand", "images", "image_count",
                           "variant_count", "sku", "upc", "ean"]
        filled = sum(
            df[col].notna() & (df[col].astype(str).str.strip() != "")
            for col in optional_fields if col in df.columns
        )
        total_optional = max(sum(1 for c in optional_fields if c in df.columns), 1)
        features["completeness_score"] = filled / total_optional

        # --- Identifiers ---
        if "upc" in df.columns or "ean" in df.columns or "gtin" in df.columns:
            features["has_barcode"] = (
                df.get("upc", pd.Series(dtype=str)).notna()
                | df.get("ean", pd.Series(dtype=str)).notna()
                | df.get("gtin", pd.Series(dtype=str)).notna()
            ).astype(int)

        if "brand" in df.columns:
            features["has_brand"] = (
                df["brand"].notna() & (df["brand"].astype(str).str.strip() != "")
            ).astype(int)

        # --- Recency ---
        if "last_modified" in df.columns:
            mod = pd.to_datetime(df["last_modified"], errors="coerce")
            now = pd.Timestamp.now()
            features["days_since_modified"] = (now - mod).dt.days.fillna(9999)
            features["modified_last_30d"] = (features["days_since_modified"] <= 30).astype(int)
            features["modified_last_90d"] = (features["days_since_modified"] <= 90).astype(int)

        return features

    @staticmethod
    def _title_contains_brand(row):
        brand = str(row.get("brand", "")).strip().lower()
        title = str(row.get("title", "")).lower()
        return int(brand != "" and brand in title)


# ---------------------------------------------------------------------------
# 2. Heuristic Scorer – no training data needed
# ---------------------------------------------------------------------------

@dataclass
class ScoringWeights:
    """Tunable weights for heuristic scoring (0-1 scale each)."""
    completeness: float = 0.20
    title_quality: float = 0.15
    description_quality: float = 0.10
    price_positioning: float = 0.15
    image_quality: float = 0.15
    recency: float = 0.15
    identifiers: float = 0.10


class HeuristicScorer:
    """
    Rule-based scorer that assigns a 0-100 selling likelihood score.
    Works on any feed without training data.
    """

    def __init__(self, weights: Optional[ScoringWeights] = None):
        self.weights = weights or ScoringWeights()

    def score(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        scores = pd.Series(0.0, index=df.index)

        # Completeness
        if "completeness_score" in features:
            scores += features["completeness_score"] * self.weights.completeness

        # Title quality
        title_score = pd.Series(0.0, index=df.index)
        if "title_word_count" in features:
            # Ideal title: 5-15 words
            wc = features["title_word_count"].clip(0, 20)
            title_score += np.where((wc >= 5) & (wc <= 15), 0.4, 0.1)
        if "title_power_words" in features:
            title_score += (features["title_power_words"].clip(0, 3) / 3) * 0.3
        if "title_spam_signals" in features:
            title_score -= features["title_spam_signals"] * 0.3
        if "title_has_brand" in features:
            title_score += features["title_has_brand"] * 0.3
        scores += title_score.clip(0, 1) * self.weights.title_quality

        # Description quality
        desc_score = pd.Series(0.0, index=df.index)
        if "has_description" in features:
            desc_score += features["has_description"] * 0.4
        if "desc_word_count" in features:
            desc_score += (features["desc_word_count"].clip(0, 200) / 200) * 0.3
        if "desc_bullet_count" in features:
            desc_score += (features["desc_bullet_count"].clip(0, 10) / 10) * 0.3
        scores += desc_score.clip(0, 1) * self.weights.description_quality

        # Price positioning
        if "price_category_zscore" in features:
            # Prefer products within 1 std dev of category mean
            z = features["price_category_zscore"].abs()
            price_score = np.where(z <= 0.5, 1.0, np.where(z <= 1.0, 0.7, 0.3))
            if "price_ends_99" in features:
                price_score = price_score + features["price_ends_99"].values * 0.1
            scores += np.clip(price_score, 0, 1) * self.weights.price_positioning

        # Images
        if "image_count" in features:
            img_score = (features["image_count"].clip(0, 8) / 8)
            scores += img_score * self.weights.image_quality

        # Recency
        if "modified_last_30d" in features:
            recency_score = (
                features["modified_last_30d"] * 1.0
                + features.get("modified_last_90d", 0) * 0.5
            ) / 1.5
            scores += recency_score * self.weights.recency

        # Identifiers
        id_score = pd.Series(0.0, index=df.index)
        if "has_barcode" in features:
            id_score += features["has_barcode"] * 0.5
        if "has_brand" in features:
            id_score += features["has_brand"] * 0.5
        scores += id_score.clip(0, 1) * self.weights.identifiers

        return (scores * 100).round(1).clip(0, 100)


# ---------------------------------------------------------------------------
# 3. ML Classifier – train on labeled feeds, predict on unlabeled
# ---------------------------------------------------------------------------

class SellingClassifier:
    """
    Trains on feeds that have sales labels (is_selling: 0/1),
    then predicts on feeds that don't.
    """

    def __init__(self, model_type: str = "gradient_boosting"):
        self.scaler = StandardScaler()
        if model_type == "logistic":
            self.model = LogisticRegression(max_iter=1000, class_weight="balanced")
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                subsample=0.8, random_state=42
            )
        self.feature_columns = []
        self.is_fitted = False

    def _prepare_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Select numeric columns and fill NaN."""
        numeric = features.select_dtypes(include=[np.number])
        return numeric.fillna(0)

    def train(self, features: pd.DataFrame, labels: pd.Series):
        """Train on a labeled dataset."""
        X = self._prepare_features(features)
        self.feature_columns = X.columns.tolist()
        y = labels.astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

        # Evaluation
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True)
        auc = roc_auc_score(y_test, y_prob)

        return {
            "classification_report": report,
            "auc_roc": round(auc, 4),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "feature_count": len(self.feature_columns),
        }

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Predict selling probability on unlabeled data."""
        if not self.is_fitted:
            raise RuntimeError("Model not trained yet. Call train() first.")

        X = self._prepare_features(features)
        # Align columns with training
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_columns]

        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)[:, 1]

        return pd.DataFrame({
            "ml_selling_probability": (proba * 100).round(1),
            "ml_is_selling": (proba >= 0.5).astype(int),
        }, index=features.index)

    def get_feature_importance(self) -> dict:
        if hasattr(self.model, "feature_importances_"):
            imp = dict(zip(self.feature_columns, self.model.feature_importances_))
            return dict(sorted(imp.items(), key=lambda x: -x[1]))
        elif hasattr(self.model, "coef_"):
            imp = dict(zip(self.feature_columns, abs(self.model.coef_[0])))
            return dict(sorted(imp.items(), key=lambda x: -x[1]))
        return {}


# ---------------------------------------------------------------------------
# 4. Pipeline Orchestrator
# ---------------------------------------------------------------------------

class ProductScoringPipeline:
    """
    End-to-end pipeline:
      1. Engineer features from raw product feed
      2. Score with heuristics (always available)
      3. Optionally train ML model on labeled data
      4. Combine scores into final ranking
    """

    def __init__(self, weights: Optional[ScoringWeights] = None):
        self.feature_engineer = FeatureEngineer()
        self.heuristic_scorer = HeuristicScorer(weights)
        self.classifier = SellingClassifier()

    def process_feed(
        self,
        df: pd.DataFrame,
        labels: Optional[pd.Series] = None,
        threshold: float = 50.0,
    ) -> pd.DataFrame:
        """
        Process a product feed. If labels are provided, also trains ML model.
        Returns DataFrame with original data + scores.
        """
        features = self.feature_engineer.transform(df)
        heuristic_scores = self.heuristic_scorer.score(df, features)

        result = df.copy()
        result["heuristic_score"] = heuristic_scores

        training_report = None
        if labels is not None:
            training_report = self.classifier.train(features, labels)
            ml_results = self.classifier.predict(features)
            result["ml_probability"] = ml_results["ml_selling_probability"]
            result["ml_is_selling"] = ml_results["ml_is_selling"]
            # Combined score: weighted average
            result["combined_score"] = (
                result["heuristic_score"] * 0.3 + result["ml_probability"] * 0.7
            ).round(1)
        else:
            result["combined_score"] = result["heuristic_score"]

        result["predicted_selling"] = (result["combined_score"] >= threshold).astype(int)

        return result, features, training_report

    def predict_unlabeled(self, df: pd.DataFrame, threshold: float = 50.0) -> pd.DataFrame:
        """Score an unlabeled feed using trained ML model + heuristics."""
        features = self.feature_engineer.transform(df)
        heuristic_scores = self.heuristic_scorer.score(df, features)

        result = df.copy()
        result["heuristic_score"] = heuristic_scores

        if self.classifier.is_fitted:
            ml_results = self.classifier.predict(features)
            result["ml_probability"] = ml_results["ml_selling_probability"]
            result["combined_score"] = (
                result["heuristic_score"] * 0.3 + result["ml_probability"] * 0.7
            ).round(1)
        else:
            result["combined_score"] = result["heuristic_score"]

        result["predicted_selling"] = (result["combined_score"] >= threshold).astype(int)
        return result
