"""
Best-Seller Prediction Pipeline
================================
Learns patterns from best-selling products across multiple stores (eBay, AliExpress),
then predicts whether new products will be best sellers.

Two modes:
  1. One-class (Phase 1) — only best-seller data available, uses Isolation Forest
  2. Binary    (Phase 2) — best-seller + general feeds, uses LightGBM

Both modes use TF-IDF text features + engineered tabular features.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack as sparse_hstack
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


# --- Currency conversion rate (GBP → EUR) ---
GBP_TO_EUR = 1.15


# ---------------------------------------------------------------------------
# 1. Data Loading & Normalization
# ---------------------------------------------------------------------------

def load_feed(path: str, store_source: str = None) -> pd.DataFrame:
    """Load a CSV/TSV feed file, normalize columns, and optionally tag with store source."""
    path = str(path)
    if path.endswith(".tsv"):
        df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    else:
        df = pd.read_csv(path, dtype=str, low_memory=False)
    return normalize_columns(df, store_source=store_source)


def normalize_columns(df: pd.DataFrame, store_source: str = None) -> pd.DataFrame:
    """Normalize column names and parse store-specific formats."""
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()

    # Alias map
    rename = {}
    if "item_id" in df.columns:
        rename["item_id"] = "product_id"
    if "id" in df.columns and "product_id" not in df.columns and "item_id" not in df.columns:
        rename["id"] = "product_id"
    if "image_link" in df.columns:
        rename["image_link"] = "image_url"
    if rename:
        df.rename(columns=rename, inplace=True)

    # --- eBay PRICE parsing: "27.90 EUR" or "2.95 GBP" ---
    if "price" in df.columns and df["price"].dtype == object:
        price_parts = df["price"].str.extract(r"([\d.]+)\s*([A-Za-z]+)?")
        df["price"] = pd.to_numeric(price_parts[0], errors="coerce")
        df["price_currency"] = price_parts[1].fillna("")
        # Convert GBP to EUR
        gbp_mask = df["price_currency"].str.upper() == "GBP"
        df.loc[gbp_mask, "price"] = df.loc[gbp_mask, "price"] * GBP_TO_EUR

    # --- eBay SHIPPING parsing: "DE::STANDARD:2.99 EUR" ---
    if "shipping" in df.columns and df["shipping"].dtype == object:
        ship_parts = df["shipping"].str.extract(
            r"([A-Z]{2})::([A-Z_]*):?([\d.]+)?\s*([A-Za-z]+)?"
        )
        df["shipping_cost"] = pd.to_numeric(ship_parts[2], errors="coerce").fillna(0)
        df["shipping_method"] = ship_parts[1].fillna("")
        # Convert GBP shipping to EUR
        gbp_ship = ship_parts[3].fillna("").str.upper() == "GBP"
        df.loc[gbp_ship, "shipping_cost"] = df.loc[gbp_ship, "shipping_cost"] * GBP_TO_EUR

    # --- eBay CONDITION normalization ---
    if "condition" in df.columns:
        cond = df["condition"].fillna("").str.lower().str.strip()
        df["condition_normalized"] = "other"
        df.loc[cond.str.startswith("neu") | cond.str.startswith("new"), "condition_normalized"] = "new"
        df.loc[cond.str.contains("refurbished"), "condition_normalized"] = "refurbished"
        df.loc[
            cond.str.startswith("gebraucht") | cond.str.startswith("used") |
            cond.str.contains("pre-owned") | cond.str.contains("geöffnete"),
            "condition_normalized"
        ] = "used"
        df.loc[
            cond.isin(["gut", "sehr gut", "neuwertig", "akzeptabel", "good", "very good", "like new"]),
            "condition_normalized"
        ] = "good"

    # --- AliExpress BRAND cleanup ---
    if "brand" in df.columns:
        df.loc[df["brand"].fillna("").str.upper().isin(["NONE", ""]), "brand"] = np.nan

    # --- Parse numeric columns ---
    for col in ["sale_price", "discount_rate", "delivery_days", "product_score", "review_number"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # --- Parse insert_time ---
    if "insert_time" in df.columns:
        df["insert_time"] = pd.to_datetime(df["insert_time"], errors="coerce")

    # --- Store source ---
    if store_source:
        df["store_source"] = store_source

    return df


# ---------------------------------------------------------------------------
# 2. Feature Engineering
# ---------------------------------------------------------------------------

class FeatureEngineer:
    """Extract predictive features from normalized product data."""

    POWER_WORDS = {
        "premium", "bestseller", "best seller", "top rated", "new",
        "sale", "deal", "limited", "exclusive", "official", "genuine",
        "organic", "free shipping", "fast shipping", "warranty",
        "upgraded", "latest", "2024", "2025", "2026",
        # German
        "neu", "angebot", "reduziert", "limitiert", "versandkostenfrei",
        "exklusiv", "original", "garantie",
    }

    SPAM_SIGNALS = {
        "wholesale", "bulk lot", "dropship", "test listing", "do not buy",
        "placeholder", "sample", "draft",
        # German
        "großhandel", "testlisting", "nicht kaufen", "entwurf",
    }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)

        # --- Title signals (7) ---
        if "title" in df.columns:
            title = df["title"].fillna("").astype(str)
            features["title_length"] = title.str.len()
            features["title_word_count"] = title.str.split().str.len().fillna(0)
            features["title_has_brand"] = (
                df.apply(self._title_contains_brand, axis=1)
                if "brand" in df.columns else 0
            )
            title_lower = title.str.lower()
            features["title_power_words"] = title_lower.apply(
                lambda t: sum(1 for w in self.POWER_WORDS if w in t)
            )
            features["title_spam_signals"] = title_lower.apply(
                lambda t: sum(1 for w in self.SPAM_SIGNALS if w in t)
            )
            features["title_caps_ratio"] = title.apply(
                lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1)
            )
            features["title_has_special_chars"] = title.apply(
                lambda t: len(re.findall(r"[!★☆♥♦✓✔️]", t))
            )

        # --- Description signals (5) ---
        if "description" in df.columns:
            desc = df["description"].fillna("").astype(str)
            features["desc_length"] = desc.str.len()
            features["desc_word_count"] = desc.str.split().str.len().fillna(0)
            features["has_description"] = (desc.str.len() > 20).astype(int)
            features["desc_has_html"] = desc.str.contains(r"<[a-z]", case=False, regex=True).astype(int)
            features["desc_bullet_count"] = desc.apply(
                lambda d: d.count("•") + d.count("✓") + d.count("- ")
            )

        # --- Price signals (8) ---
        if "price" in df.columns:
            price = pd.to_numeric(df["price"], errors="coerce").fillna(0)
            features["price"] = price
            features["price_log"] = np.log1p(price)
            features["price_is_round"] = (price % 1 == 0).astype(int)
            features["price_ends_99"] = (
                (price * 100 % 100).round().isin([99, 95, 49])
            ).astype(int)

            # Region-relative z-score
            if "region" in df.columns:
                grp = price.groupby(df["region"])
                r_mean = grp.transform("mean")
                r_std = grp.transform("std").replace(0, 1)
                features["price_region_zscore"] = ((price - r_mean) / r_std).fillna(0)
            else:
                features["price_region_zscore"] = 0

            # Discount signals
            if "sale_price" in df.columns:
                sale = pd.to_numeric(df["sale_price"], errors="coerce")
                features["has_sale_price"] = sale.notna().astype(int)
                features["discount_amount"] = (price - sale.fillna(price)).clip(lower=0)
                features["discount_pct"] = np.where(
                    price > 0, features["discount_amount"] / price * 100, 0
                )
            else:
                features["has_sale_price"] = 0
                features["discount_amount"] = 0
                features["discount_pct"] = 0

        # --- Image signal (1) ---
        if "image_url" in df.columns:
            features["has_image"] = (
                df["image_url"].fillna("").astype(str).str.len() > 5
            ).astype(int)
        else:
            features["has_image"] = 0

        # --- Shipping signals (3) ---
        if "shipping_cost" in df.columns:
            features["shipping_cost"] = pd.to_numeric(df["shipping_cost"], errors="coerce").fillna(0)
        elif "shipping" in df.columns:
            # Try to extract numeric from raw shipping string
            features["shipping_cost"] = pd.to_numeric(
                df["shipping"].astype(str).str.extract(r"([\d.]+)")[0], errors="coerce"
            ).fillna(0)
        else:
            features["shipping_cost"] = 0
        features["is_free_shipping"] = (features["shipping_cost"] == 0).astype(int)

        if "delivery_days" in df.columns:
            features["delivery_days"] = pd.to_numeric(df["delivery_days"], errors="coerce").fillna(0)
        else:
            features["delivery_days"] = 0

        # --- Recency signals (4) ---
        if "insert_time" in df.columns:
            insert = pd.to_datetime(df["insert_time"], errors="coerce")
            now = pd.Timestamp.now()
            days = (now - insert).dt.days.fillna(9999)
            features["days_since_insert"] = days
            features["inserted_last_7d"] = (days <= 7).astype(int)
            features["inserted_last_30d"] = (days <= 30).astype(int)
            features["inserted_last_90d"] = (days <= 90).astype(int)

        # --- Identifiers (2) ---
        if "gtin" in df.columns:
            features["has_gtin"] = (
                df["gtin"].fillna("").astype(str).str.strip().str.len() > 0
            ).astype(int)
        if "brand" in df.columns:
            features["has_brand"] = (
                df["brand"].notna() & (df["brand"].astype(str).str.strip() != "")
            ).astype(int)

        # --- Condition signals — eBay (3) ---
        if "condition_normalized" in df.columns:
            features["condition_is_new"] = (df["condition_normalized"] == "new").astype(int)
            features["condition_is_refurbished"] = (df["condition_normalized"] == "refurbished").astype(int)
            features["condition_is_used"] = (df["condition_normalized"] == "used").astype(int)

        # --- AliExpress-specific (6) ---
        if "product_score" in df.columns:
            ps = pd.to_numeric(df["product_score"], errors="coerce")
            features["product_score"] = ps.fillna(0)
            features["has_product_score"] = ps.notna().astype(int)

        if "review_number" in df.columns:
            rn = pd.to_numeric(df["review_number"], errors="coerce")
            features["review_number"] = rn.fillna(0)
            features["review_number_log"] = np.log1p(rn.fillna(0))
            features["has_reviews"] = (rn.fillna(0) > 0).astype(int)

        if "discount_rate" in df.columns:
            features["discount_rate_raw"] = pd.to_numeric(
                df["discount_rate"], errors="coerce"
            ).fillna(0)

        # --- Availability (1) ---
        if "availability" in df.columns:
            avail = df["availability"].fillna("").astype(str).str.lower()
            features["is_in_stock"] = (
                avail.str.contains("in.?stock", regex=True)
            ).astype(int)

        # --- Store source (2) ---
        if "store_source" in df.columns:
            features["is_aliexpress"] = (df["store_source"] == "aliexpress").astype(int)
            features["is_ebay"] = (df["store_source"] == "ebay").astype(int)

        # --- Completeness score (1) ---
        optional_fields = [
            "description", "brand", "image_url", "gtin", "sale_price",
            "shipping", "product_score", "review_number", "color", "size",
        ]
        present_fields = [c for c in optional_fields if c in df.columns]
        if present_fields:
            filled = sum(
                df[col].notna() & (df[col].astype(str).str.strip() != "")
                for col in present_fields
            )
            features["completeness_score"] = filled / len(present_fields)
        else:
            features["completeness_score"] = 0

        # --- Feature interactions (4) ---
        features["price_x_has_image"] = features.get("price_log", 0) * features.get("has_image", 0)
        features["completeness_x_recency"] = (
            features.get("completeness_score", 0) * features.get("inserted_last_30d", 0)
        )
        features["title_quality_x_image"] = (
            features.get("title_word_count", 0) * features.get("has_image", 0)
        )
        features["discount_x_reviews"] = (
            features.get("discount_pct", 0) * features.get("review_number_log", 0)
        )

        return features

    @staticmethod
    def _title_contains_brand(row):
        brand = str(row.get("brand", "")).strip().lower()
        title = str(row.get("title", "")).lower()
        return int(brand != "" and brand in title)


# ---------------------------------------------------------------------------
# 2b. Product Text Serialization (for embedding models)
# ---------------------------------------------------------------------------

class ProductTextSerializer:
    """Serialize product rows into text strings for embedding models."""

    def serialize(self, df: pd.DataFrame) -> list:
        return df.apply(self._serialize_row, axis=1).tolist()

    @staticmethod
    def _serialize_row(row) -> str:
        parts = []

        title = str(row.get("title", "")).strip()
        if title:
            parts.append(title)

        brand = str(row.get("brand", "")).strip()
        if brand and brand.upper() not in ("NONE", "", "NAN"):
            parts.append(f"Brand: {brand}")

        price = row.get("price")
        try:
            price = float(price)
            if price > 0:
                parts.append(f"Price: {price:.2f} EUR")
        except (TypeError, ValueError):
            pass

        condition = str(row.get("condition_normalized", "")).strip()
        if condition and condition not in ("other", "", "nan"):
            parts.append(f"Condition: {condition}")

        sale_price = row.get("sale_price")
        try:
            sale_price = float(sale_price)
            if price and price > 0 and sale_price < price:
                pct = round((1 - sale_price / price) * 100)
                parts.append(f"Discount: {pct}%")
        except (TypeError, ValueError):
            pass

        store = str(row.get("store_source", "")).strip()
        if store and store != "nan":
            parts.append(f"Store: {store}")

        region = str(row.get("region", "")).strip()
        if region and region != "nan":
            parts.append(f"Region: {region.upper()}")

        return " | ".join(parts) if parts else "unknown product"


# ---------------------------------------------------------------------------
# 2c. Embedding Fine-Tuning (TSDAE)
# ---------------------------------------------------------------------------

class EmbeddingFineTuner:
    """TSDAE fine-tuning of a sentence-transformer on product text."""

    def __init__(
        self,
        model_name: str = "Alibaba-NLP/gte-multilingual-base",
        max_train_samples: int = 200_000,
        batch_size: int = 32,
        epochs: int = 1,
        lr: float = 3e-5,
        deletion_ratio: float = 0.6,
    ):
        self.model_name = model_name
        self.max_train_samples = max_train_samples
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.deletion_ratio = deletion_ratio

    def fine_tune(self, texts: list, output_dir: str) -> dict:
        """Fine-tune with TSDAE. Returns training metadata."""
        import time
        from sentence_transformers import SentenceTransformer, losses
        from sentence_transformers.datasets import DenoisingAutoEncoderDataset
        from torch.utils.data import DataLoader

        print(f"  Loading base model: {self.model_name}")
        model = SentenceTransformer(self.model_name)

        # Subsample if needed
        if len(texts) > self.max_train_samples:
            rng = np.random.RandomState(42)
            indices = rng.choice(len(texts), self.max_train_samples, replace=False)
            texts = [texts[i] for i in indices]
            print(f"  Subsampled to {len(texts):,} texts")

        # Create TSDAE dataset and dataloader
        train_dataset = DenoisingAutoEncoderDataset(texts)
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True,
        )

        # TSDAE loss
        train_loss = losses.DenoisingAutoEncoderLoss(
            model,
            decoder_name_or_path=self.model_name,
            tie_encoder_decoder=False,  # True is broken with transformers>=5.0
        )

        total_steps = len(train_dataloader) * self.epochs
        warmup_steps = min(500, total_steps // 10)
        print(f"  Training: {len(texts):,} texts, {total_steps:,} steps, "
              f"{self.epochs} epoch(s), batch_size={self.batch_size}")

        start = time.time()
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.epochs,
            weight_decay=0,
            scheduler="constantlr",
            optimizer_params={"lr": self.lr},
            show_progress_bar=True,
            warmup_steps=warmup_steps,
        )
        duration = time.time() - start

        # Save fine-tuned model
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model.save(output_dir)
        print(f"  Model saved to {output_dir}/")

        return {
            "base_model": self.model_name,
            "train_samples": len(texts),
            "total_steps": total_steps,
            "epochs": self.epochs,
            "duration_seconds": round(duration, 1),
            "output_dir": output_dir,
        }


# ---------------------------------------------------------------------------
# 2d. Embedding Scorer (centroid-based one-class scoring)
# ---------------------------------------------------------------------------

class EmbeddingScorer:
    """
    Centroid-based one-class scoring using embeddings.

    Computes the centroid of all best-seller embeddings, then scores
    new products by cosine similarity to the centroid.
    """

    def __init__(
        self,
        model_name_or_path: str = "Alibaba-NLP/gte-multilingual-base",
        batch_size: int = 256,
    ):
        self.model_name_or_path = model_name_or_path
        self.batch_size = batch_size
        self.model = None
        self.global_centroid = None
        self.similarity_stats = {}
        self.is_fitted = False

    def _load_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name_or_path)

    def _encode(self, texts: list) -> np.ndarray:
        self._load_model()
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def fit(self, texts: list) -> dict:
        """Compute centroid from best-seller product texts."""
        print(f"  Encoding {len(texts):,} products...")
        embeddings = self._encode(texts).astype(np.float64)

        # Compute centroid (L2-normalized mean)
        centroid = embeddings.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        self.global_centroid = centroid

        # Compute similarity distribution for calibration
        sims = embeddings @ centroid
        self.similarity_stats = {
            "mean": round(float(sims.mean()), 6),
            "std": round(float(sims.std()), 6),
            "min": round(float(sims.min()), 6),
            "max": round(float(sims.max()), 6),
            "p5": round(float(np.percentile(sims, 5)), 6),
            "p25": round(float(np.percentile(sims, 25)), 6),
            "p50": round(float(np.percentile(sims, 50)), 6),
            "p75": round(float(np.percentile(sims, 75)), 6),
            "p95": round(float(np.percentile(sims, 95)), 6),
        }
        self.is_fitted = True

        print(f"  Centroid computed. Similarity stats:")
        for k, v in self.similarity_stats.items():
            print(f"    {k}: {v:.4f}")

        return {
            "embedding_dim": len(centroid),
            "train_size": len(texts),
            "similarity_stats": self.similarity_stats,
        }

    def predict(self, texts: list, threshold: float = 50.0) -> pd.DataFrame:
        """Score products by cosine similarity to centroid."""
        if not self.is_fitted:
            raise RuntimeError("EmbeddingScorer not fitted. Call fit() first.")

        embeddings = self._encode(texts).astype(np.float64)
        sims = embeddings @ self.global_centroid

        # Calibrate to 0-100 using percentile-based scaling
        p5 = self.similarity_stats["p5"]
        p95 = self.similarity_stats["p95"]
        if p95 - p5 > 0:
            calibrated = (sims - p5) / (p95 - p5) * 90 + 5  # maps p5->5, p95->95
        else:
            calibrated = np.full_like(sims, 50.0)
        calibrated = np.clip(calibrated, 0, 100).round(1)

        return pd.DataFrame({
            "embedding_similarity": sims.round(4),
            "embedding_probability": calibrated,
            "embedding_is_bestseller": (calibrated >= threshold).astype(int),
        })


# ---------------------------------------------------------------------------
# 3. Heuristic Scorer
# ---------------------------------------------------------------------------

@dataclass
class ScoringWeights:
    completeness: float = 0.15
    title_quality: float = 0.13
    description_quality: float = 0.10
    price_positioning: float = 0.12
    image_quality: float = 0.08
    recency: float = 0.10
    identifiers: float = 0.10
    discount_signals: float = 0.08
    shipping_signals: float = 0.07
    review_signals: float = 0.07


class HeuristicScorer:
    """Rule-based 0-100 scorer. Works on any feed without training."""

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
        if "price_region_zscore" in features:
            z = features["price_region_zscore"].abs()
            price_score = np.where(z <= 0.5, 1.0, np.where(z <= 1.0, 0.7, 0.3))
            if "price_ends_99" in features:
                price_score = price_score + features["price_ends_99"].values * 0.1
            scores += np.clip(price_score, 0, 1) * self.weights.price_positioning

        # Image
        if "has_image" in features:
            scores += features["has_image"] * self.weights.image_quality

        # Recency
        if "inserted_last_30d" in features:
            recency_score = (
                features["inserted_last_30d"] * 1.0
                + features.get("inserted_last_90d", 0) * 0.5
            ) / 1.5
            scores += recency_score * self.weights.recency

        # Identifiers
        id_score = pd.Series(0.0, index=df.index)
        if "has_gtin" in features:
            id_score += features["has_gtin"] * 0.5
        if "has_brand" in features:
            id_score += features["has_brand"] * 0.5
        scores += id_score.clip(0, 1) * self.weights.identifiers

        # Discount signals
        discount_score = pd.Series(0.0, index=df.index)
        if "has_sale_price" in features:
            discount_score += features["has_sale_price"] * 0.4
        if "discount_pct" in features:
            discount_score += (features["discount_pct"].clip(0, 70) / 70) * 0.6
        scores += discount_score.clip(0, 1) * self.weights.discount_signals

        # Shipping signals
        if "is_free_shipping" in features:
            scores += features["is_free_shipping"] * self.weights.shipping_signals

        # Review signals
        review_score = pd.Series(0.0, index=df.index)
        if "has_reviews" in features:
            review_score += features["has_reviews"] * 0.3
        if "product_score" in features:
            review_score += (features["product_score"].clip(0, 5) / 5) * 0.4
        if "review_number_log" in features:
            review_score += (features["review_number_log"].clip(0, 10) / 10) * 0.3
        scores += review_score.clip(0, 1) * self.weights.review_signals

        return (scores * 100).round(1).clip(0, 100)


# ---------------------------------------------------------------------------
# 4. Text Feature Extraction (TF-IDF)
# ---------------------------------------------------------------------------

class TextFeatureExtractor:
    """TF-IDF on title + description text."""

    def __init__(self, max_title_features=200, max_desc_features=300):
        self.title_vectorizer = TfidfVectorizer(
            max_features=max_title_features,
            sublinear_tf=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.95,
        )
        self.desc_vectorizer = TfidfVectorizer(
            max_features=max_desc_features,
            sublinear_tf=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.95,
        )
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "TextFeatureExtractor":
        titles = df.get("title", pd.Series(dtype=str)).fillna("").astype(str)
        descs = df.get("description", pd.Series(dtype=str)).fillna("").astype(str)
        self.title_vectorizer.fit(titles)
        self.desc_vectorizer.fit(descs)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> csr_matrix:
        if not self.is_fitted:
            raise RuntimeError("TextFeatureExtractor not fitted. Call fit() first.")
        titles = df.get("title", pd.Series(dtype=str)).fillna("").astype(str)
        descs = df.get("description", pd.Series(dtype=str)).fillna("").astype(str)
        return sparse_hstack([
            self.title_vectorizer.transform(titles),
            self.desc_vectorizer.transform(descs),
        ])

    def fit_transform(self, df: pd.DataFrame) -> csr_matrix:
        self.fit(df)
        return self.transform(df)


# ---------------------------------------------------------------------------
# 5. Model Classes
# ---------------------------------------------------------------------------

def _prepare_features(features: pd.DataFrame, text_features=None, numeric_columns=None):
    """Combine numeric features with optional TF-IDF sparse matrix.

    numeric_columns: list of numeric column names from training (excludes TF-IDF).
    """
    numeric = features.select_dtypes(include=[np.number]).fillna(0)

    if numeric_columns is not None:
        # Align columns with training
        missing = [c for c in numeric_columns if c not in numeric.columns]
        if missing:
            zeros = pd.DataFrame(0, index=numeric.index, columns=missing)
            numeric = pd.concat([numeric, zeros], axis=1)
        numeric = numeric[numeric_columns]

    X = csr_matrix(numeric.values)
    if text_features is not None:
        X = sparse_hstack([X, text_features])

    return X, numeric.columns.tolist()


class BestSellerProfiler:
    """Phase 1: One-class model trained on best-seller data only."""

    def __init__(self, contamination=0.05):
        self.isolation_forest = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler(with_mean=False)
        self.numeric_columns = []
        self.is_fitted = False

    def train(self, features: pd.DataFrame, text_features=None):
        X, self.numeric_columns = _prepare_features(features, text_features)
        X_scaled = self.scaler.fit_transform(X)
        self.isolation_forest.fit(X_scaled)
        self.is_fitted = True

        scores = self.isolation_forest.decision_function(X_scaled)
        return {
            "mode": "one_class",
            "train_size": X.shape[0],
            "feature_count": X.shape[1],
            "anomaly_score_mean": round(float(scores.mean()), 4),
            "anomaly_score_std": round(float(scores.std()), 4),
            "anomaly_score_percentiles": {
                "p5": round(float(np.percentile(scores, 5)), 4),
                "p25": round(float(np.percentile(scores, 25)), 4),
                "p50": round(float(np.percentile(scores, 50)), 4),
                "p75": round(float(np.percentile(scores, 75)), 4),
                "p95": round(float(np.percentile(scores, 95)), 4),
            },
        }

    def predict(self, features: pd.DataFrame, text_features=None) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model not trained. Call train() first.")
        X, _ = _prepare_features(features, text_features, self.numeric_columns)
        X_scaled = self.scaler.transform(X)
        raw_scores = self.isolation_forest.decision_function(X_scaled)
        # Normalize to 0-100: higher = more best-seller-like
        s_min, s_max = raw_scores.min(), raw_scores.max()
        if s_max - s_min > 0:
            normalized = (raw_scores - s_min) / (s_max - s_min) * 100
        else:
            normalized = np.full_like(raw_scores, 50.0)
        return pd.DataFrame({
            "ml_selling_probability": normalized.round(1),
            "ml_is_selling": (self.isolation_forest.predict(X_scaled) == 1).astype(int),
        }, index=features.index)


class SellingClassifier:
    """Phase 2: Binary classifier trained on labeled data (best-sellers + negatives)."""

    def __init__(self, model_type: str = "lightgbm"):
        if model_type == "lightgbm" and HAS_LIGHTGBM:
            self.model = LGBMClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                class_weight="balanced", verbose=-1, n_jobs=-1,
            )
            self.scaler = None
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                subsample=0.8, random_state=42,
            )
            self.scaler = StandardScaler(with_mean=False)
        self.numeric_columns = []
        self.is_fitted = False

    def train(self, features: pd.DataFrame, labels: pd.Series, text_features=None):
        X, self.numeric_columns = _prepare_features(features, text_features)
        y = labels.astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42,
        )

        if self.scaler:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        self.model.fit(X_train, y_train)
        self.is_fitted = True

        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        return {
            "mode": "binary",
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "auc_roc": round(float(roc_auc_score(y_test, y_prob)), 4),
            "train_size": len(y_train),
            "test_size": len(y_test),
            "feature_count": X.shape[1],
        }

    def predict(self, features: pd.DataFrame, text_features=None) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model not trained. Call train() first.")
        X, _ = _prepare_features(features, text_features, self.numeric_columns)
        if self.scaler:
            X = self.scaler.transform(X)
        proba = self.model.predict_proba(X)[:, 1]
        return pd.DataFrame({
            "ml_selling_probability": (proba * 100).round(1),
            "ml_is_selling": (proba >= 0.5).astype(int),
        }, index=features.index)

    def get_feature_importance(self) -> dict:
        if hasattr(self.model, "feature_importances_"):
            # Feature names = numeric columns + tfidf indices
            n_numeric = len(self.numeric_columns)
            n_total = len(self.model.feature_importances_)
            names = list(self.numeric_columns)
            for i in range(n_total - n_numeric):
                names.append(f"tfidf_{i}")
            imp = dict(zip(names, self.model.feature_importances_))
            return dict(sorted(imp.items(), key=lambda x: -x[1]))
        return {}


# ---------------------------------------------------------------------------
# 6. Model Serialization
# ---------------------------------------------------------------------------

class ModelStore:
    @staticmethod
    def save(pipeline: "ProductScoringPipeline", path: str):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        components = {
            "feature_engineer": pipeline.feature_engineer,
            "text_extractor": pipeline.text_extractor,
            "heuristic_scorer": pipeline.heuristic_scorer,
        }
        if pipeline.profiler and pipeline.profiler.is_fitted:
            components["profiler"] = pipeline.profiler
        if pipeline.classifier and pipeline.classifier.is_fitted:
            components["classifier"] = pipeline.classifier
        if pipeline.text_serializer:
            components["text_serializer"] = pipeline.text_serializer

        # Save embedding scorer separately (centroid + stats, not the model weights)
        if pipeline.embedding_scorer and pipeline.embedding_scorer.is_fitted:
            joblib.dump({
                "global_centroid": pipeline.embedding_scorer.global_centroid,
                "similarity_stats": pipeline.embedding_scorer.similarity_stats,
                "model_name_or_path": pipeline.embedding_scorer.model_name_or_path,
                "batch_size": pipeline.embedding_scorer.batch_size,
            }, p / "embedding_scorer.joblib")
            components["embedding_scorer"] = True  # marker for metadata

        for name, obj in components.items():
            if name != "embedding_scorer":
                joblib.dump(obj, p / f"{name}.joblib")

        metadata = {
            "mode": pipeline.mode,
            "trained_at": pd.Timestamp.now().isoformat(),
            "components": list(components.keys()),
        }
        with open(p / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    @staticmethod
    def load(path: str) -> "ProductScoringPipeline":
        p = Path(path)
        with open(p / "metadata.json") as f:
            metadata = json.load(f)

        pipeline = ProductScoringPipeline(mode=metadata["mode"])
        for name in metadata["components"]:
            if name == "embedding_scorer":
                data = joblib.load(p / "embedding_scorer.joblib")
                scorer = EmbeddingScorer(
                    model_name_or_path=data["model_name_or_path"],
                    batch_size=data["batch_size"],
                )
                scorer.global_centroid = data["global_centroid"]
                scorer.similarity_stats = data["similarity_stats"]
                scorer.is_fitted = True
                pipeline.embedding_scorer = scorer
            else:
                obj = joblib.load(p / f"{name}.joblib")
                setattr(pipeline, name, obj)
        return pipeline


# ---------------------------------------------------------------------------
# 7. Pipeline Orchestrator
# ---------------------------------------------------------------------------

class ProductScoringPipeline:
    """
    End-to-end pipeline:
      - 'one_class' mode: learns best-seller profile (Isolation Forest)
      - 'binary' mode: supervised classification (LightGBM)
      - 'embedding' mode: fine-tuned embedding model + centroid scoring
    """

    def __init__(self, mode: str = "one_class", weights: Optional[ScoringWeights] = None,
                 embedding_model: Optional[str] = None):
        self.mode = mode
        self.feature_engineer = FeatureEngineer()
        self.text_extractor = TextFeatureExtractor()
        self.heuristic_scorer = HeuristicScorer(weights)
        self.text_serializer = ProductTextSerializer()
        self.profiler = BestSellerProfiler() if mode == "one_class" else None
        self.classifier = SellingClassifier() if mode == "binary" else None
        self.embedding_scorer = (
            EmbeddingScorer(model_name_or_path=embedding_model or "Alibaba-NLP/gte-multilingual-base")
            if mode == "embedding" else None
        )

    def train(self, df: pd.DataFrame, labels: Optional[pd.Series] = None):
        """
        Train the pipeline.
        - one_class: df = best-seller data, labels ignored
        - binary: df = combined data, labels required (0/1)
        - embedding: df = best-seller data, computes centroid
        """
        features = self.feature_engineer.transform(df)

        if self.mode == "embedding":
            texts = self.text_serializer.serialize(df)
            report = self.embedding_scorer.fit(texts)
            report["mode"] = "embedding"
            return report

        text_features = self.text_extractor.fit_transform(df)

        if self.mode == "one_class":
            report = self.profiler.train(features, text_features)
        else:
            if labels is None:
                raise ValueError("Binary mode requires labels (0/1 Series)")
            report = self.classifier.train(features, labels, text_features)

        return report

    def score(self, df: pd.DataFrame, threshold: float = 50.0) -> pd.DataFrame:
        """Score products. Returns df with added score columns."""
        features = self.feature_engineer.transform(df)
        heuristic_scores = self.heuristic_scorer.score(df, features)

        result = df.copy()
        result["heuristic_score"] = heuristic_scores

        if self.mode == "embedding" and self.embedding_scorer and self.embedding_scorer.is_fitted:
            texts = self.text_serializer.serialize(df)
            emb_results = self.embedding_scorer.predict(texts, threshold=threshold)
            result["ml_probability"] = emb_results["embedding_probability"]
            result["embedding_similarity"] = emb_results["embedding_similarity"]
            result["ml_is_selling"] = emb_results["embedding_is_bestseller"]
            result["combined_score"] = (
                result["heuristic_score"] * 0.3 + result["ml_probability"] * 0.7
            ).round(1)
        else:
            text_features = None
            if self.text_extractor.is_fitted:
                text_features = self.text_extractor.transform(df)

            if self.mode == "one_class" and self.profiler and self.profiler.is_fitted:
                ml_results = self.profiler.predict(features, text_features)
                result["ml_probability"] = ml_results["ml_selling_probability"]
                result["ml_is_selling"] = ml_results["ml_is_selling"]
                result["combined_score"] = (
                    result["heuristic_score"] * 0.3 + result["ml_probability"] * 0.7
                ).round(1)
            elif self.mode == "binary" and self.classifier and self.classifier.is_fitted:
                ml_results = self.classifier.predict(features, text_features)
                result["ml_probability"] = ml_results["ml_selling_probability"]
                result["ml_is_selling"] = ml_results["ml_is_selling"]
                result["combined_score"] = (
                    result["heuristic_score"] * 0.3 + result["ml_probability"] * 0.7
                ).round(1)
            else:
                result["combined_score"] = result["heuristic_score"]

        result["predicted_selling"] = (result["combined_score"] >= threshold).astype(int)
        return result

    def save(self, path: str):
        ModelStore.save(self, path)

    @classmethod
    def load(cls, path: str) -> "ProductScoringPipeline":
        return ModelStore.load(path)
