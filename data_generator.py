"""
Synthetic Product Feed Generator
=================================
Generates realistic product feeds for testing the scoring pipeline.
Creates both labeled (has sales data) and unlabeled feeds.
"""

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

CATEGORIES = [
    "Electronics", "Clothing", "Home & Kitchen", "Sports & Outdoors",
    "Beauty", "Toys & Games", "Books", "Automotive", "Pet Supplies", "Garden"
]

BRANDS = {
    "Electronics": ["Samsung", "Apple", "Sony", "LG", "Anker", "Bose", "", ""],
    "Clothing": ["Nike", "Adidas", "Levi's", "H&M", "Zara", "", "", ""],
    "Home & Kitchen": ["KitchenAid", "Instant Pot", "Dyson", "OXO", "", "", ""],
    "Sports & Outdoors": ["Nike", "Under Armour", "Coleman", "Yeti", "", ""],
    "Beauty": ["L'Oreal", "Maybelline", "CeraVe", "Neutrogena", "", ""],
    "Toys & Games": ["LEGO", "Hasbro", "Mattel", "Fisher-Price", "", ""],
    "Books": ["Penguin", "HarperCollins", "Simon & Schuster", "", ""],
    "Automotive": ["Bosch", "3M", "Armor All", "Chemical Guys", "", ""],
    "Pet Supplies": ["Blue Buffalo", "Kong", "Purina", "Greenies", "", ""],
    "Garden": ["Fiskars", "Miracle-Gro", "Black+Decker", "Husqvarna", "", ""],
}

POWER_WORDS = ["Premium", "Best Seller", "New", "Top Rated", "Official"]


def _generate_product(category: str, is_selling: bool) -> dict:
    """Generate a single product with characteristics correlated to selling status."""
    brand = random.choice(BRANDS.get(category, [""]))
    has_brand = brand != ""

    # Selling products tend to have better listings
    sell_boost = 0.7 if is_selling else 0.3

    # Title
    base_title = f"{random.choice(['Pro', 'Ultra', 'Classic', 'Eco', 'Smart', 'Max'])} "
    base_title += f"{category.split('&')[0].strip()} Product {random.randint(100, 9999)}"
    if has_brand and random.random() < sell_boost:
        base_title = f"{brand} {base_title}"
    if random.random() < sell_boost * 0.5:
        base_title += f" - {random.choice(POWER_WORDS)}"

    # Price - selling products tend to be competitively priced
    base_prices = {
        "Electronics": 150, "Clothing": 40, "Home & Kitchen": 60,
        "Sports & Outdoors": 55, "Beauty": 25, "Toys & Games": 30,
        "Books": 15, "Automotive": 35, "Pet Supplies": 28, "Garden": 45,
    }
    base = base_prices.get(category, 50)
    if is_selling:
        price = round(base * random.uniform(0.6, 1.4), 2)
        # Psychological pricing more common
        if random.random() < 0.6:
            price = int(price) + 0.99
    else:
        price = round(base * random.uniform(0.3, 3.0), 2)
        if random.random() < 0.3:
            price = int(price) + 0.99

    # Description
    desc_length = int(np.random.normal(300 if is_selling else 80, 100))
    desc_length = max(0, desc_length)
    if desc_length > 20:
        bullets = random.randint(2, 6) if is_selling and random.random() < 0.7 else 0
        desc = "A " * (desc_length // 2)
        if bullets:
            desc += " " + " • Feature point" * bullets
    else:
        desc = "" if random.random() < 0.5 else "Short description"

    # Images
    if is_selling:
        image_count = random.choices([1, 2, 3, 4, 5, 6, 7, 8], 
                                      weights=[5, 10, 20, 25, 20, 10, 5, 5])[0]
    else:
        image_count = random.choices([0, 1, 2, 3, 4, 5],
                                      weights=[15, 30, 25, 15, 10, 5])[0]

    # Variants
    variant_count = random.choices(
        [1, 2, 3, 4, 5, 6],
        weights=[30, 25, 20, 15, 7, 3] if is_selling else [60, 20, 10, 5, 3, 2]
    )[0]

    # Last modified
    if is_selling:
        days_ago = random.choices(
            [random.randint(0, 7), random.randint(8, 30),
             random.randint(31, 90), random.randint(91, 365)],
            weights=[40, 30, 20, 10]
        )[0]
    else:
        days_ago = random.choices(
            [random.randint(0, 7), random.randint(8, 30),
             random.randint(31, 90), random.randint(91, 365),
             random.randint(366, 730)],
            weights=[10, 15, 20, 30, 25]
        )[0]
    last_modified = datetime.now() - timedelta(days=days_ago)

    # UPC/barcode
    has_upc = random.random() < (0.8 if is_selling else 0.4)

    return {
        "product_id": f"PRD-{random.randint(100000, 999999)}",
        "title": base_title,
        "description": desc,
        "category": category,
        "price": price,
        "brand": brand if has_brand else "",
        "image_count": image_count,
        "variant_count": variant_count,
        "last_modified": last_modified.isoformat(),
        "upc": f"{random.randint(10**11, 10**12-1)}" if has_upc else "",
        "sku": f"SKU-{random.randint(1000, 99999)}",
    }


def generate_labeled_feed(n: int = 5000, selling_ratio: float = 0.35) -> pd.DataFrame:
    """
    Generate a labeled feed (simulates a store that HAS sales data).
    selling_ratio: fraction of products that are actually selling.
    """
    products = []
    for _ in range(n):
        cat = random.choice(CATEGORIES)
        is_selling = random.random() < selling_ratio
        prod = _generate_product(cat, is_selling)
        prod["is_selling"] = int(is_selling)
        products.append(prod)
    return pd.DataFrame(products)


def generate_unlabeled_feed(n: int = 5000) -> pd.DataFrame:
    """
    Generate an unlabeled feed (simulates a store WITHOUT sales data).
    Still has an underlying selling distribution, but labels are hidden.
    """
    products = []
    for _ in range(n):
        cat = random.choice(CATEGORIES)
        is_selling = random.random() < 0.30  # hidden truth
        prod = _generate_product(cat, is_selling)
        prod["_hidden_is_selling"] = int(is_selling)  # for evaluation only
        products.append(prod)
    return pd.DataFrame(products)
