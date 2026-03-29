"""
Synthetic Product Feed Generator
=================================
Generates realistic product feeds matching real eBay/AliExpress schemas
for testing the pipeline without real CSV files.
"""

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# --- Shared product names ---
PRODUCT_TYPES_EN = [
    "Wireless Bluetooth Headphones", "LED Desk Lamp", "Stainless Steel Water Bottle",
    "Phone Case Cover", "USB-C Charging Cable", "Smart Watch Band",
    "Portable Power Bank", "Kitchen Knife Set", "Running Shoes",
    "Yoga Mat", "Car Phone Mount", "Laptop Stand", "Electric Toothbrush",
    "Gaming Mouse Pad", "Sunglasses UV Protection", "Pet Dog Collar",
    "Garden Hose Nozzle", "Baby Bottle Set", "Travel Backpack",
    "Desk Organizer Storage", "Camping Tent", "Fishing Lure Kit",
]

PRODUCT_TYPES_DE = [
    "Kabellose Bluetooth Kopfhörer", "LED Schreibtischlampe", "Edelstahl Trinkflasche",
    "Handyhülle Schutzhülle", "USB-C Ladekabel", "Smartwatch Armband",
    "Tragbare Powerbank", "Küchenmesser Set", "Laufschuhe",
    "Yogamatte", "Auto Handyhalterung", "Laptop Ständer", "Elektrische Zahnbürste",
    "Gaming Mauspad", "Sonnenbrille UV Schutz", "Haustier Hundehalsband",
    "Gartenschlauch Düse", "Babyflasche Set", "Reiserucksack",
    "Schreibtisch Organizer", "Camping Zelt", "Angel Köder Set",
]

BRANDS = [
    "Samsung", "Anker", "Baseus", "Ugreen", "LIGE", "Curren", "Bandai",
    "TAKARA TOMY", "Disney", "Xtep", "podofo", "isfriday", "NONE", "NONE", "",
]

EBAY_CONDITIONS_DE = [
    "Neu", "Neu", "Neu", "Neu",  # most common
    "Gebraucht", "Neu: Sonstige (siehe Artikelbeschreibung)",
    "Neu mit Etikett", "Sehr gut - Refurbished", "Gut",
]

REGIONS = ["de", "uk"]


def generate_ebay_feed(n: int = 2000) -> pd.DataFrame:
    """Generate synthetic eBay feed matching real schema."""
    products = []
    for i in range(n):
        region = random.choices(REGIONS, weights=[70, 30])[0]
        currency = "EUR" if region == "de" else "GBP"
        price = round(random.uniform(1.50, 500.0), 2)

        # Shipping
        ship_country = "DE" if region == "de" else "GB"
        ship_method = random.choice(["STANDARD", "ECONOMY", "EXPEDITED", "PICKUP", ""])
        ship_cost = round(random.choice([0.0, 0.0, 0.0, 1.99, 3.49, 4.99, 6.99]), 2)
        shipping = f"{ship_country}::{ship_method}:{ship_cost:.2f} {currency}"

        title = random.choice(PRODUCT_TYPES_DE if region == "de" else PRODUCT_TYPES_EN)
        title += f" {random.choice(['Pro', 'Plus', 'Max', 'Lite', ''])}"
        brand = random.choice(BRANDS)

        # Sale price (rarely present in eBay data)
        sale_price = "" if random.random() < 0.9 else f"{round(price * 0.8, 2)}"

        has_gtin = random.random() < 0.6
        days_ago = random.randint(0, 90)
        insert_time = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        products.append({
            "ID": f"v1|{random.randint(100000000000, 999999999999)}|0",
            "TITLE": title,
            "DESCRIPTION": title,  # eBay often has title = description
            "PRICE": f"{price:.2f} {currency}",
            "CONDITION": random.choice(EBAY_CONDITIONS_DE),
            "LINK": f"https://www.ebay.{'de' if region == 'de' else 'co.uk'}/itm/{random.randint(100000000000, 999999999999)}",
            "AVAILABILITY": random.choice(["in_stock", "in_stock", "in_stock", "out_of_stock"]),
            "IMAGE_LINK": f"https://i.ebayimg.com/images/g/test/s-l1600.png" if random.random() < 0.95 else "",
            "GTIN": str(random.randint(10**12, 10**13 - 1)) if has_gtin else "",
            "SHIPPING": shipping,
            "CUSTOM_LABEL_0": random.choices(
                ["Zombie", "LowPerforming", "MidPerforming", "HighPerforming", "VeryHighPerforming"],
                weights=[90, 5, 2, 1, 2],
            )[0],
            "CUSTOM_LABEL_4": "Ebay_Best_Sellers",
            "REGION": region,
            "INSERT_TIME": insert_time,
            "SALE_PRICE": sale_price,
        })

    return pd.DataFrame(products)


def generate_aliexpress_feed(n: int = 1500) -> pd.DataFrame:
    """Generate synthetic AliExpress feed matching real schema."""
    products = []
    for i in range(n):
        price = round(random.uniform(2.0, 300.0), 2)
        discount_rate = random.choice([0, 0, 10, 15, 20, 30, 40, 50, 57, 60])
        sale_price = round(price * (1 - discount_rate / 100), 2) if discount_rate > 0 else price

        title = random.choice(PRODUCT_TYPES_DE)
        brand = random.choice(BRANDS)

        product_score = random.choices(
            [0.0, round(random.uniform(3.0, 4.0), 1), round(random.uniform(4.0, 4.5), 1),
             round(random.uniform(4.5, 5.0), 1), 5.0],
            weights=[5, 5, 10, 70, 10],
        )[0]
        review_number = random.choices(
            [0, random.randint(1, 50), random.randint(50, 500),
             random.randint(500, 5000), random.randint(5000, 50000)],
            weights=[10, 20, 30, 30, 10],
        )[0]

        days_ago = random.randint(0, 60)
        insert_time = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        products.append({
            "ITEM_ID": str(random.randint(1005000000000000, 1005099999999999)),
            "SKU": str(random.randint(12000000000000000, 12000099999999999)),
            "TITLE": title,
            "DESCRIPTION": title,
            "BRAND": brand,
            "PRICE": price,
            "SALE_PRICE": sale_price,
            "DISCOUNT_RATE": discount_rate if discount_rate > 0 else "null",
            "LINK": f"https://de.aliexpress.com/item/{random.randint(1005000000000000, 1005099999999999)}.html",
            "IMAGE_LINK": f"https://ae-pic-a1.aliexpress-media.com/kf/test.jpg" if random.random() < 0.98 else "",
            "AVAILABILITY": "in stock",
            "SHIPPING": round(random.choice([0, 0, 0, 1.99, 2.49, 3.99]), 2),
            "DELIVERY_DAYS": random.choice([3, 4, 5, 7, 8, 10, 15, 17, 20]),
            "COLOR": random.choice(["Black", "White", "Red", "Blue", "2 64G", ""]),
            "SIZE": random.choice(["S", "M", "L", "XL", "50 Pieces", ""]),
            "GENDER": random.choice(["", "", "", "Male", "Female"]),
            "PRODUCT_SCORE": product_score,
            "REVIEW_NUMBER": review_number,
            "REGION": "de",
            "INSERT_TIME": insert_time,
        })

    return pd.DataFrame(products)
