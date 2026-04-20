"""
Script: Generate Sample Data
Usage: python scripts/generate_sample_data.py
"""
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from pathlib import Path

def generate(n: int = 5000, output: str = "data/raw/orders_raw.csv") -> None:
    np.random.seed(42)
    random.seed(42)

    categories = ["Electronica", "Ropa", "Hogar", "Deportes", "Belleza", "Juguetes", "Libros", "Alimentos"]
    statuses = ["completed", "returned", "cancelled", "pending"]
    channels = ["web", "mobile", "marketplace", "social"]
    regions = ["CDMX", "GDL", "MTY", "PUE", "QRO", "CUN"]

    start = datetime(2023, 1, 1)
    dates = [start + timedelta(days=random.randint(0, 548)) for _ in range(n)]

    df = pd.DataFrame({
        "order_id": [f"ORD-{str(i).zfill(6)}" for i in range(1, n + 1)],
        "customer_id": [f"CUST-{str(random.randint(1, 1500)).zfill(4)}" for _ in range(n)],
        "product_id": [f"PROD-{str(random.randint(1, 300)).zfill(4)}" for _ in range(n)],
        "category": [random.choice(categories) for _ in range(n)],
        "channel": [random.choice(channels) for _ in range(n)],
        "quantity": np.random.randint(1, 10, n),
        "unit_price": np.round(np.random.uniform(50, 5000, n), 2),
        "discount_pct": np.round(np.random.choice([0.0, 5.0, 10.0, 15.0, 20.0, 25.0], n,
                                                   p=[0.4, 0.2, 0.15, 0.1, 0.1, 0.05]), 2),
        "status": [random.choice(statuses) for _ in range(n)],
        "order_date": [d.strftime("%Y-%m-%d") for d in dates],
        "region": [random.choice(regions) for _ in range(n)],
        "customer_age": np.random.randint(18, 70, n),
        "is_new_customer": np.random.choice([True, False], n, p=[0.3, 0.7]),
    })

    # Inject noise
    for idx in df.sample(frac=0.03).index:
        df.at[idx, "discount_pct"] = None
    for idx in df.sample(frac=0.02).index:
        df.at[idx, "region"] = None

    # Add duplicates
    df = pd.concat([df, df.sample(50)], ignore_index=True)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"✓ Generated {len(df):,} rows → {output}")

if __name__ == "__main__":
    generate()
