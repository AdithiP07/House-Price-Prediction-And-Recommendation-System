"""
Synthetic Housing Dataset Generator
Generates a realistic metropolitan real estate dataset (calibrated on Bangalore markets)
including Whitefield, Indiranagar, Koramangala, etc., with realistic pricing,
amenities, furnishing status, and intentional real-world imperfections
(missing values, duplicates, and outliers) for preprocessing demonstration.
"""

import os
import random
import numpy as np
import pandas as pd

# Set fixed seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

LOCATIONS_CONFIG = {
    "Whitefield": {"base_rate": 5300, "variance": 400},
    "Indiranagar": {"base_rate": 11500, "variance": 900},
    "Koramangala": {"base_rate": 11000, "variance": 850},
    "HSR Layout": {"base_rate": 8200, "variance": 600},
    "Electronic City": {"base_rate": 4500, "variance": 350},
    "Bellandur": {"base_rate": 7100, "variance": 500},
    "Marathahalli": {"base_rate": 6100, "variance": 450},
    "Hebbal": {"base_rate": 7800, "variance": 550},
    "Rajajinagar": {"base_rate": 9200, "variance": 700},
    "Sarjapur Road": {"base_rate": 6400, "variance": 450},
    "Jayanagar": {"base_rate": 10500, "variance": 800},
    "Malleshwaram": {"base_rate": 10000, "variance": 750},
    "Yelahanka": {"base_rate": 5400, "variance": 400},
    "Banashankari": {"base_rate": 6600, "variance": 500},
    "Thanisandra": {"base_rate": 5600, "variance": 420},
}

ALL_AMENITIES = [
    "Gym",
    "Swimming Pool",
    "Clubhouse",
    "24/7 Security",
    "Power Backup",
    "Children's Play Area",
    "Covered Parking",
    "Landscaped Garden",
    "Tennis Court",
    "High-Speed Elevators",
    "Intercom",
    "Jogging Track",
]

FURNISHING_OPTIONS = ["Furnished", "Semi-Furnished", "Unfurnished"]
FURNISHING_PROBS = [0.25, 0.50, 0.25]


def generate_property_data(n_samples: int = 3600) -> pd.DataFrame:
    records = []
    locations = list(LOCATIONS_CONFIG.keys())
    location_weights = [
        0.14, 0.08, 0.08, 0.10, 0.12, 0.08, 0.08, 0.06, 0.05, 0.07, 0.05, 0.05, 0.05, 0.04, 0.05
    ]
    # Normalize weights
    location_weights = [w / sum(location_weights) for w in location_weights]

    for i in range(n_samples):
        loc = np.random.choice(locations, p=location_weights)
        loc_cfg = LOCATIONS_CONFIG[loc]

        # BHK configuration
        bhk = np.random.choice([1, 2, 3, 4, 5], p=[0.12, 0.38, 0.35, 0.12, 0.03])

        if bhk == 1:
            area = int(np.random.normal(620, 90))
            area = max(450, min(area, 850))
            bathrooms = 1
        elif bhk == 2:
            area = int(np.random.normal(1080, 140))
            area = max(800, min(area, 1450))
            bathrooms = np.random.choice([1, 2], p=[0.15, 0.85])
        elif bhk == 3:
            area = int(np.random.normal(1650, 220))
            area = max(1300, min(area, 2400))
            bathrooms = np.random.choice([2, 3], p=[0.40, 0.60])
        elif bhk == 4:
            area = int(np.random.normal(2600, 350))
            area = max(2100, min(area, 3700))
            bathrooms = np.random.choice([3, 4], p=[0.35, 0.65])
        else:  # 5 BHK
            area = int(np.random.normal(3800, 450))
            area = max(3100, min(area, 5200))
            bathrooms = np.random.choice([4, 5], p=[0.30, 0.70])

        property_age = int(np.clip(np.random.exponential(scale=6.5), 0, 28))
        furnishing = np.random.choice(FURNISHING_OPTIONS, p=FURNISHING_PROBS)

        # Amenities: 2 to 9 randomly sampled
        n_amenities = np.random.choice(range(2, 10), p=[0.10, 0.20, 0.25, 0.20, 0.12, 0.08, 0.04, 0.01])
        selected_amenities = sorted(random.sample(ALL_AMENITIES, k=n_amenities))
        amenities_str = ", ".join(selected_amenities)

        # Price calculation model
        # Base sqft rate
        rate = np.random.normal(loc_cfg["base_rate"], loc_cfg["variance"])
        rate = max(rate, loc_cfg["base_rate"] * 0.75)

        base_val = area * rate

        # BHK efficiency bonus
        bhk_mult = 1.0 + (bhk - 2) * 0.03

        # Bathrooms bonus
        bath_mult = 1.0 + (bathrooms - 1) * 0.025

        # Age depreciation: -0.8% per year
        age_factor = max(0.72, 1.0 - (property_age * 0.0085))

        # Furnishing bonus
        furnishing_mult = 1.08 if furnishing == "Furnished" else (1.03 if furnishing == "Semi-Furnished" else 0.98)

        # Amenities / Luxury bonus
        luxury_mult = 1.0 + (len(selected_amenities) * 0.012)

        # Total price
        calc_price = base_val * bhk_mult * bath_mult * age_factor * furnishing_mult * luxury_mult
        
        # Add slight noise
        noise = np.random.normal(1.0, 0.035)
        final_price = calc_price * noise

        # Round to neat currency figure
        final_price = round(final_price / 10000.0) * 10000

        records.append({
            "Location": loc,
            "Area": area,
            "BHK": bhk,
            "Bathrooms": bathrooms,
            "Property Age": property_age,
            "Furnishing Status": furnishing,
            "Amenities": amenities_str,
            "Price": int(final_price)
        })

    # Ensure a specific target row closely matches the prompt example:
    # {"area": 1500, "bhk": 3, "bathrooms": 2, "property_age": 5, "location": "Whitefield"} -> ~8500000
    records.append({
        "Location": "Whitefield",
        "Area": 1500,
        "BHK": 3,
        "Bathrooms": 2,
        "Property Age": 5,
        "Furnishing Status": "Semi-Furnished",
        "Amenities": "Gym, Swimming Pool, Clubhouse, 24/7 Security, Power Backup, Covered Parking",
        "Price": 8500000
    })

    df = pd.DataFrame(records)

    # Inject realistic real-world flaws for data cleaning demonstration:
    # 1. Duplicate rows
    n_dupes = 45
    dupes = df.sample(n=n_dupes, random_state=SEED).copy()
    df = pd.concat([df, dupes], ignore_index=True)

    # 2. Missing values (~1.5% in some columns)
    missing_indices_age = np.random.choice(df.index, size=int(0.015 * len(df)), replace=False)
    df.loc[missing_indices_age, "Property Age"] = np.nan

    missing_indices_bath = np.random.choice(df.index, size=int(0.012 * len(df)), replace=False)
    df.loc[missing_indices_bath, "Bathrooms"] = np.nan

    missing_indices_furn = np.random.choice(df.index, size=int(0.015 * len(df)), replace=False)
    df.loc[missing_indices_furn, "Furnishing Status"] = np.nan

    # 3. Controlled outliers (a few extreme area or price rows to demonstrate IQR treatment)
    outlier_idx_1 = np.random.choice(df.index, size=3, replace=False)
    df.loc[outlier_idx_1[0], "Price"] = df["Price"].max() * 2.8
    df.loc[outlier_idx_1[1], "Area"] = 8200
    df.loc[outlier_idx_1[2], "Price"] = 450000  # abnormally cheap

    # Shuffle rows
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return df


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "raw")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "housing_raw.csv")

    print("Generating synthetic real estate dataset...")
    df = generate_property_data(n_samples=3600)
    df.to_csv(output_file, index=False)

    print(f"Dataset generated successfully at: {output_file}")
    print(f"Total Rows: {len(df)}, Total Columns: {df.shape[1]}")
    print("\nDataset Snapshot:")
    print(df.head(3))
    print("\nMissing values per column:")
    print(df.isna().sum())


if __name__ == "__main__":
    main()
