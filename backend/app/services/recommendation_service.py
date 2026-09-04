import os
import joblib
import pandas as pd
from typing import List, Dict, Any

from app.core.config import settings
from app.core.logger import logger
from app.schemas.recommendation import RecommendationRequest, RecommendationResponse, RecommendedProperty


def format_inr(amount: float) -> str:
    if amount >= 10000000:
        return f"₹{amount / 10000000:.2f} Cr"
    else:
        return f"₹{amount / 100000:.2f} Lakhs"


class RecommendationService:
    def __init__(self):
        self.catalog_df: pd.DataFrame = pd.DataFrame()
        self._load_catalog()

    def _load_catalog(self):
        try:
            if os.path.exists(settings.CATALOG_PATH):
                self.catalog_df = joblib.load(settings.CATALOG_PATH)
                logger.info(f"Loaded property catalog with {len(self.catalog_df)} records.")
            else:
                logger.warning(f"Property catalog not found at {settings.CATALOG_PATH}")
        except Exception as e:
            logger.error(f"Failed to load property catalog: {e}", exc_info=True)

    def recommend(self, req: RecommendationRequest) -> RecommendationResponse:
        if self.catalog_df.empty:
            self._load_catalog()
            if self.catalog_df.empty:
                raise RuntimeError("Property catalog is not loaded. Please train models first.")

        df = self.catalog_df.copy()
        user_budget = float(req.budget)
        user_loc = req.location.strip().lower() if req.location else None
        user_bhk = int(req.bhk) if req.bhk is not None else None
        user_amenities = [a.strip().lower() for a in (req.amenities or []) if a.strip()]

        scored_records = []

        for idx, row in df.iterrows():
            price = float(row["Price"])
            loc = str(row["Location"]).strip()
            bhk = int(row["BHK"])
            prop_amenities_raw = str(row.get("Amenities", ""))
            prop_amenities = [x.strip().lower() for x in prop_amenities_raw.split(",")]

            # 1. Budget proximity score (exponential decay)
            rel_diff = abs(price - user_budget) / max(user_budget, 1.0)
            budget_score = max(0.0, 1.0 - (rel_diff * 1.5))

            # 2. Location match score
            if user_loc:
                loc_score = 1.0 if loc.lower() == user_loc else 0.30
            else:
                loc_score = 1.0

            # 3. BHK match score
            if user_bhk:
                if bhk == user_bhk:
                    bhk_score = 1.0
                elif abs(bhk - user_bhk) == 1:
                    bhk_score = 0.55
                else:
                    bhk_score = 0.15
            else:
                bhk_score = 1.0

            # 4. Amenities similarity score (Jaccard-like overlap)
            matched_amenities = []
            if user_amenities:
                for am in user_amenities:
                    if any(am in p_am for p_am in prop_amenities):
                        matched_amenities.append(am.title())
                amenity_score = len(matched_amenities) / max(len(user_amenities), 1)
            else:
                amenity_score = min(1.0, float(row.get("luxury_score", 5.0)) / 10.0)

            # Weighted combination
            # Budget: 40%, Location: 25%, BHK: 20%, Amenities: 15%
            composite = (
                (0.40 * budget_score) +
                (0.25 * loc_score) +
                (0.20 * bhk_score) +
                (0.15 * amenity_score)
            )

            sim_pct = round(min(98.5, max(45.0, composite * 100)), 1)

            # Match rationale
            reasons = []
            if abs(price - user_budget) / user_budget <= 0.10:
                reasons.append(f"Price matches within 10% of target budget")
            elif abs(price - user_budget) / user_budget <= 0.20:
                reasons.append(f"Comfortable price point close to budget")

            if user_loc and loc.lower() == user_loc:
                reasons.append(f"Prime location match in {loc}")

            if user_bhk and bhk == user_bhk:
                reasons.append(f"Exact {bhk} BHK layout preference")

            if matched_amenities:
                reasons.append(f"Includes {len(matched_amenities)} requested amenities: {', '.join(matched_amenities[:3])}")

            reasons.append(f"{row.get('property_category', 'Mid-Range')} category asset")

            scored_records.append({
                "property_id": str(row.get("property_id", f"PROP_{idx}")),
                "location": loc,
                "area": float(row["Area"]),
                "bhk": bhk,
                "bathrooms": int(row["Bathrooms"]),
                "property_age": int(row["Property Age"]),
                "furnishing_status": str(row["Furnishing Status"]),
                "amenities": prop_amenities_raw,
                "price": price,
                "price_formatted": format_inr(price),
                "price_per_sqft": float(row.get("price_per_sqft", round(price / row["Area"], 2))),
                "property_category": str(row.get("property_category", "Mid-Range")),
                "similarity_score": sim_pct,
                "match_reasons": reasons[:3],
                "_composite": composite
            })

        # Rank by composite score descending
        scored_records.sort(key=lambda x: x["_composite"], reverse=True)
        top_items = scored_records[: req.top_n or 5]

        # Clean internal helper key
        for item in top_items:
            item.pop("_composite", None)

        return RecommendationResponse(
            total_recommended=len(top_items),
            query_criteria={
                "budget": req.budget,
                "budget_formatted": format_inr(req.budget),
                "location": req.location or "Any Location",
                "bhk": req.bhk if req.bhk else "Any BHK",
                "amenities": req.amenities or []
            },
            recommendations=[RecommendedProperty(**item) for item in top_items]
        )


recommendation_service = RecommendationService()
