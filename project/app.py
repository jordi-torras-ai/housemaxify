import csv
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from flask import Flask, render_template, request, send_from_directory
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.geocoders import Nominatim

# ---------------------------------------------------------------------------
# Paths & assets
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "xgboost_kc_house.pkl"
LEADS_PATH = BASE_DIR / "data" / "leads.csv"
DATA_PATH = BASE_DIR.parent / "data" / "kc_house_data_clean.csv"

# Ensure the leads file exists with headers
LEADS_PATH.parent.mkdir(parents=True, exist_ok=True)
if not LEADS_PATH.exists():
    with LEADS_PATH.open("w", newline="", encoding="utf-8") as leads_file:
        writer = csv.writer(leads_file)
        writer.writerow(["timestamp", "name", "email", "phone", "intent", "timeline_months"])

# ---------------------------------------------------------------------------
# Model & data loading
# ---------------------------------------------------------------------------
try:
    MODEL = joblib.load(MODEL_PATH)
except FileNotFoundError as exc:
    raise RuntimeError(f"Model file not found at {MODEL_PATH}") from exc

try:
    REFERENCE_DATA = pd.read_csv(DATA_PATH)
    REFERENCE_DATA["zipcode"] = REFERENCE_DATA["zipcode"].astype(str)
except FileNotFoundError as exc:
    raise RuntimeError(f"Reference dataset not found at {DATA_PATH}") from exc

ZIPCODE_COORDS = (
    REFERENCE_DATA.groupby("zipcode")[["lat", "long"]]
    .mean()
    .to_dict(orient="index")
)

# Model expects all features except 'price' and 'sale_date'
MODEL_FEATURES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "zipcode",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
    "sale_year",
    "sale_month",
    "sale_dow",
    "house_age_at_sale",
    "was_renovated",
    "years_since_renov",
]

SIMILARITY_COLUMNS = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "zipcode",
    "price",
    "sqft_lot",
]


def create_app() -> Flask:
    app = Flask(__name__, template_folder="html")

    @app.route("/css/<path:filename>")
    def css(filename):
        return send_from_directory(BASE_DIR / "css", filename)

    @app.route("/js/<path:filename>")
    def js(filename):
        return send_from_directory(BASE_DIR / "js", filename)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/estimate", methods=["GET", "POST"])
    def estimate():
        if request.method == "POST":
            form_data = request.form.to_dict()
            try:
                features = prepare_features(form_data)
                prediction = predict_price(features)
                price_range = make_price_range(prediction)
                lead_info = extract_lead_info(form_data)
                append_lead(lead_info)
                similar_homes = find_similar_properties(features, top_n=3)
            except ValueError as err:
                return render_template(
                    "estimate.html",
                    error=str(err),
                    form_values=form_data,
                )

            return render_template(
                "results.html",
                price_range=price_range,
                prediction=prediction,
                similar_properties=similar_homes,
                lead_name=lead_info["name"],
            )

        return render_template("estimate.html", form_values={})

    return app


def prepare_features(form: Dict[str, str]) -> Dict[str, Any]:
    """Convert form inputs into the feature dict expected by the model."""
    required_float = {
        "bedrooms": int,
        "bathrooms": float,
        "sqft_living": float,
        "sqft_lot": float,
        "floors": float,
        "sqft_above": float,
        "sqft_basement": float,
        "sqft_living15": float,
        "sqft_lot15": float,
    }

    required_int = {
        "view": int,
        "condition": int,
        "grade": int,
        "yr_built": int,
    }

    features: Dict[str, Any] = {}

    for key, caster in required_float.items():
        value = form.get(key)
        if value in (None, ""):
            raise ValueError(f"'{friendly_label(key)}' is required.")
        try:
            features[key] = caster(float(value))
        except ValueError as exc:
            raise ValueError(f"'{friendly_label(key)}' must be a number.") from exc

    for key, caster in required_int.items():
        value = form.get(key)
        if value in (None, ""):
            raise ValueError(f"'{friendly_label(key)}' is required.")
        try:
            features[key] = caster(value)
        except ValueError as exc:
            raise ValueError(f"'{friendly_label(key)}' must be an integer.") from exc

    zipcode = form.get("zipcode", "").strip()
    if not zipcode:
        raise ValueError("'Zip Code' is required.")
    if zipcode.isdigit() and len(zipcode) <= 5:
        zipcode = zipcode.zfill(5)
    features["zipcode"] = zipcode

    yr_renovated_input = form.get("yr_renovated", "").strip()
    try:
        yr_renovated = int(yr_renovated_input) if yr_renovated_input else 0
    except ValueError as exc:
            raise ValueError("'Year Renovated' must be an integer.") from exc
    features["yr_renovated"] = yr_renovated

    waterfront = form.get("waterfront")
    if waterfront not in {"0", "1"}:
        raise ValueError("Please select whether the property is waterfront.")
    features["waterfront"] = int(waterfront)

    intent_months = form.get("timeline")
    if intent_months not in {"3", "6", "12"}:
        raise ValueError("Timeline selection is required.")
    months = int(intent_months)

    sale_date = pd.Timestamp.today().normalize() + pd.DateOffset(months=months)
    features["sale_year"] = int(sale_date.year)
    features["sale_month"] = int(sale_date.month)
    features["sale_dow"] = int(sale_date.dayofweek)

    address = form.get("address", "").strip()
    city = form.get("city", "").strip()
    state = form.get("state", "").strip().upper()

    if not address:
        raise ValueError("Street address is required for geocoding.")
    if not city:
        raise ValueError("City is required for geocoding.")
    if not state:
        raise ValueError("State abbreviation is required for geocoding.")

    features["lat"], features["long"] = geocode_to_coordinates(
        address=address,
        city=city,
        state=state,
        zipcode=zipcode,
    )

    yr_built = features["yr_built"]
    features["house_age_at_sale"] = max(features["sale_year"] - yr_built, 0)

    if yr_renovated > 0 and yr_renovated <= features["sale_year"]:
        features["was_renovated"] = 1
        features["years_since_renov"] = max(features["sale_year"] - yr_renovated, 0)
    else:
        features["was_renovated"] = 0
        features["years_since_renov"] = 0

    for col in MODEL_FEATURES:
        if col not in features:
            # Default missing numeric values to 0
            features[col] = 0

    return features


def friendly_label(field_name: str) -> str:
    labels = {
        "bedrooms": "Number of Bedrooms",
        "bathrooms": "Number of Bathrooms",
        "sqft_living": "Living Area (sqft)",
        "sqft_lot": "Lot Size (sqft)",
        "floors": "Number of Floors",
        "sqft_above": "Above Ground Living Space (sqft)",
        "sqft_basement": "Basement Size (sqft)",
        "lat": "Latitude",
        "long": "Longitude",
        "sqft_living15": "Nearby Living Area (15)",
        "sqft_lot15": "Nearby Lot Size (15)",
        "view": "View Rating",
        "condition": "Overall Condition",
        "grade": "Construction Grade",
        "yr_built": "Year Built",
    }
    return labels.get(field_name, field_name)


def predict_price(features: Dict[str, Any]) -> float:
    input_df = pd.DataFrame([{k: features[k] for k in MODEL_FEATURES}])
    prediction = MODEL.predict(input_df)[0]
    return float(prediction)


def make_price_range(prediction: float) -> Dict[str, int]:
    buffer = prediction * 0.07
    low = max(prediction - buffer, 0)
    high = prediction + buffer
    return {"low": int(low), "high": int(high)}


def extract_lead_info(form: Dict[str, str]) -> Dict[str, str]:
    name = form.get("name", "").strip()
    email = form.get("email", "").strip()
    phone = form.get("phone", "").strip()
    intent = form.get("intent", "").strip()
    timeline = form.get("timeline", "").strip()

    if not name or not email or not phone:
        raise ValueError("Name, email, and phone are required for follow-up.")
    if intent not in {"buy", "sell"}:
        raise ValueError("Please indicate whether you plan to buy or sell.")

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "intent": intent,
        "timeline": timeline,
    }


def append_lead(lead: Dict[str, str]) -> None:
    timestamp = datetime.utcnow().isoformat()
    row = [
        timestamp,
        lead["name"],
        lead["email"],
        lead["phone"],
        lead["intent"],
        lead["timeline"],
    ]
    with LEADS_PATH.open("a", newline="", encoding="utf-8") as leads_file:
        writer = csv.writer(leads_file)
        writer.writerow(row)


def find_similar_properties(features: Dict[str, Any], top_n: int = 3) -> List[Dict[str, Any]]:
    numeric_input = {
        "bedrooms": float(features["bedrooms"]),
        "bathrooms": float(features["bathrooms"]),
        "sqft_living": float(features["sqft_living"]),
        "sqft_lot": float(features["sqft_lot"]),
    }

    df = REFERENCE_DATA.dropna(subset=SIMILARITY_COLUMNS).copy()
    df = df[df["zipcode"] == features["zipcode"]]

    if df.empty:
        df = REFERENCE_DATA.dropna(subset=SIMILARITY_COLUMNS).copy()

    weights = {
        "bedrooms": 0.25,
        "bathrooms": 0.25,
        "sqft_living": 0.35,
        "sqft_lot": 0.15,
    }

    def distance(row):
        score = 0.0
        for key, weight in weights.items():
            denom = max(numeric_input[key], 1.0)
            score += weight * abs(row[key] - numeric_input[key]) / denom
        return score

    df["similarity"] = df.apply(distance, axis=1)
    top_matches = df.nsmallest(top_n, "similarity")

    results: List[Dict[str, Any]] = []
    for _, row in top_matches.iterrows():
        results.append(
            {
                "bedrooms": int(row["bedrooms"]),
                "bathrooms": round(float(row["bathrooms"]), 1),
                "sqft_living": int(row["sqft_living"]),
                "sqft_lot": int(row["sqft_lot"]),
                "zipcode": row["zipcode"],
                "price": int(row["price"]),
            }
        )
    return results


GEOCODER = Nominatim(user_agent="house_maxify_app")


@lru_cache(maxsize=256)
def geocode_to_coordinates(address: str, city: str, state: str, zipcode: str) -> tuple[float, float]:
    query_parts = [address]
    if city:
        query_parts.append(city)
    if state:
        query_parts.append(state)
    if zipcode:
        query_parts.append(zipcode)

    query = ", ".join(part for part in query_parts if part)

    if not query:
        raise ValueError("Unable to geocode: address details are incomplete.")

    try:
        location = GEOCODER.geocode(query, timeout=10)
        if location:
            return float(location.latitude), float(location.longitude)
    except (GeocoderTimedOut, GeocoderServiceError, Exception):
        pass

    fallback = ZIPCODE_COORDS.get(zipcode)
    if fallback:
        return float(fallback["lat"]), float(fallback["long"])

    raise ValueError(
        "We could not locate this property. Please double-check the address or try a nearby landmark."
    )


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
