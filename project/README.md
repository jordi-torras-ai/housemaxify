# House Maxify Web App

House Maxify is a prototype web experience that showcases a real-estate valuation workflow powered by an existing XGBoost regression model trained on the King County (Seattle area) housing dataset.

## Project Structure

```
project/
├── app.py               # Flask application entry point
├── README.md            # This file
├── requirements.txt     # Python dependencies
├── css/
│   └── styles.css       # Custom theme
├── data/
│   └── leads.csv        # Captured leads (appended at runtime)
├── html/
│   ├── estimate.html    # Property valuation form
│   ├── index.html       # Landing page
│   └── results.html     # Prediction results view
├── js/
│   └── main.js          # Minor UI interactions
└── models/
    └── xgboost_kc_house.pkl  # Pre-trained regression pipeline
```

## Prerequisites

* Python 3.9 or newer
* The cleaned dataset located at `../data/kc_house_data_clean.csv` relative to `app.py`
* The trained model file already placed at `project/models/xgboost_kc_house.pkl`

## Setup

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask development server:

   ```bash
   python app.py
   ```

   By default the app serves on [http://localhost:5000](http://localhost:5000).

## Usage

1. Visit the landing page and click **Get Free Price Estimate**.
2. Complete the valuation form with contact details, property attributes, and a street address (used to geocode latitude/longitude automatically).
3. Submit to:
   * Persist your contact info to `data/leads.csv`.
   * Generate a price estimate and ±7% range using the XGBoost model.
   * View three similar properties surfaced from the historical dataset for additional context.

All lead data are appended to `data/leads.csv` with a UTC timestamp. Remove or archive the file as needed between runs.

## Notes

* The model pipeline relies on pandas, scikit-learn, XGBoost, and geopy; ensure your environment has compatible versions (see `requirements.txt`).
* Address lookup uses OpenStreetMap’s Nominatim service when online and falls back to zipcode averages from the dataset if the geocoder is unavailable.
* If you move the project directory, update `DATA_PATH` inside `app.py` so it can still locate `kc_house_data_clean.csv`.
* The Flask server runs in debug mode by default. Disable debug before deploying to production.
