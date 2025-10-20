# House Maxify

House Maxify is an end-to-end machine learning project that predicts single-family home prices in King County, Washington. It includes reproducible data preparation scripts, linear and gradient-boosting regression pipelines, and a Flask web experience that serves price estimates and captures buyer/seller leads.

## Repository Layout

```
.
├── data/                  # Downloaded King County housing datasets (git-ignored)
├── models/                # Trained model artifacts + evaluation metrics
├── project/               # Flask web app ("House Maxify" marketing site)
│   ├── app.py             # Web server entry point
│   ├── requirements.txt   # Runtime dependencies (covers training scripts too)
│   └── ...                # HTML, CSS, JS, and lead capture CSV
├── src/                   # Offline data preparation & training scripts
└── txt/                   # Ideation prompts / project notes
```

## Quick Start

> These steps assume macOS/Linux with Python 3.9 or newer available. Use `python3` instead of `python` if required by your shell.

1. **Create and activate a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install shared dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r project/requirements.txt
   ```

3. **Configure Kaggle credentials**
   - Log into Kaggle and create a new API token from *My Account → API → Create New API Token*.
   - Save the downloaded `kaggle.json` to `~/.kaggle/kaggle.json` and restrict its permissions:
     ```bash
     chmod 600 ~/.kaggle/kaggle.json
     ```

4. **Download the dataset**
   ```bash
   python src/download_data.py
   ```
   Use `--force` to refresh an existing copy or `--output-dir` to change the target folder.

5. **Validate and clean the dataset (optional but recommended)**
   ```bash
   # Inspect common data issues and leakage columns
   python src/data_explorer.py

   # Engineer features & drop leakage columns -> data/kc_house_data_clean.csv
   python src/data_cleaning.py
   ```

6. **Train the regression models**
   ```bash
   python src/train_linear_regression.py
   python src/train_xgboost.py
   ```
   Artifacts and metric JSON files are written to `models/`.

7. **Launch the House Maxify web app**
   ```bash
   cd project
   python app.py
   ```
   Visit `http://localhost:5000` and complete the valuation form to trigger a prediction and append a new row to `project/data/leads.csv`.

## Data Workflow

| Stage | Script | Key Outputs |
| ----- | ------ | ----------- |
| Acquisition | `src/download_data.py` | Downloads the raw Kaggle CSV into `data/` (requires Kaggle API credentials). |
| Exploration & validation | `src/data_explorer.py` | Prints checks for missing target values, duplicates, non-positive prices, and leakage columns before training. |
| Feature engineering | `src/data_cleaning.py` | Produces `data/kc_house_data_clean.csv` with enriched temporal features. |
| Modeling | `src/train_linear_regression.py`, `src/train_xgboost.py` | Serialized pipelines (`.pkl`) and evaluation metrics (`.json`) saved in `models/`. |

### Engineered Features

`src/data_cleaning.py` augments the Kaggle King County dataset with:

- `sale_date`: parsed timestamp derived from the raw `date` column.
- `sale_year`, `sale_month`, `sale_dow`: calendar breakdown for temporal patterns.
- `house_age_at_sale`: difference between sale year and `yr_built`.
- `was_renovated`: binary flag based on `yr_renovated`.
- `years_since_renov`: elapsed years since the most recent renovation.

Leakage-driven columns (`id`, `date`) are dropped, rows without a valid `sale_date` are removed, and the cleaned CSV is reused by every training script and the web app.

### Train/Test Strategy

Both training scripts enforce a time-based split anchored at `2015-01-01` to reduce leakage from future sales. The target column is `price`; all remaining features (other than `sale_date`) feed the model pipelines. Zip codes are treated categorically and go through one-hot encoding; numeric columns are median-imputed and standardized (linear model) or median-imputed (XGBoost).

## Models & Metrics

| Model | R² | MAE (USD) | RMSE (USD) | Train Time (s) | Artifact |
| ----- | -- | --------- | ---------- | -------------- | -------- |
| Linear Regression | 0.791 | 101,214 | 166,030 | 0.136 | `models/linreg_kc_house.pkl` |
| XGBoost Regressor | 0.879 | 71,812 | 126,037 | 4.397 | `models/xgboost_kc_house.pkl` |

The JSON files in `models/` mirror these scores and include the recorded training duration. Swap in alternate regression algorithms by modifying the pipeline constructors in `src/train_linear_regression.py` or `src/train_xgboost.py`.

## Web Application

The `project/` directory packages a single-file Flask app backed by the XGBoost pipeline:

- **User journey** – Landing page → valuation form → results page with predicted range and three comparable properties.
- **Lead capture** – Every submission persists contact info, intent (buy/sell), and move timeline to `project/data/leads.csv` with a UTC timestamp.
- **Geocoding** – Uses OpenStreetMap’s Nominatim service (via `geopy`) to translate the provided address into latitude/longitude. If Nominatim is unavailable, the app falls back to zipcode-level averages computed from the cleaned dataset.
- **Similar listings** – Top-3 nearest neighbors are selected within the same zipcode (or the broader dataset when no local matches exist) with weighted distance on bedrooms, bathrooms, living area, and lot size.
- **Presentation layer** – Responsive HTML/CSS/JS assets in `project/html`, `project/css`, and `project/js`.

To deploy the app elsewhere, ensure the cleaned dataset remains accessible relative to `project/app.py` (default: `../data/kc_house_data_clean.csv`) and copy the XGBoost model into `project/models/`.

## Datasets & Artifacts

- `data/kc_house_data.csv` — Raw King County home sales dataset downloaded via `src/download_data.py` (ignored by git).
- `data/kc_house_data_clean.csv` — Feature-engineered dataset generated by `src/data_cleaning.py` (ignored by git).
- `models/*.pkl` — Serialized scikit-learn pipelines (Linear Regression and XGBoost).
- `models/*_metrics.json` — Evaluation reports (R², MAE, RMSE, training time).
- `project/data/leads.csv` — Lead log automatically created/appended by the web app.

## Development Tips & Troubleshooting

- Always rerun `src/data_cleaning.py` if you refresh the raw dataset; this keeps engineered features and the web app in sync.
- The geocoder requires outbound HTTP access. Offline environments will rely on zipcode centroids instead.
- When extending the model, keep `MODEL_FEATURES` in `project/app.py` synchronized with the pipeline output to avoid missing inputs at inference time.
- Enable Flask debug logging (default) for rapid development; disable it before deploying to production infrastructure.

## Next Steps & Ideas

1. Add evaluation notebooks or dashboards that visualize feature importance and error distribution.
2. Introduce cross-validation and hyperparameter tuning (e.g., Optuna) to further boost accuracy.
3. Containerize the project with Docker to standardize local development and deployment.
4. Integrate authentication or rate limiting around the lead form before exposing the app publicly.

---

Questions or contributions? Open an issue or submit a pull request – feedback is welcome!
