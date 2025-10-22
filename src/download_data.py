"""Download the King County housing dataset from Kaggle.

This script pulls the public dataset maintained at
https://www.kaggle.com/datasets/harlfoxem/housesalesprediction and stores the
raw CSV inside the repository's ``data/`` directory. It wraps Kaggle's Python
API so we can authenticate with the usual ``~/.kaggle/kaggle.json`` credentials
and avoid committing large data files to version control.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

DATASET_REF = "harlfoxem/housesalesprediction"
SOURCE_FILE = "kc_house_data.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the King County house sales dataset from Kaggle."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Directory where the CSV should be saved (default: ./data)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if the CSV already exists.",
    )
    return parser.parse_args()


def authenticate() -> KaggleApi:
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:
        print(
            "❌ Kaggle authentication failed. "
            "Ensure you have kaggle.json configured under ~/.kaggle/",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    return api


def download_csv(api: KaggleApi, output_dir: Path, force: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / SOURCE_FILE
    zip_path = output_dir / f"{SOURCE_FILE}.zip"

    if csv_path.exists() and not force:
        print(f"✅ Dataset already present at {csv_path}. Use --force to refresh.")
        return csv_path

    print(f"⬇️  Downloading {SOURCE_FILE} from {DATASET_REF} ...")
    api.dataset_download_file(
        DATASET_REF,
        SOURCE_FILE,
        path=str(output_dir),
        force=force,
        quiet=False,
    )

    # Kaggle sometimes sends archives without a .zip suffix (named like the original file).
    # Detect that scenario, rename temporarily, and extract the CSV.
    if zipfile.is_zipfile(csv_path):
        temp_zip = csv_path.with_suffix(csv_path.suffix + ".zip")
        csv_path.rename(temp_zip)
        with zipfile.ZipFile(temp_zip, "r") as archive:
            archive.extractall(output_dir)
        temp_zip.unlink(missing_ok=True)

    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(output_dir)
        zip_path.unlink(missing_ok=True)

    if csv_path.exists():
        print(f"✅ Saved CSV to {csv_path}")
        return csv_path

    print(
        "❌ Download finished but the CSV was not found. "
        "Double-check Kaggle API output.",
        file=sys.stderr,
    )
    raise SystemExit(1)


def main() -> None:
    args = parse_args()
    api = authenticate()
    download_csv(api, args.output_dir, args.force)


if __name__ == "__main__":
    main()
