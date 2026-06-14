"""
Load Home Credit CSV files into PostgreSQL.

Workflow:
  1. make download   — pulls CSVs from Kaggle into data/raw/
  2. make ingest     — creates schema + bulk-loads all CSVs

DATABASE_URL must be set in the environment (copy .env.example → .env).
"""
from __future__ import annotations

import os
import zipfile
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from credit_risk.data.schema import metadata

RAW_DATA_DIR = Path(__file__).parents[4] / "data" / "raw"

# Map CSV filename → (table_name, load_mode)
# "replace" for the first file that owns the table; "append" for subsequent ones.
_CSV_MAP: list[tuple[str, str, str]] = [
    ("application_train.csv", "applications", "replace"),
    ("application_test.csv", "applications", "append"),
    ("bureau.csv", "bureau", "replace"),
    ("bureau_balance.csv", "bureau_balance", "replace"),
    ("previous_application.csv", "previous_applications", "replace"),
    ("POS_CASH_balance.csv", "pos_cash_balance", "replace"),
    ("installments_payments.csv", "installments_payments", "replace"),
    ("credit_card_balance.csv", "credit_card_balance", "replace"),
]


def get_engine() -> Engine:
    url = os.environ["DATABASE_URL"]
    return create_engine(url, echo=False)


def create_schema(engine: Engine) -> None:
    metadata.create_all(engine)


def download_from_kaggle(dest_dir: Path = RAW_DATA_DIR) -> None:
    """Download and unzip all Home Credit files from Kaggle."""
    import kaggle  # type: ignore[import-untyped]

    dest_dir.mkdir(parents=True, exist_ok=True)
    kaggle.api.authenticate()
    kaggle.api.competition_download_files(
        "home-credit-default-risk",
        path=str(dest_dir),
        quiet=False,
    )
    for zf in dest_dir.glob("*.zip"):
        with zipfile.ZipFile(zf) as z:
            z.extractall(dest_dir)
        zf.unlink()


def _load_csv(engine: Engine, csv_path: Path, table: str, if_exists: str) -> int:
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.lower()
    df.to_sql(table, engine, if_exists=if_exists, index=False, method="multi", chunksize=5_000)
    return len(df)


def ingest_all(data_dir: Path = RAW_DATA_DIR) -> None:
    engine = get_engine()
    create_schema(engine)

    for filename, table, mode in _CSV_MAP:
        path = data_dir / filename
        if not path.exists():
            print(f"  skip  {filename} (not found)")
            continue
        n = _load_csv(engine, path, table, mode)
        print(f"  load  {filename} → {table}  ({n:,} rows)")


if __name__ == "__main__":
    ingest_all()
