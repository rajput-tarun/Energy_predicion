from dataclasses import dataclass
from typing import Optional
import pandas as pd

from .config import DataConfig


@dataclass
class DataLoader:
    config: DataConfig

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.data_path)
        # Normalize date column to datetime
        if self.config.date_column in df.columns:
            # Try a common format first, then fallback to pandas inference
            try:
                df[self.config.date_column] = pd.to_datetime(
                    df[self.config.date_column], format="%d/%m/%Y %H:%M"
                )
            except Exception:
                df[self.config.date_column] = pd.to_datetime(df[self.config.date_column], errors="coerce")

            # Basic time-based features
            df["hour"] = df[self.config.date_column].dt.hour
            df["day"] = df[self.config.date_column].dt.day
            df["month"] = df[self.config.date_column].dt.month
            df["year"] = df[self.config.date_column].dt.year
        return df
