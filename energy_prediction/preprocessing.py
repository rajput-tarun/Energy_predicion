from dataclasses import dataclass
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import DataConfig


@dataclass
class Preprocessor:
    config: DataConfig
    label_encoders: Dict[str, LabelEncoder] = None
    scaler: StandardScaler = None

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        df = df.copy()
        self.label_encoders = {}

        # Encode categorical columns
        for col in self.config.categorical_columns + [self.config.target_column]:
            if col in df.columns:
                le = LabelEncoder()
                df[col + "_encoded"] = le.fit_transform(df[col])
                self.label_encoders[col] = le

        # Determine numerical columns that exist
        numeric_cols = [c for c in self.config.numerical_columns if c in df.columns]

        # Scale numerical columns
        self.scaler = StandardScaler()
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])

        # Build feature set
        feature_cols = numeric_cols + [
            c + "_encoded" for c in self.config.categorical_columns
            if (c + "_encoded") in df.columns
        ] + [c for c in ["hour", "day", "month", "year"] if c in df.columns]

        X = df[feature_cols]
        y = df[self.config.target_column + "_encoded"] if (self.config.target_column + "_encoded") in df.columns else None
        return X, y.to_numpy() if y is not None else None, feature_cols

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Encode cat cols using fitted encoders
        for col, le in (self.label_encoders or {}).items():
            if col in df.columns:
                df[col + "_encoded"] = le.transform(df[col])

        # Scale numerical columns
        numeric_cols = [c for c in self.config.numerical_columns if c in df.columns]
        if self.scaler is not None and numeric_cols:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])

        # Compose features
        feature_cols = numeric_cols + [
            c + "_encoded" for c in self.config.categorical_columns
            if (c + "_encoded") in df.columns
        ] + [c for c in ["hour", "day", "month", "year"] if c in df.columns]
        return df[feature_cols]
