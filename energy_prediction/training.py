from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from .config import SplitConfig


@dataclass
class Trainer:
    split_config: SplitConfig

    def split(self, X, y) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        stratify = y if self.split_config.stratify else None
        return train_test_split(
            X,
            y,
            test_size=self.split_config.test_size,
            random_state=self.split_config.random_state,
            stratify=stratify,
        )

    def fit(self, model, X_train, y_train):
        model.fit(X_train, y_train)
        return model

    def evaluate_basic(self, model, X_test, y_test) -> Dict[str, Any]:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        return {
            "accuracy": float(acc),
            "classification_report": report,
            "y_pred": y_pred,
        }
