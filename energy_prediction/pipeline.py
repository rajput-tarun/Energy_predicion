from dataclasses import dataclass
from typing import Dict, Any, List

import pandas as pd

from .config import PipelineConfig
from .data_loader import DataLoader
from .preprocessing import Preprocessor
from .models import ModelBuilder
from .training import Trainer
from .evaluation import Evaluator


@dataclass
class Pipeline:
    config: PipelineConfig

    def run(self) -> Dict[str, Any]:
        # 1) Load data
        loader = DataLoader(self.config.data)
        df = loader.load()

        # 2) Preprocess
        pre = Preprocessor(self.config.data)
        X, y, feature_cols = pre.fit_transform(df)

        # 3) Split
        trainer = Trainer(self.config.split)
        X_train, X_test, y_train, y_test = trainer.split(X, y)

        # 4) Build + Train + Evaluate across models
        builder = ModelBuilder(self.config.models.params)
        evaluator = Evaluator(save_plots=self.config.save_plots, plots_dir=self.config.plots_dir)

        results: Dict[str, Any] = {
            "features": feature_cols,
            "models": {},
        }

        for name in self.config.models.models:
            model = builder.build(name)
            model = trainer.fit(model, X_train, y_train)
            basic = trainer.evaluate_basic(model, X_test, y_test)
            evaluator.confusion_matrix_plot(y_test, basic["y_pred"], model_name=name)
            auc = evaluator.roc_auc(model, X_test, y_test, model_name=name)

            results["models"][name] = {
                "accuracy": basic["accuracy"],
                "auc": auc,
                "classification_report": basic["classification_report"],
            }

        return results
