from dataclasses import dataclass
from typing import Dict, Any, Optional
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize


@dataclass
class Evaluator:
    save_plots: bool = False
    plots_dir: str = "plots"

    def _ensure_dir(self):
        if self.save_plots:
            os.makedirs(self.plots_dir, exist_ok=True)

    def confusion_matrix_plot(self, y_true, y_pred, model_name: str):
        self._ensure_dir()
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=['Light Load', 'Medium Load', 'Maximum Load'],
                    yticklabels=['Light Load', 'Medium Load', 'Maximum Load'])
        plt.title(f"Confusion Matrix: {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        if self.save_plots:
            path = os.path.join(self.plots_dir, f"confusion_matrix_{model_name}.png")
            plt.savefig(path, bbox_inches="tight")
        plt.close()

    def roc_auc(self, model, X_test, y_test, model_name: str) -> Optional[float]:
        # Attempt to compute multi-class ROC AUC if possible
        if not hasattr(model, "predict_proba"):
            return None
        try:
            y_score = model.predict_proba(X_test)
            classes = np.unique(y_test)
            y_test_bin = label_binarize(y_test, classes=classes)
            auc = roc_auc_score(y_test_bin, y_score, multi_class="ovr")
            # Optional plot (one-vs-rest for each class)
            self._ensure_dir()
            plt.figure(figsize=(6, 4))
            for i, cls in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                plt.plot(fpr, tpr, label=f"Class {cls}")
            plt.plot([0, 1], [0, 1], 'k--')
            plt.title(f"ROC Curves: {model_name} (AUC={auc:.3f})")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc="lower right")
            if self.save_plots:
                path = os.path.join(self.plots_dir, f"roc_{model_name}.png")
                plt.savefig(path, bbox_inches="tight")
            plt.close()
            return float(auc)
        except Exception:
            return None
