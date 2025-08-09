import argparse
import json
from energy_prediction.config import PipelineConfig, DataConfig, ModelConfig
from energy_prediction.pipeline import Pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Energy Load Prediction Pipeline (OOP)")
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to the CSV dataset (e.g., Steel_Industry_Energy_Consumption_Prediction_14.csv)",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=["rf", "knn", "svm", "logreg", "nb", "dt"],
        help="Models to train/evaluate: rf, knn, svm, logreg, nb, dt",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save confusion matrix and ROC plots to disk",
    )
    parser.add_argument(
        "--plots-dir",
        default="plots",
        help="Directory to save plots if --save-plots is enabled",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = PipelineConfig(
        data=DataConfig(data_path=args.data_path),
        models=ModelConfig(models=args.models),
        save_plots=args.save_plots,
        plots_dir=args.plots_dir,
    )

    pipeline = Pipeline(cfg)
    results = pipeline.run()

    # Pretty-print a concise summary to stdout
    summary = {}
    for name, res in results["models"].items():
        summary[name] = {
            "accuracy": round(res.get("accuracy", 0.0), 4),
            "auc": (round(res.get("auc", 0.0), 4) if res.get("auc") is not None else None),
        }
    print(json.dumps({"models": summary}, indent=2))


if __name__ == "__main__":
    main()
