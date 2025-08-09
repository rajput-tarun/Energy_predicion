from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class DataConfig:
    data_path: str
    target_column: str = "Load_Type"
    date_column: str = "date"
    categorical_columns: List[str] = field(default_factory=lambda: [
        "WeekStatus",
        "Day_of_week",
    ])
    # numerical columns as they appear in the dataset
    numerical_columns: List[str] = field(default_factory=lambda: [
        "Usage_kWh",
        "Lagging_Current_Reactive.Power_kVarh",
        "Leading_Current_Reactive_Power_kVarh",
        "CO2(tCO2)",
        "Lagging_Current_Power_Factor",
        "Leading_Current_Power_Factor",
        "NSM",
    ])


@dataclass
class SplitConfig:
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True


@dataclass
class ModelConfig:
    # model names to build; available: ["logreg", "nb", "rf", "svm", "knn", "dt"]
    models: List[str] = field(default_factory=lambda: ["rf", "knn", "svm", "logreg", "nb", "dt"])
    params: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    data: DataConfig
    split: SplitConfig = field(default_factory=SplitConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    save_plots: bool = False
    plots_dir: str = "plots"
