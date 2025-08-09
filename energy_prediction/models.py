from dataclasses import dataclass
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


@dataclass
class ModelBuilder:
    params: Dict[str, Dict[str, Any]]

    def build(self, name: str):
        name = name.lower()
        p = self.params.get(name, {})
        if name == "logreg":
            # multinomial handles multiclass with softmax
            return LogisticRegression(max_iter=1000, multi_class="multinomial", **p)
        if name == "nb":
            return GaussianNB(**p)
        if name == "rf":
            return RandomForestClassifier(random_state=42, **p)
        if name == "svm":
            # enable probability for ROC curves
            return SVC(kernel=p.pop("kernel", "rbf"), probability=True, random_state=42, **p)
        if name == "knn":
            return KNeighborsClassifier(**p)
        if name == "dt":
            return DecisionTreeClassifier(random_state=42, **p)
        raise ValueError(f"Unknown model name: {name}")
