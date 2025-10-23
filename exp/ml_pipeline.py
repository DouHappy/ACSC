"""Modular machine learning training/testing pipeline.

This module provides a lightweight yet flexible experiment framework that
loads experiment settings from a JSON configuration file.  The configuration
controls dataset loading, feature extraction, model construction, training,
and evaluation which enables rapid prototyping with different algorithms.

Key features:

* Config driven – modify JSON files instead of touching code when iterating.
* Vector extractor hook – customise how raw samples are converted to vectors.
* Algorithm registry – easily switch between scikit-learn/LightGBM models.
* Reusable evaluation utilities.

Example usage::

    python -m exp.ml_pipeline --config exp/config/ml_config.json

The example configuration demonstrates how to train LightGBM on two binary
class datasets stored as ``.npy`` files.
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


# ---------------------------------------------------------------------------
# Configuration data structures
# ---------------------------------------------------------------------------


@dataclass
class DatasetSplits:
    """Convenience container for dataset splits."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    X_val: Optional[np.ndarray] = None
    y_val: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Vector extractors
# ---------------------------------------------------------------------------


def identity_extractor(vectors: np.ndarray, **_: Dict) -> np.ndarray:
    """Return the input vectors unchanged."""

    return np.asarray(vectors)


def flatten_extractor(vectors: np.ndarray, **_: Dict) -> np.ndarray:
    """Flatten samples to ``(n_samples, -1)`` while preserving batch size."""

    array = np.asarray(vectors)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array.reshape(array.shape[0], -1)


EXTRACTOR_REGISTRY: Dict[str, Callable[..., np.ndarray]] = {
    "identity": identity_extractor,
    "flatten": flatten_extractor,
}


def build_vector_extractor(config: Optional[Dict]) -> Callable[[np.ndarray], np.ndarray]:
    """Construct a vector extractor from configuration."""

    if config is None:
        config = {}

    name = config.get("name", "identity").lower()
    params = config.get("params", {})

    if name not in EXTRACTOR_REGISTRY:
        raise ValueError(f"Unknown vector extractor: {name}")

    extractor = EXTRACTOR_REGISTRY[name]
    return partial(extractor, **params)


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------


class AlgorithmFactory:
    """Factory for constructing estimators via a registry."""

    def __init__(self) -> None:
        self._registry: Dict[str, Callable[[Dict], object]] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register("gaussian_nb", lambda params: GaussianNB(**params))
        self.register(
            "logistic_regression",
            lambda params: LogisticRegression(max_iter=1000, **params),
        )
        self.register("lightgbm", lambda params: lgb.LGBMClassifier(**params))

    def register(self, name: str, builder: Callable[[Dict], object]) -> None:
        self._registry[name.lower()] = builder

    def create(self, name: str, params: Optional[Dict] = None):
        if params is None:
            params = {}

        key = name.lower()
        if key not in self._registry:
            available = ", ".join(sorted(self._registry))
            raise ValueError(f"Unknown algorithm '{name}'. Available: {available}")

        return self._registry[key](params)


# ---------------------------------------------------------------------------
# Dataset loading utilities
# ---------------------------------------------------------------------------


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_sources(
    sources: Sequence[Dict], base_path: Path
) -> Tuple[np.ndarray, np.ndarray]:
    features: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    for source in sources:
        path = source.get("path")
        label = source.get("label")

        if path is None or label is None:
            raise ValueError("Each dataset source must define 'path' and 'label'")

        array_path = Path(path)
        if not array_path.is_absolute():
            array_path = base_path / array_path

        data = np.load(array_path)
        features.append(data)
        labels.append(np.full(shape=(data.shape[0],), fill_value=label))

    X = np.concatenate(features, axis=0)
    y = np.concatenate(labels, axis=0)
    return X, y


def load_dataset(
    dataset_cfg: Dict, base_path: Path
) -> Tuple[np.ndarray, np.ndarray] | Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load datasets specified in the configuration.

    Preferred usage is to provide explicit dataset splits under
    ``dataset.splits`` where each split (e.g. ``train``/``test``) contains a
    list of ``path``/``label`` pairs.  When the legacy ``dataset.sources``
    format is detected a warning is raised and the combined dataset is
    returned so that default splitting can be applied downstream.
    """

    if "splits" in dataset_cfg:
        splits_cfg = dataset_cfg["splits"]
        if not isinstance(splits_cfg, dict) or not splits_cfg:
            raise ValueError("'dataset.splits' must be a non-empty mapping")

        explicit_splits: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for split_name, sources in splits_cfg.items():
            if not isinstance(sources, SequenceABC) or not sources:
                raise ValueError(
                    f"Split '{split_name}' must provide a non-empty list of sources"
                )
            explicit_splits[split_name.lower()] = _load_sources(sources, base_path)

        if "train" not in explicit_splits or "test" not in explicit_splits:
            raise KeyError("Explicit dataset splits must include both 'train' and 'test'")

        return explicit_splits

    if "sources" not in dataset_cfg:
        raise KeyError(
            "Either 'dataset.splits' or the legacy 'dataset.sources' must be provided"
        )

    warnings.warn(
        "No explicit dataset splits defined. Please configure 'dataset.splits' "
        "with dedicated train/test entries to control data partitioning. "
        "Falling back to automatic train/test splitting.",
        UserWarning,
    )
    return _load_sources(dataset_cfg["sources"], base_path)


def create_dataset_splits(
    X: np.ndarray,
    y: np.ndarray,
    training_cfg: Dict,
) -> DatasetSplits:
    """Create train/validation/test splits according to configuration."""

    test_size = training_cfg.get("test_size", 0.2)
    stratify = training_cfg.get("stratify", True)
    random_state = training_cfg.get("random_state", 42)
    validation_size = training_cfg.get("validation_size")

    stratify_targets = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_targets,
    )

    X_val = y_val = None

    if validation_size:
        # Validation size is relative to the original dataset. Adjust split.
        adjusted_size = validation_size / (1 - test_size)
        stratify_inner = y_train if stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=adjusted_size,
            random_state=random_state,
            stratify=stratify_inner,
        )

    return DatasetSplits(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X_val=X_val,
        y_val=y_val,
    )


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def train_model(
    estimator,
    algorithm_name: str,
    splits: DatasetSplits,
    training_cfg: Dict,
):
    """Fit the estimator according to configuration options."""

    fit_params = dict(training_cfg.get("fit_params", {}))

    if splits.X_val is not None and training_cfg.get("use_validation_for_fit", True):
        # Provide evaluation set for algorithms that support it (e.g. LightGBM).
        fit_params.setdefault("eval_set", [(splits.X_val, splits.y_val)])
        fit_params.setdefault("eval_metric", "binary_logloss")

    if algorithm_name.lower() == "lightgbm":
        early_stopping_rounds = training_cfg.get("early_stopping_rounds")
        if early_stopping_rounds is not None:
            callbacks = list(fit_params.get("callbacks", []))
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=early_stopping_rounds,
                    verbose=training_cfg.get("early_stopping_verbose", True),
                )
            )
            fit_params["callbacks"] = callbacks

    estimator.fit(splits.X_train, splits.y_train, **fit_params)
    return estimator


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------


MetricFn = Callable[[np.ndarray, np.ndarray], float]


def _build_metrics(average: str) -> Dict[str, MetricFn]:
    return {
        "accuracy": accuracy_score,
        "precision": partial(precision_score, average=average),
        "recall": partial(recall_score, average=average),
        "f1": partial(f1_score, average=average),
        "log_loss": log_loss,
        "roc_auc": roc_auc_score,
    }


PROBABILITY_METRICS = {"roc_auc", "log_loss"}


def evaluate_model(
    estimator,
    splits: DatasetSplits,
    evaluation_cfg: Dict,
) -> Dict[str, object]:
    """Evaluate the estimator on the test split."""

    metrics: Sequence[str] = evaluation_cfg.get("metrics", ["accuracy", "f1"])
    average = evaluation_cfg.get("average", "binary")
    report = evaluation_cfg.get("classification_report", True)

    metric_registry = _build_metrics(average)

    y_pred = estimator.predict(splits.X_test)
    results: Dict[str, object] = {}

    for metric_name in metrics:
        metric_key = metric_name.lower()
        if metric_key not in metric_registry:
            available = ", ".join(sorted(metric_registry))
            raise ValueError(
                f"Unknown metric '{metric_name}'. Available metrics: {available}"
            )

        if metric_key in PROBABILITY_METRICS:
            if not hasattr(estimator, "predict_proba"):
                raise AttributeError(
                    f"Metric '{metric_name}' requires 'predict_proba' support"
                )
            y_scores = estimator.predict_proba(splits.X_test)
            if y_scores.ndim == 2 and y_scores.shape[1] > 1:
                y_scores = y_scores[:, 1]
            results[metric_key] = metric_registry[metric_key](splits.y_test, y_scores)
        else:
            results[metric_key] = metric_registry[metric_key](splits.y_test, y_pred)

    if report:
        results["classification_report"] = classification_report(
            splits.y_test,
            y_pred,
            digits=evaluation_cfg.get("report_digits", 4),
        )

    return results


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------


def maybe_save_model(estimator, output_cfg: Optional[Dict], base_path: Path) -> None:
    if not output_cfg:
        return

    path = output_cfg.get("model_path")
    if not path:
        return

    output_path = Path(path)
    if not output_path.is_absolute():
        output_path = base_path / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, output_path)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_experiment(config_path: Path) -> Dict[str, object]:
    config = load_config(config_path)
    base_path = config_path.parent

    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    vector_cfg = config.get("vectorizer")
    training_cfg = config.get("training", {})
    evaluation_cfg = config.get("evaluation", {})
    output_cfg = config.get("output")

    extractor = build_vector_extractor(vector_cfg)

    dataset = load_dataset(dataset_cfg, base_path)

    if isinstance(dataset, dict):
        train_X_raw, train_y = dataset["train"]
        test_X_raw, test_y = dataset["test"]

        X_val = y_val = None
        for candidate_key in ("validation", "val", "dev"):
            if candidate_key in dataset:
                X_val_raw, y_val = dataset[candidate_key]
                X_val = extractor(X_val_raw)
                break

        splits = DatasetSplits(
            X_train=extractor(train_X_raw),
            y_train=train_y,
            X_test=extractor(test_X_raw),
            y_test=test_y,
            X_val=X_val,
            y_val=y_val,
        )
    else:
        X_raw, y = dataset
        X = extractor(X_raw)
        splits = create_dataset_splits(X, y, training_cfg)

    algorithm_name = model_cfg.get("name")
    if not algorithm_name:
        raise KeyError("'model.name' must be specified in the configuration")

    algorithm_params = model_cfg.get("params", {})

    factory = AlgorithmFactory()
    estimator = factory.create(algorithm_name, algorithm_params)

    estimator = train_model(estimator, algorithm_name, splits, training_cfg)

    results = evaluate_model(estimator, splits, evaluation_cfg)

    maybe_save_model(estimator, output_cfg, base_path)

    return {
        "model": estimator,
        "results": results,
        "splits": splits,
    }


def format_results(results: Dict[str, object]) -> str:
    lines = []
    for key, value in results.items():
        if key == "classification_report":
            lines.append("Classification report:\n" + str(value))
        else:
            lines.append(f"{key}: {value:.4f}")
    return "\n".join(lines)


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Config driven ML experiment")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("exp/config/ml_config.json"),
        help="Path to the experiment configuration file",
    )
    return parser.parse_args(args=args)


def main(args: Optional[Sequence[str]] = None) -> None:
    parsed = parse_args(args)
    outcome = run_experiment(parsed.config)
    print(format_results(outcome["results"]))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

