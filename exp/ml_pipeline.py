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

def get_instruction(vectors: np.ndarray, **_: Dict) -> np.ndarray:
    """默认最后一个维度的第一个元素是 instruction，提取出来作为特征"""
    array = np.asarray(vectors)

    # 提取最后一个维度的第一个元素
    instruction_features = array[..., 0]

    # 如果输入是二维以上，保持其他维度不变
    if instruction_features.ndim == 1:
        return instruction_features.reshape(-1, 1)
    return instruction_features.reshape(instruction_features.shape[0], -1)

def ignore_instruction(vectors: np.ndarray, **_: Dict) -> np.ndarray:
    """忽略最后一个维度的第一个元素作为特征"""
    array = np.asarray(vectors)

    # 忽略最后一个维度的第一个元素
    features = array[..., 1:]

    # 如果输入是二维以上，保持其他维度不变
    if features.ndim == 1:
        return features.reshape(-1, 1)
    return features.reshape(features.shape[0], -1)

def get_l22h25(vectors: np.ndarray, **_: Dict) -> np.ndarray:
    """获取Layer22Head25的特征"""
    array = np.asarray(vectors)

    l22h25_features = array[:, 18, 14, 1]

    # 如果输入是二维以上，保持其他维度不变
    if l22h25_features.ndim == 1:
        return l22h25_features.reshape(-1, 1)
    return l22h25_features.reshape(l22h25_features.shape[0], -1)

EXTRACTOR_REGISTRY: Dict[str, Callable[..., np.ndarray]] = {
    "identity": identity_extractor,
    "flatten": flatten_extractor,
    "get_instruction": get_instruction,
    "ignore_instruction": ignore_instruction,
    "get_l22h25": get_l22h25,
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

from sklearn.metrics import precision_recall_curve

def find_best_threshold_by_f1(estimator, X_val, y_val):
    proba = estimator.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, proba)
    # F1 = 2PR/(P+R)；注意 thresholds 长度= len(precision) - 1
    f1s = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = f1s.argmax()
    return float(thresholds[best_idx]), float(f1s[best_idx]), float(precision[best_idx]), float(recall[best_idx])


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
    else:
        fit_params.setdefault("eval_set", [(splits.X_test, splits.y_test)])

    if algorithm_name.lower() == "lightgbm":
        early_stopping_rounds = training_cfg.get("early_stopping_rounds")
        if early_stopping_rounds is not None:
            fit_params.setdefault("eval_metric", "binary_logloss")
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

# evaluate_model() 里，先取 y_proba
def _predict_with_optional_threshold(estimator, X, threshold=None):
    if threshold is None or not hasattr(estimator, "predict_proba"):
        return estimator.predict(X)
    proba = estimator.predict_proba(X)
    if proba.ndim == 2:
        proba = proba[:, 1]
    return (proba >= threshold).astype(int)


def evaluate_model(
    estimator,
    splits: DatasetSplits,
    evaluation_cfg: Dict,
) -> Dict[str, Dict[str, object]]:
    """Evaluate the estimator on the train and test splits."""

    metrics: Sequence[str] = evaluation_cfg.get("metrics", ["accuracy", "f1"])
    average = evaluation_cfg.get("average", "binary")
    report = evaluation_cfg.get("classification_report", True)

    metric_registry = _build_metrics(average)

    def evaluate_split(X: np.ndarray, y: np.ndarray) -> Dict[str, object]:
        split_results: Dict[str, object] = {}
        # y_pred = estimator.predict(X)
        threshold = evaluation_cfg.get("decision_threshold")  # e.g. 0.1 ~ 0.3
        y_pred = _predict_with_optional_threshold(estimator, X, threshold)


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
                y_scores = estimator.predict_proba(X)
                if y_scores.ndim == 2 and y_scores.shape[1] > 1:
                    y_scores = y_scores[:, 1]
                split_results[metric_key] = metric_registry[metric_key](y, y_scores)
            else:
                split_results[metric_key] = metric_registry[metric_key](y, y_pred)

        if report:
            split_results["classification_report"] = classification_report(
                y,
                y_pred,
                digits=evaluation_cfg.get("report_digits", 4),
            )

        return split_results

    return {
        "test": evaluate_split(splits.X_test, splits.y_test),
        "train": evaluate_split(splits.X_train, splits.y_train),
    }


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
    print("\n" + "-"*30 + "\n")
    print(f"# Experiment name: {config.get('exp_name', 'unnamed_experiment')}")
    print("\n" + "-"*30)
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

    if splits.X_val is not None and hasattr(estimator, "predict_proba"):
        thr, f1v, pv, rv = find_best_threshold_by_f1(estimator, splits.X_val, splits.y_val)
        print(f"[VAL] best F1 threshold={thr:.4f} (F1={f1v:.4f}, P={pv:.4f}, R={rv:.4f})")
        evaluation_cfg["decision_threshold"] = thr

    results = evaluate_model(estimator, splits, evaluation_cfg)

    maybe_save_model(estimator, output_cfg, base_path)

    return {
        "model": estimator,
        "results": results,
        "splits": splits,
    }


def format_results(results: Dict[str, Dict[str, object]]) -> str:
    lines = []

    for split in ("test", "train"):
        if split not in results:
            continue

        lines.append(f"{split.capitalize()} metrics:")
        split_results = results[split]
        for key, value in split_results.items():
            if key == "classification_report":
                lines.append("Classification report:\n" + str(value))
            else:
                lines.append(f"{key}: {value:.4f}")
        lines.append("")

    if lines and lines[-1] == "":
        lines.pop()

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
    '''
    python exp/ml_pipeline.py  --config /home/yangchunhao/csc/exp/config/p2p_w_ins.json >> logs/ml_pipeline.log 2>&1
    python exp/ml_pipeline.py  --config /home/yangchunhao/csc/exp/config/p2p_3.json >> logs/ml_pipeline.log 2>&1
    python exp/ml_pipeline.py  --config /home/yangchunhao/csc/exp/config/p2p_3_w_ins.json >> logs/ml_pipeline.log 2>&1
    python exp/ml_pipeline.py  --config /home/yangchunhao/csc/exp/config/p2p_4.json >> logs/ml_pipeline.log 2>&1
    python exp/ml_pipeline.py  --config /home/yangchunhao/csc/exp/config/p2p_4_w_ins.json >> logs/ml_pipeline.log 2>&1
    python exp/ml_pipeline.py  --config /home/yangchunhao/csc/exp/config/p2p_w_ins_l22h25.json >> logs/ml_pipeline.log 2>&1
    python exp/ml_pipeline.py  --config /home/yangchunhao/csc/exp/config/p2p_full.json >> logs/ml_pipeline.log 2>&1
    '''
    main()
