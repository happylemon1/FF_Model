"""WR rookie projection training, evaluation, and inference pipeline.

The pipeline is intentionally tabular-first. The historical WR sample is small,
so tree ensembles are expected to be strong baselines while neural nets are
included as measured contenders rather than assumed winners.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "models" / "combined_df.csv"
DEFAULT_ROOKIE_PATH = PROJECT_ROOT / "models" / "merged_df.csv"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "models" / "artifacts"
DEFAULT_PREDICTION_PATH = PROJECT_ROOT / "models" / "rookie_predictions.csv"

TARGET_COLUMNS = ["nfl_YPG", "nfl_RPG", "nfl_TDPG"]
PRIMARY_TARGET = "nfl_YPG"
MIN_NFL_GAMES = 4

BASE_FEATURES = [
    "YPG",
    "Yards_percentage",
    "draft_pick",
    "RPG",
    "TDPG",
    "second_last_TDPG",
    "second_last_RPG",
    "second_last_YPG",
    "conference_SEC",
    "conference_ACC",
    "conference_Big Ten",
    "conference_Big 12",
    "conference_Other",
    "age",
]

DERIVED_FEATURES = [
    "YPG_diff",
    "RPG_diff",
    "TDPG_diff",
    "ypc",
    "draft_pick_log",
    "draft_pick_inverse",
    "production_score",
    "trend_score",
    "td_rate_per_rec",
    "volume_score",
]

FEATURE_COLUMNS = BASE_FEATURES + DERIVED_FEATURES
LEAKAGE_PREFIXES = ("nfl_", "predicted_")


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    factory: Callable[[], Any]


def require_ml_dependencies() -> dict[str, Any]:
    """Import ML dependencies lazily and raise a helpful error if missing."""

    missing = []
    modules: dict[str, Any] = {}
    for name in ["joblib", "sklearn"]:
        try:
            modules[name] = importlib.import_module(name)
        except ImportError:
            missing.append(name)
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"Missing required ML dependencies: {joined}. "
            "Install them with `python -m pip install -r requirements.txt`."
        )
    return modules


def optional_import(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def detect_gpu_status() -> dict[str, Any]:
    status: dict[str, Any] = {
        "nvidia_smi_available": False,
        "nvidia_gpu": None,
        "torch_available": False,
        "torch_cuda_available": False,
        "torch_device_count": 0,
        "torch_version": None,
        "xgboost_available": False,
        "xgboost_version": None,
    }

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            status["nvidia_smi_available"] = True
            status["nvidia_gpu"] = result.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError):
        pass

    torch = optional_import("torch")
    if torch is not None:
        status["torch_available"] = True
        status["torch_version"] = getattr(torch, "__version__", None)
        try:
            status["torch_cuda_available"] = bool(torch.cuda.is_available())
            status["torch_device_count"] = int(torch.cuda.device_count())
        except Exception:
            pass

    xgb = optional_import("xgboost")
    if xgb is not None:
        status["xgboost_available"] = True
        status["xgboost_version"] = getattr(xgb, "__version__", None)

    return status


def read_csv_checked(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV is empty: {path}")
    return df


def coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_inputs = {
        "YPG",
        "RPG",
        "TDPG",
        "second_last_YPG",
        "second_last_RPG",
        "second_last_TDPG",
        "draft_pick",
        "recs",
        "recYDS",
        "recTDS",
        "age",
        "Yards_percentage",
        "TEAM_YARDS",
    }
    df = coerce_numeric(df, numeric_inputs)

    if "Yards_percentage" not in df.columns and {"recYDS", "TEAM_YARDS"}.issubset(df.columns):
        df["Yards_percentage"] = df["recYDS"] / df["TEAM_YARDS"]
    if "YPG_diff" not in df.columns:
        df["YPG_diff"] = df.get("YPG", np.nan) - df.get("second_last_YPG", np.nan)
    if "RPG_diff" not in df.columns:
        df["RPG_diff"] = df.get("RPG", np.nan) - df.get("second_last_RPG", np.nan)
    if "TDPG_diff" not in df.columns:
        df["TDPG_diff"] = df.get("TDPG", np.nan) - df.get("second_last_TDPG", np.nan)

    safe_recs = df.get("recs", pd.Series(np.nan, index=df.index)).replace(0, np.nan)
    df["ypc"] = df.get("recYDS", np.nan) / safe_recs
    df["draft_pick_log"] = np.log1p(df.get("draft_pick", np.nan))
    df["draft_pick_inverse"] = 1.0 / np.sqrt(df.get("draft_pick", np.nan).clip(lower=1))
    df["production_score"] = (
        df.get("YPG", np.nan).fillna(0.0)
        + 6.0 * df.get("RPG", np.nan).fillna(0.0)
        + 14.0 * df.get("TDPG", np.nan).fillna(0.0)
    )
    df["trend_score"] = (
        df.get("YPG_diff", np.nan).fillna(0.0)
        + 6.0 * df.get("RPG_diff", np.nan).fillna(0.0)
        + 14.0 * df.get("TDPG_diff", np.nan).fillna(0.0)
    )
    df["td_rate_per_rec"] = df.get("recTDS", np.nan) / safe_recs
    df["volume_score"] = (
        df.get("Yards_percentage", np.nan).fillna(0.0)
        * df.get("RPG", np.nan).fillna(0.0)
        * 100.0
    )

    for conference in [
        "conference_SEC",
        "conference_ACC",
        "conference_Big Ten",
        "conference_Big 12",
        "conference_Other",
    ]:
        if conference not in df.columns:
            df[conference] = 0.0

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def validate_feature_columns(feature_columns: list[str]) -> None:
    leaked = [
        column
        for column in feature_columns
        if column.startswith(LEAKAGE_PREFIXES) or column in TARGET_COLUMNS
    ]
    if leaked:
        raise ValueError(f"Target leakage detected in feature list: {leaked}")


def prepare_training_data(
    train_path: Path = DEFAULT_TRAIN_PATH,
    min_nfl_games: int = MIN_NFL_GAMES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = read_csv_checked(train_path)
    df = add_engineered_features(df)
    df = coerce_numeric(df, FEATURE_COLUMNS + TARGET_COLUMNS + ["nfl_games", "season"])
    validate_feature_columns(FEATURE_COLUMNS)

    required_columns = FEATURE_COLUMNS + TARGET_COLUMNS + ["nfl_games", "season", "name"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Training data is missing columns: {missing_columns}")

    df = df[df["nfl_games"] >= min_nfl_games].copy()
    df = df.dropna(subset=FEATURE_COLUMNS + TARGET_COLUMNS + ["season"])
    df = df.reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid training rows remain after cleaning.")

    x = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMNS].copy()
    meta = df[["name", "season", "nfl_games"]].copy()
    return x, y, meta


def prepare_rookie_data(
    rookie_path: Path = DEFAULT_ROOKIE_PATH,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_columns = feature_columns or FEATURE_COLUMNS
    df = read_csv_checked(rookie_path)
    df = add_engineered_features(df)
    df = coerce_numeric(df, feature_columns)
    validate_feature_columns(feature_columns)

    missing_columns = [column for column in feature_columns + ["name"] if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Rookie data is missing columns: {missing_columns}")

    usable = df.dropna(subset=feature_columns).copy()
    if usable.empty:
        raise ValueError("No valid rookie rows remain after feature cleaning.")
    return usable[feature_columns].copy(), usable.reset_index(drop=True)


def make_time_splits(meta: pd.DataFrame, max_folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    seasons = sorted(pd.to_numeric(meta["season"], errors="coerce").dropna().unique())
    if len(seasons) < 4:
        return make_kfold_splits(len(meta), max_folds)

    validation_seasons = seasons[-max_folds:]
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    season_values = pd.to_numeric(meta["season"], errors="coerce").to_numpy()
    for season in validation_seasons:
        train_idx = np.where(season_values < season)[0]
        valid_idx = np.where(season_values == season)[0]
        if len(train_idx) >= 50 and len(valid_idx) >= 5:
            splits.append((train_idx, valid_idx))
    if not splits:
        return make_kfold_splits(len(meta), max_folds)
    return splits


def make_kfold_splits(n_rows: int, max_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    require_ml_dependencies()
    from sklearn.model_selection import KFold

    n_splits = max(2, min(max_folds, n_rows))
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    indices = np.arange(n_rows)
    return [(train, valid) for train, valid in kfold.split(indices)]


def rankdata(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=float)


def spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    true_rank = rankdata(y_true)
    pred_rank = rankdata(y_pred)
    if np.nanstd(true_rank) == 0 or np.nanstd(pred_rank) == 0:
        return float("nan")
    corr = np.corrcoef(true_rank, pred_rank)[0, 1]
    return float(corr) if np.isfinite(corr) else float("nan")


def top_k_overlap(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    if len(y_true) == 0:
        return float("nan")
    k = min(k, len(y_true))
    true_top = set(np.argsort(y_true)[-k:])
    pred_top = set(np.argsort(y_pred)[-k:])
    return len(true_top & pred_top) / k


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    require_ml_dependencies()
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "spearman": spearman_correlation(y_true, y_pred),
        "top10_overlap": top_k_overlap(y_true, y_pred, k=10),
    }


def mean_metric(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    keys = metric_rows[0].keys()
    output: dict[str, float] = {}
    for key in keys:
        values = np.asarray([row[key] for row in metric_rows], dtype=float)
        finite = values[np.isfinite(values)]
        output[key] = float(np.mean(finite)) if len(finite) else float("nan")
    return output


def get_model_builders(
    include_neural: bool,
    use_xgboost: bool,
    prefer_gpu: bool,
) -> dict[str, Callable[..., Any]]:
    require_ml_dependencies()
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def scaled_model(model: Any) -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )

    def tree_model(model: Any) -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", model),
            ]
        )

    builders: dict[str, Callable[..., Any]] = {
        "median": lambda **_: DummyRegressor(strategy="median"),
        "ridge": lambda alpha=10.0, **_: scaled_model(Ridge(alpha=alpha, random_state=42)),
        "elastic_net": lambda alpha=0.05, l1_ratio=0.25, **_: scaled_model(
            ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=20000)
        ),
        "random_forest": lambda n_estimators=350, max_depth=None, min_samples_leaf=2, max_features=0.8, **_: tree_model(
            RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=42,
                n_jobs=-1,
            )
        ),
        "extra_trees": lambda n_estimators=400, max_depth=None, min_samples_leaf=2, max_features=0.9, **_: tree_model(
            ExtraTreesRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=42,
                n_jobs=-1,
            )
        ),
        "gradient_boosting": lambda n_estimators=180, learning_rate=0.04, max_depth=2, subsample=0.85, **_: tree_model(
            GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                random_state=42,
            )
        ),
    }

    xgb = optional_import("xgboost")
    if use_xgboost and xgb is not None:
        device = "cuda" if prefer_gpu else "cpu"

        def xgboost_builder(
            n_estimators: int = 350,
            learning_rate: float = 0.035,
            max_depth: int = 2,
            min_child_weight: float = 2.0,
            subsample: float = 0.85,
            colsample_bytree: float = 0.85,
            reg_lambda: float = 2.0,
            **_: Any,
        ) -> Pipeline:
            model = xgb.XGBRegressor(
                objective="reg:squarederror",
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_lambda=reg_lambda,
                random_state=42,
                tree_method="hist",
                device=device,
                n_jobs=-1,
            )
            return tree_model(model)

        builders[f"xgboost_{device}"] = xgboost_builder

    if include_neural and optional_import("torch") is not None:
        builders["mlp"] = lambda hidden=(128, 64), dropout=0.15, lr=0.002, weight_decay=0.0005, epochs=180, **_: scaled_model(
            TorchRegressor(
                architecture="mlp",
                hidden=hidden,
                dropout=dropout,
                lr=lr,
                weight_decay=weight_decay,
                epochs=epochs,
                batch_size=32,
                patience=25,
                prefer_gpu=prefer_gpu,
                random_state=42,
            )
        )
        builders["residual_mlp"] = lambda hidden=(128, 64), dropout=0.15, lr=0.002, weight_decay=0.0005, epochs=180, **_: scaled_model(
            TorchRegressor(
                architecture="residual_mlp",
                hidden=hidden,
                dropout=dropout,
                lr=lr,
                weight_decay=weight_decay,
                epochs=epochs,
                batch_size=32,
                patience=25,
                prefer_gpu=prefer_gpu,
                random_state=42,
            )
        )
        builders["tabular_transformer"] = lambda d_model=32, dropout=0.1, lr=0.0015, weight_decay=0.001, epochs=160, **_: scaled_model(
            TorchRegressor(
                architecture="tabular_transformer",
                d_model=d_model,
                dropout=dropout,
                lr=lr,
                weight_decay=weight_decay,
                epochs=epochs,
                batch_size=32,
                patience=25,
                prefer_gpu=prefer_gpu,
                random_state=42,
            )
        )

    return builders


def default_candidates(builders: dict[str, Callable[..., Any]]) -> list[Candidate]:
    candidates = [
        Candidate("median", "baseline", builders["median"]),
        Candidate("ridge", "linear", builders["ridge"]),
        Candidate("elastic_net", "linear", builders["elastic_net"]),
        Candidate("random_forest", "bagging_tree", builders["random_forest"]),
        Candidate("extra_trees", "bagging_tree", builders["extra_trees"]),
        Candidate("gradient_boosting", "boosted_tree", builders["gradient_boosting"]),
    ]
    for name in ["xgboost_cuda", "xgboost_cpu", "mlp", "residual_mlp", "tabular_transformer"]:
        if name in builders:
            family = "neural_net" if name in {"mlp", "residual_mlp", "tabular_transformer"} else "boosted_tree"
            candidates.append(Candidate(name, family, builders[name]))
    return candidates


def clone_estimator(estimator: Any) -> Any:
    require_ml_dependencies()
    from sklearn.base import clone

    return clone(estimator)


def evaluate_candidate(
    candidate: Candidate,
    x: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    target: str,
    allow_gpu_retry: bool = True,
) -> tuple[dict[str, Any], np.ndarray]:
    predictions = np.full(len(y), np.nan, dtype=float)
    fold_metrics: list[dict[str, float]] = []
    used_cpu_fallback = False

    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        estimator = candidate.factory()
        try:
            estimator.fit(x.iloc[train_idx], y.iloc[train_idx])
        except Exception as exc:
            if allow_gpu_retry and "xgboost_cuda" in candidate.name:
                cpu_candidate = Candidate(
                    name=candidate.name.replace("cuda", "cpu_fallback"),
                    family=candidate.family,
                    factory=get_model_builders(False, True, False)["xgboost_cpu"],
                )
                estimator = cpu_candidate.factory()
                estimator.fit(x.iloc[train_idx], y.iloc[train_idx])
                used_cpu_fallback = True
            else:
                raise exc
        fold_pred = np.asarray(estimator.predict(x.iloc[valid_idx]), dtype=float)
        predictions[valid_idx] = fold_pred
        fold_metrics.append(regression_metrics(y.iloc[valid_idx].to_numpy(dtype=float), fold_pred))

    metrics = mean_metric(fold_metrics)
    metrics.update(
        {
            "target": target,
            "model": candidate.name,
            "family": candidate.family,
            "folds": len(splits),
            "used_cpu_fallback": used_cpu_fallback,
        }
    )
    return metrics, predictions


def optimize_candidates(
    base_builders: dict[str, Callable[..., Any]],
    x: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    target: str,
    trials: int,
    prefer_gpu: bool,
    include_neural: bool,
) -> list[Candidate]:
    optuna = optional_import("optuna")
    if optuna is None or trials <= 0:
        return []

    tuned: list[Candidate] = []

    def objective_for(name: str, family: str, suggest_fn: Callable[[Any], dict[str, Any]]) -> Candidate | None:
        if name not in base_builders:
            return None

        def objective(trial: Any) -> float:
            params = suggest_fn(trial)
            candidate = Candidate(
                name=f"{name}_trial",
                family=family,
                factory=lambda params=params: base_builders[name](**params),
            )
            metrics, _ = evaluate_candidate(candidate, x, y, splits, target)
            return metrics["mae"]

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=trials, show_progress_bar=False)
        best_params = study.best_params
        return Candidate(
            name=f"{name}_optuna",
            family=family,
            factory=lambda best_params=best_params: base_builders[name](**best_params),
        )

    search_spaces: list[tuple[str, str, Callable[[Any], dict[str, Any]]]] = [
        (
            "random_forest",
            "bagging_tree",
            lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 150, 700),
                "max_depth": trial.suggest_categorical("max_depth", [None, 2, 3, 4, 6, 8]),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
                "max_features": trial.suggest_float("max_features", 0.45, 1.0),
            },
        ),
        (
            "extra_trees",
            "bagging_tree",
            lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 150, 800),
                "max_depth": trial.suggest_categorical("max_depth", [None, 2, 3, 4, 6, 8]),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
                "max_features": trial.suggest_float("max_features", 0.45, 1.0),
            },
        ),
        (
            "gradient_boosting",
            "boosted_tree",
            lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 60, 450),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
                "max_depth": trial.suggest_int("max_depth", 1, 3),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            },
        ),
    ]

    xgb_name = "xgboost_cuda" if prefer_gpu and "xgboost_cuda" in base_builders else "xgboost_cpu"
    if xgb_name in base_builders:
        search_spaces.append(
            (
                xgb_name,
                "boosted_tree",
                lambda trial: {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 700),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
                    "max_depth": trial.suggest_int("max_depth", 1, 4),
                    "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 6.0),
                    "subsample": trial.suggest_float("subsample", 0.55, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
                },
            )
        )

    if include_neural:
        for name in ["mlp", "residual_mlp"]:
            if name in base_builders:
                search_spaces.append(
                    (
                        name,
                        "neural_net",
                        lambda trial: {
                            "hidden": trial.suggest_categorical(
                                "hidden",
                                [(64, 32), (128, 64), (128, 64, 32), (256, 128)],
                            ),
                            "dropout": trial.suggest_float("dropout", 0.05, 0.35),
                            "lr": trial.suggest_float("lr", 0.0005, 0.005, log=True),
                            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 5e-3, log=True),
                            "epochs": trial.suggest_int("epochs", 100, 240),
                        },
                    )
                )

    for name, family, suggest_fn in search_spaces:
        tuned_candidate = objective_for(name, family, suggest_fn)
        if tuned_candidate is not None:
            tuned.append(tuned_candidate)
    return tuned


def sort_leaderboard(leaderboard: pd.DataFrame, target: str) -> pd.DataFrame:
    sort_columns = ["mae", "rmse"]
    ascending = [True, True]
    if target == PRIMARY_TARGET:
        sort_columns = ["spearman", "mae", "rmse"]
        ascending = [False, True, True]
    return leaderboard.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)


def train_pipeline(args: argparse.Namespace) -> None:
    require_ml_dependencies()
    joblib = importlib.import_module("joblib")

    x, y, meta = prepare_training_data(args.train_csv, args.min_nfl_games)
    gpu_status = detect_gpu_status()
    prefer_gpu = bool(args.prefer_gpu and gpu_status.get("torch_cuda_available"))
    xgb_prefer_gpu = bool(args.prefer_gpu and gpu_status.get("nvidia_smi_available"))

    builders = get_model_builders(
        include_neural=args.include_neural,
        use_xgboost=not args.no_xgboost,
        prefer_gpu=xgb_prefer_gpu,
    )
    candidates = default_candidates(builders)
    splits = make_time_splits(meta, max_folds=args.folds)

    all_rows: list[dict[str, Any]] = []
    selected: dict[str, Any] = {}
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    for target in TARGET_COLUMNS:
        target_candidates = list(candidates)
        if args.trials > 0:
            target_candidates.extend(
                optimize_candidates(
                    builders,
                    x,
                    y[target],
                    splits,
                    target,
                    trials=args.trials,
                    prefer_gpu=xgb_prefer_gpu,
                    include_neural=args.include_neural,
                )
            )

        rows: list[dict[str, Any]] = []
        for candidate in target_candidates:
            try:
                metrics, _ = evaluate_candidate(candidate, x, y[target], splits, target)
                rows.append(metrics)
                print(
                    f"{target:8s} {candidate.name:24s} "
                    f"MAE={metrics['mae']:.3f} Spearman={metrics['spearman']:.3f}"
                )
            except Exception as exc:
                rows.append(
                    {
                        "target": target,
                        "model": candidate.name,
                        "family": candidate.family,
                        "folds": len(splits),
                        "mae": float("inf"),
                        "rmse": float("inf"),
                        "r2": float("nan"),
                        "spearman": float("nan"),
                        "top10_overlap": float("nan"),
                        "used_cpu_fallback": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(f"{target:8s} {candidate.name:24s} failed: {exc}")

        leaderboard = sort_leaderboard(pd.DataFrame(rows), target)
        all_rows.extend(leaderboard.to_dict(orient="records"))
        successful = leaderboard[np.isfinite(leaderboard["mae"])]
        if successful.empty:
            raise RuntimeError(f"No successful models for {target}.")

        top_n = successful.head(args.top_models)
        target_dir = args.artifact_dir / target
        target_dir.mkdir(parents=True, exist_ok=True)
        saved_models: list[dict[str, Any]] = []

        for _, row in top_n.iterrows():
            candidate = next(candidate for candidate in target_candidates if candidate.name == row["model"])
            estimator = candidate.factory()
            try:
                estimator.fit(x, y[target])
            except Exception:
                if "xgboost_cuda" in candidate.name:
                    cpu_builder = get_model_builders(False, True, False)["xgboost_cpu"]
                    estimator = cpu_builder()
                    estimator.fit(x, y[target])
                    row["used_cpu_fallback"] = True
                else:
                    raise

            model_path = target_dir / f"{safe_filename(candidate.name)}.joblib"
            joblib.dump(estimator, model_path)
            saved_models.append(
                {
                    "name": candidate.name,
                    "family": candidate.family,
                    "path": str(model_path.relative_to(args.artifact_dir)),
                    "metrics": {key: normalize_json_value(row[key]) for key in leaderboard.columns if key in row},
                }
            )

        selected[target] = {
            "best_model": saved_models[0],
            "top_models": saved_models,
        }

    leaderboard_df = pd.DataFrame(all_rows)
    leaderboard_path = args.artifact_dir / "validation_leaderboard.csv"
    leaderboard_df.to_csv(leaderboard_path, index=False)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train_csv": str(args.train_csv),
        "rookie_csv": str(args.rookie_csv),
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "primary_target": PRIMARY_TARGET,
        "rank_score": "0.75*z(predicted_nfl_YPG)+0.15*z(predicted_nfl_RPG)+0.10*z(predicted_nfl_TDPG)",
        "min_nfl_games": args.min_nfl_games,
        "training_rows": len(x),
        "cv_folds": len(splits),
        "selected": selected,
        "gpu_status": gpu_status,
        "prefer_gpu_requested": bool(args.prefer_gpu),
        "torch_cuda_used_for_neural_candidates": prefer_gpu,
        "xgboost_cuda_requested": xgb_prefer_gpu,
        "package_versions": package_versions(["pandas", "numpy", "sklearn", "xgboost", "optuna", "torch"]),
    }
    write_json(args.artifact_dir / "metadata.json", metadata)
    print(f"\nSaved artifacts to {args.artifact_dir}")
    print(f"Saved leaderboard to {leaderboard_path}")


def predict_pipeline(args: argparse.Namespace) -> pd.DataFrame:
    require_ml_dependencies()
    joblib = importlib.import_module("joblib")
    metadata = read_json(args.artifact_dir / "metadata.json")
    feature_columns = metadata["feature_columns"]
    x_rookie, rookie_meta = prepare_rookie_data(args.rookie_csv, feature_columns)

    predictions: dict[str, np.ndarray] = {}
    uncertainty: dict[str, np.ndarray] = {}
    model_family_for_target: dict[str, str] = {}

    for target in metadata["target_columns"]:
        target_models = metadata["selected"][target]["top_models"]
        model_preds = []
        for model_info in target_models:
            model = joblib.load(args.artifact_dir / model_info["path"])
            model_preds.append(np.asarray(model.predict(x_rookie), dtype=float))
        stacked = np.vstack(model_preds)
        predictions[target] = stacked[0]
        uncertainty[target] = np.std(stacked, axis=0) if len(model_preds) > 1 else np.zeros(stacked.shape[1])
        model_family_for_target[target] = target_models[0]["family"]

    output = rookie_meta[["name", "school", "draft_pick"]].copy()
    for target in metadata["target_columns"]:
        output[f"predicted_{target}"] = predictions[target]
        output[f"uncertainty_{target}"] = uncertainty[target]
        output[f"model_family_{target}"] = model_family_for_target[target]

    output["rank_score"] = compute_rank_score(output)
    output["model_family"] = output[f"model_family_{PRIMARY_TARGET}"]
    output = output.sort_values("rank_score", ascending=False).reset_index(drop=True)
    output.insert(0, "rank", np.arange(1, len(output) + 1))
    output.to_csv(args.output_csv, index=False)
    print(f"Saved rookie predictions to {args.output_csv}")
    print(output.head(args.preview_rows).to_string(index=False))
    return output


def evaluate_pipeline(args: argparse.Namespace) -> None:
    metadata_path = args.artifact_dir / "metadata.json"
    leaderboard_path = args.artifact_dir / "validation_leaderboard.csv"
    if not metadata_path.exists() or not leaderboard_path.exists():
        raise FileNotFoundError("No saved artifacts found. Run `train` first.")

    metadata = read_json(metadata_path)
    leaderboard = pd.read_csv(leaderboard_path)
    print("GPU status")
    print(json.dumps(metadata.get("gpu_status", {}), indent=2))
    print("\nSelected models")
    for target, info in metadata["selected"].items():
        best = info["best_model"]
        metrics = best["metrics"]
        print(
            f"{target}: {best['name']} ({best['family']}) "
            f"MAE={float(metrics['mae']):.3f} Spearman={float(metrics['spearman']):.3f}"
        )
    print("\nLeaderboard")
    visible = ["target", "model", "family", "mae", "rmse", "r2", "spearman", "top10_overlap"]
    print(leaderboard[visible].head(args.preview_rows).to_string(index=False))


def compute_rank_score(output: pd.DataFrame) -> np.ndarray:
    weights = {
        "predicted_nfl_YPG": 0.75,
        "predicted_nfl_RPG": 0.15,
        "predicted_nfl_TDPG": 0.10,
    }
    score = np.zeros(len(output), dtype=float)
    for column, weight in weights.items():
        values = pd.to_numeric(output[column], errors="coerce").to_numpy(dtype=float)
        std = np.nanstd(values)
        if std == 0 or not np.isfinite(std):
            z = np.zeros_like(values)
        else:
            z = (values - np.nanmean(values)) / std
        score += weight * z
    return score


def safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)


def normalize_json_value(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def package_versions(package_names: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in package_names:
        module = optional_import(package)
        versions[package] = getattr(module, "__version__", None) if module is not None else None
    return versions


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=normalize_json_value), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


from sklearn.base import BaseEstimator, RegressorMixin


class TorchRegressor(BaseEstimator, RegressorMixin):
    """Small sklearn-compatible PyTorch regressor for tabular data."""

    def __init__(
        self,
        architecture: str = "mlp",
        hidden: tuple[int, ...] = (128, 64),
        d_model: int = 32,
        dropout: float = 0.15,
        lr: float = 0.002,
        weight_decay: float = 0.0005,
        epochs: int = 180,
        batch_size: int = 32,
        patience: int = 25,
        prefer_gpu: bool = True,
        random_state: int = 42,
    ) -> None:
        self.architecture = architecture
        self.hidden = hidden
        self.d_model = d_model
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.prefer_gpu = prefer_gpu
        self.random_state = random_state

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "hidden": self.hidden,
            "d_model": self.d_model,
            "dropout": self.dropout,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "prefer_gpu": self.prefer_gpu,
            "random_state": self.random_state,
        }

    def set_params(self, **params: Any) -> "TorchRegressor":
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, x: Any, y: Any) -> "TorchRegressor":
        torch = importlib.import_module("torch")
        nn = torch.nn
        rng = np.random.default_rng(self.random_state)
        torch.manual_seed(self.random_state)

        x_np = np.asarray(x, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        self.n_features_in_ = x_np.shape[1]
        self.y_mean_ = float(np.mean(y_np))
        self.y_std_ = float(np.std(y_np) or 1.0)
        y_scaled = (y_np - self.y_mean_) / self.y_std_

        indices = np.arange(len(x_np))
        rng.shuffle(indices)
        valid_size = max(1, int(0.2 * len(indices)))
        valid_idx = indices[:valid_size]
        train_idx = indices[valid_size:]
        if len(train_idx) == 0:
            train_idx = valid_idx

        device = torch.device("cuda" if self.prefer_gpu and torch.cuda.is_available() else "cpu")
        self.device_name_ = str(device)
        model = self._build_model(self.n_features_in_, nn).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.SmoothL1Loss()

        x_train = torch.as_tensor(x_np[train_idx], dtype=torch.float32, device=device)
        y_train = torch.as_tensor(y_scaled[train_idx], dtype=torch.float32, device=device)
        x_valid = torch.as_tensor(x_np[valid_idx], dtype=torch.float32, device=device)
        y_valid = torch.as_tensor(y_scaled[valid_idx], dtype=torch.float32, device=device)

        best_state = None
        best_loss = float("inf")
        stale_epochs = 0
        for _epoch in range(int(self.epochs)):
            model.train()
            permutation = torch.randperm(len(x_train), device=device)
            for start in range(0, len(x_train), int(self.batch_size)):
                batch_idx = permutation[start : start + int(self.batch_size)]
                optimizer.zero_grad()
                pred = model(x_train[batch_idx])
                loss = loss_fn(pred, y_train[batch_idx])
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                valid_loss = float(loss_fn(model(x_valid), y_valid).detach().cpu().item())
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= int(self.patience):
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model_state_ = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        self.model_config_ = {
            "architecture": self.architecture,
            "hidden": self.hidden,
            "d_model": self.d_model,
            "dropout": self.dropout,
            "n_features": self.n_features_in_,
        }
        return self

    def predict(self, x: Any) -> np.ndarray:
        torch = importlib.import_module("torch")
        nn = torch.nn
        if not hasattr(self, "model_state_"):
            raise RuntimeError("TorchRegressor must be fitted before predict().")
        x_np = np.asarray(x, dtype=np.float32)
        device = torch.device("cuda" if self.prefer_gpu and torch.cuda.is_available() else "cpu")
        model = self._build_model(self.n_features_in_, nn).to(device)
        model.load_state_dict({key: value.to(device) for key, value in self.model_state_.items()})
        model.eval()
        with torch.no_grad():
            pred = model(torch.as_tensor(x_np, dtype=torch.float32, device=device)).detach().cpu().numpy().ravel()
        return pred * self.y_std_ + self.y_mean_

    def _build_model(self, n_features: int, nn: Any) -> Any:
        if self.architecture == "mlp":
            layers: list[Any] = []
            in_features = n_features
            for width in self.hidden:
                layers.extend(
                    [
                        nn.Linear(in_features, width),
                        nn.LayerNorm(width),
                        nn.SiLU(),
                        nn.Dropout(self.dropout),
                    ]
                )
                in_features = width
            layers.append(nn.Linear(in_features, 1))
            return nn.Sequential(*layers)

        if self.architecture == "residual_mlp":
            return ResidualMLP(n_features, self.hidden, self.dropout, nn)

        if self.architecture == "tabular_transformer":
            return TabularTransformer(n_features, self.d_model, self.dropout, nn)

        raise ValueError(f"Unknown neural architecture: {self.architecture}")


class ResidualMLP:
    def __init__(self, n_features: int, hidden: tuple[int, ...], dropout: float, nn: Any) -> None:
        import torch

        class Block(nn.Module):
            def __init__(self, width: int) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(width, width),
                    nn.LayerNorm(width),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(width, width),
                )
                self.activation = nn.SiLU()

            def forward(self, x: Any) -> Any:
                return self.activation(x + self.net(x))

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                width = hidden[0] if hidden else 64
                self.input = nn.Sequential(nn.Linear(n_features, width), nn.SiLU())
                self.blocks = nn.Sequential(*[Block(width) for _ in range(max(1, len(hidden)))])
                self.output = nn.Linear(width, 1)

            def forward(self, x: Any) -> Any:
                return self.output(self.blocks(self.input(x)))

        self.model = Model()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)


class TabularTransformer:
    def __init__(self, n_features: int, d_model: int, dropout: float, nn: Any) -> None:
        import torch

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.value_projection = nn.Linear(1, d_model)
                self.feature_embedding = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=4,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
                self.output = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

            def forward(self, x: Any) -> Any:
                tokens = self.value_projection(x.unsqueeze(-1)) + self.feature_embedding.unsqueeze(0)
                encoded = self.encoder(tokens)
                pooled = encoded.mean(dim=1)
                return self.output(pooled)

        self.model = Model()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/evaluate/predict WR rookie projection models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_PATH)
        subparser.add_argument("--rookie-csv", type=Path, default=DEFAULT_ROOKIE_PATH)
        subparser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)

    train = subparsers.add_parser("train", help="Run sweeps, select models, and save artifacts.")
    add_common(train)
    train.add_argument("--trials", type=int, default=6, help="Optuna trials per search space. Use 0 to disable.")
    train.add_argument("--folds", type=int, default=5)
    train.add_argument("--top-models", type=int, default=3)
    train.add_argument("--min-nfl-games", type=int, default=MIN_NFL_GAMES)
    train.add_argument("--include-neural", action="store_true")
    train.add_argument("--no-xgboost", action="store_true")
    train.add_argument("--prefer-gpu", action="store_true", default=True)

    predict = subparsers.add_parser("predict", help="Load artifacts and write rookie predictions.")
    add_common(predict)
    predict.add_argument("--output-csv", type=Path, default=DEFAULT_PREDICTION_PATH)
    predict.add_argument("--preview-rows", type=int, default=20)

    evaluate = subparsers.add_parser("evaluate", help="Print saved validation metrics and GPU diagnostics.")
    add_common(evaluate)
    evaluate.add_argument("--preview-rows", type=int, default=30)

    subparsers.add_parser("gpu-info", help="Print GPU and ML package detection.")
    subparsers.add_parser("smoke", help="Validate data loading, feature engineering, and leakage checks.")
    return parser


def run_smoke() -> None:
    x, y, meta = prepare_training_data()
    x_rookie, rookie = prepare_rookie_data(feature_columns=FEATURE_COLUMNS)
    validate_feature_columns(FEATURE_COLUMNS)
    assert list(x.columns) == FEATURE_COLUMNS
    assert list(x_rookie.columns) == FEATURE_COLUMNS
    assert not any(column.startswith("nfl_") for column in x.columns)
    print(f"Training rows: {len(x)}")
    print(f"Rookie rows: {len(x_rookie)}")
    print(f"Targets: {list(y.columns)}")
    print(f"Seasons: {int(meta['season'].min())}-{int(meta['season'].max())}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "train":
            train_pipeline(args)
        elif args.command == "predict":
            predict_pipeline(args)
        elif args.command == "evaluate":
            evaluate_pipeline(args)
        elif args.command == "gpu-info":
            print(json.dumps(detect_gpu_status(), indent=2))
        elif args.command == "smoke":
            run_smoke()
        else:
            parser.error(f"Unknown command: {args.command}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if os.environ.get("FF_MODEL_DEBUG"):
            traceback.print_exc()
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
