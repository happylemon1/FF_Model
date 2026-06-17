"""Multi-position rookie projection training, prediction, and reporting."""

from __future__ import annotations

import argparse
import importlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src import rookie_data_pipeline as data_pipeline
from src.wr_rookie_pipeline import detect_gpu_status, spearman_correlation, top_k_overlap


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "models" / "artifacts" / "multi_position"
PREDICTION_DIR = PROJECT_ROOT / "data" / "final"
REPORT_DIR = PROJECT_ROOT / "reports"


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    factory: Callable[[], Any]


def require_ml() -> None:
    missing = []
    for module in ["sklearn", "joblib"]:
        if importlib.util.find_spec(module) is None:
            missing.append(module)
    if missing:
        raise RuntimeError(f"Missing ML dependencies: {missing}. Run `python -m pip install -r requirements.txt`.")


def package_versions() -> dict[str, str | None]:
    versions = {}
    for name in ["pandas", "numpy", "sklearn", "xgboost", "optuna", "torch"]:
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", None)
        except ImportError:
            versions[name] = None
    return versions


def candidate_builders(prefer_gpu: bool, include_neural: bool) -> list[Candidate]:
    require_ml()
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def scaled(model: Any) -> Pipeline:
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", model)])

    def tree(model: Any) -> Pipeline:
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])

    candidates = [
        Candidate("median", "baseline", lambda: DummyRegressor(strategy="median")),
        Candidate("ridge", "linear", lambda: scaled(Ridge(alpha=10.0, random_state=42))),
        Candidate(
            "elastic_net",
            "linear",
            lambda: scaled(ElasticNet(alpha=0.05, l1_ratio=0.25, random_state=42, max_iter=20000)),
        ),
        Candidate(
            "random_forest",
            "bagging_tree",
            lambda: tree(
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=4,
                    min_samples_leaf=3,
                    max_features=0.8,
                    random_state=42,
                    n_jobs=-1,
                )
            ),
        ),
        Candidate(
            "extra_trees",
            "bagging_tree",
            lambda: tree(
                ExtraTreesRegressor(
                    n_estimators=350,
                    max_depth=4,
                    min_samples_leaf=3,
                    max_features=0.8,
                    random_state=42,
                    n_jobs=-1,
                )
            ),
        ),
        Candidate(
            "gradient_boosting",
            "boosted_tree",
            lambda: tree(
                GradientBoostingRegressor(
                    n_estimators=160,
                    learning_rate=0.04,
                    max_depth=2,
                    subsample=0.85,
                    random_state=42,
                )
            ),
        ),
    ]

    xgb = optional_import("xgboost")
    if xgb is not None:
        device = "cuda" if prefer_gpu else "cpu"
        candidates.append(
            Candidate(
                f"xgboost_{device}",
                "boosted_tree",
                lambda: tree(
                    xgb.XGBRegressor(
                        objective="reg:squarederror",
                        n_estimators=250,
                        learning_rate=0.035,
                        max_depth=2,
                        min_child_weight=2.0,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_lambda=2.0,
                        random_state=42,
                        tree_method="hist",
                        device=device,
                        n_jobs=-1,
                    )
                ),
            )
        )

    if include_neural and optional_import("torch") is not None:
        from src.wr_rookie_pipeline import TorchRegressor

        candidates.extend(
            [
                Candidate(
                    "mlp",
                    "neural_net",
                    lambda: scaled(
                        TorchRegressor(
                            architecture="mlp",
                            hidden=(96, 48),
                            dropout=0.15,
                            epochs=120,
                            prefer_gpu=prefer_gpu,
                            random_state=42,
                        )
                    ),
                ),
                Candidate(
                    "residual_mlp",
                    "neural_net",
                    lambda: scaled(
                        TorchRegressor(
                            architecture="residual_mlp",
                            hidden=(96, 48),
                            dropout=0.15,
                            epochs=120,
                            prefer_gpu=prefer_gpu,
                            random_state=42,
                        )
                    ),
                ),
            ]
        )
    return candidates


def optional_import(name: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def position_features(position: str) -> list[str]:
    return data_pipeline.COMMON_FEATURES + data_pipeline.POSITION_FEATURES[position]


def target_columns(position: str) -> list[str]:
    return list(data_pipeline.POSITION_SPECS[position].target_columns)


def clean_xy(df: pd.DataFrame, features: list[str], target: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    needed = ["name", "position", "school", "draft_pick", "season", *features, target]
    data = df[[column for column in needed if column in df.columns]].copy()
    data = data.loc[:, ~data.columns.duplicated()]
    features = list(dict.fromkeys(features))
    for column in [*features, target, "season"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=features + [target])
    return data[features], data[target], data[["name", "position", "school", "draft_pick", "season"]]


def make_splits(meta: pd.DataFrame, folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    seasons = sorted(pd.to_numeric(meta["season"], errors="coerce").dropna().unique())
    splits = []
    for season in seasons[-folds:]:
        train_idx = np.where(meta["season"].to_numpy() < season)[0]
        valid_idx = np.where(meta["season"].to_numpy() == season)[0]
        if len(train_idx) >= 20 and len(valid_idx) >= 3:
            splits.append((train_idx, valid_idx))
    if splits:
        return splits
    from sklearn.model_selection import KFold

    kfold = KFold(n_splits=min(folds, max(2, len(meta))), shuffle=True, random_state=42)
    return list(kfold.split(np.arange(len(meta))))


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "spearman": spearman_correlation(y_true, y_pred),
        "top10_overlap": top_k_overlap(y_true, y_pred, 10),
    }


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    output = {}
    for key in rows[0]:
        values = np.asarray([row[key] for row in rows], dtype=float)
        values = values[np.isfinite(values)]
        output[key] = float(values.mean()) if len(values) else float("nan")
    return output


def evaluate_candidate(candidate: Candidate, x: pd.DataFrame, y: pd.Series, splits: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, float]:
    fold_rows = []
    for train_idx, valid_idx in splits:
        model = candidate.factory()
        try:
            model.fit(x.iloc[train_idx], y.iloc[train_idx])
        except Exception:
            if "xgboost_cuda" not in candidate.name:
                raise
            cpu_candidate = [c for c in candidate_builders(False, False) if c.name == "xgboost_cpu"][0]
            model = cpu_candidate.factory()
            model.fit(x.iloc[train_idx], y.iloc[train_idx])
        pred = np.asarray(model.predict(x.iloc[valid_idx]), dtype=float)
        fold_rows.append(metrics(y.iloc[valid_idx].to_numpy(dtype=float), pred))
    return mean_metrics(fold_rows)


def train_position_models(position: str, df: pd.DataFrame, args: argparse.Namespace, prefer_gpu: bool) -> list[dict[str, Any]]:
    require_ml()
    joblib = importlib.import_module("joblib")
    features = position_features(position)
    candidates = candidate_builders(prefer_gpu, args.include_neural)
    rows = []
    position_dir = args.artifact_dir / position
    position_dir.mkdir(parents=True, exist_ok=True)

    for target in target_columns(position):
        x, y, meta = clean_xy(df, features, target)
        if len(x) < 25:
            continue
        splits = make_splits(meta, args.folds)
        scored = []
        for candidate in candidates:
            try:
                score = evaluate_candidate(candidate, x, y, splits)
                scored.append({**score, "position": position, "target": target, "model": candidate.name, "family": candidate.family})
                print(f"{position} {target:16s} {candidate.name:18s} MAE={score['mae']:.3f} Spearman={score['spearman']:.3f}")
            except Exception as exc:
                print(f"{position} {target:16s} {candidate.name:18s} failed: {exc}")
        if not scored:
            continue
        leaderboard = pd.DataFrame(scored)
        sort_cols = ["spearman", "mae"] if target == "fantasy_points" else ["mae", "rmse"]
        ascending = [False, True] if target == "fantasy_points" else [True, True]
        leaderboard = leaderboard.sort_values(sort_cols, ascending=ascending)
        best = leaderboard.iloc[0]
        best_candidate = next(c for c in candidates if c.name == best["model"])
        model = best_candidate.factory()
        model.fit(x, y)
        model_path = position_dir / f"{target}_{best_candidate.name}.joblib"
        joblib.dump(model, model_path)
        rows.extend(leaderboard.to_dict(orient="records"))
    return rows


def train_pooled_model(processed: dict[str, pd.DataFrame], args: argparse.Namespace, prefer_gpu: bool) -> list[dict[str, Any]]:
    require_ml()
    joblib = importlib.import_module("joblib")
    combined = pd.concat(processed.values(), ignore_index=True, sort=False)
    features = data_pipeline.COMMON_FEATURES
    x, y, meta = clean_xy(combined, features, "fantasy_points")
    candidates = candidate_builders(prefer_gpu, args.include_neural)
    splits = make_splits(meta, args.folds)
    rows = []
    for candidate in candidates:
        try:
            score = evaluate_candidate(candidate, x, y, splits)
            rows.append({**score, "position": "ALL", "target": "fantasy_points", "model": candidate.name, "family": candidate.family})
            print(f"ALL fantasy_points  {candidate.name:18s} MAE={score['mae']:.3f} Spearman={score['spearman']:.3f}")
        except Exception as exc:
            print(f"ALL fantasy_points  {candidate.name:18s} failed: {exc}")
    leaderboard = pd.DataFrame(rows).sort_values(["spearman", "mae"], ascending=[False, True])
    best = leaderboard.iloc[0]
    best_candidate = next(c for c in candidates if c.name == best["model"])
    model = best_candidate.factory()
    model.fit(x, y)
    all_dir = args.artifact_dir / "ALL"
    all_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, all_dir / f"fantasy_points_{best_candidate.name}.joblib")
    return leaderboard.to_dict(orient="records")


def load_processed() -> dict[str, pd.DataFrame]:
    processed = {}
    for position in data_pipeline.POSITIONS:
        path = data_pipeline.PROCESSED_DIR / f"{position.lower()}_training_dataset.csv"
        if not path.exists():
            data_pipeline.build_all_processed()
        processed[position] = pd.read_csv(path)
    return processed


def train(args: argparse.Namespace) -> None:
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    processed = load_processed()
    gpu = detect_gpu_status()
    prefer_gpu = bool(args.prefer_gpu and gpu.get("nvidia_smi_available"))
    rows = []
    for position, df in processed.items():
        rows.extend(train_position_models(position, df, args, prefer_gpu))
    rows.extend(train_pooled_model(processed, args, prefer_gpu))
    leaderboard = pd.DataFrame(rows)
    leaderboard.to_csv(args.artifact_dir / "leaderboard.csv", index=False)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "positions": data_pipeline.POSITIONS,
        "artifact_dir": str(args.artifact_dir),
        "gpu_status": gpu,
        "package_versions": package_versions(),
        "include_neural": args.include_neural,
        "leaderboard": str(args.artifact_dir / "leaderboard.csv"),
    }
    (args.artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=json_default), encoding="utf-8")


def find_best_model(leaderboard: pd.DataFrame, position: str, target: str) -> pd.Series:
    subset = leaderboard[(leaderboard["position"] == position) & (leaderboard["target"] == target)].copy()
    if subset.empty:
        raise ValueError(f"No model found for {position}/{target}")
    if target == "fantasy_points":
        subset = subset.sort_values(["spearman", "mae"], ascending=[False, True])
    else:
        subset = subset.sort_values(["mae", "rmse"], ascending=[True, True])
    return subset.iloc[0]


def predict(args: argparse.Namespace) -> pd.DataFrame:
    require_ml()
    joblib = importlib.import_module("joblib")
    leaderboard = pd.read_csv(args.artifact_dir / "leaderboard.csv")
    all_predictions = []
    for position in data_pipeline.POSITIONS:
        feature_path = data_pipeline.FINAL_DIR / f"{position.lower()}_rookies_{args.year}_features.csv"
        if not feature_path.exists():
            continue
        df = pd.read_csv(feature_path)
        features = position_features(position)
        for column in features:
            if column not in df.columns:
                df[column] = 0.0
            df[column] = pd.to_numeric(df[column], errors="coerce")
        out = df[["name", "position", "school", "draft_pick"]].copy()
        for target in target_columns(position):
            try:
                best = find_best_model(leaderboard, position, target)
                model_path = args.artifact_dir / position / f"{target}_{best['model']}.joblib"
                model = joblib.load(model_path)
                out[f"predicted_{target}"] = model.predict(df[features])
                out[f"model_{target}"] = best["model"]
                out[f"uncertainty_{target}"] = best["mae"]
            except Exception:
                continue
        if "predicted_fantasy_points" not in out.columns:
            out["predicted_fantasy_points"] = fallback_fantasy(position, out)
        all_predictions.append(out)
    if not all_predictions:
        raise FileNotFoundError(f"No rookie feature files found for {args.year}. Run crawl first.")
    combined = pd.concat(all_predictions, ignore_index=True, sort=False)
    combined = combined.sort_values("predicted_fantasy_points", ascending=False).reset_index(drop=True)
    combined.insert(0, "rank", np.arange(1, len(combined) + 1))
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    combined_path = PREDICTION_DIR / f"rookie_predictions_{args.year}_combined.csv"
    combined.to_csv(combined_path, index=False)
    for position, group in combined.groupby("position"):
        group.to_csv(PREDICTION_DIR / f"{position.lower()}_rookie_predictions_{args.year}.csv", index=False)
    print(combined.head(args.preview_rows).to_string(index=False))
    return combined


def fallback_fantasy(position: str, df: pd.DataFrame) -> pd.Series:
    values = pd.Series(0.0, index=df.index)
    for column, weight in [
        ("predicted_nfl_passingYDS", 0.04),
        ("predicted_nfl_passingTDS", 4),
        ("predicted_nfl_passingINT", -2),
        ("predicted_nfl_rushingYDS", 0.1),
        ("predicted_nfl_rushingTDS", 6),
        ("predicted_nfl_recs", 1),
        ("predicted_nfl_recYDS", 0.1),
        ("predicted_nfl_recTDS", 6),
    ]:
        if column in df.columns:
            values += pd.to_numeric(df[column], errors="coerce").fillna(0) * weight
    return values


def report(args: argparse.Namespace) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    leaderboard = pd.read_csv(args.artifact_dir / "leaderboard.csv")
    metadata = json.loads((args.artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    lines = [
        "# Modeling Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## GPU Status",
        "",
        "```json",
        json.dumps(metadata.get("gpu_status", {}), indent=2),
        "```",
        "",
        "## Selected Models",
        "",
    ]
    for position in [*data_pipeline.POSITIONS, "ALL"]:
        subset = leaderboard[leaderboard["position"] == position]
        if subset.empty:
            continue
        lines.append(f"### {position}")
        for target in sorted(subset["target"].unique()):
            best = find_best_model(leaderboard, position, target)
            lines.append(
                f"- `{target}`: `{best['model']}` ({best['family']}), "
                f"MAE={best['mae']:.3f}, Spearman={best['spearman']:.3f}"
            )
        lines.append("")
    lines.extend(
        [
            "## Outputs",
            "",
            f"- Artifacts: `{args.artifact_dir}`",
            f"- Leaderboard: `{args.artifact_dir / 'leaderboard.csv'}`",
            f"- Combined predictions: `{PREDICTION_DIR / f'rookie_predictions_{args.year}_combined.csv'}`",
            "",
            "## Caveats",
            "",
            "- Historical data is small by modern ML standards, so tree ensembles and regularized linear models may beat neural nets.",
            "- 2026 rookie inference quality depends on the availability and shape of Sports Reference college profile tables.",
            "- GPU acceleration is attempted for XGBoost when available; PyTorch CUDA requires a CUDA-enabled PyTorch wheel.",
            "",
        ]
    )
    (PROJECT_ROOT / "docs").mkdir(exist_ok=True)
    (PROJECT_ROOT / "docs" / "modeling_report.md").write_text("\n".join(lines), encoding="utf-8")
    (REPORT_DIR / f"modeling_report_{args.year}.md").write_text("\n".join(lines), encoding="utf-8")
    print(PROJECT_ROOT / "docs" / "modeling_report.md")


def evaluate(args: argparse.Namespace) -> None:
    leaderboard = pd.read_csv(args.artifact_dir / "leaderboard.csv")
    visible = ["position", "target", "model", "family", "mae", "rmse", "r2", "spearman", "top10_overlap"]
    print(leaderboard[visible].sort_values(["position", "target", "mae"]).head(args.preview_rows).to_string(index=False))


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-position rookie projection pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build-data")
    crawl = sub.add_parser("crawl")
    crawl.add_argument("--year", type=int, default=2026)
    crawl.add_argument("--refresh", action="store_true")
    crawl.add_argument("--max-players", type=int, default=None)
    train_cmd = sub.add_parser("train")
    train_cmd.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    train_cmd.add_argument("--folds", type=int, default=3)
    train_cmd.add_argument("--include-neural", action="store_true")
    train_cmd.add_argument("--prefer-gpu", action="store_true", default=True)
    pred = sub.add_parser("predict")
    pred.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    pred.add_argument("--year", type=int, default=2026)
    pred.add_argument("--preview-rows", type=int, default=25)
    eval_cmd = sub.add_parser("evaluate")
    eval_cmd.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    eval_cmd.add_argument("--preview-rows", type=int, default=50)
    report_cmd = sub.add_parser("report")
    report_cmd.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    report_cmd.add_argument("--year", type=int, default=2026)
    sub.add_parser("gpu-info")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "build-data":
        outputs = data_pipeline.build_all_processed()
        for key, path in outputs.items():
            print(f"{key}: {path}")
    elif args.command == "crawl":
        outputs = data_pipeline.scrape_rookie_class(args.year, refresh=args.refresh, max_players=args.max_players)
        for key, path in outputs.items():
            print(f"{key}: {path}")
    elif args.command == "train":
        train(args)
    elif args.command == "predict":
        predict(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "report":
        report(args)
    elif args.command == "gpu-info":
        print(json.dumps(detect_gpu_status(), indent=2))


if __name__ == "__main__":
    main()
