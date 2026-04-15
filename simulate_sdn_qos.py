from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


BASE_SEED = 42
METRICS = ["accuracy", "f1_weighted", "kappa", "roc_auc_ovr"]


@dataclass
class ScenarioConfig:
    name: str
    n_samples: int
    n_features: int
    n_informative: int
    n_redundant: int
    n_classes: int
    weights: Tuple[float, ...]
    class_sep: float


def get_scenarios() -> List[ScenarioConfig]:
    return [
        ScenarioConfig(
            name="multi_class",
            n_samples=9000,
            n_features=20,
            n_informative=12,
            n_redundant=4,
            n_classes=7,
            weights=(0.30, 0.22, 0.16, 0.12, 0.10, 0.06, 0.04),
            class_sep=1.1,
        ),
        ScenarioConfig(
            name="binary",
            n_samples=6000,
            n_features=20,
            n_informative=10,
            n_redundant=3,
            n_classes=2,
            weights=(0.75, 0.25),
            class_sep=1.3,
        ),
    ]


def build_dataset(cfg: ScenarioConfig, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    return make_classification(
        n_samples=cfg.n_samples,
        n_features=cfg.n_features,
        n_informative=cfg.n_informative,
        n_redundant=cfg.n_redundant,
        n_classes=cfg.n_classes,
        n_clusters_per_class=1,
        weights=list(cfg.weights),
        class_sep=cfg.class_sep,
        random_state=random_state,
    )


def get_models(seed: int) -> Dict[str, object]:
    models: Dict[str, object] = {
        "DT": DecisionTreeClassifier(max_depth=8, random_state=seed),
        "SVM": SVC(probability=True, kernel="rbf", C=2.0, gamma="scale", random_state=seed),
        "KNN": KNeighborsClassifier(n_neighbors=7),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=120,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=seed,
        )
    else:
        print("Warning: xgboost is not installed, skipping XGBoost model.")

    models["Hybrid SVM-DT"] = VotingClassifier(
        estimators=[("svm", clone(models["SVM"])), ("dt", clone(models["DT"]))],
        voting="soft",
    )
    models["Hybrid KNN-SVM"] = VotingClassifier(
        estimators=[("knn", clone(models["KNN"])), ("svm", clone(models["SVM"]))],
        voting="soft",
    )
    return models


def one_vs_rest_auc(y_true: np.ndarray, probas: np.ndarray) -> float:
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y_true)
    if y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])
    return roc_auc_score(y_bin, probas, average="macro", multi_class="ovr")


def evaluate_model(model, X_train, y_train, X_test, y_test) -> Dict[str, float]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
    else:
        scores = model.decision_function(X_test)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        y_proba = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "kappa": cohen_kappa_score(y_test, y_pred),
        "roc_auc_ovr": one_vs_rest_auc(y_test, y_proba),
    }


def run_experiment(output_dir: Path, repeats: int, test_sizes: List[float]) -> pd.DataFrame:
    scalers = {
        "none": None,
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }
    all_rows: List[Dict[str, object]] = []

    for scenario in get_scenarios():
        for repeat in range(repeats):
            seed = BASE_SEED + repeat
            X, y = build_dataset(scenario, random_state=seed)
            for test_size in test_sizes:
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=seed,
                    stratify=y,
                )
                for scaler_name, scaler in scalers.items():
                    for model_name, model in get_models(seed).items():
                        if scaler is None:
                            estimator = clone(model)
                        else:
                            estimator = Pipeline([("scaler", clone(scaler)), ("model", clone(model))])
                        metrics = evaluate_model(estimator, X_train, y_train, X_test, y_test)
                        all_rows.append(
                            {
                                "scenario": scenario.name,
                                "repeat": repeat,
                                "seed": seed,
                                "test_size": test_size,
                                "scaling": scaler_name,
                                "model": model_name,
                                **metrics,
                            }
                        )

    run_df = pd.DataFrame(all_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_df.to_csv(output_dir / "raw_runs.csv", index=False)
    return run_df


def summarize_results(run_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    summary = (
        run_df.groupby(["scenario", "test_size", "scaling", "model"], as_index=False)[METRICS]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join([str(c) for c in col if c]).rstrip("_") if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]
    summary.to_csv(output_dir / "summary_mean_std.csv", index=False)

    best_idx = summary.groupby(["scenario", "test_size"])["accuracy_mean"].idxmax()
    best = summary.loc[best_idx].sort_values(["scenario", "test_size"])
    best.to_csv(output_dir / "best_per_setting.csv", index=False)
    return summary


def paired_scaling_test(run_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for scenario in sorted(run_df["scenario"].unique()):
        for test_size in sorted(run_df["test_size"].unique()):
            for model in sorted(run_df["model"].unique()):
                sub = run_df[
                    (run_df["scenario"] == scenario)
                    & (run_df["test_size"] == test_size)
                    & (run_df["model"] == model)
                ]
                baseline = sub[sub["scaling"] == "none"].sort_values("repeat")
                if baseline.empty:
                    continue
                for scaling in ["standard", "minmax", "robust"]:
                    target = sub[sub["scaling"] == scaling].sort_values("repeat")
                    if len(target) != len(baseline):
                        continue
                    for metric in METRICS:
                        t_stat, p_val = ttest_rel(target[metric], baseline[metric], nan_policy="omit")
                        rows.append(
                            {
                                "scenario": scenario,
                                "test_size": test_size,
                                "model": model,
                                "metric": metric,
                                "scaling_vs_none": scaling,
                                "delta_mean": float(target[metric].mean() - baseline[metric].mean()),
                                "p_value": float(p_val),
                                "significant_0_05": bool(p_val < 0.05),
                                "t_stat": float(t_stat),
                            }
                        )
    test_df = pd.DataFrame(rows).sort_values(["scenario", "test_size", "model", "metric", "p_value"])
    test_df.to_csv(output_dir / "scaling_gain_tests.csv", index=False)
    return test_df


def plot_results(summary_df: pd.DataFrame, output_dir: Path) -> None:
    for scenario in summary_df["scenario"].unique():
        data_s = summary_df[summary_df["scenario"] == scenario]
        for metric in METRICS:
            fig, ax = plt.subplots(figsize=(12, 5))
            for scaling in sorted(data_s["scaling"].unique()):
                data_ss = data_s[data_s["scaling"] == scaling]
                model_scores = data_ss.groupby("model")[f"{metric}_mean"].mean().sort_values(ascending=False)
                ax.plot(model_scores.index, model_scores.values, marker="o", label=scaling)
            ax.set_title(f"{scenario} - {metric} (average over split ratios)")
            ax.set_ylabel(metric)
            ax.set_xlabel("model")
            ax.tick_params(axis="x", rotation=25)
            ax.grid(alpha=0.3)
            ax.legend(title="scaling")
            plt.tight_layout()
            plt.savefig(output_dir / f"{scenario}_{metric}_line.png", dpi=150)
            plt.close()


def print_key_findings(summary_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    for scenario in sorted(summary_df["scenario"].unique()):
        print(f"\n=== Scenario: {scenario} ===")
        local = summary_df[summary_df["scenario"] == scenario]
        for test_size in sorted(local["test_size"].unique()):
            best = local[local["test_size"] == test_size].sort_values("accuracy_mean", ascending=False).iloc[0]
            print(
                f"test_size={test_size:.2f} -> best accuracy: {best['model']} | scaling={best['scaling']} | "
                f"mean={best['accuracy_mean']:.4f} +- {best['accuracy_std']:.4f}"
            )

    if not test_df.empty:
        sig = test_df[test_df["significant_0_05"]]
        print(f"\nSignificant scaling improvements (p < 0.05): {len(sig)}/{len(test_df)} tests")


def parse_test_sizes(text: str) -> List[float]:
    vals = [float(v.strip()) for v in text.split(",") if v.strip()]
    for v in vals:
        if not (0.1 <= v <= 0.9):
            raise ValueError("Each test_size must be in [0.1, 0.9].")
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(description="Research-grade SDN QoS traffic classification simulator.")
    parser.add_argument("--output", type=str, default="outputs", help="Output folder for CSV tables and plots.")
    parser.add_argument("--repeats", type=int, default=10, help="Number of random repeats per configuration.")
    parser.add_argument(
        "--test-sizes",
        type=str,
        default="0.3,0.4,0.5",
        help="Comma-separated test ratios, e.g. '0.3,0.4,0.5'.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    test_sizes = parse_test_sizes(args.test_sizes)

    run_df = run_experiment(output_dir, repeats=args.repeats, test_sizes=test_sizes)
    summary_df = summarize_results(run_df, output_dir)
    test_df = paired_scaling_test(run_df, output_dir)
    plot_results(summary_df, output_dir)
    print_key_findings(summary_df, test_df)
    print(f"\nSaved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
