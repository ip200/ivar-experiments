import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import make_friedman1, make_friedman2, make_friedman3
from sklearn.svm import SVR
import os, sys
import random

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from venn_abers import VennAberRegressor
from data.other_datasets.datasets import GetDataset
from data.uci_repository_datasets.datasets import load_dataset

import warnings
warnings.filterwarnings("ignore")

artificial_datasets = ['linear_gaussian', 'nonlinear_sine', 'heteroscedastic',
                       'heavy_tailed', 'outliers', 'sparse_highdim', 'covariate_shift']
friedman_datasets = ['friedman1', 'friedman2', 'friedman3']

uci_datasets = ['airfoil', 'climate_bias', 'electricity']
other_datasets = ['star']

dataset_dict = dict()
dataset_dict['synthetic_datasets'] = artificial_datasets + friedman_datasets
dataset_dict['real_datasets'] = uci_datasets + other_datasets

m_parameters = [1, 10]

@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    meta: Dict


def _train_test_split(X, y, test_size: float, rng: np.random.Generator):
    n = X.shape[0]
    idx = rng.permutation(n)
    n_test = int(np.round(n * test_size))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def make_synth_regression(
        *,
        n_samples: int = 10000,
        n_features: int = 10,
        test_size: float = 0.2,
        scenario: str = "linear_gaussian",
        noise_scale: float = 1.0,
        random_state: int = 0,
        standardize: bool = True,
) -> Dataset:
    """
    Scenarios:
      - linear_gaussian: linear model + iid Gaussian noise
      - nonlinear_sine: linear + (sin/cos/quadratic) nonlinearities
      - heteroscedastic: noise variance depends on X
      - heavy_tailed: Student-t noise (robustness stress)
      - outliers: mixture noise with rare large outliers
      - sparse_highdim: sparse signal in high-dimensional space
      - covariate_shift: train and test have different X distributions
    """
    rng = np.random.default_rng(random_state)

    # For most scenarios we generate a single pool then split.
    X = rng.normal(size=(n_samples, n_features))

    if scenario == "linear_gaussian":
        w = rng.normal(size=n_features)
        y_clean = X @ w
        eps = rng.normal(scale=noise_scale, size=n_samples)
        y = y_clean + eps
        meta = {"scenario": scenario, "noise": "gaussian", "w": w}

    elif scenario == "nonlinear_sine":
        w = rng.normal(size=n_features)
        lin = X @ w
        nl = (
                2.0 * np.sin(X[:, 0])
                + 0.5 * (X[:, 1] ** 2)
                - 1.0 * np.cos(2.0 * X[:, 2])
        )
        y_clean = lin + nl
        eps = rng.normal(scale=noise_scale, size=n_samples)
        y = y_clean + eps
        meta = {"scenario": scenario, "noise": "gaussian", "w": w}

    elif scenario == "heteroscedastic":
        w = rng.normal(size=n_features)
        y_clean = X @ w
        local_scale = noise_scale * (0.5 + np.abs(X[:, 0]))  # depends on feature 0
        eps = rng.normal(scale=local_scale, size=n_samples)
        y = y_clean + eps
        meta = {"scenario": scenario, "noise": "heteroscedastic", "w": w}

    elif scenario == "heavy_tailed":
        w = rng.normal(size=n_features)
        y_clean = X @ w
        dof = 3.0
        eps = rng.standard_t(df=dof, size=n_samples) * noise_scale
        y = y_clean + eps
        meta = {"scenario": scenario, "noise": f"student_t(df={dof})", "w": w}

    elif scenario == "outliers":
        w = rng.normal(size=n_features)
        y_clean = X @ w
        p_out = 0.01
        base = rng.normal(scale=noise_scale, size=n_samples)
        out = rng.normal(scale=10.0 * noise_scale, size=n_samples)
        mask = rng.random(n_samples) < p_out
        eps = np.where(mask, out, base)
        y = y_clean + eps
        meta = {"scenario": scenario, "noise": "mixture_outliers", "p_out": p_out, "w": w}

    elif scenario == "sparse_highdim":
        k = max(1, n_features // 10)
        w = np.zeros(n_features)
        support = rng.choice(n_features, size=k, replace=False)
        w[support] = rng.normal(loc=0.0, scale=1.0, size=k)
        y_clean = X @ w
        eps = rng.normal(scale=noise_scale, size=n_samples)
        y = y_clean + eps
        meta = {"scenario": scenario, "noise": "gaussian", "k": k, "support": support, "w": w}

    elif scenario == "covariate_shift":
        # Force shift by sampling train/test from different distributions.
        n_test = int(np.round(n_samples * test_size))
        n_train = n_samples - n_test

        X_train = rng.normal(loc=0.0, scale=1.0, size=(n_train, n_features))
        X_test = rng.normal(loc=1.0, scale=1.0, size=(n_test, n_features))

        w = rng.normal(size=n_features)
        y_train = X_train @ w + rng.normal(scale=noise_scale, size=n_train)
        y_test = X_test @ w + rng.normal(scale=noise_scale, size=n_test)

        if standardize:
            mu = X_train.mean(axis=0, keepdims=True)
            sig = X_train.std(axis=0, keepdims=True) + 1e-12
            X_train = (X_train - mu) / sig
            X_test = (X_test - mu) / sig

        return Dataset(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            meta={"scenario": scenario, "noise": "gaussian", "shift": True, "w": w},
        )

    else:
        try:
            if scenario in other_datasets:
                X, y = GetDataset(scenario, base_path=module_path + '/src/data/other_datasets')
                standardize = True
                meta = {"scenario": scenario, "noise": "dataset"}
            elif scenario in uci_datasets:
                X, y, _ = load_dataset(scenario, split=False)
                standardize = True
                meta = {"scenario": scenario, "noise": "dataset"}
            elif scenario in friedman_datasets:
                if scenario == friedman_datasets[0]:
                    X, y = make_friedman1(n_samples=n_samples, noise=noise_scale, random_state=random_state)
                    meta = {"scenario": scenario, "noise": "dataset"}
                elif scenario == friedman_datasets[2]:
                    X, y = make_friedman2(n_samples=n_samples, noise=noise_scale, random_state=random_state)
                    meta = {"scenario": scenario, "noise": "dataset"}
                else:
                    X, y = make_friedman3(n_samples=n_samples, noise=noise_scale, random_state=random_state)
                    meta = {"scenario": scenario, "noise": "dataset"}

        except:
            raise ValueError(f"Unknown scenario: {scenario}")

    X_train, X_test, y_train, y_test = _train_test_split(X, y, test_size, rng)

    if standardize:
        mu = X_train.mean(axis=0, keepdims=True)
        sig = X_train.std(axis=0, keepdims=True) + 1e-12
        X_train = (X_train - mu) / sig
        X_test = (X_test - mu) / sig

    return Dataset(
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        meta=meta,
    )


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def baseline_models(random_state: int = 0):
    # Reasonable "standard" baselines.
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet(),
        "SVR(RBF)": SVR(),
        "RandomForest": RandomForestRegressor(random_state=random_state, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=random_state)

    }


def run_one_scenario(
        *,
        scenario: str,
        n_samples: int = 10000,
        n_features: int = 10,
        noise_scale: float = 1.0,
        seed: int = 0,
):
    ds = make_synth_regression(
        n_samples=n_samples,
        n_features=n_features,
        scenario=scenario,
        noise_scale=noise_scale,
        random_state=seed,
    )

    results = {}
    print(scenario)
    for name, model in baseline_models(random_state=seed).items():
        model.fit(ds.X_train, ds.y_train)
        base_pred = model.predict(ds.X_test)
        results[name] = rmse(ds.y_test, base_pred)
        va = VennAberRegressor(estimator=model, inductive=False, n_splits=10, random_state=seed)
        for m in m_parameters:
            va.fit(ds.X_train, ds.y_train, m=m)
            va_preds, _ = va.predict(ds.X_test)
            results[name + ' CVAP - ' + str(m)] = rmse(ds.y_test, va_preds)

    return results, ds.meta

def run_benchmark(
    scenarios,
    seeds=range(10),
    n_samples=5000,
    n_features=10,
    noise_scale=1.0,
):
    rows = []
    for sc in scenarios:
        for seed in seeds:
            res, meta = run_one_scenario(
                scenario=sc,
                n_samples=n_samples,
                n_features=n_features,
                noise_scale=noise_scale,
                seed=seed,
            )
            for model_name, score in res.items():
                rows.append({
                    "scenario": sc,
                    "seed": seed,
                    "model": model_name,
                    "rmse": score,
                })

    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["scenario", "model"])["rmse"]
          .agg(["mean", "std", "count"])
          .reset_index()
          .sort_values(["scenario", "mean"])
    )
    return df, summary



import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run experiments on synthetic or real datasets."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["synthetic_datasets", "real_datasets"],
        help="Dataset to use: synthetic_datasets or real_datasets"
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        choices=[1000, 10000],
        default=None,
        help="Number of samples is 1000 or 10000"
    )

    parser.add_argument(
        "--noise_level",
        type=int,
        choices=[1, 3],
        default=None,
        help="Noise level (only applicable for synthetic_datasets)"
    )

    return parser.parse_args()


def main():

    args = parse_args()

    # Validate argument combination
    if args.dataset == "real_datasets" and args.noise_level is not None:
        raise ValueError(
            "--noise_level is only applicable when dataset=synthetic_datasets"
        )

    if args.dataset == "synthetic_datasets" and args.noise_level is None:
        raise ValueError(
            "--noise_level must be specified when dataset=synthetic_datasets"
        )

    print(f"Dataset: {args.dataset}")
    print(f"Number of samples: {args.n_samples}")
    print(f"Noise level: {args.noise_level}")

    random.seed(0)
    if args.dataset == "synthetic_datasets":
        _, df_summary = run_benchmark(
            scenarios=dataset_dict[args.dataset],
            n_samples=args.n_samples,
            n_features=10,
            noise_scale=args.noise_level)

        df_summary.to_csv('output/'+ args.dataset + '_noise_' +
                      str(int(args.noise_level)) + '_' + str(int(args.n_samples)) + '.csv')
    else:

        _, df_summary = run_benchmark(
            scenarios=dataset_dict[args.dataset])

        df_summary.to_csv('output/' + args.dataset + '.csv')


if __name__ == "__main__":
    main()








