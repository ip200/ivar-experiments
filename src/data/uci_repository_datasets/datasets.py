import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ----------------- Internal util ----------------- #

def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"Downloading {url} → {dest}")
        urllib.request.urlretrieve(url, dest)
    return dest


# ----------------- URLs ----------------- #

UCI_WINE_RED_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"
    "winequality-red.csv"
)
UCI_WINE_WHITE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"
    "winequality-white.csv"
)

UCI_AIRFOIL_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/"
    "airfoil_self_noise.dat"
)

UCI_CONCRETE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/"
    "Concrete_Data.xls"
)


# Climate prediction dataset (bias correction of NWP temperature forecast, UCI id=514)
UCI_GRID_ID = 471 # used via ucimlrepo

# electrical_grid_stability_simulated_data = fetch_ucirepo(id=471) 


# ----------------- Main unified loader ----------------- #

from typing import Union

def load_dataset(
    name: str,
    *,
    split: bool = True,
    test_size: float = 0.2,
    random_state: int = 0,
    cache_dir: Union[str, Path] = "./uci_cache",
):
    """
    Unified loader for several standard regression datasets.

    Parameters
    ----------
    name : {"wine_red", "wine_white", "wine_both",
            "airfoil", "concrete", "boston",
            "climate_bias"}
        Which dataset to load.
    split : bool, default=True
        If True, return (X_train, X_test, y_train, y_test, meta).
        If False, return (X, y, meta).
    test_size : float, default=0.2
        Fraction of data to use as test set when split=True.
    random_state : int, default=0
        Random seed used for train/test split.
    cache_dir : str or Path, default="./uci_cache"
        Directory where downloaded files will be cached.

    Returns
    -------
    If split=True:
        X_train, X_test, y_train, y_test, meta
    If split=False:
        X, y, meta
    """
    name = name.lower()
    cache_dir = Path(cache_dir)

    # ---- Airfoil Self-Noise ----
    if name == "airfoil":
        path = _download(UCI_AIRFOIL_URL, cache_dir / "airfoil_self_noise.dat")
        cols = [
            "Freq",
            "Angle_of_attack",
            "Chord_length",
            "Free_stream_velocity",
            "Suction_side_displacement_thickness",
            "Scaled_sound_pressure",
        ]
        df = pd.read_csv(path, sep="\t", header=None, names=cols)

        X = df.drop(columns=["Scaled_sound_pressure"]).to_numpy(dtype=float)
        y = df["Scaled_sound_pressure"].to_numpy(dtype=float)

        meta = {
            "name": "Airfoil Self-Noise (UCI)",
            "n_samples": len(df),
            "n_features": X.shape[1],
        }

    # ---- Climate prediction: bias correction of NWP temperature forecast ----
    elif name == "climate_bias":
        module_path = Path(__file__).parent
        data = pd.read_csv(module_path / 'Bias_correction_ucl.csv').dropna()
        features = ['Present_Tmax', 'Present_Tmin', 'LDAPS_RHmin',
       'LDAPS_RHmax', 'LDAPS_Tmax_lapse', 'LDAPS_Tmin_lapse', 'LDAPS_WS',
       'LDAPS_LH', 'LDAPS_CC1', 'LDAPS_CC2', 'LDAPS_CC3', 'LDAPS_CC4',
       'LDAPS_PPT1', 'LDAPS_PPT2', 'LDAPS_PPT3', 'LDAPS_PPT4', 'lat', 'lon',
       'DEM', 'Slope', 'Solar radiation']
        targets = ['Next_Tmax', 'Next_Tmin']
        X_df = data[features]
        y_df = data[targets]
        
        X = X_df.to_numpy(dtype=float)

        # Use next-day maximum temperature as the primary regression target
        if "Next_Tmax" in y_df.columns:
            y = y_df["Next_Tmax"].to_numpy(dtype=float)
        else:
            # fallback: take the first column
            y = y_df.iloc[:, 0].to_numpy(dtype=float)

        meta = {
            "name": (
                "Bias correction of numerical prediction model "
                "temperature forecast (UCI)"
            ),
            "target": "Next_Tmax",
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
        }
    elif name == "electricity":
        # uses the official ucimlrepo helper suggested on the UCI page
        from ucimlrepo import fetch_ucirepo

        ds = fetch_ucirepo(id=UCI_GRID_ID)
        X=ds.data.features.values
        y=ds.data.targets.iloc[:,0].values
        meta={}
 
        
    # ---- Wine Quality (Red) ----
    elif name == "wine_red":
        path = _download(UCI_WINE_RED_URL, cache_dir / "winequality-red.csv")
        df = pd.read_csv(path, sep=";")
        X = df.drop(columns=["quality"]).to_numpy(dtype=float)
        y = df["quality"].to_numpy(dtype=float)
        meta = {
            "name": "Wine Quality (Red) (UCI)",
            "n_samples": len(df),
            "n_features": X.shape[1],
        }

    # ---- Wine Quality (White) ----
    elif name == "wine_white":
        path = _download(UCI_WINE_WHITE_URL, cache_dir / "winequality-white.csv")
        df = pd.read_csv(path, sep=";")
        X = df.drop(columns=["quality"]).to_numpy(dtype=float)
        y = df["quality"].to_numpy(dtype=float)
        meta = {
            "name": "Wine Quality (White) (UCI)",
            "n_samples": len(df),
            "n_features": X.shape[1],
        }

    else:
        raise ValueError(
            f"Unknown dataset name '{name}'. "
            "Supported: 'electricity', 'airfoil', 'climate_bias', 'wine_red', 'wine_white'."
        )

    # ---- Return with or without split ----
    if not split:
        return X, y, meta

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    return X_train, X_test, y_train, y_test, meta

