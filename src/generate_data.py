import json
import os
import platform
import sys
import numpy as np
import pandas as pd
try:
    import yaml
except ImportError:
    yaml = None


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        if yaml is not None:
            return yaml.safe_load(f)
    return {
        "project_name": "5G_Project_sim",
        "seeds": [42, 1024, 2026],
        "data": {"n_steps": 2400, "freq": "5min", "start_date": "2026-01-01 00:00:00", "outlier_cap_percentile": 99.5, "split_ratios": {"train": 0.70, "calibration": 0.15, "test": 0.15}},
        "features": {"lookback_lags": 15, "rolling_window": 12, "horizon": 5},
        "models": {"random_forest": {"n_estimators": 60, "max_depth": 8, "min_samples_leaf": 3}, "xgboost": {"n_estimators": 80, "learning_rate": 0.08, "max_depth": 5, "subsample": 0.9}, "lstm": {"hidden_size": 32, "num_layers": 1, "epochs": 40, "batch_size": 64, "lr": 0.01}},
        "safety_margins": {"fixed_mbps": 35.0, "error_based_quantile": 0.90, "conservative_quantile": 0.95, "conservative_z_multiplier": 1.96},
        "allocation_and_qos": {"link_capacity_C_mbps": 1000.0, "urllc_priority": True, "base_propagation_delay_ms": 4.0, "congestion_delay_scale_ms": 45.0},
    }


def ensure_directories() -> None:
    for folder in [
        "data/raw",
        "data/processed",
        "src",
        "notebooks",
        "results/logs",
        "results/tables",
        "results/figures",
    ]:
        os.makedirs(folder, exist_ok=True)


def log_runtime_environment(seed: int, output_path: str = "results/logs/runtime_env.json") -> dict:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    versions = {}
    for pkg_name, import_name in [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
        ("xgboost", "xgboost"),
        ("torch", "torch"),
        ("scipy", "scipy"),
        ("matplotlib", "matplotlib"),
    ]:
        try:
            mod = __import__(import_name)
            versions[pkg_name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[pkg_name] = "unavailable"

    env_info = {
        "python_version": sys.version.split()[0],
        "os_platform": f"{platform.system()} {platform.release()}",
        "processor": platform.processor() or "unknown",
        "primary_seed": seed,
        "packages": versions,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=4)
    return env_info


def generate_telemetry(config: dict, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_steps = int(config["data"]["n_steps"])
    freq = config["data"]["freq"]
    start_date = config["data"]["start_date"]

    ts = pd.date_range(start=start_date, periods=n_steps, freq=freq)
    t = np.arange(n_steps, dtype=float)
    steps_per_day = 288.0

    diurnal = 0.5 + 0.5 * np.sin(2.0 * np.pi * (t % steps_per_day) / steps_per_day - np.pi / 2.0)
    wknd_factor = np.where(ts.dayofweek >= 5, 0.82, 1.05)

    ar = np.zeros(n_steps, dtype=float)
    eps = rng.normal(0.0, 12.0, size=n_steps)
    for i in range(1, n_steps):
        ar[i] = 0.65 * ar[i - 1] + eps[i]

    bursts = np.where(rng.random(n_steps) < 0.025, rng.uniform(60.0, 180.0, size=n_steps), 0.0)

    embb = np.clip(260.0 * diurnal * wknd_factor + ar * 1.4 + bursts, 30.0, 750.0)
    mmtc = np.clip(95.0 + 25.0 * np.cos(2.0 * np.pi * t / steps_per_day) + rng.normal(0.0, 8.0, size=n_steps), 15.0, 220.0)
    urllc = np.clip(55.0 + 15.0 * diurnal + rng.normal(0.0, 6.0, size=n_steps) + np.where(bursts > 0, 25.0, 0.0), 10.0, 180.0)

    df = pd.DataFrame({
        "timestamp": ts,
        "eMBB_mbps": np.round(embb, 2),
        "mMTC_mbps": np.round(mmtc, 2),
        "URLLC_mbps": np.round(urllc, 2),
    })
    df["total_demand_mbps"] = np.round(df[["eMBB_mbps", "mMTC_mbps", "URLLC_mbps"]].sum(axis=1), 2)
    return df


def clean_slice_table(raw_df: pd.DataFrame, cap_percentile: float = 99.5) -> pd.DataFrame:
    df = raw_df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    slice_cols = ["eMBB_mbps", "mMTC_mbps", "URLLC_mbps"]
    df[slice_cols] = df[slice_cols].ffill().bfill()
    for col in slice_cols:
        cap = float(np.percentile(df[col].values, cap_percentile))
        df[col] = np.clip(df[col].values, 0.0, cap)
    df["total_demand_mbps"] = np.round(df[slice_cols].sum(axis=1), 2)
    return df


if __name__ == "__main__":
    ensure_directories()
    cfg = load_config("config.yaml")
    seed0 = int(cfg["seeds"][0])
    log_runtime_environment(seed0, "results/logs/runtime_env.json")
    raw_df = generate_telemetry(cfg, seed=seed0)
    raw_df.to_csv("data/raw/traffic_telemetry.csv", index=False)
    clean_df = clean_slice_table(raw_df, cap_percentile=float(cfg["data"]["outlier_cap_percentile"]))
    clean_df.to_csv("data/processed/clean_slice_table.csv", index=False)
    print(f"Generated raw ({len(raw_df)} rows) and cleaned ({len(clean_df)} rows) telemetry.")
