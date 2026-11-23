import json
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
try:
    from xgboost import XGBRegressor
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor as _GBR
    class XGBRegressor(_GBR):
        def __init__(self, n_estimators=80, learning_rate=0.08, max_depth=5, subsample=0.9, random_state=42, n_jobs=-1):
            self.n_jobs = n_jobs
            super().__init__(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, subsample=subsample, random_state=random_state)

from generate_data import clean_slice_table, ensure_directories, generate_telemetry, load_config, log_runtime_environment


def build_features(df: pd.DataFrame, lookback_lags: int = 15, rolling_window: int = 12, horizon: int = 5):
    s = df['total_demand_mbps'].values.astype(float)
    hrs = df['timestamp'].dt.hour.values + df['timestamp'].dt.minute.values / 60.0
    wknd = (df['timestamp'].dt.dayofweek.values >= 5).astype(float)

    X_rows, y_vals, t_vals = [], [], []
    for idx in range(max(lookback_lags, rolling_window), len(s) - horizon):
        lags = s[idx - lookback_lags + 1 : idx + 1][::-1]
        win = s[idx - rolling_window + 1 : idx + 1]
        feat = np.concatenate([
            lags,
            [
                np.mean(win),
                np.std(win),
                np.sin(2.0 * np.pi * hrs[idx] / 24.0),
                np.cos(2.0 * np.pi * hrs[idx] / 24.0),
                wknd[idx],
            ],
        ])
        X_rows.append(feat)
        y_vals.append(s[idx + horizon])
        t_vals.append(df['timestamp'].iloc[idx + horizon])

    cols = [f'lag_{i}' for i in range(1, lookback_lags + 1)] + [
        'rolling_mean', 'rolling_std', 'sin_tod', 'cos_tod', 'weekend_flag'
    ]
    return pd.DataFrame(X_rows, columns=cols), np.array(y_vals, dtype=float), pd.Series(t_vals)


def chronological_split(X: pd.DataFrame, y: np.ndarray, t_series: pd.Series, ratios: dict):
    n = len(X)
    i_tr = int(n * ratios['train'])
    i_ca = i_tr + int(n * ratios['calibration'])

    X_tr, y_tr, t_tr = X.iloc[:i_tr], y[:i_tr], t_series.iloc[:i_tr]
    X_ca, y_ca, t_ca = X.iloc[i_tr:i_ca], y[i_tr:i_ca], t_series.iloc[i_tr:i_ca]
    X_te, y_te, t_te = X.iloc[i_ca:], y[i_ca:], t_series.iloc[i_ca:]

    assert t_tr.max() < t_ca.min(), 'Leakage error: train overlaps calibration!'
    assert t_ca.max() < t_te.min(), 'Leakage error: calibration overlaps test!'
    return (X_tr, y_tr, t_tr), (X_ca, y_ca, t_ca), (X_te, y_te, t_te)


class TrafficLSTM(nn.Module):
    def __init__(self, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def train_and_predict_models(X_tr, y_tr, X_ca, y_ca, X_te, y_te, cfg: dict, seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    results = []
    cal_preds = {}
    test_preds = {}

    cal_preds['Baseline_LastValue'] = X_ca['lag_1'].values
    test_preds['Baseline_LastValue'] = X_te['lag_1'].values
    cal_preds['Baseline_MovingAvg'] = X_ca['rolling_mean'].values
    test_preds['Baseline_MovingAvg'] = X_te['rolling_mean'].values

    rf_cfg = cfg['models']['random_forest']
    t0 = time.perf_counter()
    rf = RandomForestRegressor(
        n_estimators=int(rf_cfg['n_estimators']),
        max_depth=int(rf_cfg['max_depth']),
        min_samples_leaf=int(rf_cfg['min_samples_leaf']),
        random_state=seed,
        n_jobs=-1,
    ).fit(X_tr, y_tr)
    rf_sec = round(time.perf_counter() - t0, 4)
    cal_preds['RandomForest'] = rf.predict(X_ca)
    test_preds['RandomForest'] = rf.predict(X_te)

    shuf_pred = RandomForestRegressor(n_estimators=25, max_depth=6, random_state=seed, n_jobs=-1).fit(
        X_tr, rng.permutation(y_tr)
    ).predict(X_te)

    xgb_cfg = cfg['models']['xgboost']
    t0 = time.perf_counter()
    xgb = XGBRegressor(
        n_estimators=int(xgb_cfg['n_estimators']),
        learning_rate=float(xgb_cfg['learning_rate']),
        max_depth=int(xgb_cfg['max_depth']),
        subsample=float(xgb_cfg['subsample']),
        random_state=seed,
        n_jobs=-1,
    ).fit(X_tr, y_tr)
    xgb_sec = round(time.perf_counter() - t0, 4)
    cal_preds['XGBoost'] = xgb.predict(X_ca)
    test_preds['XGBoost'] = xgb.predict(X_te)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    f_mean = X_tr.mean(axis=0).values.astype(np.float32)
    f_std = np.where(X_tr.std(axis=0).values == 0, 1.0, X_tr.std(axis=0).values).astype(np.float32)

    X_tr_s = ((X_tr.values.astype(np.float32) - f_mean) / f_std)[:, :, None]
    X_ca_s = ((X_ca.values.astype(np.float32) - f_mean) / f_std)[:, :, None]
    X_te_s = ((X_te.values.astype(np.float32) - f_mean) / f_std)[:, :, None]

    y_mean = float(np.mean(y_tr))
    y_std = float(np.std(y_tr)) if float(np.std(y_tr)) > 0 else 1.0
    y_tr_norm = ((y_tr - y_mean) / y_std).astype(np.float32).reshape(-1, 1)

    lstm_cfg = cfg['models']['lstm']
    loader = DataLoader(
        TensorDataset(torch.tensor(X_tr_s), torch.tensor(y_tr_norm)),
        batch_size=int(lstm_cfg['batch_size']),
        shuffle=False,
    )
    lstm_net = TrafficLSTM(int(lstm_cfg['hidden_size']), int(lstm_cfg['num_layers'])).to(device)
    opt = torch.optim.Adam(lstm_net.parameters(), lr=float(lstm_cfg['lr']))
    loss_fn = nn.L1Loss()

    best_cal_mae = float('inf')
    best_weights = None
    x_ca_t = torch.tensor(X_ca_s, device=device)

    t0 = time.perf_counter()
    for _ in range(int(lstm_cfg['epochs'])):
        lstm_net.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(lstm_net(xb), yb)
            loss.backward()
            opt.step()

        lstm_net.eval()
        with torch.no_grad():
            ca_p = lstm_net(x_ca_t).cpu().numpy().ravel() * y_std + y_mean
            ca_mae = float(mean_absolute_error(y_ca, ca_p))
            if ca_mae < best_cal_mae:
                best_cal_mae = ca_mae
                best_weights = {k: v.cpu().clone() for k, v in lstm_net.state_dict().items()}

    lstm_sec = round(time.perf_counter() - t0, 4)
    lstm_net.load_state_dict(best_weights)
    lstm_net.eval()
    with torch.no_grad():
        cal_preds['LSTM'] = lstm_net(x_ca_t).cpu().numpy().ravel() * y_std + y_mean
        test_preds['LSTM'] = lstm_net(torch.tensor(X_te_s, device=device)).cpu().numpy().ravel() * y_std + y_mean

    timings = {
        'Baseline_LastValue': 0.0,
        'Baseline_MovingAvg': 0.0,
        'RandomForest': rf_sec,
        'XGBoost': xgb_sec,
        'LSTM': lstm_sec,
    }

    for name in ['Baseline_LastValue', 'Baseline_MovingAvg', 'RandomForest', 'XGBoost', 'LSTM']:
        c_mae = float(mean_absolute_error(y_ca, cal_preds[name]))
        t_mae = float(mean_absolute_error(y_te, test_preds[name]))
        t_rmse = float(root_mean_squared_error(y_te, test_preds[name]))
        t_r2 = float(r2_score(y_te, test_preds[name]))
        results.append({
            'model': name,
            'is_ml_candidate': name in ['RandomForest', 'XGBoost', 'LSTM'],
            'cal_mae': round(c_mae, 3),
            'test_mae': round(t_mae, 3),
            'test_rmse': round(t_rmse, 3),
            'test_r2': round(t_r2, 4),
            'train_sec': timings[name],
        })

    shuf_mae = round(float(mean_absolute_error(y_te, shuf_pred)), 3)
    return pd.DataFrame(results), cal_preds, test_preds, shuf_mae


def compute_safety_margins(y_ca: np.ndarray, cal_pred: np.ndarray, y_te: np.ndarray, test_pred: np.ndarray, cfg: dict):
    sm_cfg = cfg['safety_margins']
    residuals = y_ca - cal_pred

    fixed_buf = float(sm_cfg['fixed_mbps'])
    err_q = float(sm_cfg['error_based_quantile'])
    error_based_buf = float(np.quantile(residuals, err_q))

    cons_q = float(sm_cfg['conservative_quantile'])
    z_mult = float(sm_cfg['conservative_z_multiplier'])
    gaussian_upper = float(np.mean(residuals) + z_mult * np.std(residuals))
    conformal_upper = float(np.quantile(residuals, cons_q))
    conservative_buf = max(gaussian_upper, conformal_upper)

    margin_dict = {
        'No_Margin': 0.0,
        'Fixed_Margin': round(fixed_buf, 3),
        'Error_Based_Margin': round(error_based_buf, 3),
        'Conservative_Margin': round(conservative_buf, 3),
    }

    rows = []
    for rule_name, buf in margin_dict.items():
        emp_cov = float(np.mean(y_te <= (test_pred + buf)))
        rows.append({
            'margin_rule': rule_name,
            'margin_buffer_mbps': buf,
            'empirical_test_coverage_pct': round(emp_cov * 100.0, 2),
        })
    return margin_dict, pd.DataFrame(rows)


def distribute_bandwidth_and_evaluate_qos(y_te: np.ndarray, test_pred: np.ndarray, margin_mbps: float, slice_tail: pd.DataFrame, cfg: dict) -> dict:
    qos_cfg = cfg['allocation_and_qos']
    cap_C = float(qos_cfg['link_capacity_C_mbps'])
    base_delay = float(qos_cfg['base_propagation_delay_ms'])
    cong_scale = float(qos_cfg['congestion_delay_scale_ms'])

    tot_hist = np.maximum(slice_tail['total_demand_mbps'].values, 1e-6)
    u_share = slice_tail['URLLC_mbps'].values / tot_hist
    e_share = slice_tail['eMBB_mbps'].values / tot_hist
    m_share = slice_tail['mMTC_mbps'].values / tot_hist

    requested_total = np.clip(test_pred + margin_mbps, 0.0, cap_C)

    throughputs, shortages, wastages, latencies_ms, packet_losses_pct = [], [], [], [], []
    urllc_shortages, embb_shortages, mmtc_shortages = [], [], []

    for i in range(len(y_te)):
        actual_tot = float(y_te[i])
        u_act = actual_tot * u_share[i]
        e_act = actual_tot * e_share[i]
        m_act = actual_tot * m_share[i]

        req_tot = float(requested_total[i])
        u_req = req_tot * u_share[i]
        e_req = req_tot * e_share[i]
        m_req = req_tot * m_share[i]

        u_alloc = min(u_req, cap_C)
        rem_cap = max(0.0, cap_C - u_alloc)
        other_req = e_req + m_req
        if other_req > 0.0 and rem_cap > 0.0:
            scale = min(1.0, rem_cap / other_req)
            e_alloc = e_req * scale
            m_alloc = m_req * scale
        else:
            e_alloc, m_alloc = 0.0, 0.0

        total_alloc = u_alloc + e_alloc + m_alloc

        u_served = min(u_act, u_alloc)
        e_served = min(e_act, e_alloc)
        m_served = min(m_act, m_alloc)
        served_tot = u_served + e_served + m_served

        shortage_i = max(0.0, actual_tot - served_tot)
        wastage_i = max(0.0, total_alloc - served_tot)
        pkt_loss_i = (shortage_i / actual_tot * 100.0) if actual_tot > 0 else 0.0

        load_ratio = actual_tot / max(total_alloc, 1.0)
        latency_i = base_delay + cong_scale * (max(0.0, load_ratio - 0.75) ** 1.5)

        throughputs.append(served_tot)
        shortages.append(shortage_i)
        wastages.append(wastage_i)
        latencies_ms.append(latency_i)
        packet_losses_pct.append(pkt_loss_i)
        urllc_shortages.append(max(0.0, u_act - u_served))
        embb_shortages.append(max(0.0, e_act - e_served))
        mmtc_shortages.append(max(0.0, m_act - m_served))

    lat_arr = np.array(latencies_ms)
    jitter_ms = float(np.mean(np.abs(np.diff(lat_arr)))) if len(lat_arr) > 1 else 0.0

    return {
        'margin_buffer_mbps': round(margin_mbps, 3),
        'mean_throughput_mbps': round(float(np.mean(throughputs)), 3),
        'mean_latency_ms': round(float(np.mean(lat_arr)), 3),
        'jitter_ms': round(jitter_ms, 3),
        'mean_packet_loss_pct': round(float(np.mean(packet_losses_pct)), 3),
        'mean_bandwidth_shortage_mbps': round(float(np.mean(shortages)), 3),
        'mean_bandwidth_wastage_mbps': round(float(np.mean(wastages)), 3),
        'mean_urllc_shortage_mbps': round(float(np.mean(urllc_shortages)), 3),
        'mean_embb_shortage_mbps': round(float(np.mean(embb_shortages)), 3),
        'mean_mmtc_shortage_mbps': round(float(np.mean(mmtc_shortages)), 3),
    }


def generate_all_figures(t_te, y_te, test_preds: dict, best_model_name: str, margin_dict: dict, qos_df: pd.DataFrame):
    os.makedirs('results/figures', exist_ok=True)
    win = min(288, len(y_te))
    plt.figure(figsize=(11, 4.5))
    plt.plot(t_te.iloc[:win], y_te[:win], label='Actual Demand', color='black', linewidth=1.8)
    for m_name, color in [('RandomForest', '#1f77b4'), ('XGBoost', '#ff7f0e'), ('LSTM', '#2ca02c')]:
        plt.plot(t_te.iloc[:win], test_preds[m_name][:win], label=m_name, linestyle='--', alpha=0.85, color=color)
    plt.title('5G Bandwidth Demand Forecast Comparison (24-Hour Window)')
    plt.xlabel('Timestamp')
    plt.ylabel('Bandwidth Demand (Mbps)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/figures/fig1_model_forecast_comparison.png', dpi=150)
    plt.close()

    best_pred = test_preds[best_model_name][:win]
    plt.figure(figsize=(11, 4.5))
    plt.plot(t_te.iloc[:win], y_te[:win], label='Actual Demand', color='black', linewidth=1.6)
    plt.plot(t_te.iloc[:win], best_pred, label=f'Best Model ({best_model_name})', color='#1f77b4', linewidth=1.4)
    for rule, style in [('Fixed_Margin', ':'), ('Error_Based_Margin', '--'), ('Conservative_Margin', '-.')]:
        buf = margin_dict[rule]
        plt.plot(t_te.iloc[:win], best_pred + buf, label=f'{rule} (+{buf:.1f} Mbps)', linestyle=style)
    plt.title(f'Safety Margin Envelopes Applied to Best Model ({best_model_name})')
    plt.xlabel('Timestamp')
    plt.ylabel('Allocated Bandwidth Target (Mbps)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/figures/fig2_safety_margin_envelopes.png', dpi=150)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(qos_df))
    w = 0.35
    ax1.bar(x - w / 2, qos_df['mean_bandwidth_shortage_mbps'], width=w, label='Bandwidth Shortage (Mbps)', color='#d62728')
    ax1.bar(x + w / 2, qos_df['mean_bandwidth_wastage_mbps'], width=w, label='Bandwidth Wastage (Mbps)', color='#1f77b4')
    ax1.set_xticks(x)
    ax1.set_xticklabels(qos_df['margin_rule'], rotation=15)
    ax1.set_ylabel('Bandwidth (Mbps)')
    ax1.set_title('QoS Evaluation Across Safety Margin Rules')
    ax2 = ax1.twinx()
    ax2.plot(x, qos_df['mean_latency_ms'], color='#2ca02c', marker='o', linewidth=2, label='Mean Latency (ms)')
    ax2.set_ylabel('Mean Latency (ms)', color='#2ca02c')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    plt.tight_layout()
    plt.savefig('results/figures/fig3_qos_margin_tradeoff.png', dpi=150)
    plt.close()





def load_source_telemetry(cfg: dict, seed: int) -> tuple[pd.DataFrame, str]:
    data_cfg = cfg.get("data", {})
    real_cfg = data_cfg.get("real_dataset", {}) or {}
    enabled = bool(real_cfg.get("enabled", False))
    csv_path = real_cfg.get("csv_path")

    if enabled and csv_path and os.path.exists(csv_path):
        col_map = real_cfg.get("column_mapping", {}) or {}
        df = pd.read_csv(csv_path)

        # Reverse map user column names -> canonical names
        rename_dict = {}
        for canonical_col, source_col in col_map.items():
            if source_col and source_col in df.columns and source_col != canonical_col:
                rename_dict[source_col] = canonical_col
        if rename_dict:
            df = df.rename(columns=rename_dict)

        if "timestamp" not in df.columns:
            raise ValueError(f"Real dataset '{csv_path}' is missing a 'timestamp' column or valid mapping.")
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        shares = real_cfg.get("default_slice_shares", {}) or {}
        e_share = float(shares.get("eMBB", 0.65))
        m_share = float(shares.get("mMTC", 0.20))
        u_share = float(shares.get("URLLC", 0.15))
        share_sum = max(e_share + m_share + u_share, 1e-9)
        e_share, m_share, u_share = e_share / share_sum, m_share / share_sum, u_share / share_sum

        slice_cols = ["eMBB_mbps", "mMTC_mbps", "URLLC_mbps"]
        has_all_slices = all(c in df.columns for c in slice_cols)

        if has_all_slices:
            for c in slice_cols:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            if "total_demand_mbps" not in df.columns:
                df["total_demand_mbps"] = df[slice_cols].sum(axis=1)
            else:
                df["total_demand_mbps"] = pd.to_numeric(df["total_demand_mbps"], errors="coerce")
        elif "total_demand_mbps" in df.columns:
            tot = pd.to_numeric(df["total_demand_mbps"], errors="coerce")
            df["eMBB_mbps"] = tot * e_share
            df["mMTC_mbps"] = tot * m_share
            df["URLLC_mbps"] = tot * u_share
            df["total_demand_mbps"] = tot
        else:
            raise ValueError(
                f"Real dataset '{csv_path}' must contain slice columns {slice_cols} or 'total_demand_mbps'."
            )

        keep_cols = ["timestamp", "eMBB_mbps", "mMTC_mbps", "URLLC_mbps", "total_demand_mbps"]
        return df[keep_cols].copy(), f"real:{csv_path}"

    # Fallback / default: synthetic dataset
    raw_path = "data/raw/traffic_telemetry.csv"
    if not enabled and os.path.exists(raw_path):
        raw_df = pd.read_csv(raw_path, parse_dates=["timestamp"])
        return raw_df, f"synthetic_cached:{raw_path}"

    raw_df = generate_telemetry(cfg, seed=seed)
    if not enabled:
        raw_df.to_csv(raw_path, index=False)
    return raw_df, f"synthetic_generated(seed={seed})"


def run_across_seeds(cfg: dict):
    all_model_rows = []
    all_margin_rows = []
    all_qos_rows = []
    last_bundle = None

    for s in [int(x) for x in cfg["seeds"]]:
        raw_df, _ = load_source_telemetry(cfg, seed=s)
        clean_df = clean_slice_table(raw_df, cap_percentile=float(cfg["data"]["outlier_cap_percentile"]))

        X, y, t_ser = build_features(
            clean_df,
            lookback_lags=int(cfg["features"]["lookback_lags"]),
            rolling_window=int(cfg["features"]["rolling_window"]),
            horizon=int(cfg["features"]["horizon"]),
        )
        (X_tr, y_tr, _), (X_ca, y_ca, _), (X_te, y_te, t_te) = chronological_split(
            X, y, t_ser, cfg["data"]["split_ratios"]
        )

        metrics_df, cal_preds, test_preds, shuf_mae = train_and_predict_models(
            X_tr, y_tr, X_ca, y_ca, X_te, y_te, cfg, s
        )
        metrics_df["seed"] = s
        all_model_rows.append(metrics_df)

        ml_only = metrics_df[metrics_df["is_ml_candidate"]].sort_values("cal_mae")
        best_model_name = str(ml_only.iloc[0]["model"])

        margin_dict, margin_df = compute_safety_margins(
            y_ca, cal_preds[best_model_name], y_te, test_preds[best_model_name], cfg
        )
        margin_df["seed"] = s
        margin_df["best_model"] = best_model_name
        all_margin_rows.append(margin_df)

        slice_tail = clean_df.iloc[-len(y_te) :].reset_index(drop=True)
        for rule_name, buf in margin_dict.items():
            q_res = distribute_bandwidth_and_evaluate_qos(
                y_te, test_preds[best_model_name], buf, slice_tail, cfg
            )
            all_qos_rows.append({
                "seed": s,
                "best_model": best_model_name,
                "margin_rule": rule_name,
                **q_res,
            })

        if s == int(cfg["seeds"][0]):
            last_bundle = (clean_df, t_te, y_te, test_preds, best_model_name, margin_dict, shuf_mae, metrics_df, margin_df)

    raw_models_df = pd.concat(all_model_rows, ignore_index=True)
    raw_margins_df = pd.concat(all_margin_rows, ignore_index=True)
    raw_qos_df = pd.DataFrame(all_qos_rows)

    raw_models_df.to_csv("results/tables/multi_seed_model_metrics_raw.csv", index=False)
    raw_margins_df.to_csv("results/tables/multi_seed_safety_margins_raw.csv", index=False)
    raw_qos_df.to_csv("results/tables/multi_seed_qos_raw.csv", index=False)

    # Mean +/- Std summary for Chapter 5 tables
    model_agg = (
        raw_models_df.groupby(["model", "is_ml_candidate"])[
            ["cal_mae", "test_mae", "test_rmse", "test_r2", "train_sec"]
        ]
        .agg(["mean", "std"])
    )
    model_agg.columns = [f"{col}_{stat}" for col, stat in model_agg.columns]
    model_agg = model_agg.reset_index()
    model_agg = model_agg.round(4)
    model_agg.to_csv("results/tables/model_metrics_mean_std.csv", index=False)

    margin_agg = (
        raw_margins_df.groupby("margin_rule")[
            ["margin_buffer_mbps", "empirical_test_coverage_pct"]
        ]
        .agg(["mean", "std"])
    )
    margin_agg.columns = [f"{col}_{stat}" for col, stat in margin_agg.columns]
    margin_agg = margin_agg.reset_index()
    margin_agg = margin_agg.round(3)
    margin_agg.to_csv("results/tables/safety_margins_mean_std.csv", index=False)

    qos_metric_cols = [
        "margin_buffer_mbps",
        "mean_throughput_mbps",
        "mean_latency_ms",
        "jitter_ms",
        "mean_packet_loss_pct",
        "mean_bandwidth_shortage_mbps",
        "mean_bandwidth_wastage_mbps",
        "mean_urllc_shortage_mbps",
        "mean_embb_shortage_mbps",
        "mean_mmtc_shortage_mbps",
    ]
    qos_agg = raw_qos_df.groupby("margin_rule")[qos_metric_cols].agg(["mean", "std"])
    qos_agg.columns = [f"{col}_{stat}" for col, stat in qos_agg.columns]
    qos_agg = qos_agg.reset_index()
    qos_agg = qos_agg.round(3)
    qos_agg.to_csv("results/tables/qos_metrics_mean_std.csv", index=False)

    # Figure 4: Multi-Seed Mean Test MAE with Error Bars
    plt.figure(figsize=(8.5, 4.2))
    m_names = model_agg["model"].values
    m_means = model_agg["test_mae_mean"].values
    m_stds = model_agg["test_mae_std"].fillna(0.0).values
    plt.bar(m_names, m_means, yerr=m_stds, capsize=5, color=["#8c96c6", "#9ebcda", "#1f77b4", "#ff7f0e", "#2ca02c"], alpha=0.9)
    plt.xticks(rotation=15)
    plt.ylabel("Test MAE (Mbps)")
    plt.title(f"Multi-Seed Test MAE (Mean +/- Std across {len(cfg['seeds'])} Seeds)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/figures/fig4_multiseed_model_mae.png", dpi=150)
    plt.close()

    # Figure 5: Multi-Seed Mean Shortage vs Wastage Across Margin Rules
    plt.figure(figsize=(8.5, 4.2))
    r_names = qos_agg["margin_rule"].values
    s_means = qos_agg["mean_bandwidth_shortage_mbps_mean"].values
    w_means = qos_agg["mean_bandwidth_wastage_mbps_mean"].values
    x = np.arange(len(r_names))
    bw = 0.36
    plt.bar(x - bw / 2, s_means, width=bw, label="Mean Shortage (Mbps)", color="#d62728")
    plt.bar(x + bw / 2, w_means, width=bw, label="Mean Wastage (Mbps)", color="#1f77b4")
    plt.xticks(x, r_names, rotation=15)
    plt.ylabel("Bandwidth (Mbps)")
    plt.title(f"Multi-Seed Mean Shortage vs. Wastage ({len(cfg['seeds'])} Seeds)")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/figures/fig5_multiseed_shortage_wastage.png", dpi=150)
    plt.close()

    return model_agg, margin_agg, qos_agg, last_bundle


def main():
    ensure_directories()
    cfg = load_config('config.yaml')
    seed = int(cfg['seeds'][0])
    log_runtime_environment(seed, 'results/logs/runtime_env.json')

    raw_df, data_source_label = load_source_telemetry(cfg, seed=seed)
    print(f'Active dataset source: {data_source_label}')

    clean_df = clean_slice_table(raw_df, cap_percentile=float(cfg['data']['outlier_cap_percentile']))
    clean_df.to_csv('data/processed/clean_slice_table.csv', index=False)

    X, y, t_ser = build_features(
        clean_df,
        lookback_lags=int(cfg['features']['lookback_lags']),
        rolling_window=int(cfg['features']['rolling_window']),
        horizon=int(cfg['features']['horizon']),
    )
    (X_tr, y_tr, _), (X_ca, y_ca, _), (X_te, y_te, t_te) = chronological_split(X, y, t_ser, cfg['data']['split_ratios'])

    metrics_df, cal_preds, test_preds, shuf_mae = train_and_predict_models(X_tr, y_tr, X_ca, y_ca, X_te, y_te, cfg, seed)
    metrics_df.to_csv('results/tables/forecast_model_comparison.csv', index=False)

    ml_only = metrics_df[metrics_df['is_ml_candidate']].sort_values('cal_mae')
    best_model_name = str(ml_only.iloc[0]['model'])
    margin_dict, margin_df = compute_safety_margins(y_ca, cal_preds[best_model_name], y_te, test_preds[best_model_name], cfg)
    margin_df.to_csv('results/tables/safety_margin_comparison.csv', index=False)

    slice_tail = clean_df.iloc[-len(y_te) :].reset_index(drop=True)
    qos_rows = []
    for rule_name, buf in margin_dict.items():
        qos_metrics = distribute_bandwidth_and_evaluate_qos(y_te, test_preds[best_model_name], buf, slice_tail, cfg)
        qos_rows.append({'best_model': best_model_name, 'margin_rule': rule_name, **qos_metrics})
    qos_df = pd.DataFrame(qos_rows)
    qos_df.to_csv('results/tables/qos_evaluation_summary.csv', index=False)

    pred_out = pd.DataFrame({
        'timestamp': t_te.values,
        'actual_mbps': np.round(y_te, 2),
        'RandomForest_mbps': np.round(test_preds['RandomForest'], 2),
        'XGBoost_mbps': np.round(test_preds['XGBoost'], 2),
        'LSTM_mbps': np.round(test_preds['LSTM'], 2),
        'BestModel_ErrorBased_Alloc_mbps': np.round(
            np.clip(test_preds[best_model_name] + margin_dict['Error_Based_Margin'], 0.0, float(cfg['allocation_and_qos']['link_capacity_C_mbps'])),
            2,
        ),
    })
    pred_out.to_csv('results/tables/test_predictions_and_allocations.csv', index=False)

    generate_all_figures(t_te, y_te, test_preds, best_model_name, margin_dict, qos_df)

    model_agg, margin_agg, qos_agg, _ = run_across_seeds(cfg)

    summary_json = {
        'dataset_source': data_source_label,
        'selected_best_ml_model': best_model_name,
        'shuffled_target_sanity_mae': shuf_mae,
        'safety_margins_mbps': margin_dict,
        'qos_by_margin_rule': qos_rows,
        'multi_seed_count': len(cfg['seeds']),
        'multi_seed_model_summary': model_agg.to_dict(orient='records'),
        'multi_seed_qos_summary': qos_agg.to_dict(orient='records'),
    }
    with open('results/tables/final_application_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary_json, f, indent=4)

    print('Forecasting & QoS application complete.')
    print(f'Automatically selected best ML model: {best_model_name}')
    print(metrics_df[['model', 'cal_mae', 'test_mae', 'test_rmse', 'test_r2']].to_string(index=False))
    print('\nQoS Summary:')
    print(qos_df[['margin_rule', 'mean_throughput_mbps', 'mean_latency_ms', 'jitter_ms', 'mean_packet_loss_pct', 'mean_bandwidth_shortage_mbps', 'mean_bandwidth_wastage_mbps']].to_string(index=False))


if __name__ == '__main__':
    main()