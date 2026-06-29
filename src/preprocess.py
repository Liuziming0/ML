"""预处理：分钟级用电 → 日级聚合，合并天气，划分 train/test。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import (
    fit_scaler,
    get_project_root,
    load_config,
    resolve_path,
    save_json,
    save_scaler,
)

POWER_COLS_SUM = [
    "global_active_power",
    "global_reactive_power",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
]
POWER_COLS_MEAN = ["voltage", "global_intensity"]
CALENDAR_FEATURES = ["day_of_week", "month", "day_of_year", "is_weekend"]


def load_minute_power(raw_path: Path) -> pd.DataFrame:
    """读取 UCI 分钟级数据（分号分隔，? 为缺失）。"""
    df = pd.read_csv(
        raw_path,
        sep=";",
        na_values="?",
        low_memory=False,
    )
    df.columns = [c.strip().lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(
        df["date"] + " " + df["time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    df = df.drop(columns=["date", "time"])
    numeric_cols = POWER_COLS_SUM + POWER_COLS_MEAN
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def _minute_remainder(df: pd.DataFrame) -> pd.Series:
    return (df["global_active_power"] * 1000.0 / 60.0) - (
        df["sub_metering_1"] + df["sub_metering_2"] + df["sub_metering_3"]
    )


def aggregate_to_daily(minute_df: pd.DataFrame) -> pd.DataFrame:
    """按作业规则聚合为日特征。"""
    df = minute_df.copy()
    df["date"] = df["datetime"].dt.normalize()
    df["sub_metering_remainder"] = _minute_remainder(df)

    agg_map: dict[str, str] = {col: "sum" for col in POWER_COLS_SUM}
    for col in POWER_COLS_MEAN:
        agg_map[col] = "mean"
    agg_map["sub_metering_remainder"] = "sum"

    daily = df.groupby("date", as_index=False).agg(agg_map)
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month
    daily["day_of_year"] = daily["date"].dt.dayofyear
    daily["is_weekend"] = (daily["day_of_week"] >= 5).astype(np.float32)
    return daily.sort_values("date").reset_index(drop=True)


def load_weather_monthly(weather_path: Path, rr_already_mm: bool = True) -> pd.DataFrame | None:
    """读取 download_weather.py 生成的月气象 CSV。"""
    if not weather_path.exists():
        return None

    weather = pd.read_csv(weather_path)
    weather.columns = [c.strip() for c in weather.columns]

    rename = {}
    for col in weather.columns:
        lower = col.lower()
        if lower in ("annee", "year"):
            rename[col] = "year"
        elif lower in ("mois", "month"):
            rename[col] = "month"
    weather = weather.rename(columns=rename)

    if "year" not in weather.columns or "month" not in weather.columns:
        print(f"[警告] 气象文件缺少年/月列，已跳过：{weather_path}")
        return None

    if "RR" in weather.columns:
        weather["RR"] = pd.to_numeric(weather["RR"], errors="coerce")
        if not rr_already_mm:
            weather["RR"] = weather["RR"] / 10.0

    for col in ["NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"]:
        if col in weather.columns:
            weather[col] = pd.to_numeric(weather[col], errors="coerce")

    keep = ["year", "month"] + [
        c for c in ["RR", "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"] if c in weather.columns
    ]
    return weather[keep].drop_duplicates(subset=["year", "month"])


def merge_weather(daily: pd.DataFrame, weather: pd.DataFrame | None) -> pd.DataFrame:
    if weather is None:
        return daily
    out = daily.copy()
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out = out.merge(weather, on=["year", "month"], how="left")
    out = out.drop(columns=["year"])
    for col in ["RR", "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"]:
        if col in out.columns:
            out[col] = out[col].fillna(0.0)
    return out


def fill_missing_daily(daily: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = daily.copy()
    out[feature_cols] = out[feature_cols].ffill().bfill()
    return out


def get_feature_columns(daily: pd.DataFrame, config: dict) -> list[str]:
    base = POWER_COLS_SUM + POWER_COLS_MEAN + ["sub_metering_remainder"] + CALENDAR_FEATURES
    weather_cols = [c for c in config.get("weather_cols", []) if c in daily.columns]
    return [c for c in base + weather_cols if c in daily.columns]


def split_train_test(daily: pd.DataFrame, split_cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按时间切分：date 模式用 test_start_date，ratio 模式用 train_ratio。"""
    method = split_cfg.get("method", "date")
    daily = daily.sort_values("date").reset_index(drop=True)

    if method == "ratio":
        ratio = float(split_cfg.get("train_ratio", 0.75))
        cut = int(len(daily) * ratio)
        train = daily.iloc[:cut].copy()
        test = daily.iloc[cut:].copy()
    else:
        split_dt = pd.Timestamp(split_cfg["test_start_date"])
        train = daily[daily["date"] < split_dt].copy()
        test = daily[daily["date"] >= split_dt].copy()

    return train.reset_index(drop=True), test.reset_index(drop=True)


def run_preprocess(config_path: str | None = None) -> dict:
    config = load_config(config_path)
    paths = config["paths"]

    raw_path = resolve_path(paths["raw_power"])
    weather_path = resolve_path(paths["raw_weather"])
    processed_dir = resolve_path(paths["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"读取分钟级数据：{raw_path}")
    minute_df = load_minute_power(raw_path)
    print(f"  共 {len(minute_df):,} 条记录")

    daily = aggregate_to_daily(minute_df)
    weather = load_weather_monthly(weather_path)
    if weather is not None:
        print(f"合并气象数据（{len(weather)} 个月）")
    else:
        print("未找到气象文件，仅使用电力特征")
    daily = merge_weather(daily, weather)

    feature_cols = get_feature_columns(daily, config)
    daily = fill_missing_daily(daily, feature_cols)

    daily_path = resolve_path(paths["daily_csv"])
    daily.to_csv(daily_path, index=False)
    print(f"已保存日级数据：{daily_path}（{len(daily)} 天）")

    train, test = split_train_test(daily, config["split"])

    min_test_days = config["window"]["input_len"] + config["window"]["long_output_len"]
    if len(test) < min_test_days:
        print(
            f"[警告] 测试集仅 {len(test)} 天，不足长期窗口所需 {min_test_days} 天，"
            f"请考虑提前 test_start_date"
        )

    train_out = resolve_path(paths["train_csv"])
    test_out = resolve_path(paths["test_csv"])
    train.to_csv(train_out, index=False)
    test.to_csv(test_out, index=False)
    print(f"训练集：{len(train)} 天 -> {train_out}")
    print(f"测试集：{len(test)} 天 -> {test_out}")

    train_features = train[feature_cols].to_numpy(dtype=np.float32)
    target_idx = feature_cols.index(config["target_col"])
    feature_scaler, target_scaler = fit_scaler(train_features, target_idx)
    scaler_path = resolve_path(paths["scaler_path"])
    save_scaler(
        feature_scaler,
        target_scaler,
        scaler_path,
        feature_cols,
        config["target_col"],
    )
    print(f"已保存标准化器：{scaler_path}")

    meta = {
        "n_days_total": len(daily),
        "n_train": len(train),
        "n_test": len(test),
        "date_range": [str(daily["date"].min().date()), str(daily["date"].max().date())],
        "train_range": [str(train["date"].min().date()), str(train["date"].max().date())],
        "test_range": [str(test["date"].min().date()), str(test["date"].max().date())],
        "feature_cols": feature_cols,
        "target_col": config["target_col"],
        "has_weather": weather is not None,
        "split_method": config["split"].get("method", "date"),
        "split_config": config["split"],
    }
    meta_path = processed_dir / "preprocess_meta.json"
    save_json(meta, meta_path)
    print(f"已保存元数据：{meta_path}")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="预处理家庭用电数据")
    parser.add_argument("--config", default=None, help="配置文件路径")
    args = parser.parse_args()
    run_preprocess(args.config)


if __name__ == "__main__":
    main()
