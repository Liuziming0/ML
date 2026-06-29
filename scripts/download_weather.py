"""下载 Météo-France 92 省月尺度气象数据（Sceaux 附近）。"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import get_project_root

# UCI 数据集位于 Sceaux
SCEAUX_LAT, SCEAUX_LON = 48.776, 2.290

MENSQ_URL = (
    "https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/"
    "BASE/MENS/MENSQ_92_previous-1950-2024.csv.gz"
)
WEATHER_COLS = ["RR", "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"]


def _distance(lat: float, lon: float) -> float:
    return math.hypot(lat - SCEAUX_LAT, lon - SCEAUX_LON)


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"正在下载 {url}")
    with urllib.request.urlopen(url, timeout=180) as resp:
        data = resp.read()
    dest.write_bytes(data)
    print(f"已保存 {dest}（{len(data):,} 字节）")


def pick_station(raw: pd.DataFrame, start_ym: int, end_ym: int) -> tuple[int, str]:
    """选取时段内距 Sceaux 最近的观测站。"""
    raw = raw.copy()
    raw["AAAAMM"] = raw["AAAAMM"].astype(int)
    in_period = raw[(raw["AAAAMM"] >= start_ym) & (raw["AAAAMM"] <= end_ym)]
    if in_period.empty:
        raise ValueError(f"{start_ym}–{end_ym} 时段内无气象记录")

    stations = in_period[["NUM_POSTE", "NOM_USUEL", "LAT", "LON"]].drop_duplicates()
    stations = stations.copy()
    stations["dist"] = stations.apply(lambda r: _distance(r["LAT"], r["LON"]), axis=1)
    best = stations.sort_values("dist").iloc[0]
    return int(best["NUM_POSTE"]), str(best["NOM_USUEL"])


def build_monthly_weather(raw: pd.DataFrame, station_id: int) -> pd.DataFrame:
    station = raw[raw["NUM_POSTE"] == station_id].copy()
    station["AAAAMM"] = station["AAAAMM"].astype(int)
    station["year"] = station["AAAAMM"] // 100
    station["month"] = station["AAAAMM"] % 100

    out = station[["year", "month"]].copy()
    for col in WEATHER_COLS:
        out[col] = pd.to_numeric(station[col], errors="coerce")

    out["RR"] = out["RR"] / 10.0  # 原始单位为 0.1 mm
    return out.sort_values(["year", "month"]).reset_index(drop=True)


def run_download(
    output_csv: Path | None = None,
    raw_gz: Path | None = None,
    start_ym: int = 200612,
    end_ym: int = 201011,
) -> dict:
    project_root = get_project_root()
    raw_gz = raw_gz or project_root / "datasets" / "raw" / "weather_dept92_1950-2024.csv.gz"
    output_csv = output_csv or project_root / "datasets" / "raw" / "weather_monthly.csv"
    meta_path = project_root / "datasets" / "raw" / "weather_meta.json"

    if not raw_gz.exists():
        download_file(MENSQ_URL, raw_gz)

    raw = pd.read_csv(raw_gz, sep=";", low_memory=False)
    station_id, station_name = pick_station(raw, start_ym, end_ym)
    weather = build_monthly_weather(raw, station_id)

    weather = weather[
        (weather["year"] * 100 + weather["month"] >= start_ym)
        & (weather["year"] * 100 + weather["month"] <= end_ym)
    ].reset_index(drop=True)

    if weather.empty:
        raise ValueError(f"站点 {station_name}（{station_id}）在 {start_ym}–{end_ym} 无数据")

    weather.to_csv(output_csv, index=False)
    meta = {
        "source_url": MENSQ_URL,
        "department": "92 (Hauts-de-Seine)",
        "household_location": "Sceaux, France (UCI dataset)",
        "selected_station_id": station_id,
        "selected_station_name": station_name,
        "period": [start_ym, end_ym],
        "columns": ["year", "month"] + WEATHER_COLS,
        "rr_unit_note": "RR converted from 1/10 mm to mm",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"选用站点：{station_name}（{station_id}）")
    print(f"已保存气象 CSV：{output_csv}（{len(weather)} 个月）")
    print(f"已保存元数据：{meta_path}")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 Sceaux 附近月气象数据")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()
    output = Path(args.output) if args.output else None
    run_download(output_csv=output)


if __name__ == "__main__":
    main()
