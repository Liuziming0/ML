"""检查滑动窗口 Dataset 构建是否正确。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import Horizon, build_dataset, describe_dataset

SPLIT_CN = {"train": "训练集", "test": "测试集"}
HORIZON_CN = {"short": "短期", "long": "长期"}


def main() -> None:
    for split in ("train", "test"):
        for horizon in (Horizon.SHORT, Horizon.LONG):
            ds = build_dataset(split, horizon)
            info = describe_dataset(ds)
            print(f"\n[{SPLIT_CN[split]} | {HORIZON_CN[horizon.value]}]")
            for k, v in info.items():
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
