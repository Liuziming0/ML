"""运行数据预处理。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocess import run_preprocess

if __name__ == "__main__":
    run_preprocess()
