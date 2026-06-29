# 家庭电力消耗多变量时间序列预测

2026 年专硕机器学习课程大作业：基于 UCI Household Power Consumption 数据集，实现 LSTM、Transformer 与 CNN-Transformer 三种 Direct Multi-step 预测模型，分别完成短期（90→90 天）与长期（90→365 天）预测任务。

## 环境要求

- Python 3.10+
- 建议使用 GPU（CPU 亦可运行，完整实验耗时较长）

## 1. 克隆仓库并安装依赖

```bash
git clone https://github.com/Liuziming0/ML.git
cd 机器学习

pip install -r requirements.txt
```

## 2. 下载原始数据

### 2.1 电力消耗数据（UCI）

从 UCI 下载分钟级家庭用电数据，解压后将 `household_power_consumption.txt` 放到：

```
datasets/raw/household_power_consumption.txt
```

下载地址：[Household Power Consumption](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)

### 2.2 气象数据（Météo-France）

运行脚本自动下载 92 省（Sceaux 附近）月尺度气象数据，并选取距 Sceaux 最近的观测站：

```bash
python scripts/download_weather.py
```

输出文件：

- `datasets/raw/weather_dept92_1950-2024.csv.gz` — 原始压缩包
- `datasets/raw/weather_monthly.csv` — 预处理后的月气象特征
- `datasets/raw/weather_meta.json` — 站点与字段说明

若跳过此步，预处理仍可运行，但不会合并气象特征。

## 3. 数据预处理

将分钟级数据聚合为日级，合并气象与日历特征，并按时间切分 train/test：

```bash
python scripts/run_preprocess.py
```

默认配置（见 `configs/default.yaml`）：

- 训练集：2006-12-16 至 2008-12-31
- 测试集：2009-01-01 起
- 滑动窗口：输入 90 天，短期输出 90 天，长期输出 365 天

生成文件：

| 路径 | 说明 |
|------|------|
| `datasets/processed/daily.csv` | 日级全量特征 |
| `datasets/train.csv` / `datasets/test.csv` | 训练/测试划分 |
| `datasets/processed/scaler.joblib` | 特征标准化参数 |
| `datasets/processed/preprocess_meta.json` | 预处理元信息 |

可选：验证滑动窗口样本是否正确构建：

```bash
python scripts/verify_dataset.py
```

## 4. 运行实验

完整实验：3 模型 × 2 预测长度 × 5 随机种子 = **30 次训练**。

```bash
python scripts/run_experiments.py
```

常用选项：

```bash
# 快速冒烟测试（1 个 seed、3 个 epoch，用于验证流程）
python scripts/run_experiments.py --quick

# 只跑部分模型
python scripts/run_experiments.py --models lstm transformer

# 指定配置文件
python scripts/run_experiments.py --config configs/default.yaml
```

训练完成后，终端会打印各模型 MSE/MAE 的 mean±std，并在结果目录生成汇总文件与对比图。

---

## 项目结构

```
机器学习/
├── configs/default.yaml       # 路径、划分、窗口、训练超参数
├── datasets/
│   ├── raw/                   # 原始数据（需自行下载/脚本生成）
│   ├── processed/             # 预处理中间产物
│   ├── train.csv / test.csv   # 划分后的日级数据
├── src/
│   ├── preprocess.py          # 数据预处理
│   ├── dataset.py             # 滑动窗口 Dataset
│   ├── train.py / evaluate.py # 训练与评估
│   └── models/                # LSTM、Transformer、CNN-Transformer
├── scripts/
│   ├── download_weather.py
│   ├── run_preprocess.py
│   ├── run_experiments.py
│   └── verify_dataset.py
├── experiments/results/       # 模型权重与实验结果（见下节）
└── docs/实验报告.tex
```

---

## 模型与结果存储位置

所有实验输出默认写入 **`experiments/results/`**（可在 `configs/default.yaml` 的 `paths.results_dir` 修改）。

### 模型权重

每次训练保存在独立子目录，结构如下：

```
experiments/results/
├── lstm_short/seed_42/model.pt
├── lstm_short/seed_123/model.pt
├── ...
├── lstm_long/seed_42/model.pt
├── transformer_short/seed_42/model.pt
├── transformer_long/seed_42/model.pt
├── cnn_transformer_short/seed_42/model.pt
└── cnn_transformer_long/seed_42/model.pt
```

目录命名规则：`{模型名}_{short|long}/seed_{随机种子}/`

- `model.pt`：PyTorch checkpoint，含 `state_dict`、模型名、horizon、特征维度等
- 共 30 个 checkpoint（3 模型 × 2 horizon × 5 seeds）

### 评估结果与图表

| 文件 | 说明 |
|------|------|
| `experiments/results/summary.json` | 全部 30 次 run 的 MSE/MAE 及汇总统计 |
| `experiments/results/summary_table.txt` | 可读的 mean±std 表格 |
| `experiments/results/comparison_short.png` | 三模型短期预测对比图（同一样本） |
| `experiments/results/comparison_long.png` | 三模型长期预测对比图 |
| `experiments/results/{模型}_{horizon}/seed_{seed}/plot_seed{seed}.png` | 单次 run 的预测 vs 真实值曲线 |

报告中的表格与图片均引用上述路径下的汇总文件与对比图。

### 预处理相关（非模型，但训练依赖）

| 文件 | 说明 |
|------|------|
| `datasets/processed/scaler.joblib` | 训练集拟合的标准化器，评估时反归一化用 |
| `datasets/processed/preprocess_meta.json` | 特征列、划分日期等元数据 |

---

## 模型说明

| 模型 | 文件 | 任务 |
|------|------|------|
| LSTM | `src/models/lstm_model.py` | 短期 / 长期（分别训练） |
| Transformer | `src/models/transformer_model.py` | 短期 / 长期 |
| CNN-Transformer | `src/models/cnn_transformer.py` | 短期 / 长期（自研） |

三种模型均采用 Direct Multi-step：一次前向传播直接输出未来 90 或 365 天的 `global_active_power` 预测序列。
