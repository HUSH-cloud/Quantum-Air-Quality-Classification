# Quantum Air Quality Classification | 量子空气质量四分类

A dual-model solution (classical MLP + quantum–classical hybrid QNN) for **4-class air-quality level classification**, built with [pyVQNet](https://vqnet20-tutorial.readthedocs.io/) (OriginQ). Submitted to the **CCF "Sinan Cup" Quantum Computing Programming Competition (司南杯)**.

基于 pyVQNet 的空气质量等级**四分类**双模型方案（经典 MLP + 量子经典混合 QNN），参赛于 **CCF 司南杯量子计算编程比赛**。

---

## Problem | 赛题

Predict the air-quality level — **Good / Moderate / Poor / Hazardous** — from 9 environmental features (Temperature, Humidity, PM2.5, PM10, NO₂, SO₂, CO, proximity to industrial areas, population density).

- Training set: 4000 samples × 10 columns (9 features + 1 label)
- Test set: 1000 samples × 10 columns
- Class imbalance present (Good 1610 / Moderate 1214 / Poor 778 / Hazardous 398)

根据 9 个环境特征预测空气质量等级（优/中/差/有害），数据存在明显类别不平衡。

## Approach | 方法

**1. Classical model — lightweight MLP (`src/classical_mlp.py`)**
- `9 → 8 → 6 → 4` compression architecture, parameter budget ≤ 200
- Weighted cross-entropy loss to handle class imbalance
- Anomaly-sample down-weighting (coeff 0.4), Adam + early stopping

**2. Quantum–classical hybrid QNN (`src/quantum_hybrid_qnn.py`)**
- Dual-channel encoding: **angle encoding** for key features + **amplitude encoding** to compress the rest
- Parameterized quantum circuit with nearest-neighbour entanglement (noise-robust)
- Classical post-processing layer (LeakyReLU) to compensate quantum non-linearity

## Results | 结果

| Model | Accuracy | Notes |
|-------|----------|-------|
| Classical MLP | **94.2%** (paper) / **94.4–94.8%** (reproduced) | F1 ≈ 0.926 |
| Quantum–classical hybrid QNN | **91.6%** (paper) | hybrid encoding gives +5–7% over single encoding |

> ✅ **Reproducibility verified**: the classical MLP was re-run from this repo and reproduces ~94% accuracy. Both scripts run end-to-end on the original pyVQNet environment.

详细方法与实验分析见 [`paper/`](paper/) 中的论文。

## Repository layout | 目录结构
```
.
├── src/
│   ├── classical_mlp.py        # 经典 MLP 模型 (94%)
│   └── quantum_hybrid_qnn.py   # 量子经典混合 QNN (91.6%)
├── data/
│   ├── train_data.csv          # 4000 × 10
│   └── test_data.csv           # 1000 × 10
├── paper/                      # 论文 (PDF)
├── requirements.txt
└── README.md
```

## How to run | 运行方法
```bash
pip install -r requirements.txt        # needs Python 3.10–3.12 for pyvqnet wheels
python src/classical_mlp.py            # run from the repo root (reads data/)
python src/quantum_hybrid_qnn.py
```
> `pyvqnet` (OriginQ VQNet) provides the quantum primitives. Run scripts **from the repository root** so the relative `data/` paths resolve.

## Usage Terms | 使用须知

Copyright © 2026 Zongyuan Ge (葛宗源). **All rights reserved.**

This repository (code, data, and paper) is published for portfolio and reference purposes only.

> **Any commercial use or academic / research use of this project requires prior authorization from the author.**
> 任何将本项目（代码／数据／论文）用于**商业用途或科研用途**的行为，**必须事先获得作者本人授权**。

If you would like to use this work for such purposes, please contact the author by email first to obtain permission.
如需上述用途，请先邮件联系作者获得许可：

📧 **gezongyuan876@gmail.com**
