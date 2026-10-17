# MNIST 手写数字识别（PyTorch CNN，从零手写）

用 PyTorch 从零实现的卷积神经网络手写数字识别。数据集是 MNIST，模型是本项目手写的三层卷积网络，
训练、评估、真实图片推理、实验管理全部自己实现，没有用现成的训练框架封装。

## 结果

| 配置 | 验证集 | 测试集（10,000 张） | 错例 |
|---|---|---|---|
| 5 轮，无增强 | 98.98% | 98.98% | 102 |
| 5 轮，数据增强 | 99.20% | 99.16% | 84 |
| **15 轮，数据增强** | **99.58%** | **99.51%** | **49** |

测试集全程只在最后使用一次。验证集从训练集里固定切出 5,000 张（`manual_seed(42)`），
不参与参数更新。

## 模型结构

```
输入 (1, 28, 28)
  Conv2d(1, 32, 3, padding=1) → BatchNorm → ReLU → MaxPool(2)    28→14
  Conv2d(32, 64, 3, padding=1) → BatchNorm → ReLU → MaxPool(2)   14→7
  Conv2d(64, 128, 3, padding=1) → BatchNorm → ReLU
  Flatten → (128, 7, 7) → 6272
  Linear(6272, 128) → ReLU → Dropout(0.3)
  Linear(128, 10)
```

参数量 897,354。

## 文件

| 文件 | 职责 |
|---|---|
| `model.py` | 网络结构，dropout 比例由构造参数传入 |
| `train.py` | 单次训练：五步循环 + 每轮验证 + 峰值权重归档 |
| `experiment.py` | 实验记录：写 `experiments.csv`、归档权重、打印汇总表 |
| `run_experiments.py` | 批量跑多组超参，改 `CONFIGS` 列表即可 |
| `plot_history.py` | 单次训练曲线 `curve.png`、多组对比曲线 `compare.png` |
| `my_eval.py` | 测试集评估、混淆矩阵、错例图 |
| `my_train.py` | 第一版数据加载脚本（保留作记录） |
| `任务清单.md` | 项目分关实现的任务与验收标准 |

## 环境

```
pip install -r requirements.txt
```

PyTorch 有 GPU 版会自动用 CUDA，没有就退回 CPU（MNIST 在 CPU 上也能跑，15 轮约 10 分钟）。

## 怎么跑

```bash
# 训练 15 轮（数据不存在会自动下载）
python train.py

# 批量跑多组超参
python run_experiments.py

# 画曲线
python plot_history.py

# 测试集评估 + 混淆矩阵 + 错例
python my_eval.py
```

产物都在 `outputs/`：

```
outputs/
├── model.pt                验证集上最好的一版权重
├── history.csv             每次训练的逐轮指标
├── experiments.csv         跨实验对比表
├── archive/                每个峰值的权重归档
├── curve.png               单次训练曲线
├── compare.png             多组对比曲线
├── confusion_matrix.png    混淆矩阵
└── misclassified.png       25 张错例
```

## 实验结论

同一份验证集、固定随机种子的对照实验：

| 改了什么 | 验证集 | 判断 |
|---|---|---|
| 基准（5 轮，无增强） | 98.98% | 基线 |
| 加 Dropout 0.3 | 99.02% | 噪声内，无差别 |
| 加 Dropout 0.5 | 98.88% | 欠拟合，变差 |
| 加数据增强 | 99.08% ~ 99.20% | **有效** |
| 5 轮 → 15 轮 | 99.58% | **有效** |
| 换优化器（SGD / RMSprop） | 99.36% ~ 99.44% | 噪声内，无差别 |

同配置三个随机种子：99.10% / 99.10% / 99.28%，波动 0.18 个点。
**所以小于 0.2 个点的差异不能当成改进。**

结论：在这个任务上，数据增强和训练轮数是主要杠杆，Dropout 和优化器选择都不重要。

## 错例分析

49 个错例里最多的是 `7 → 2`（横笔长、竖笔短）、`9 → 4`、`5 ↔ 3`。
其中相当一部分人类也会犹豫，属于标签本身有歧义的样本。

## 数据

MNIST 原始数据由脚本自动下载到 `data/mnist/`，不进版本库。
