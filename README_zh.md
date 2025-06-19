# REISD: Detecting LLM-Generated Text via Iterative Semantic Difference

本仓库包含论文 **“REISD: Detecting LLM-Generated Text via Iterative Semantic Difference”** 的全部代码与数据。

Read this in [English](README.md) or [Chinese](README_zh.md) 
## 🚀 功能特色

* **支持中英文**：可处理 **中文** 与 **英文** 文本。
* **迭代重写**：调用 Ollama API（如 `qwen2.5:7b`）对每条输入进行两轮重写。
* **语义分析**：计算连续重写文本之间的 **BERTScore** 与 **MoverScore**。
* **特征提取**：从重写链中提取语义变化特征。
* **分类判断**：使用预训练的 **XGBoost** 模型判断文本为人类撰写或 LLM 生成。
* **高度可配置**：输入路径、使用模型、重写轮数等参数均支持命令行配置。

## 🧩 工作流程

1. **文本重写**
   每条输入文本通过指定的大模型（LLM）进行两次重写，形成一条三段式的重写链：原文 → 第一次重写 → 第二次重写。

2. **相似度计算**
   对每对相邻文本，计算以下语义相似度指标：

   * **BERTScore**
   * **MoverScore**

3. **特征提取**
   提取以下特征用于分类：

   * 两次重写之间 BERTScore 的差值
   * 两次重写之间 MoverScore 的差值
   * 原始与第一次重写之间的 BERTScore 和 MoverScore

4. **模型预测**
   将上述特征输入到预训练的 XGBoost 分类器中，判断该文本是否为 AI 所生成。

## ✅ 域内实验结果

| 方法             | CLTS  | XSUM  | SQuAD | GovReport | Billsum | HC3   |
| -------------- | ----- | ----- | ----- | --------- | ------- | ----- |
| Log-Likelihood | 55.41 | 55.07 | 48.28 | 36.17     | 34.15   | 51.79 |
| Rank           | 52.21 | 56.61 | 41.57 | 42.44     | 37.03   | 48.31 |
| Log-Rank       | 50.37 | 56.42 | 47.64 | 36.93     | 35.79   | 49.98 |
| Entropy        | 59.35 | 52.31 | 43.78 | 31.54     | 42.64   | 54.64 |
| Fast-detectGPT | 49.40 | 40.05 | 43.40 | 51.30     | 55.40   | 74.46 |
| BERTScore      | 82.71 | 41.78 | 72.22 | 53.24     | 50.42   | 61.17 |
| Raidar         | 84.38 | 97.02 | 98.58 | 98.73     | 96.84   | 59.24 |
| **REISD (本文)** | 85.93 | 95.23 | 96.58 | 95.34     | 93.23   | 81.20 |

## 🌐 域外实验结果

| 方法        | CLTS（Claude 3.7Sonnet） | CLTS（Gemini 2.0） | XSUM（Claude 3.7Sonnet） | XSUM（Gemini 2.0） |
| --------- | ---------------------- | ---------------- | ---------------------- | ---------------- |
| BERTScore | 63.12                  | 66.42            | 74.16                  | 66.90            |
| Raidar    | 66.00                  | 62.32            | 84.22                  | 66.02            |
| **REISD** | 76.24                  | 76.64            | 83.96                  | 83.35            |

## ⚙️ 安装说明

```bash
git clone https://github.com/qsyqkq/REISD.git
cd REISD
pip install -r requirements.txt
```

然后下载所需的模型文件：

```bash
python -c "from transformers import AutoTokenizer, AutoModel; \
AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext'); \
AutoModel.from_pretrained('hfl/chinese-roberta-wwm-ext')"
```

## 🛠 依赖项

* `transformers`
* `torch`
* `scikit-learn`
* `xgboost`
* `numpy==2.1.3`
* `ollama`（Ollama Python 客户端）

## 🎯 使用方式

### 训练与验证

* 使用 **train.py** 可复现论文中的训练结果。
* **正样本**：人类文本 + 重写后的版本。
* **负样本**：LLM 生成文本 + 重写后的版本。
* 所有用于训练验证的数据保存在 `/output` 文件夹中。
* 可使用 `exclude_features` 参数控制特征消融。

### 启动检测器

```bash
python REISD.py \
  --text_input ./example/human_300.txt \
  --ai_model qwen2.5:7b
```

* `--text_input`：支持 `.txt` 文件（每行一条样本）或直接输入文本字符串。
* `--ai_model`：Ollama 模型名称，例如 `qwen2.5:7b`、`llama3` 等。

## 📂 项目结构

```
REISD/
├── REISD.py              # 主检测脚本
├── train.py              # 模型训练与验证
├── xgboost_model.json    # 预训练分类器模型
├── example/
│   └── human_300.txt     # 示例输入
├── output/               # 验证数据集
├── requirements.txt
└── README.md
```

## 📜 许可证

本项目基于 MIT 协议开源，详情见 `LICENSE` 文件。

---

如有任何问题或建议，欢迎提交 issue 或 pull request！

---
