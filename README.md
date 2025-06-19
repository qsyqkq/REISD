# REISD: Detecting LLM-Generated Text via Iterative Semantic Difference

This repository contains the code and data for the paper **“REISD: Detecting LLM-Generated Text via Iterative Semantic Difference.”**

Read this in [English](README.md) or [Chinese](README_zh.md) 
## 🚀 Features

* **Bilingual Support**: Handles both **Chinese** and **English** texts.
* **Iterative Rewriting**: Uses the Ollama API (e.g., `qwen2.5:7b`) to produce two rounds of rewrites per input.
* **Semantic Analysis**: Computes **BERTScore** and **MoverScore** between consecutive rewrites.
* **Feature Extraction**: Derives semantic-shift features from the rewrite chain.
* **Classification**: Applies a pre-trained **XGBoost** model to label text as AI- or human-generated.
* **Configurable**: All key parameters—input, model—are exposed via command‑line flags.

## 🧩 How It Works

1. **Text Rewriting**
   Each input is passed through the chosen LLM twice, generating a chain of three texts: original → first rewrite → second rewrite.

2. **Similarity Computation**
   For each adjacent pair in the rewrite chain, we compute:

   * **BERTScore**
   * **MoverScore**

3. **Feature Derivation**
   We extract the following features:

   * Difference of BERTScore between rewrite steps
   * Difference of MoverScore between rewrite steps
   * Initial BERTScore and initial MoverScore

4. **Prediction**
   The resulting feature vector is fed into a pre-trained XGBoost classifier to decide if the text is AI‑generated.

## ✅ In‑Domain Experiments

| Method           | CLTS  | XSUM  | SQuAD | GovReport | Billsum | HC3   |
| ---------------- | ----- | ----- | ----- | --------- | ------- | ----- |
| Log-Likelihood   | 55.41 | 55.07 | 48.28 | 36.17     | 34.15   | 51.79 |
| Rank             | 52.21 | 56.61 | 41.57 | 42.44     | 37.03   | 48.31 |
| Log-Rank         | 50.37 | 56.42 | 47.64 | 36.93     | 35.79   | 49.98 |
| Entropy          | 59.35 | 52.31 | 43.78 | 31.54     | 42.64   | 54.64 |
| Fast-detectGPT   | 49.40 | 40.05 | 43.40 | 51.30     | 55.40   | 74.46 |
| BERTScore        | 82.71 | 41.78 | 72.22 | 53.24     | 50.42   | 61.17 |
| Raidar           | 84.38 | 97.02 | 98.58 | 98.73     | 96.84   | 59.24 |
| **REISD (Ours)** | 85.93 | 95.23 | 96.58 | 95.34     | 93.23   | 81.20 |

## 🌐 Out‑of‑Domain Experiments

| Method    | CLTS (Claude 3.7Sonnet) | CLTS (Gemini 2.0) | XSUM (Claude 3.7Sonnet) | XSUM (Gemini 2.0) |
| --------- | ----------------------- | ----------------- | ----------------------- | ----------------- |
| BERTScore | 63.12                   | 66.42             | 74.16                   | 66.90             |
| Raidar    | 66.00                   | 62.32             | 84.22                   | 66.02             |
| **REISD (Ours)** | 76.24              | 76.64         | 83.96               | 83.35         |

## ⚙️ Installation

```bash
git clone https://github.com/qsyqkq/REISD.git
cd REISD
pip install -r requirements.txt
```

Then download the model:

```bash
python -c "from transformers import AutoTokenizer, AutoModel; \
AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext'); \
AutoModel.from_pretrained('hfl/chinese-roberta-wwm-ext')"
```

## 🛠 Dependencies

* `transformers`
* `torch`
* `scikit-learn`
* `xgboost`
* `numpy==2.1.3`
* `ollama` (Python API client)

## 🎯 Usage

### Train & Validate

* **train.py**: Script for reproducing the paper’s results.
* **Positive files**: Human text + its LLM rewrites.
* **Negative files**: LLM-generated text + its rewrites.
* All data used for paper stored in `/output`.
* Use the `exclude_features` flag to ablate specific features.

### Run the Detector

```bash
python REISD.py \
  --text_input ./example/human_300.txt \
  --ai_model qwen2.5:7b
```

* `--text_input`: Path to a `.txt` file (one sample per line) or a direct string.
* `--ai_model`: Ollama model identifier (e.g., `qwen2.5:7b`, `llama3`).



## 📂 Repository Structure

```
REISD/
├── REISD.py              # Main detection pipeline
├── train.py              # Training/validation script
├── xgboost_model.json    # Pretrained XGBoost model
├── example/
│   └── human_300.txt     # Sample input
├── output/               # Validation dataset
├── requirements.txt
└── README.md
```

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.

---

Feel free to open issues or submit pull requests for improvements!

