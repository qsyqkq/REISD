# -*- coding: utf-8 -*-
import os
import json
import xgboost as xgb
import re
import time
import torch
import numpy as np
from ollama import chat, ChatResponse
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from mover_score import compute_moverscore  # Imported from external library

# =============== Configuration Parameters ===============
TEXT_INPUT = "./example/human_300.txt"  # Input text or file path
AI_MODEL = 'qwen2.5:7b'  # Ollama model name
NUM_REWRITES = 2
XGB_MODEL_PATH = './xgboost_model.json'
CACHE_DIR = "./models"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# =============== Model Initialization ===============
# Chinese and English models and tokenizers
ZH_MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
en_tokenizer = AutoTokenizer.from_pretrained(ZH_MODEL_NAME, cache_dir=CACHE_DIR)
en_model = AutoModel.from_pretrained(ZH_MODEL_NAME, cache_dir=CACHE_DIR).to(DEVICE)
zh_tokenizer = AutoTokenizer.from_pretrained(ZH_MODEL_NAME, cache_dir=CACHE_DIR)
zh_model = AutoModel.from_pretrained(ZH_MODEL_NAME, cache_dir=CACHE_DIR).to(DEVICE)

# =============== Core Functionality ===============
def is_chinese(text):
    """Check if the text contains Chinese characters"""
    return any('\u4e00' <= char <= '\u9fff' for char in text)

def detect_language(text):
    """Detect the language of the input text"""
    return 'zh' if any('\u4e00' <= char <= '\u9fff' for char in text) else 'en'

def get_prompt_template(lang):
    """Get prompt templates based on the detected language"""
    templates = {
        'zh': {
            'system': "你是一个专业的中文文本重写助手，擅长在保持原意的前提下优化文本的表达方式。请遵守以下规则：\n1. 只输出改写后的段落内容\n2. 保持与原文相近的句子长度\n3. 使用符合中文表达习惯的句式",
            'instruction': "请根据以下要求改写文本：\n1. 保持专业但自然的语气\n2. 优化句子结构\n3. 保留所有关键信息\n原文："
        },
        'en': {
            'system': "You are a professional English writing assistant. Follow these rules:\n1. Output only the rephrased text\n2. Maintain similar sentence length to the original\n3. Use natural English expressions",
            'instruction': "Please rephrase the following text according to these guidelines:\n1. Maintain a professional tone\n2. Improve sentence structure\n3. Preserve all key information\nOriginal text: "
        }
    }
    return templates.get(lang, templates['en'])

def call_ollama_api(text, lang, model_name):
    """Call Ollama API to rewrite the text"""
    try:
        prompt = get_prompt_template(lang)
        messages = [
            {'role': 'system', 'content': prompt['system']},
            {'role': 'user', 'content': f"{prompt['instruction']}{text}"}
        ]
        response = chat(model=model_name, messages=messages)
        content = response.message.content.strip() if isinstance(response, ChatResponse) else response['message']['content'].strip()
        return re.sub(r'[\x00-\x1F\x7F]', '', content)  # Remove control characters
    except Exception as e:
        print(f"API call error: {str(e)}")
        return text

def rewrite_chain(text, ai_model, num_rewrites):
    """Generate a chain of rewritten texts"""
    lang = 'zh' if is_chinese(text) else 'en'
    chain = [text]
    current = text

    for _ in range(num_rewrites):
        rewritten = call_ollama_api(current, lang, ai_model)
        chain.append(rewritten)
        current = rewritten

    return chain

def get_embeddings(text):
    """Generate embedding vectors for the given text"""
    if not text.strip():
        return [], np.array([]).reshape(0, 0)

    lang = detect_language(text)
    tokenizer, model = (en_tokenizer, en_model) if lang == 'en' else (zh_tokenizer, zh_model)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.squeeze(0).cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze(0))
    return tokens, embeddings

def compute_bertscore(tokens1, embeddings1, tokens2, embeddings2):
    """
    Compute BERTScore (Precision part) - without custom attention weights
    Formula: S_bertscore = 1/N2 * Σ(max(cos(e_i^(2), e_j^(1))))
    """
    if not tokens1 or not tokens2 or embeddings1.shape[0] == 0 or embeddings2.shape[0] == 0:
        return 0.0

    embeddings1 = np.array(embeddings1)
    embeddings2 = np.array(embeddings2)

    if embeddings1.size == 0 or embeddings2.size == 0:
        return 0.0

    similarity_matrix = cosine_similarity(embeddings2, embeddings1)
    max_similarities = similarity_matrix.max(axis=1)

    return np.mean(max_similarities)

def process_similarity(segment_data):
    """Process similarity calculation, only compute BERTScore and MoverScore"""
    processed_entries = [{
        "text": e.get('text', ''),
        "label": e.get('label', 'unknown'),
        "bertscore": 0.0,
        "mover_score": 0.0
    } for e in segment_data]

    for i in range(len(segment_data)):
        if i == 0 or processed_entries[i]["label"] == "human":
            continue

        prev_text = segment_data[i - 1].get('text', '')
        curr_text = segment_data[i].get('text', '')

        # Get embeddings
        prev_tokens, prev_emb = get_embeddings(prev_text)
        curr_tokens, curr_emb = get_embeddings(curr_text)

        # BERTScore
        if prev_tokens and curr_tokens and prev_emb.size > 0 and curr_emb.size > 0:
            bs_val = round(compute_bertscore(
                prev_tokens, prev_emb,
                curr_tokens, curr_emb
            ), 4)
            processed_entries[i]["bertscore"] = float(bs_val)
        else:
            processed_entries[i]["bertscore"] = 0.0  # Default if inputs are insufficient

        # MoverScore
        if prev_tokens and curr_tokens and prev_emb.size > 0 and curr_emb.size > 0:
            try:
                src_weights = np.ones(len(prev_tokens), dtype=np.float32)
                trg_weights = np.ones(len(curr_tokens), dtype=np.float32)

                mover_val = compute_moverscore(
                    prev_tokens, prev_emb,
                    curr_tokens, curr_emb,
                    src_weights,
                    trg_weights,
                    device=DEVICE
                )
                processed_entries[i]["mover_score"] = round(float(mover_val), 4)
            except Exception as e:
                print(f"Error computing MoverScore for entry {i} (texts: '{prev_text[:20]}...', '{curr_text[:20]}...'): {e}")
                processed_entries[i]["mover_score"] = 0.0
        else:
            processed_entries[i]["mover_score"] = 0.0  # Default if inputs are insufficient

    return processed_entries

def calculate_features(chain):
    """Calculate feature vector based on the rewriting chain"""
    if len(chain) < 3:
        return [0.0] * 6

    processed = process_similarity([{'text': t, 'label': 'ai' if i > 0 else 'human'} for i, t in enumerate(chain)])

    bert_scores = [p['bertscore'] for p in processed if p['label'] != 'human']
    mover_scores = [p['mover_score'] for p in processed if p['label'] != 'human']

    if len(bert_scores) < 2 or len(mover_scores) < 2:
        return [0.0] * 6

    bert_prev, bert_curr = bert_scores[:2]
    mover_prev, mover_curr = mover_scores[:2]

    return [
        bert_curr - bert_prev,
        bert_curr / bert_prev if bert_prev != 0 else 0.0,
        mover_curr - mover_prev,
        mover_curr / mover_prev if mover_prev != 0 else 0.0,
        bert_prev,
        mover_prev
    ]

def predict_text(text, model, ai_model, num_rewrites):
    """Execute the full prediction pipeline for a given text"""
    print(f"\nInput Text: {text[:200]}...")  # Limit display length

    # Generate rewriting chain
    chain = rewrite_chain(text, ai_model, num_rewrites)

    print("Rewriting Chain:")
    for idx, txt in enumerate(chain):
        print(f"  [{idx}] {txt[:100]}..." if len(txt) > 100 else f"  [{idx}] {txt}")

    # Extract features
    features = calculate_features(chain)
    print(f"\nFeature Vector: {features}")

    # Make prediction
    pred_prob = model.predict_proba([features])[0][1]
    pred_label = model.predict([features])[0]
    print(f"\nPrediction Result: Label={pred_label}, Confidence={pred_prob:.4f}")

# =============== Main Program Entry ===============
def main():
    """Main function"""
    # Load XGBoost model
    if not os.path.exists(XGB_MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {XGB_MODEL_PATH}")

    model = xgb.XGBClassifier()
    model.load_model(XGB_MODEL_PATH)

    # Process input
    if os.path.isfile(TEXT_INPUT):
        with open(TEXT_INPUT, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        for idx, line in enumerate(lines):
            print(f"\n========= Sample {idx + 1} =========")
            predict_text(line, model, AI_MODEL, NUM_REWRITES)
    else:
        predict_text(TEXT_INPUT, model, AI_MODEL, NUM_REWRITES)

if __name__ == '__main__':
    main()