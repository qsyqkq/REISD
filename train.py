import json
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd
from sklearn.utils import resample
plt.rcParams['font.family'] = 'Microsoft YaHei'

def calculate_features(current, next1, next2):
    """Calculate specified 6 feature vectors based on three consecutive data segments"""
    bert_prev = next1.get('bertscore', 0.0)
    bert_curr = next2.get('bertscore', 0.0)
    bert_diff = bert_curr - bert_prev
    bert_ratio = bert_curr / bert_prev if bert_prev != 0 else 0.0
    mover_prev = next1.get('mover_score', 0.0)
    mover_curr = next2.get('mover_score', 0.0)
    mover_diff = mover_curr - mover_prev
    mover_ratio = mover_curr / mover_prev if mover_prev != 0 else 0.0
    return [
        bert_diff, bert_ratio, mover_diff, mover_ratio,
        bert_prev, mover_prev,
    ]

def load_data(file_path, example_type=None):
    """Load data from single file, supports specifying sample type"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    features, labels, texts = [], [], []
    for seg in data:
        seg_data = seg['data']
        if example_type in (None, 'positive') and len(seg_data) >= 3:
            try:
                t0, t1, t2 = seg_data[:3]
                features.append(calculate_features(t0, t1, t2))
                labels.append(1)
                texts.append([t['text'] for t in (t0, t1, t2)])
            except Exception as e:
                print(f"Positive example processing error: {e} in file {file_path}")
        if example_type in (None, 'negative'):
            required_length = 3 if example_type == 'negative' else 4
            if len(seg_data) >= required_length:
                try:
                    start_idx = 0 if example_type == 'negative' else 1
                    t1_neg, t2_neg, t3_neg = seg_data[start_idx:start_idx + 3]
                    features.append(calculate_features(t1_neg, t2_neg, t3_neg))
                    labels.append(0)
                    texts.append([t['text'] for t in (t1_neg, t2_neg, t3_neg)])
                except Exception as e:
                    print(f"Negative example processing error: {e} in file {file_path}")
    # Ensure features is a 2D numpy array even if empty
    if not features:
        # Determine number of columns from dummy call if features is empty
        # This ensures (0, num_cols) shape for empty features
        num_cols = len(calculate_features({}, {}, {}))
        return np.empty((0, num_cols)), np.array(labels), texts
    return np.array(features), np.array(labels), texts

def load_datasets(file_paths, example_type=None):
    """Load multiple dataset files, supports specifying sample type"""
    features_list, labels_list, texts_list = [], [], []
    for path in file_paths:
        try:
            f, l, t = load_data(path, example_type=example_type)
            if len(f) > 0:  # Check if any features were loaded
                features_list.append(f)
                labels_list.append(l)
                texts_list.extend(t)
        except Exception as e:
            print(f"File loading failed {path}: {e}")
    if not features_list:
        num_cols = len(calculate_features({}, {}, {}))  # Default num_cols if all files are empty/fail
        return np.empty((0, num_cols)), np.array([]), []
    return np.concatenate(features_list), np.concatenate(labels_list), texts_list

def visualize_features(X_train, y_train, X_val, y_val, feature_names):
    selected_features_to_plot = [  # Renamed to avoid conflict
        'bart_diff', 'bart_prev', 'bart_curr', 'bert_diff', 'bert_ratio', 'bert_prev',
        'mover_prev', 'mover_diff', 'mover_ratio', 't0_mle', 'trend_mle_diff',
        'mover_curr', 'bert_curr',
    ]
    # Filter selected_features_to_plot to those actually present in current feature_names
    valid_features_to_plot = [f for f in selected_features_to_plot if f in feature_names]
    if not valid_features_to_plot:
        print("No valid features for visualization (selected features don't intersect with current model features or current model has no features).")
        return
    palette = {'LLM': '#FF7F0E', 'Human': '#1F77B4'}
    y_train_mapped = np.where(y_train == 0, 'LLM', 'Human')
    y_val_mapped = np.where(y_val == 0, 'LLM', 'Human')
    for feature_name_iter in valid_features_to_plot:
        try:
            feature_idx = feature_names.index(feature_name_iter)
        except ValueError:
            print(f"Warning: Feature '{feature_name_iter}' was selected in visualize_features but not found in current model feature list. Skipping.")
            continue  # Should not happen if valid_features_to_plot derived from feature_names
        # Training set
        if X_train.ndim == 2 and X_train.shape[1] > feature_idx and X_train.shape[0] > 0:
            train_df = pd.DataFrame({'value': X_train[:, feature_idx], 'label': y_train_mapped})
            plt.figure(figsize=(8, 5))
            sns.histplot(data=train_df, x='value', hue='label', palette=palette, kde=True, stat='density',
                         common_norm=False)
            plt.title(f"Training Set - {feature_name_iter}")
            plt.tight_layout()
            plt.show()
            plt.close()
        elif X_train.shape[0] > 0:  # Has rows but not enough columns or not 2D
            print(f"Warning: Training set data format incorrect or insufficient columns for feature '{feature_name_iter}' (index {feature_idx}). Shape: {X_train.shape}")
        # Validation set
        if X_val.ndim == 2 and X_val.shape[1] > feature_idx and X_val.shape[0] > 0:
            val_df = pd.DataFrame({'value': X_val[:, feature_idx], 'label': y_val_mapped})
            plt.figure(figsize=(8, 5))
            sns.histplot(data=val_df, x='value', hue='label', palette=palette, kde=True, stat='density',
                         common_norm=False)
            plt.title(f"Validation Set - {feature_name_iter}")
            plt.tight_layout()
            plt.show()
            plt.close()
        elif X_val.shape[0] > 0:  # Has rows but not enough columns or not 2D
            print(f"Warning: Validation set data format incorrect or insufficient columns for feature '{feature_name_iter}' (index {feature_idx}). Shape: {X_val.shape}")

def main(train_files=None, pos_train_files=None, neg_train_files=None,
         val_files=None, pos_val_files=None, neg_val_files=None,
         exclude_features=None, visualize=False, balance_validation_set=False):
    print("\n=== Training Dataset ===")
    if train_files: print(f"Mixed training files: {train_files}")
    if pos_train_files: print(f"Positive training files: {pos_train_files}")
    if neg_train_files: print(f"Negative training files: {neg_train_files}")
    X_train_parts, y_train_parts, train_texts_list = [], [], []  # Renamed to avoid confusion
    if train_files:
        X_temp, y_temp, texts_temp = load_datasets(train_files)
        if len(X_temp) > 0: X_train_parts.append(X_temp); y_train_parts.append(y_temp); train_texts_list.extend(texts_temp)
    if pos_train_files:
        X_temp, y_temp, texts_temp = load_datasets(pos_train_files, example_type='positive')
        if len(X_temp) > 0: X_train_parts.append(X_temp); y_train_parts.append(y_temp); train_texts_list.extend(texts_temp)
    if neg_train_files:
        X_temp, y_temp, texts_temp = load_datasets(neg_train_files, example_type='negative')
        if len(X_temp) > 0: X_train_parts.append(X_temp); y_train_parts.append(y_temp); train_texts_list.extend(texts_temp)
    if not X_train_parts: raise ValueError("Training data loading failed or empty (X_train_parts is empty)")
    X_train = np.concatenate(X_train_parts)
    y_train = np.concatenate(y_train_parts)
    # train_texts is now train_texts_list
    print("\n=== Validation Dataset ===")
    if val_files: print(f"Mixed validation files: {val_files}")
    if pos_val_files: print(f"Positive validation files: {pos_val_files}")
    if neg_val_files: print(f"Negative validation files: {neg_val_files}")
    X_val_parts, y_val_parts, val_texts_list = [], [], []  # Renamed
    if val_files:
        X_temp, y_temp, texts_temp = load_datasets(val_files)
        if len(X_temp) > 0: X_val_parts.append(X_temp); y_val_parts.append(y_temp); val_texts_list.extend(texts_temp)
    if pos_val_files:
        X_temp, y_temp, texts_temp = load_datasets(pos_val_files, example_type='positive')
        if len(X_temp) > 0: X_val_parts.append(X_temp); y_val_parts.append(y_temp); val_texts_list.extend(texts_temp)
    if neg_val_files:
        X_temp, y_temp, texts_temp = load_datasets(neg_val_files, example_type='negative')
        if len(X_temp) > 0: X_val_parts.append(X_temp); y_val_parts.append(y_temp); val_texts_list.extend(texts_temp)
    if X_val_parts:
        X_val = np.concatenate(X_val_parts)
        y_val = np.concatenate(y_val_parts)
    elif len(X_train) > 0:
        print("No validation set provided, automatically splitting from training set (80/20).")
        if len(train_texts_list) != len(X_train):
            print(f"Warning: train_texts_list (len {len(train_texts_list)}) does not match X_train (len {len(X_train)}). Text list may be inaccurate after split.")
            # Decide handling: either proceed risking error or don't pass texts to split
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
            )
            # Texts are not split if lengths mismatch, val_texts_list remains potentially empty or from files
            val_texts_list = []  # Or handle more gracefully if possible
        else:
            X_train, X_val, y_train, y_val, train_texts_list, val_texts_list = train_test_split(
                X_train, y_train, train_texts_list, test_size=0.2, stratify=y_train, random_state=42
            )
    else:  # X_train is empty and no val_files provided
        num_cols = len(calculate_features({}, {}, {}))
        X_val, y_val, val_texts_list = np.empty((0, num_cols)), np.array([]), []
        print("Warning: Training data empty and no validation files provided. Validation set is empty.")
    train_files_set = set(train_files or []) | set(pos_train_files or []) | set(neg_train_files or [])
    val_files_set = set(val_files or []) | set(pos_val_files or []) | set(neg_val_files or [])
    common_files = train_files_set.intersection(val_files_set)
    if common_files:
        print(f"\nDetected overlapping files between training and validation sets: {common_files}")
        print("Warning: Overlapping file handling logic may cause data duplication between training and validation sets. Non-overlapping datasets are recommended.")
        print("Current strategy: Load data from overlapping files, split into 80% training and 20% validation, and append to existing sets.")
        common_X_data_parts, common_y_data_parts, common_texts_data_list_temp = [], [], []
        for file_path in list(common_files):
            X_temp_common, y_temp_common, texts_temp_common = load_datasets([file_path])
            if len(X_temp_common) > 0:
                common_X_data_parts.append(X_temp_common)
                common_y_data_parts.append(y_temp_common)
                common_texts_data_list_temp.extend(texts_temp_common)
        if common_X_data_parts:
            X_common_all = np.concatenate(common_X_data_parts)
            y_common_all = np.concatenate(common_y_data_parts)
            texts_common_train, texts_common_val = [], []  # Initialize
            if len(common_texts_data_list_temp) == len(X_common_all):
                X_common_train, X_common_val, \
                y_common_train, y_common_val, \
                texts_common_train, texts_common_val = train_test_split(
                    X_common_all, y_common_all, common_texts_data_list_temp,
                    test_size=0.2, stratify=y_common_all, random_state=42
                )
            else:
                print(f"Warning: common_texts_data_list_temp (len {len(common_texts_data_list_temp)}) does not match X_common_all (len {len(X_common_all)}). Splitting overlapping data without text.")
                X_common_train, X_common_val, y_common_train, y_common_val = train_test_split(
                    X_common_all, y_common_all, test_size=0.2, stratify=y_common_all, random_state=42)
            X_train = np.concatenate([X_train, X_common_train]) if len(X_train) > 0 else X_common_train
            y_train = np.concatenate([y_train, y_common_train]) if len(y_train) > 0 else y_common_train
            train_texts_list.extend(texts_common_train)
            X_val = np.concatenate([X_val, X_common_val]) if len(X_val) > 0 else X_common_val
            y_val = np.concatenate([y_val, y_common_val]) if len(y_val) > 0 else y_common_val
            val_texts_list.extend(texts_common_val)
        else:
            print("Overlapping files loaded no data, skipping append operation.")
    if balance_validation_set and len(X_val) > 0:
        print("\nStarting validation set balancing...")
        unique_labels_val, counts_val = np.unique(y_val, return_counts=True)
        label_counts_val = dict(zip(unique_labels_val, counts_val))
        print(f"Validation set sample distribution before balancing: {label_counts_val}")
        if len(unique_labels_val) == 2:
            label0_count_val = label_counts_val.get(0, 0)
            label1_count_val = label_counts_val.get(1, 0)
            if label0_count_val != label1_count_val and label0_count_val > 0 and label1_count_val > 0:  # Ensure both classes exist
                min_samples_val = min(label0_count_val, label1_count_val)
                print(f"Downsampling majority class to target sample count: {min_samples_val} per class")
                val_texts_np_for_resample = np.array(val_texts_list, dtype=object)
                if len(val_texts_np_for_resample) != len(y_val):  # Check alignment
                    print("Warning: Validation text list misaligned with labels/features, cannot safely resample text. Balancing features and labels only.")
                    # Fallback: resample X and y only
                    df_val_to_resample = pd.DataFrame(X_val)
                    df_val_to_resample['label'] = y_val
                    df_val_balanced = df_val_to_resample.groupby('label', group_keys=False).apply(
                        lambda x: x.sample(min_samples_val, random_state=42))
                    X_val = df_val_balanced.drop('label', axis=1).values
                    y_val = df_val_balanced['label'].values
                    val_texts_list = []  # Texts lost or need complex realignment
                else:
                    X_val_pos = X_val[y_val == 1]
                    y_val_pos = y_val[y_val == 1]
                    texts_val_pos_list_resample = list(val_texts_np_for_resample[y_val == 1])
                    X_val_neg = X_val[y_val == 0]
                    y_val_neg = y_val[y_val == 0]
                    texts_val_neg_list_resample = list(val_texts_np_for_resample[y_val == 0])
                    if label1_count_val > min_samples_val:
                        X_val_pos, y_val_pos, texts_val_pos_list_resample = resample(
                            X_val_pos, y_val_pos, texts_val_pos_list_resample,
                            replace=False, n_samples=min_samples_val, random_state=42)
                    elif label0_count_val > min_samples_val:
                        X_val_neg, y_val_neg, texts_val_neg_list_resample = resample(
                            X_val_neg, y_val_neg, texts_val_neg_list_resample,
                            replace=False, n_samples=min_samples_val, random_state=42)
                    X_val = np.concatenate((X_val_pos, X_val_neg), axis=0)
                    y_val = np.concatenate((y_val_pos, y_val_neg), axis=0)
                    val_texts_list = texts_val_pos_list_resample + texts_val_neg_list_resample
                shuffle_indices_val = np.random.RandomState(seed=42).permutation(len(X_val))
                X_val = X_val[shuffle_indices_val]
                y_val = y_val[shuffle_indices_val]
                if val_texts_list:  # Only shuffle texts if part of resampling
                    val_texts_list = [val_texts_list[i] for i in shuffle_indices_val]
                unique_labels_after_val, counts_after_val = np.unique(y_val, return_counts=True)
                print(f"Validation set sample distribution after balancing: {dict(zip(unique_labels_after_val, counts_after_val))}")
            elif label0_count_val == 0 or label1_count_val == 0:
                print("One class has zero samples in validation set, skipping balancing.")
            else:  # Already balanced
                print("Validation set already balanced or no action needed.")
        else:
            print("Validation set not binary or has single class, skipping balancing.")
    _initial_feature_names = ['bert_diff', 'bert_ratio', 'mover_diff', 'mover_ratio', 'bert_prev', 'mover_prev']
    feature_names = list(_initial_feature_names)
    if X_train.ndim == 1 and X_train.shape[0] == 0: X_train = np.empty((0, len(_initial_feature_names)))
    if X_val.ndim == 1 and X_val.shape[0] == 0: X_val = np.empty((0, len(_initial_feature_names)))
    if X_train.shape[1] != len(_initial_feature_names):
        raise ValueError(f"X_train column count ({X_train.shape[1]}) != initial names length ({len(_initial_feature_names)}).")
    if len(X_val) > 0 and X_val.shape[1] != len(_initial_feature_names):
        raise ValueError(f"X_val column count ({X_val.shape[1]}) != initial names length ({len(_initial_feature_names)}).")
    if exclude_features:
        print("\n--- Feature Exclusion ---")
        print(f"Original feature list ({len(feature_names)}): {list(feature_names)}")
        print(f"Features to exclude: {exclude_features}")
        keep_indices = [i for i, name in enumerate(feature_names) if name not in exclude_features]
        actually_excluded_names = [name for name in feature_names if name in exclude_features]
        if not keep_indices:
            raise ValueError("Error: All features excluded. Please check `exclude_features` list.")
        X_train = X_train[:, keep_indices]
        if len(X_val) > 0: X_val = X_val[:, keep_indices]
        feature_names = [name for i, name in enumerate(feature_names) if i in keep_indices]
        print(f"Actually excluded features: {actually_excluded_names}")
        print(f"Remaining features after exclusion ({len(feature_names)}): {feature_names}")
        print("--- Feature Exclusion Complete ---\n")
    X_train_scaled, X_val_scaled = X_train, X_val
    print("Training set feature shape (after processing):", X_train_scaled.shape)
    if len(X_val_scaled) > 0:
        print("Validation set feature shape (after processing):", X_val_scaled.shape)
    else:
        print("Validation set empty or became empty after processing.")
    param_grid = {'n_estimators': [150], 'max_depth': [7], 'learning_rate': [0.05], 'subsample': [0.8],
                  'colsample_bytree': [0.8]}
    xgb_clf = xgb.XGBClassifier(random_state=42, objective='binary:logistic', eval_metric='logloss',
                                use_label_encoder=False)
    if len(X_train_scaled) == 0: raise ValueError("Training data empty, cannot perform model training.")
    grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, scoring='roc_auc', cv=3, n_jobs=-1, verbose=2)
    print("\nStarting Grid Search Hyperparameter Optimization...")
    grid_search.fit(X_train_scaled, y_train)
    print("\nBest Parameters:", grid_search.best_params_)
    best_model = grid_search.best_estimator_
    if len(X_val_scaled) > 0 and len(y_val) > 0:
        print("\nModel Evaluation Report (using best parameters):")
        val_preds = best_model.predict(X_val_scaled)
        print(classification_report(y_val, val_preds, digits=4))
        val_probs = best_model.predict_proba(X_val_scaled)[:, 1]
        auroc = roc_auc_score(y_val, val_probs)
        print(f"\nAUROC Value: {auroc:.4f}")
    else:
        print("\nValidation set empty, skipping model evaluation.")
    if hasattr(best_model, 'feature_importances_'):
        importance = best_model.feature_importances_
        print("\nFeature Importance Ranking:")
        if len(feature_names) == len(importance):
            for name, imp in sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True):
                print(f"{name:20} {imp:.4f}")
        else:
            print(f"Warning: Feature name list length ({len(feature_names)}) does not match importance scores length ({len(importance)}).")
            for i, imp_val in enumerate(importance): print(f"Feature_Index_{i}: {imp_val:.4f}")
    else:
        print("Model has no feature_importances_ attribute.")
    if visualize and len(X_train_scaled) > 0 and len(y_train) > 0 and \
            len(X_val_scaled) > 0 and len(y_val) > 0:
        print("\nStarting Feature Visualization...")
        visualize_features(X_train_scaled, y_train, X_val_scaled, y_val, feature_names)
    elif visualize:
        print("\nInsufficient data, skipping feature visualization.")
    if len(X_val_scaled) > 0 and len(y_val) > 0:
        print("\nSample Detailed Analysis:")
        val_probs = best_model.predict_proba(X_val_scaled)[:, 1]
        val_preds = best_model.predict(X_val_scaled)
        correct_indices = np.where(val_preds == y_val)[0]
        error_indices = np.where(val_preds != y_val)[0]
        sample_size = 0
        if val_texts_list: sample_size = min(3, len(val_texts_list))  # Use updated val_texts_list
        if sample_size > 0:
            print("\nCorrect Prediction Examples:")
            if len(correct_indices) > 0:
                valid_correct_indices = [i for i in correct_indices if i < len(val_texts_list)]
                if valid_correct_indices:
                    chosen_indices_correct = np.random.choice(valid_correct_indices,
                                                              size=min(sample_size, len(valid_correct_indices)),
                                                              replace=False)
                    for idx in chosen_indices_correct: print(
                        f"\nText Combination: {val_texts_list[idx]}\nActual Label: {y_val[idx]} | Predicted Label: {val_preds[idx]}\nPrediction Probability: {val_probs[idx]:.3f}")
                else:
                    print("No valid index correct prediction samples.")
            else:
                print("No correct prediction samples")
            print("\nError Prediction Examples:")
            if len(error_indices) > 0:
                valid_error_indices = [i for i in error_indices if i < len(val_texts_list)]
                if valid_error_indices:
                    chosen_indices_error = np.random.choice(valid_error_indices,
                                                            size=min(sample_size, len(valid_error_indices)),
                                                            replace=False)
                    for idx in chosen_indices_error: print(
                        f"\nText Combination: {val_texts_list[idx]}\nActual Label: {y_val[idx]} | Predicted Label: {val_preds[idx]}\nPrediction Probability: {val_probs[idx]:.3f}")
                else:
                    print("No valid index error prediction samples.")
            else:
                print("No error prediction samples")
        else:
            print("val_texts_list empty or too few samples, skipping detailed example analysis.")
    else:
        print("\nValidation set empty, skipping sample detailed analysis.")
    model_path = "xgboost_model.json"
    booster = best_model.get_booster()
    booster.save_model(model_path)
    print(f"\nModel saved to: {model_path}")

if __name__ == "__main__":
    main(
        pos_train_files=["./output/CLFS_human.json","./output/XSUM_human.json","./output/squad_human.json","./output/govreport_human.json","./output/billsum_human.json",],
        neg_train_files=["./output/CLFS_gpt.json", "./output/XSUM_gpt.json", "./output/squad_gpt.json",
                         "./output/govreport_gpt.json", "./output/billsum_gpt.json", ],
        pos_val_files=["./output/XSUM_human.json"],
        neg_val_files=["./output/XSUM_Claude.json"],
        exclude_features=[
            # "bert_diff", "bert_ratio",
            # "bert_prev",
            # "mover_prev",
            # "mover_diff", "mover_ratio"
        ],
        visualize=False,
        balance_validation_set=True
    )