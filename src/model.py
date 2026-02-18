#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import pytorch_lightning as pl
from pytorch_forecasting.models import TemporalFusionTransformer

from typing import Optional

class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels=1, output_size=16):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)  # MaxPooling層を追加
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)  # MaxPooling層を追加
        self.fc = nn.Linear(32, output_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)  # MaxPoolingを適用
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)  # MaxPoolingを適用
        x = torch.mean(x, dim=2)
        x = self.fc(x)
        return x

# Temporal Fusion Transformer部分
class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.2):
        super(TemporalFusionTransformer, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()  # ゲートの出力は0〜1の範囲
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)  # 出力次元はhidden_dimのまま
        )
        self._init_weights()

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        x, _ = self.lstm(x)
        gate_output = self.gate(x)  # ゲートの出力を計算
        x = x * gate_output  # ゲートの出力を要素ごとに乗算
        x = self.mlp(x)  # ゲートの後にMLPを追加
        attention_output, attention_weights = self.attention(x, x, x)  # Get attention weights
        x = self.dropout(attention_output)
        self.variable_selection_weights = attention_weights.mean(dim=1)  # Store average attention weights
        return x[:, -1, :] # Return the last hidden state

    def _init_weights(self):
          for name, param in self.named_parameters():
              if 'weight' in name and param.dim() > 1:
                  init.xavier_uniform_(param)
              elif 'bias' in name:
                  init.constant_(param, 0.0)
class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.2):
        super(SimpleLSTM, self).__init__()
        self.hidden_dim = hidden_dim # hidden_dimを属性として設定
        self.sequence_length = sequence_length
        self.lstm = nn.LSTM(num_features, hidden_dim, batch_first=True, bidirectional=True) # 双方向LSTM
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        x, _ = self.lstm(x)  # LSTM
        # 双方向LSTMの出力を結合
        x = torch.cat([x[:, -1, :hidden_dim], x[:, 0, hidden_dim:]], dim=1)
        x = self.dropout(x)  # ドロップアウト
        return x  # 結合した隠れ状態を出力

class LSTMFC(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.2):
        super(LSTMFC, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)  # 全結合層を追加

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        x, _ = self.lstm(x)  # LSTM
        x = self.dropout(x)  # ドロップアウト
        x = x[:, -1, :]  # 最後の隠れ状態
        x = self.fc(x)  # 全結合層
        return x

# # StockPricePredictionModelの変更
# class StockPricePredictionModel(nn.Module):
#     def __init__(self, num_features, hidden_dim, embedding_dim, sequence_length, num_classes=3):
#         super(StockPricePredictionModel, self).__init__()
#         self.tft = TemporalFusionTransformer(num_features, hidden_dim)
#         # self.cnn = CNNFeatureExtractor(input_channels, output_channels)
#         # self.static_embedding = nn.Embedding(num_static_features, embedding_dim)  # 静的変数埋め込み層
#         self.fc = nn.Linear(hidden_dim + embedding_dim, 3)  # 結合した特徴量の次元に合わせて変更

#     def forward(self, numerical_sequence):
#         # cnn_output = self.cnn(numerical_sequence)
#         # numerical_features = self.tft(cnn_output)
#         numerical_features = self.tft(numerical_sequence)
#         # static_embeddings = self.static_embedding(static_features)  # 静的変数を埋め込み
#         # combined_features = torch.cat([numerical_features], dim=1)  # 特徴量を結合
#         # output = self.fc(combined_features)
#         output = self.fc(numerical_features)
#         return torch.softmax(output, dim=1)
######################分類モデル用##########################################################
class StockPricePredictionModel(nn.Module):
    def __init__(self, num_features, hidden_dim, sequence_length, num_classes=3):
        super(StockPricePredictionModel, self).__init__()
        # self.tft = TemporalFusionTransformer(num_features, hidden_dim)
        self.SimpleLSTM = SimpleLSTM(num_features, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)  # hidden_dim に修正

    def forward(self, numerical_sequence):
        numerical_features = self.SimpleLSTM(numerical_sequence)
        # numerical_features = self.tft(numerical_sequence)
        output = self.fc(numerical_features)
        # output = torch.softmax(output, dim=1) #WKLのとき
        return output
######################回帰モデル用##########################################################
class StockPricePredictionModel(nn.Module):
    def __init__(self, num_features, hidden_dim, sequence_length):  # num_classes を削除
        super(StockPricePredictionModel, self).__init__()
        # self.tft = TemporalFusionTransformer(num_features, hidden_dim)
        self.SimpleLSTM = SimpleLSTM(num_features, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)  # 出力層を1ユニットに変更

    def forward(self, numerical_sequence):
        numerical_features = self.SimpleLSTM(numerical_sequence)
        # numerical_features = self.tft(numerical_sequence)
        output = self.fc(numerical_features)
        return output  # 活性化関数を削除

class CNN_TFT(nn.Module):
    def __init__(self, tft_dataset, cnn_input_size, cnn_output_size):
        super().__init__()
        self.cnn = CNNFeatureExtractor(input_size=cnn_input_size, output_size=cnn_output_size)
        self.tft = TemporalFusionTransformer.from_dataset(
            tft_dataset,
            learning_rate=0.03,
            hidden_size=16,
            attention_head_size=4,
            dropout=0.1,
            loss=torch.nn.MSELoss(),
            output_size=1,
            optimizer="adam"
        )

    def forward(self, x):
        # CNNの特徴抽出
        cnn_features = self.cnn(x["encoder_cont"])  # 数値時系列データをCNNに入力

        # TFTへの入力を更新
        x["encoder_cont"] = torch.cat([x["encoder_cont"], cnn_features.unsqueeze(1)], dim=1)

        return self.tft(x)

class CustomTrainer(pl.LightningModule):
    def __init__(self, dataset, decay_lambda=0.01):
        """
        dataset: TimeSeriesDataSet（TFTのデータセット）
        decay_lambda: 損失関数の減衰率
        cnn_input_size: CNNへの入力サイズ
        cnn_output_size: CNNからの出力サイズ
        """
        super().__init__()
        self.cnn_tft = CNN_TFT(dataset, cnn_input_size, cnn_output_size)
        self.weighted_loss = WeightedMSELoss(decay_lambda)

    def training_step(self, batch, batch_idx):
        """
        学習ステップ。カスタム損失を適用。
        """
        y_pred, _ = self.tft(batch)  # モデルの出力
        y_true = batch["decoder_target"]  # 正解ラベル
        time_idx = batch["decoder_time_idx"]  # 時間インデックス

        loss = self.weighted_loss(y_pred, y_true, time_idx)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.03)

############　分類モデル用　###################################
# Train function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for numerical_data, targets in dataloader:
        numerical_data, targets = numerical_data.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(numerical_data)
        # loss = criterion(outputs, targets, device)# WKLのとき
        loss = criterion(outputs, targets)
        # 重みで損失をスケール
        # weighted_loss = loss * weights
        # weighted_loss.mean().backward()  # 平均を取って勾配を計算
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Evaluate function
def evaluate_loss(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for numerical_data, targets in dataloader:
            numerical_data, targets = numerical_data.to(device), targets.to(device)
            outputs = model(numerical_data)
            # loss = criterion(outputs, targets, device)  # WKLのとき
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)  # 平均損失を計算
    return avg_loss  # 損失値を返す

def evaluate(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_actuals = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)


            predictions = torch.softmax(outputs, dim=1) # ここで確率値に変換
            _, predictions = torch.max(predictions, 1)  # 確率値が最大のインデックスを取得

            all_predictions.extend(predictions.cpu().numpy())
            all_actuals.extend(labels.cpu().numpy())

    return np.array(all_predictions), np.array(all_actuals)


# In[ ]:


# ############　回帰モデル用　###################################
# # Train function
# def train(model, dataloader, criterion, optimizer, device):
#     model.train()
#     total_loss = 0
#     for numerical_data, targets, weights, _ in dataloader:
#         numerical_data, targets, weights = numerical_data.to(device), targets.to(device), weights.to(device)
#         optimizer.zero_grad()
#         outputs = model(numerical_data)
#         # loss = criterion(outputs, targets, device)# WKLのとき
#         loss = criterion(outputs, targets)
#         # 重みで損失をスケール
#         # weighted_loss = loss * weights
#         # weighted_loss.mean().backward()  # 平均を取って勾配を計算
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     return total_loss / len(dataloader)

# # Evaluate function
# def evaluate_loss(model, dataloader, device, criterion):
#     model.eval()
#     total_loss = 0
#     with torch.no_grad():
#         for numerical_data, targets, _ , _ in dataloader:
#             numerical_data, targets = numerical_data.to(device), targets.to(device)
#             outputs = model(numerical_data)
#             # loss = criterion(outputs, targets, device)  # WKLのとき
#             loss = criterion(outputs, targets)
#             total_loss += loss.item()
#     avg_loss = total_loss / len(dataloader)  # 平均損失を計算
#     return avg_loss  # 損失値を返す

# def evaluate(model, dataloader, device):
#     model.eval()
#     all_predictions = []
#     all_actuals = []
#     all_vol = []

#     with torch.no_grad():
#         for inputs, labels, weights, vol in dataloader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             vol = vol.to(device)

#             outputs = model(inputs)

#             predictions = outputs

#             all_predictions.extend(predictions.cpu().numpy())
#             all_actuals.extend(labels.cpu().numpy())
#             all_vol.extend(vol.cpu().numpy())

#     return np.array(all_predictions), np.array(all_actuals), np.array(all_vol)


# In[ ]:


#################Softmax変換した確率分布を用いて、順位情報を学習するカスタム損失関数###########################################
import torch
import torch.nn as nn
import torch.nn.functional as F

class ListNetLoss(nn.Module):
    def __init__(self):
        super(ListNetLoss, self).__init__()

    def forward(self, y_pred, y_true):
        # y_trueをfloat型に変換
        y_true = y_true.type(torch.float32)
        # log_softmaxによる対数確率分布化
        P_pred = F.log_softmax(y_pred, dim=0)
        P_true = F.softmax(y_true, dim=0)

        # KLダイバージェンスによる確率分布の距離測定
        loss = -(P_true * P_pred).sum(dim=0).mean()

        return loss


# In[ ]:


def calculate_accuracy_for_specific_classes(y_class_true, y_class_pred):
    """y_class_predが2もしくは0のときに y_class_trueが一致している割合を算出する。

    Args:
        y_class_true: 正解ラベルのリスト。
        y_class_pred: 予測ラベルのリスト。

    Returns:
        y_class_predが2もしくは0のときに y_class_trueが一致している割合。
    """
    # y_class_predが2もしくは0の要素のインデックスを取得
    indices = [i for i, pred in enumerate(y_class_pred) if pred in (0, 2)]

    # 一致している要素数をカウント
    correct_count = sum(1 for i in indices if y_class_true[i] == y_class_pred[i])

    # 割合を計算
    accuracy = correct_count / len(indices) if indices else 0  # インデックスが空の場合は0を返す

    return accuracy


# In[ ]:


# 最適パラメータによる学習と予測を実行する関数
def best_model_training_and_prediction(df_train, train_labels, weights, df_test, test_labels, best_params):

    # デバイスの設定 (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sequence_length = best_params["sequence_length"]
    hidden_dim = best_params["hidden_dim"]
    learning_rate = best_params["learning_rate"]
    num_epochs = best_params["num_epochs"]
    # future_period = best_params["future_period"]
    # volatility_multiplier = best_params["volatility_multiplier"]
    # decay_rate = best_params["decay_rate"]
    # quantile = best_params["quantile"]

    num_static_features = 0
    embedding_dim = 16
    num_classes = 3
    num_features = df_train.shape[1]


    # 最適なハイパーパラメータでモデルを初期化
    #　分類モデル
    best_model = StockPricePredictionModel(num_features, hidden_dim, sequence_length, num_classes)
    #　回帰モデル
    # best_model = StockPricePredictionModel(num_features, hidden_dim, sequence_length)
    best_model.to(device)

    #　分類モデル
    # criterion = OrdinalCrossEntropyLoss()
    criterion = WeightedKappaLoss(num_classes)
    #　回帰モデル
    # criterion = ListNetLoss()
    # criterion = nn.MSELoss()
    #criterion = nn.HuberLoss()

    optimizer = torch.optim.Adam(best_model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # 学習率スケジューラの種類を選択
    scheduler_type = best_params.get("scheduler", "StepLR")  # デフォルトはStepLR

    # StepLRの場合
    if scheduler_type == "StepLR":
        step_size = best_params.get("step_size", 10)  # デフォルト値を設定
        gamma = best_params.get("gamma", 0.1)  # デフォルト値を設定
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # CosineAnnealingLRの場合
    else:
        T_max = best_params.get("T_max", 5)  # デフォルト値を設定
        eta_min = best_params.get("eta_min", 1e-5)  # デフォルト値を設定
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    # 全学習データでトレーニング
    # transform = create_transform(input_size=(3, image_size, image_size), is_training=True)
    dataset = StockDataset(df_train, train_labels, weights, sequence_length)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)


    # split_index = int(len(dataset) * 0.8)

    # train_data = torch.utils.data.Subset(dataset, range(split_index))
    # val_data = torch.utils.data.Subset(dataset, range(split_index, len(dataset)))
    # train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=False, num_workers=4)
    # val_loader = torch.utils.data.DataLoader(val_data, batch_size=16, shuffle=False, num_workers=4)

    # # Early Stoppingのパラメータ
    # patience = 3  # 検証データセットでの性能が向上しなくなってから、学習を継続するエポック数
    # best_loss = float('inf')  # 最良の検証データセットでの損失
    # epochs_without_improvement = 0  # 性能が向上していないエポック数



    for epoch in range(num_epochs):
        # if epoch == 3:
        #     for param in best_model.swin_transformer.parameters():
        #         param.requires_grad = True

        train_loss = train(best_model, dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")
        scheduler.step()  # 学習率スケジューラの更新



    # テストデータで検証
    # test_time_series_data = df_test.iloc[time_step-1:]
    # test_labels = test_labels.iloc[time_step-1:]
    # test_weights = test_weights.iloc[time_step-1:]

    test_dataset = StockDataset(df_test, test_labels, sequence_length=sequence_length)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    predictions, actuals = evaluate(best_model, test_dataloader, device)

    qwk = cohen_kappa_score(actuals, predictions, weights='quadratic')
    print(f"Test QWK: {qwk:.4f}")
    return


# In[ ]:





# In[ ]:


def create_timeseries_dataframe(X, y, original_indices, cluster_ids_all, label_strategy, feature_names):
    """
    Creates a pandas DataFrame suitable for PyTorch Forecasting's TimeSeriesDataSet.

    Args:
        X (np.ndarray): Input features for sequences (samples, sequence_length, num_features).
        y (np.ndarray): Target labels for sequences (samples,).
        original_indices (pd.DatetimeIndex or np.ndarray): Original start index in the full dataset for each sample.
                                       Shape (n_samples,). Can be a DatetimeIndex or a numpy array of integers.
                                       If DatetimeIndex, assumed to be the start time of each sequence.
        cluster_ids_all (np.ndarray or None): Cluster ID for each sample in the *full* original dataset.
                                             Shape (total_original_samples,). Can be None.
        label_strategy (dict or None): Mapping from cluster ID to strategy name. Can be None.
        feature_names (list): List of names for the features in X.

    Returns:
        pd.DataFrame: DataFrame with columns required for TimeSeriesDataSet.
    """
    num_samples, sequence_length, num_features = X.shape
    all_data = []

    # Determine if cluster and strategy info can be added
    add_cluster_info = cluster_ids_all is not None and isinstance(cluster_ids_all, np.ndarray)
    use_label_strategy = label_strategy is not None and isinstance(label_strategy, dict)

    if original_indices is None or len(original_indices) != num_samples:
         raise ValueError("original_indices must have the same number of samples as X.")

    # Determine the base time index type and calculate minimum if DatetimeIndex
    base_time_indices = None
    min_timestamp = None
    if isinstance(original_indices, pd.DatetimeIndex):
        print("original_indices is DatetimeIndex. Calculating time_idx based on hours from minimum timestamp.")
        min_timestamp = original_indices.min() # Minimum timestamp in this batch of sequences
        # Calculate base time index for each sequence (hours from min_timestamp)
        # Ensure time difference is in hours and is an integer
        base_time_indices = ((original_indices - min_timestamp).total_seconds() / 3600).astype(int)
    elif isinstance(original_indices, np.ndarray) and (np.issubdtype(original_indices.dtype, np.integer) or np.issubdtype(original_indices.dtype, np.floating)):
         base_time_indices = original_indices.astype(int)
         print("original_indices is a numeric numpy array. Using it as base time_idx.")
    else:
         raise ValueError("original_indices must be a pandas DatetimeIndex or a numpy array of integers/floats.")


    # If adding cluster info, ensure cluster_ids_all has the expected length
    if add_cluster_info and len(cluster_ids_all) != num_samples:
         print(f"Warning: cluster_ids_all length ({len(cluster_ids_all)}) does not match the number of samples ({num_samples}). original_cluster_id and strategy_name columns may be incorrect.")


    for i in range(num_samples):
        # Determine group_id (Using sequence position as group_id)
        group_id = i

        # Determine original cluster ID and strategy for this sequence
        original_cluster_id = None # Initialize to None or a placeholder
        if add_cluster_info:
             original_cluster_id = cluster_ids_all[i] # Index cluster_ids_all by sequence position
        else:
             original_cluster_id = -2 # Use -2 for missing/invalid if cluster info not added

        # Map cluster ID to strategy name, handle unknown clusters or missing strategy dict
        strategy_name = "unknown" # Default strategy name
        if use_label_strategy and original_cluster_id in label_strategy:
             strategy_name = label_strategy[original_cluster_id]
        elif add_cluster_info and original_cluster_id == -1:
             strategy_name = "noise" # Explicitly label noise cluster strategy


        # Generate time_idx for the sequence
        # time_indices_sequence = np.arange(sequence_length) + base_time_indices[i]
        # If original_indices is DatetimeIndex (start time), the time_idx for the t-th step is the base time index + t
        if isinstance(original_indices, pd.DatetimeIndex):
             time_indices_sequence = np.arange(sequence_length) + base_time_indices[i]
        elif isinstance(original_indices, np.ndarray):
             # If original_indices is integer, assume it's the time_idx of the *start* of the sequence
             time_indices_sequence = np.arange(sequence_length) + base_time_indices[i]
        else:
             # This case should not be reached due to the initial check, but for safety
             raise TypeError("original_indices must be DatetimeIndex or numpy array.")


        # Flatten the sequence features and create records for each timestep
        for t in range(sequence_length):
            record = {
                "group_id": group_id, # Use sequence position as group_id
                "time_idx": int(time_indices_sequence[t]), # Ensure time_idx is integer
                "target": float(y[i]) if pd.notna(y[i]) else np.nan # Use the sequence's target label y[i]
            }

            # Add feature columns
            for j in range(num_features):
                feature_col_name = feature_names[j] if j < len(feature_names) else f"feature_{j}"
                feature_value = X[i, t, j] # Get feature value for the current sequence (i) and time step (t)
                try:
                    record[feature_col_name] = float(feature_value) if pd.notna(feature_value) else np.nan
                except (ValueError, TypeError):
                    # If conversion to float fails, store as object or string.
                    record[feature_col_name] = feature_value


            # Add cluster and strategy info as static categoricals (if applicable)
            record["original_cluster_id"] = original_cluster_id
            record["strategy_name"] = strategy_name

            all_data.append(record)

    # Create DataFrame
    df = pd.DataFrame(all_data)

    # Ensure group_id and time_idx are correctly typed
    df['group_id'] = df['group_id'].astype(str) # group_id should be categorical
    df['time_idx'] = df['time_idx'].astype(int)

    # --- Filter out rows where original_cluster_id is -1 ---
    if "original_cluster_id" in df.columns:
        print(f"DataFrame shape before filtering noise cluster (-1): {df.shape}")
        df = df[~df['original_cluster_id'].isin([-1, -2])].copy()
        print(f"DataFrame shape after filtering noise cluster (-1) and invalid (-2): {df.shape}")
    else:
        print("Warning: 'original_cluster_id' column not found in DataFrame. Noise filtering skipped.")

    # Ensure target column exists and is numeric
    if 'target' not in df.columns:
        print("Warning: 'target' column not found in the created DataFrame.")
    else:
        if not pd.api.types.is_numeric_dtype(df['target']):
             print(f"Warning: 'target' column is not numeric (dtype: {df['target'].dtype}). Attempting to convert.")
             df['target'] = pd.to_numeric(df['target'], errors='coerce')


    return df


# In[ ]:


# def move_to_device(obj, device):
#     """
#     Recursively moves tensors within a nested structure (dict, list, tuple, namedtuple)
#     to the specified device. Handles TimeSeriesDataSet output structure.
#     """
#     if torch.is_tensor(obj):
#         return obj.to(device)
#     elif isinstance(obj, dict):
#         # Process dict keys in a defined order if necessary, but simple recursion is usually fine
#         return {k: move_to_device(v, device) for k, v in obj.items()}
#     elif isinstance(obj, (list, tuple)):
#         # Convert tuple to list for modification, then back to tuple
#         # Use type(obj) to preserve the original type (list/tuple/namedtuple)
#         return type(obj)(move_to_device(v, device) for v in obj)
#     # Add this case to handle namedtuples more explicitly if needed,
#     # but the list/tuple case should handle them if they behave like tuples.
#     # elif hasattr(obj, '_fields'): # Check if it's a namedtuple
#     #     fields = [move_to_device(getattr(obj, f), device) for f in obj._fields]
#     #     return type(obj)(*fields)
#     else:
#         # Return non-tensor objects as is
#         return obj


# In[ ]:


# 損失マスク作成
def compute_predictability_score(labels, window_size=5):
    half_w = window_size // 2
    scores = []

    for i in range(len(labels)):
        start = max(0, i - half_w)
        end = min(len(labels), i + half_w + 1)
        window = labels[start:end]
        center_label = labels[i]
        score = np.mean([1 if l == center_label else 0 for l in window])
        scores.append(score)

    return np.array(scores)


# In[ ]:


import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_forecasting.models import TemporalFusionTransformer
import torchmetrics.functional as F
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score # torchmetrics を使用


# バッチをデバイスに移動するためのヘルパー関数
def move_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    else:
        return batch


class TFTModel(pl.LightningModule):
    """
    Temporal Fusion Transformer モデルを二値分類タスク用にカスタマイズ。
    BCEWithLogitsLoss を使用し、pos_weight でラベルの不均衡に対応。
    評価指標 (Accuracy, Precision, Recall, F1) を組み込みます。
    """
    def __init__(self, dataset, pos_weight=None, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        tft_kwargs = kwargs.copy()
        if 'output_size' in tft_kwargs:
            print("Warning: output_size is provided in kwargs and will be overridden to 1 for binary classification.")
            del tft_kwargs['output_size']
        tft_kwargs['output_size'] = 1

        if 'loss' in tft_kwargs:
            print("Warning: 'loss' is provided in kwargs and will be ignored by TemporalFusionTransformer.from_dataset. Using BCEWithLogitsLoss with pos_weight instead.")
            del tft_kwargs['loss']

        self.model = TemporalFusionTransformer.from_dataset(
            dataset=dataset,
            **tft_kwargs
        )

        # pos_weight は初期化時ではなく、モデルがデバイスに移動された後に適用
        # CPUテンソルとしてバッファに登録
        if pos_weight is not None:
            if not isinstance(pos_weight, torch.Tensor):
                self.register_buffer('pos_weight_tensor', torch.tensor(pos_weight, dtype=torch.float32))
            else:
                self.register_buffer('pos_weight_tensor', pos_weight.to("cpu")) # 初期はCPUに
        else:
            self.register_buffer('pos_weight_tensor', None)

        # 損失関数は on_fit_start で pos_weight を考慮して初期化/再初期化される
        self.criterion = None


    def on_fit_start(self):
        """
        トレーニング開始時にpos_weightと損失関数を適切にデバイスに配置/初期化します。
        """
        # pos_weight があれば、それを現在のデバイスに移動し、損失関数を初期化/再初期化
        if self.pos_weight_tensor is not None:
            if self.pos_weight_tensor.device != self.device:
                self.pos_weight_tensor = self.pos_weight_tensor.to(self.device)
                print(f"Debug: Moved pos_weight_tensor to {self.device}.")
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_tensor)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        print(f"Debug: BCEWithLogitsLoss initialized on device {self.device}.")


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # PyTorch Lightningがバッチをデバイスに移動するため、手動での移動は不要
        # ただし、batchの要素がネストされている場合や、カスタムのバッチ構造の場合は
        # 明示的な移動が必要になることがあります。
        # TimeSeriesDataSetのバッチはdictが含まれるタプルなので、ここで移動するのが安全です。
        batch_on_device = move_to_device(batch, self.device)
        x, y_list = batch_on_device

        y = y_list[0].float()
        if y.ndim == 1:
            y = y.unsqueeze(-1)

        # x をモデルに渡す前にデバイスを確認・移動
        x_on_device = move_to_device(x, self.device)
        output = self.model(x_on_device)
        pred_logits = output["prediction"].squeeze(-1)

        if pred_logits.ndim != y.ndim:
            if pred_logits.ndim == 1 and y.ndim == 2 and y.shape[1] == 1:
                y = y.squeeze(-1)
            elif pred_logits.ndim == 2 and pred_logits.shape[1] == 1 and y.ndim == 1:
                pred_logits = pred_logits.squeeze(-1)

        # 損失計算時のテンソルのデバイスは、Lightningが保証する
        # ただし、念のため明示的に移動
        loss = self.criterion(pred_logits.to(self.device), y.to(self.device))

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        with torch.no_grad():
            pred_probs = torch.sigmoid(pred_logits)
            predictions = (pred_probs > 0.5).int()
            y_int = y.int()

            if predictions.ndim != y_int.ndim:
                if predictions.ndim == 1 and y_int.ndim == 2 and y_int.shape[1] == 1:
                    y_int = y_int.squeeze(-1)
                elif predictions.ndim == 2 and predictions.shape[1] == 1 and y_int.ndim == 1:
                    predictions = predictions.squeeze(-1)

            # メトリクス計算時のテンソルのデバイスも、Lightningが保証する
            # ただし、念のため明示的に移動
            self.log('train_accuracy', F.binary_accuracy(predictions.to(self.device), y_int.to(self.device)), on_step=False, on_epoch=True, prog_bar=True)
            self.log('train_precision', F.binary_precision(predictions.to(self.device), y_int.to(self.device)), on_step=False, on_epoch=True, prog_bar=True)
            self.log('train_recall', F.binary_recall(predictions.to(self.device), y_int.to(self.device)), on_step=False, on_epoch=True, prog_bar=True)
            self.log('train_f1', F.binary_f1_score(predictions.to(self.device), y_int.to(self.device)), on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # バッチ全体をデバイスに移動
        batch_on_device = move_to_device(batch, self.device)
        x, y_list = batch_on_device

        y = y_list[0].float()
        if y.ndim == 1:
            y = y.unsqueeze(-1)

        # x をモデルに渡す前にデバイスを確認・移動
        x_on_device = move_to_device(x, self.device)
        output = self.model(x_on_device)
        pred_logits = output["prediction"].squeeze(-1)

        if pred_logits.ndim != y.ndim:
            if pred_logits.ndim == 1 and y.ndim == 2 and y.shape[1] == 1:
                y = y.squeeze(-1)
            elif pred_logits.ndim == 2 and pred_logits.shape[1] == 1 and y.ndim == 1:
                pred_logits = pred_logits.squeeze(-1)

        # 損失計算時のテンソルのデバイスも、Lightningが保証する
        # ただし、念のため明示的に移動
        loss = self.criterion(pred_logits.to(self.device), y.to(self.device))

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        with torch.no_grad():
            pred_probs = torch.sigmoid(pred_logits)
            predictions = (pred_probs > 0.5).int()
            y_int = y.int()

            if predictions.ndim != y_int.ndim:
                if predictions.ndim == 1 and y_int.ndim == 2 and y_int.shape[1] == 1:
                    y_int = y_int.squeeze(-1)
                elif predictions.ndim == 2 and predictions.shape[1] == 1 and y_int.ndim == 1:
                    predictions = predictions.squeeze(-1)

            # メトリクス計算時のテンソルのデバイスも、Lightningが保証する
            # ただし、念のため明示的に移動
            self.log('val_accuracy', F.binary_accuracy(predictions.to(self.device), y_int.to(self.device)), on_step=False, on_epoch=True, prog_bar=True)
            self.log('val_precision', F.binary_precision(predictions.to(self.device), y_int.to(self.device)), on_step=False, on_epoch=True, prog_bar=True)
            self.log('val_recall', F.binary_recall(predictions.to(self.device), y_int.to(self.device)), on_step=False, on_epoch=True, prog_bar=True)
            self.log('val_f1', F.binary_f1_score(predictions.to(self.device), y_int.to(self.device)), on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        # バッチ全体をデバイスに移動
        batch_on_device = move_to_device(batch, self.device)
        x, y_list = batch_on_device

        y = y_list[0].float()
        if y.ndim == 1:
            y = y.unsqueeze(-1)

        # x をモデルに渡す前にデバイスを確認・移動
        x_on_device = move_to_device(x, self.device)
        output = self.model(x_on_device)
        pred_logits = output["prediction"].squeeze(-1)

        if pred_logits.ndim != y.ndim:
            if pred_logits.ndim == 1 and y.ndim == 2 and y.shape[1] == 1:
                y = y.squeeze(-1)
            elif pred_logits.ndim == 2 and pred_logits.shape[1] == 1 and y.ndim == 1:
                pred_logits = pred_logits.squeeze(-1)

        # 損失計算時のテンソルのデバイスも、Lightningが保証する
        # ただし、念のため明示的に移動
        loss = self.criterion(pred_logits.to(self.device), y.to(self.device))

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        with torch.no_grad():
            pred_probs = torch.sigmoid(pred_logits)
            predictions = (pred_probs > 0.5).int()
            y_int = y.int()

            if predictions.ndim != y_int.ndim:
                if predictions.ndim == 1 and y_int.ndim == 2 and y_int.shape[1] == 1:
                    y_int = y_int.squeeze(-1)
                elif predictions.ndim == 2 and predictions.shape[1] == 1 and y_int.ndim == 1:
                    predictions = predictions.squeeze(-1)

            # メトリクス計算時のテンソルのデバイスも、Lightningが保証する
            # ただし、念のため明示的に移動
            self.log('test_accuracy', F.binary_accuracy(predictions.to(self.device), y_int.to(self.device)), on_step=False, on_epoch=True, prog_bar=True)
            self.log('test_precision', F.binary_precision(predictions.to(self.device), y_int.to(self.device)), on_step=False, on_epoch=True, prog_bar=True)
            self.log('test_recall', F.binary_recall(predictions.to(self.device), y_int.to(self.device)), on_step=False, on_epoch=True, prog_bar=True)
            self.log('test_f1', F.binary_f1_score(predictions.to(self.device), y_int.to(self.device)), on_step=False, on_epoch=True, prog_bar=True)

        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # move_to_device ヘルパー関数が定義されていることを前提とする
        # PyTorch Lightning 2.0以降では、predict_stepも自動的にデバイスに移動される
        # しかし、念のため明示的に移動するか、move_to_device 関数を使う
        batch_on_device = move_to_device(batch, self.device)
        x = batch_on_device # Assuming batch_on_device is the x_dict

        # x をモデルに渡す前にデバイスを確認・移動
        x_on_device = move_to_device(x, self.device)
        return self.model.predict(x_on_device, mode="raw")


# In[ ]:


# import torch
# import torch.nn as nn
# import pytorch_lightning as pl
# from pytorch_forecasting.models import TemporalFusionTransformer
# import torchmetrics.functional as F
# from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score # torchmetrics を使用

# # CrossEntropy は不要になるかもしれません
# # from pytorch_forecasting.metrics import CrossEntropy


# class TFTModel(pl.LightningModule):
#     """
#     Temporal Fusion Transformer モデルを二値分類タスク用にカスタマイズ。
#     BCEWithLogitsLoss を使用し、pos_weight でラベルの不均衡に対応。
#     評価指標 (Accuracy, Precision, Recall, F1) を組み込みます。
#     """
#     def __init__(self, dataset, pos_weight=None, **kwargs):
#         # dataset 引数を明示的に受け取る
#         super().__init__()
#         self.save_hyperparameters() # ロギング用にハイパーパラメータを保存

#         # 二値分類のため output_size は 1 に強制
#         # TemporalFusionTransformer.from_dataset に渡す kwargs から loss を削除
#         tft_kwargs = kwargs.copy()
#         if 'output_size' in tft_kwargs:
#             print("Warning: output_size is provided in kwargs and will be overridden to 1 for binary classification.")
#             del tft_kwargs['output_size'] # output_size を上書き
#         tft_kwargs['output_size'] = 1 # 二値分類のため出力サイズは 1

#         if 'loss' in tft_kwargs:
#              print("Warning: 'loss' is provided in kwargs and will be ignored by TemporalFusionTransformer.from_dataset. Using BCEWithLogitsLoss with pos_weight instead.")
#              del tft_kwargs['loss']


#         # TemporalFusionTransformer のインスタンスを内部に持つ
#         self.model = TemporalFusionTransformer.from_dataset(
#             dataset=dataset,
#             **tft_kwargs
#         )


#         # BCEWithLogitsLoss をカスタムクラスの属性として保持
#         # __init__の引数としてpos_weightを受け取る必要がある
#         if pos_weight is not None:
#             # pos_weight も属性として保持しておくと便利
#             # Ensure pos_weight is a tensor and on the correct device if passed as float/list
#             if not isinstance(pos_weight, torch.Tensor):
#                  # This might be tricky if device is not available yet in __init__
#                  # A better approach is to ensure pos_weight is a tensor before passing
#                  # Or move it in the first training_step
#                  # For now, assume it's a tensor already on the correct device or can be moved.
#                  # Registering as buffer is good practice for tensors not requiring gradients.
#                  self.register_buffer('pos_weight', torch.tensor(pos_weight, dtype=torch.float32))
#             else:
#                  # Assume it's already a tensor, register it as a buffer
#                  self.register_buffer('pos_weight', pos_weight.to(self.device)) # Ensure it's on the model's device

#             self.pos_weight = self.pos_weight.to(self.device)
#             self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
#         else:
#             self.criterion = nn.BCEWithLogitsLoss()
#             self.register_buffer('pos_weight', None) # Register None if no pos_weight

#         # torchmetrics のメトリクスオブジェクトはここでインスタンス化しない
#         # メトリクス計算はステップメソッド内で functional API を使用

#         # learning_rate などは super() で設定されるため、ここで self. として保持する必要はありません。
#         # configure_optimizers では self.hparams.learning_rate などとしてアクセスできるはずです。


#         # configure_optimizers メソッドを定義する必要があります（もし定義されていなければ）
#         # 例:
#         # from torch.optim import Adam
#         # def configure_optimizers(self):
#         #     optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate) # self.hparams を使用
#         #     return optimizer


#     def forward(self, x):
#         # 内部の TemporalFusionTransformer インスタンスの forward を呼び出す
#         # デバッグ用: 入力テンソルのデバイスを確認
#         # if isinstance(x, dict):
#         #     for key, value in x.items():
#         #         if isinstance(value, torch.Tensor):
#         #             print(f"Debug: forward input '{key}' tensor device: {value.device}")
#         #         elif isinstance(value, (list, tuple)):
#         #             print(f"Debug: forward input '{key}' type: {type(value)}, first element type: {type(value[0]) if value else 'empty'}")
#         #         else:
#         #             print(f"Debug: forward input '{key}' type: {type(value)}")
#         # elif isinstance(x, torch.Tensor):
#         #     print(f"Debug: forward input tensor device: {x.device}")
#         # else:
#         #     print(f"Debug: forward input type: {type(x)}")

#         # Ensure input x is on the model's device before passing to self.model
#         # This is redundant if move_to_device is used in the step methods,
#         # but provides an extra layer of safety.
#         # x_on_device = move_to_device(x, self.device) # move_to_device might be slow here if not needed


#         return self.model(x) # Pass the input as is, assuming step methods handle device placement


#     def training_step(self, batch, batch_idx):

#         # デバッグ用: ステップメソッドに渡されたバッチのデバイスを確認
#         if isinstance(batch, (list, tuple)):
#             print(f"Debug: training_step batch type: {type(batch)}")
#             if len(batch) > 0:
#                   if isinstance(batch[0], dict):
#                       print(f"Debug: training_step batch[0] (x) type: dict")
#                       for key, value in batch[0].items():
#                             if isinstance(value, torch.Tensor):
#                                 print(f"Debug: training_step batch[0]['{key}'] tensor device: {value.device}")
#                   elif isinstance(batch[0], torch.Tensor):
#                       print(f"Debug: training_step batch[0] tensor device: {batch[0].device}")
#                   else:
#                       print(f"Debug: training_step batch[0] type: {type(batch[0])}")

#             if len(batch) > 1 and isinstance(batch[1], torch.Tensor):
#                   print(f"Debug: training_step batch[1] (y) tensor device: {batch[1].device}")

#         # バッチ全体をデバイスに移動
#         batch_on_device = {key: val.to(self.device) if isinstance(val, torch.Tensor) else val for key, val in batch.items()}

#         # batch_on_device をアンパックして x と y を取り出す
#         # TimeSeriesDataSet のバッチ構造: (x_dict, y_tuple_or_tensor)
#         x, y_list = batch_on_device


#         # ターゲットをバッチから直接取得し、float型かつ正しい形状であることを確認
#         # y_listは通常、タプルやリストで、各予測ステップのターゲットが含まれる
#         # ここでは最初の予測ステップのターゲット (y_list[0]) を使用すると仮定
#         # Ensure y is a tensor and on the device
#         y = y_list[0].float() # .to(self.device) は move_to_device で既に行われている
#         if y.ndim == 1:
#             y = y.unsqueeze(-1) # BCEWithLogitsLoss のターゲットは (B, 1) または (B,) が必要

#         # モデルの予測結果を取得（ロジット）
#         # self.model(x) は辞書を返す
#         output = self.model(x)
#         # prediction キーから予測値を取り出す。形状は (B, T, output_size)。
#         # 二値分類なので output_size=1。通常は (B, T, 1)
#         # Note: TemporalFusionTransformer's prediction is typically (Batch_size, Prediction_length, output_size)
#         # For binary classification with output_size=1 and prediction_length=1, this is (B, 1, 1).
#         # Squeezing is needed to get (B,) or (B, 1).
#         pred_logits = output["prediction"].squeeze(-1) # Should be (B,) or (B, 1)


#         # Handing potential shape mismatch between pred_logits and y for BCEWithLogitsLoss
#         # BCEWithLogitsLoss expects inputs and targets to have the same shape.
#         # If pred_logits is (B, 1) and y is (B,), or vice versa, adjust.
#         if pred_logits.ndim != y.ndim:
#              if pred_logits.ndim == 1 and y.ndim == 2 and y.shape[1] == 1:
#                  y = y.squeeze(-1) # Make y (B,) if pred_logits is (B,)
#              elif pred_logits.ndim == 2 and pred_logits.shape[1] == 1 and y.ndim == 1:
#                  pred_logits = pred_logits.squeeze(-1) # Make pred_logits (B,) if y is (B,)
#              # Add other cases if needed, but (B,) and (B,1) are common for BCE.


#         # 手動で損失を計算
#         # BCEWithLogitsLoss は (B,) と (B,) または (B, 1) と (B, 1) を受け付ける
#         # if pred_logits.ndim == 1 and y.ndim == 2 and y.shape[1] == 1:
#         #     y = y.squeeze(-1) # (B,) - Redundant if handled above


#         # FocalLossを使用する場合 (Ensure FocalLoss can handle the shapes)
#         # loss = self.criterion(pred_logits, y)

#         # BCEWithLogitsLossを使用する場合
#         # Ensure both inputs and targets are on the same device before calculating loss
#         loss = self.criterion(pred_logits.to(self.device), y.to(self.device)) # Explicitly move before loss


#         # 損失をログに記録 (epoch 平均も記録)
#         self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

#         # 評価指標の計算とログ記録 (functional API を使用)
#         with torch.no_grad():
#             # ロジットを確率に変換
#             pred_probs = torch.sigmoid(pred_logits)
#             # 確率を0.5の閾値で二値予測に変換
#             predictions = (pred_probs > 0.5).int()

#             # ターゲットを int 型に変換
#             y_int = y.int()
#             # 形状を合わせる - Redundant if handled above, but good to be explicit
#             if predictions.ndim != y_int.ndim:
#                  if predictions.ndim == 1 and y_int.ndim == 2 and y_int.shape[1] == 1:
#                      y_int = y_int.squeeze(-1)
#                  elif predictions.ndim == 2 and predictions.shape[1] == 1 and y_int.ndim == 1:
#                      predictions = predictions.squeeze(-1)


#             # functional API でメトリクスを計算し、直接ログに記録
#             # on_step=False, on_epoch=True で epoch 平均を記録するのが一般的
#             # Step ごとのログが必要なら on_step=True も追加
#             # Ensure both predictions and y_int are on the same device
#             self.log('train_accuracy', F.binary_accuracy(predictions.to(self.device), y_int.to(self.device)), on_step=False, on_epoch=True, prog_bar=True)
#             self.log('train_precision', F.binary_precision(predictions.to(self.device), y_int.to(self.device)), on_step=False, on_epoch=True, prog_bar=True)
#             self.log('train_recall', F.binary_recall(predictions.to(self.device), y_int.to(self.device)), on_step=False, on_epoch=True, prog_bar=True)
#             self.log('train_f1', F.binary_f1_score(predictions.to(self.device), y_int.to(self.device)), on_step=False, on_epoch=True, prog_bar=True)


#         return loss


#     def validation_step(self, batch, batch_idx):

#         # デバッグ用: ステップメソッドに渡されたバッチのデバイスを確認
#         # if isinstance(batch, (list, tuple)):
#         #     print(f"Debug: training_step batch type: {type(batch)}")
#         #     if len(batch) > 0:
#         #           if isinstance(batch[0], dict):
#         #               print(f"Debug: training_step batch[0] (x) type: dict")
#         #               for key, value in batch[0].items():
#         #                     if isinstance(value, torch.Tensor):
#         #                         print(f"Debug: training_step batch[0]['{key}'] tensor device: {value.device}")
#         #           elif isinstance(batch[0], torch.Tensor):
#         #               print(f"Debug: training_step batch[0] tensor device: {batch[0].device}")
#         #           else:
#         #               print(f"Debug: training_step batch[0] type: {type(batch[0])}")

#         #     if len(batch) > 1 and isinstance(batch[1], torch.Tensor):
#         #           print(f"Debug: training_step batch[1] (y) tensor device: {batch[1].device}")
#         # バッチ全体をデバイスに移動
#         batch_on_device = {key: val.to(self.device) if isinstance(val, torch.Tensor) else val for key, val in batch.items()}

#         # batch_on_device をアンパックして x と y を取り出す
#         x, y_list = batch_on_device

#         # ターゲットを取得 (ここでは最初の予測ステップのターゲットを使用)
#         y = y_list[0].float() # .to(self.device) は move_to_device で既に行われている
#         if y.ndim == 1:
#             y = y.unsqueeze(-1)

#         # モデルの予測結果を取得（ロジット）
#         output = self.model(x)
#         pred_logits = output["prediction"].squeeze(-1)

#         # Handing potential shape mismatch between pred_logits and y for BCEWithLogitsLoss
#         if pred_logits.ndim != y.ndim:
#              if pred_logits.ndim == 1 and y.ndim == 2 and y.shape[1] == 1:
#                  y = y.squeeze(-1)
#              elif pred_logits.ndim == 2 and pred_logits.shape[1] == 1 and y.ndim == 1:
#                  pred_logits = pred_logits.squeeze(-1)


#         # 手動で損失を計算
#         # if pred_logits.ndim == 1 and y.ndim == 2 and y.shape[1] == 1:
#         #     y = y.squeeze(-1)

#         # FocalLossを使用する場合
#         # loss = self.criterion(pred_logits, y)

#         # BCEWithLogitsLossを使用する場合
#         loss = self.criterion(pred_logits.to(self.device), y.to(self.device))


#         # 損失をログに記録 (epoch 平均を記録)
#         self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

#         # 評価指標の計算とログ記録 (functional API を使用)
#         with torch.no_grad():
#             pred_probs = torch.sigmoid(pred_logits)
#             predictions = (pred_probs > 0.5).int()

#             y_int = y.int()
#             # 形状を合わせる - Redundant if handled above
#             if predictions.ndim != y_int.ndim:
#                  if predictions.ndim == 1 and y_int.ndim == 2 and y_int.shape[1] == 1:
#                      y_int = y_int.squeeze(-1)
#                  elif predictions.ndim == 2 and predictions.shape[1] == 1 and y_int.ndim == 1:
#                      predictions = predictions.squeeze(-1)


#             # functional API でメトリクスを計算し、直接ログに記録
#             # on_step=False, on_epoch=True で epoch 平均を記録
#             self.log('val_accuracy', F.binary_accuracy(predictions.to(self.device), y_int.to(self.device)), on_step=False, on_epoch=True, prog_bar=True)
#             self.log('val_precision', F.binary_precision(predictions.to(self.device), y_int.to(self.device)), on_step=False, on_epoch=True, prog_bar=True)
#             self.log('val_recall', F.binary_recall(predictions.to(self.device), y_int.to(self.device)), on_step=False, on_epoch=True, prog_bar=True)
#             self.log('val_f1', F.binary_f1_score(predictions.to(self.device), y_int.to(self.device)), on_step=False, on_epoch=True, prog_bar=True)

#         return loss

#     def test_step(self, batch, batch_idx):
#         # バッチ全体をデバイスに移動
#         batch_on_device = {key: val.to(self.device) if isinstance(val, torch.Tensor) else val for key, val in batch.items()}

#         # batch_on_device をアンパックして x と y を取り出す
#         x, y_list = batch_on_device

#         # ターゲットを取得 (ここでは最初の予測ステップのターゲットを使用)
#         y = y_list[0].float() # .to(self.device) は move_to_device で既に行われている
#         if y.ndim == 1:
#             y = y.unsqueeze(-1)

#         # モデルの予測結果を取得（ロジット）
#         output = self.model(x)
#         pred_logits = output["prediction"].squeeze(-1)


#         # Handing potential shape mismatch between pred_logits and y for BCEWithLogitsLoss
#         if pred_logits.ndim != y.ndim:
#              if pred_logits.ndim == 1 and y.ndim == 2 and y.shape[1] == 1:
#                  y = y.squeeze(-1)
#              elif pred_logits.ndim == 2 and pred_logits.shape[1] == 1 and y.ndim == 1:
#                  pred_logits = pred_logits.squeeze(-1)


#         # 手動で損失を計算
#         # if pred_logits.ndim == 1 and y.ndim == 2 and y.shape[1] == 1:
#         #     y = y.squeeze(-1)

#         # FocalLossを使用する場合
#         # loss = self.criterion(pred_logits, y)

#         # BCEWithLogitsLossを使用する場合
#         loss = self.criterion(pred_logits.to(self.device), y.to(self.device))

#         # 損失をログに記録 (テスト実行全体の平均を記録)
#         self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

#         # 評価指標の計算とログ記録 (functional API を使用)
#         with torch.no_grad():
#             pred_probs = torch.sigmoid(pred_logits)
#             predictions = (pred_probs > 0.5).int()

#             y_int = y.int()
#             # 形状を合わせる - Redundant if handled above
#             if predictions.ndim != y_int.ndim:
#                  if predictions.ndim == 1 and y_int.ndim == 2 and y_int.shape[1] == 1:
#                      y_int = y_int.squeeze(-1)
#                  elif predictions.ndim == 2 and predictions.shape[1] == 1 and y_int.ndim == 1:
#                      predictions = predictions.squeeze(-1)


#             # functional API でメトリクスを計算し、直接ログに記録
#             # on_step=False, on_epoch=True でテスト実行全体の平均を記録
#             self.log('test_accuracy', F.binary_accuracy(predictions.to(self.device), y_int.to(self.device)), on_step=False, on_epoch=True, prog_bar=True)
#             self.log('test_precision', F.binary_precision(predictions.to(self.device), y_int.to(self.device)), on_step=False, on_epoch=True, prog_bar=True)
#             self.log('test_recall', F.binary_recall(predictions.to(self.device), y_int.to(self.device)), on_step=False, on_epoch=True, prog_bar=True)
#             self.log('test_f1', F.binary_f1_score(predictions.to(self.device), y_int.to(self.device)), on_step=False, on_epoch=True, prog_bar=True)

#         # test_step からは通常何も返しません (または None を返します)
#         return

#     def configure_optimizers(self):
#         # TemporalFusionTransformer の configure_optimizers を呼び出す
#         # デフォルトでは Adam を使用
#         # ここでoptimizerとschedulerを設定することも可能
#         # self.hparams から learning_rate を取得して optimizer を設定するのが一般的
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
#         return optimizer

#     def predict_step(self, batch, batch_idx, dataloader_idx=0):
#         # バッチ全体をデバイスに移動
#         batch_on_device = move_to_device(batch, self.device)

#         # batch_on_device から x を取り出す (予測時は y は不要だが、構造を合わせる)
#         # predict_step の batch 構造は train/val/test と異なる場合があるため注意が必要です。
#         # TimeSeriesDataSet の predict_dataloader は通常、x_dict のみを返します。
#         # predict_step の batch 引数は DataLoader から渡されるので、その出力構造を確認してください。
#         # ここでは、batch が直接 x_dict であると仮定します。
#         x = batch_on_device # Assume batch_on_device is the x_dict after move_to_device

#         # model.predict は通常、入力 x を受け取ります。
#         # mode="raw" はロジットを返します。
#         # predict_step は通常、モデルの forward 出力をそのまま返すため、ここでは model.predict の結果をそのまま返します。
#         return self.model.predict(x, mode="raw") # raw: ロジット, prediction: 確率, quantiles: 分位数

#     def on_train_start(self):
#         # トレーニング開始時に pos_weight が確実にモデルと同じデバイスにあることを確認
#         if self.pos_weight is not None and self.pos_weight.device != self.device:
#             print(f"Debug: Moving pos_weight from {self.pos_weight.device} to {self.device} at train start.")
#             self.pos_weight = self.pos_weight.to(self.device)
#             # BCEWithLogitsLoss を再初期化する必要があるかもしれない
#             self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

#         # 念のため、self.model もデバイスに移動
#         # これは trainer.fit(model.to(device), ...) で行われるはずだが、念のため
#         if self.model.device != self.device:
#             print(f"Debug: Moving self.model from {self.model.device} to {self.device} at train start.")
#             self.model.to(self.device)

#         # また、optimizer も self.parameters() を参照するので、
#         # on_train_start で optimizer を再構築することでも解決する場合がある。
#         # configure_optimizers は fit の前に呼ばれるので、通常は問題ないはずだが。

# # --- インスタンス化の例 ---
# # train_dataset は TimeSeriesDataSet のインスタンスであると仮定
# # pos_weight_value は計算済みの pos_weight (float) であると仮定

# # # 計算済みの pos_weight (float)
# # calculated_pos_weight = 0.8571 # 例

# # # pos_weight を PyTorch の Tensor に変換し、デバイスに移動
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # pos_weight_tensor = torch.tensor(calculated_pos_weight, dtype=torch.float32).to(device)

# # # TFTModel のインスタンス化
# # # dataset は必須引数
# # # pos_weight をカスタム引数として渡す
# # # learning_rate は configure_optimizers で使用されるため、hparams に保存されるように渡す
# # model = TFTModel(
# #     dataset=train_dataset, # TimeSeriesDataSet インスタンスを渡す
# #     pos_weight=pos_weight_tensor, # 計算済みの pos_weight_tensor を渡す
# #     hidden_size=64,
# #     lstm_layers=2,
# #     attention_head_size=64, # attention_head_size は hidden_size と同じにすることが多い
# #     dropout=0.3,
# #     hidden_continuous_size=64,
# #     learning_rate=0.5e-3, # learning_rate を渡す
# #     log_interval=10,
# #     reduce_on_plateau_patience=4,
# #     # output_size=1 は TFTModel の __init__ で設定されるため不要
# #     # loss=nn.BCEWithLogitsLoss() も TFTModel の __init__ で設定されるため不要
# # )

# # print("TFTModel インスタンスが作成されました。")


# In[ ]:


# import pytorch_lightning as pl
# import torch
# from torchmetrics import Accuracy, Precision, Recall, F1Score

# class TFTModel(pl.LightningModule):
#     def __init__(self, dataset,
#                  hidden_size,
#                  lstm_layers,
#                  attention_head_size,
#                  dropout,
#                  hidden_continuous_size,
#                  learning_rate,
#                  **kwargs):
#         super().__init__()
#         self.save_hyperparameters()  # ロギング用にハイパーパラメータを保存
#         self.model = TemporalFusionTransformer.from_dataset(
#             dataset,
#             hidden_size=hidden_size,
#             lstm_layers=lstm_layers,
#             attention_head_size=attention_head_size,
#             dropout=dropout,
#             hidden_continuous_size=hidden_continuous_size,
#             **kwargs
#         )
#         # self.criterion = FocalLoss(alpha=0.9, gamma=3.0)
#         weight = 6
#         pos_weight = torch.tensor([weight]).to(device)
#         self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

#         self.val_accuracy = Accuracy(task="binary")
#         self.val_precision = Precision(task="binary")
#         self.val_recall = Recall(task="binary")
#         self.val_f1 = F1Score(task="binary")
#         self.test_accuracy = Accuracy(task="binary")
#         self.test_precision = Precision(task="binary")
#         self.test_recall = Recall(task="binary")
#         self.test_f1 = F1Score(task="binary")

#     def forward(self, x):
#         return self.model(x)

#     def training_step(self, batch, batch_idx):


#         x, y_list = batch  # 正しいアンパック方法



#         # 入力全体を適切にデバイスへ移動
#         x = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}

#         y = y_list[0]

#         y_1h = y[0].squeeze(-1)# 1h target (first prediction step, first target variable)
#         y_3h = y[1].squeeze(-1) # 3h target (second prediction step, second target variable)
#         y_6h = y[2].squeeze(-1) # 6h target (third prediction step, third target variable)

#         loss_mask = torch.tensor(compute_predictability_score(y_6h.cpu().numpy(), window_size=5), device=self.device, dtype=torch.float32)

#         output = self.model(x)
#         # pred_logits_1h = output[0][0][:, 0, 1] # ロジットのPositiveクラス側 (Index 1) を取得
#         # pred_logits_3h = output[0][0][:, 0, 3]
#         # pred_logits_6h = output[0][0][:, 0, 5]

#         pred_logits_1h = output["prediction"][0][:, 0, 1] # ロジットのPositiveクラス側 (Index 1) を取得
#         pred_logits_3h = output["prediction"][0][:, 0, 3]
#         pred_logits_6h = output["prediction"][0][:, 0, 5]

#         loss_1h = self.criterion(pred_logits_1h, y_1h.float())
#         loss_3h = self.criterion(pred_logits_3h, y_3h.float())
#         loss_6h = self.criterion(pred_logits_6h, y_6h.float())

#         # loss = 0.1*loss_1h + 0.2*loss_3h + 0.7 * loss_6h
#         loss = loss_6h

#         # loss = self.criterion(y_pred, y.float())  # yをfloatに変換
#         loss = (loss * loss_mask).sum() / loss_mask.sum()  # マスク付き平均損失
#         self.log('train_loss', loss)
#         return loss


#     def validation_step(self, batch, batch_idx):
#         x, y_list = batch  # バッチを正しくアンパック


#         x = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}


#         y = y_list[0]

#         y_1h = y[0].squeeze(-1) # 1h target (first prediction step, first target variable)
#         y_3h = y[1].squeeze(-1) # 3h target (second prediction step, second target variable)
#         y_6h = y[2].squeeze(-1) # 6h target (third prediction step, third target variable)

#         output = self.model(x)

#         # # --- Debugging Prints ---
#         # print("\n--- Validation Step Debug ---")
#         # print(f"Type of output: {type(output)}")
#         # if isinstance(output, (list, tuple)) and len(output) > 0:
#         #     print(f"Type of output[0][1]: {type(output[0][1])}")
#         #     if isinstance(output[0][1], torch.Tensor):
#         #         print(f"Shape of output[0][1]: {output[0][1].shape}")
#         #     elif isinstance(output[0][1], dict):
#         #         print(f"Keys in output[0][1]: {output[0][1].keys()}")
#         # elif isinstance(output, dict):
#         #      print(f"Keys in output: {output.keys()}")
#         #      if "prediction" in output and isinstance(output["prediction"], torch.Tensor):
#         #          print(f"Shape of output['prediction']: {output['prediction'].shape}")
#         # print("---------------------------\n")
#         # print("\n--- Validation Step Debug ---")
#         # print(f"Type of output: {type(output)}")
#         # if isinstance(output, (list, tuple)) and len(output) > 0:
#         #     print(f"Type of output[0][2]: {type(output[0][2])}")
#         #     if isinstance(output[0][2], torch.Tensor):
#         #         print(f"Shape of output[0][2]: {output[0][2].shape}")
#         #     elif isinstance(output[0][2], dict):
#         #         print(f"Keys in output[0][2]: {output[0][2].keys()}")
#         # elif isinstance(output, dict):
#         #      print(f"Keys in output: {output.keys()}")
#         #      if "prediction" in output and isinstance(output["prediction"], torch.Tensor):
#         #          print(f"Shape of output['prediction']: {output['prediction'].shape}")
#         # print("---------------------------\n")
#         # print("\n--- Validation Step Debug ---")
#         # print(f"Type of output: {type(output)}")
#         # if isinstance(output, (list, tuple)) and len(output) > 0:
#         #     print(f"Type of output[0][3]: {type(output[0][3])}")
#         #     if isinstance(output[0][1], torch.Tensor):
#         #         print(f"Shape of output[0][3]: {output[0][3].shape}")
#         #     elif isinstance(output[0][3], dict):
#         #         print(f"Keys in output[0][3]: {output[0][3].keys()}")
#         # elif isinstance(output, dict):
#         #      print(f"Keys in output: {output.keys()}")
#         #      if "prediction" in output and isinstance(output["prediction"], torch.Tensor):
#         #          print(f"Shape of output['prediction']: {output['prediction'].shape}")
#         # print("---------------------------\n")
#         # # --- End Debugging Prints ---

#         pred_logits_1h = output["prediction"][0][:, 0, 1] # ロジットのPositiveクラス側 (Index 1) を取得
#         pred_logits_3h = output["prediction"][0][:, 0, 3]
#         pred_logits_6h = output["prediction"][0][:, 0, 5]

#         # Focal Lossの計算
#         loss_1h = self.criterion(pred_logits_1h, y_1h.float())
#         loss_3h = self.criterion(pred_logits_3h, y_3h.float())
#         loss_6h = self.criterion(pred_logits_6h, y_6h.float())

#         # loss = 0.1*loss_1h + 0.2*loss_3h + 0.7 * loss_6h
#         loss = loss_6h
#         self.log('val_loss', loss, prog_bar=True)
#         # 評価指標の更新 (6hラベルを使用)
#         with torch.no_grad():
#             pred_probs_6h = torch.sigmoid(pred_logits_6h)
#             predictions_6h = (pred_probs_6h > 0.5).int()
#             self.val_accuracy(predictions_6h, y_6h.int())
#             self.val_precision(predictions_6h, y_6h.int())
#             self.val_recall(predictions_6h, y_6h.int())
#             self.val_f1(predictions_6h, y_6h.int()) # Update val_f1 metric

#         self.log('val_accuracy', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
#         self.log('val_precision', self.val_precision, on_step=False, on_epoch=True, prog_bar=True)
#         self.log('val_recall', self.val_recall, on_step=False, on_epoch=True, prog_bar=True)
#         self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True) # Log val_f1 at epoch end


#         return loss

#     def test_step(self, batch, batch_idx):
#         x, y_list = batch

#         x = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}

#         y = y_list[0]

#         y_1h = y[0].squeeze(-1)
#         y_3h = y[1].squeeze(-1)
#         y_6h = y[2].squeeze(-1)

#         output = self.model(x)

#         # prediction attribute からロジットを取得
#         pred_logits_1h = output["prediction"][0][:, 0, 1]
#         pred_logits_3h = output["prediction"][0][:, 0, 3]
#         pred_logits_6h = output["prediction"][0][:, 0, 5]

#         # Focal Lossの計算
#         loss_1h = self.criterion(pred_logits_1h, y_1h.float())
#         loss_3h = self.criterion(pred_logits_3h, y_3h.float())
#         loss_6h = self.criterion(pred_logits_6h, y_6h.float())

#         # loss = 0.1 * loss_1h + 0.2 * loss_3h + 0.7 * loss_6h
#         loss = loss_6h
#         self.log('test_loss', loss, prog_bar=True)

#         pred_probs_6h = torch.sigmoid(pred_logits_6h)

#         # 必要に応じて print で直接表示
#         print(f"[Test Batch {batch_idx}] y_hat: {pred_probs_6h.detach().cpu().numpy()[:5]}")
#         print(f"[Test Batch {batch_idx}] y_true: {y_6h.detach().cpu().numpy()[:5]}")

#         # 評価指標の更新 (6hラベルを使用)
#         with torch.no_grad():
#             pred_probs_6h = torch.sigmoid(pred_logits_6h)
#             predictions_6h = (pred_probs_6h > 0.4).int()
#             self.test_accuracy(predictions_6h, y_6h.int())
#             self.test_precision(predictions_6h, y_6h.int())
#             self.test_recall(predictions_6h, y_6h.int())
#             self.test_f1(predictions_6h, y_6h.int())

#         self.log('test_accuracy', self.test_accuracy, on_step=False, on_epoch=True, prog_bar=True)
#         self.log('test_precision', self.test_precision, on_step=False, on_epoch=True, prog_bar=True)
#         self.log('test_recall', self.test_recall, on_step=False, on_epoch=True, prog_bar=True)
#         self.log('test_f1', self.test_f1, on_step=False, on_epoch=True, prog_bar=True)

#         return loss

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

#     def predict_step(self, batch, batch_idx, dataloader_idx=0):
#         # batchはタプルなので、辞書に変換する (TimeSeriesDataSetからの出力は通常dict for x)
#         # yは予測時には不要なので、xだけ取り出す
#         x, _ = batch # Unpack, ignore y

#         # predict_stepでもデバイス移動が必要
#         x = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}

#         # model(x)は辞書を返すので、それをそのまま返す
#         return self(x)


# In[ ]:


import numpy as np

def filter_predictions_by_threshold(predictions, test_labels, threshold):
    """
    確率値の閾値に基づいて予測結果をフィルタリングし、
    確率が高い結果のみを出力してテストラベルと比較する関数

    Args:
        predictions (np.ndarray): TFTモデルの予測結果 (確率値)
        test_labels (np.ndarray): テストデータのラベル
        threshold (float): 確率値の閾値

    Returns:
        tuple: フィルタリングされた予測結果と対応するテストラベル
    """

    # 閾値以上の確率値を持つインデックスを取得
    high_prob_indices = np.where(predictions >= threshold)[0]

    # 予測結果とテストラベルをフィルタリング
    filtered_predictions = predictions[high_prob_indices]
    filtered_test_labels = test_labels[high_prob_indices]

    return filtered_predictions, filtered_test_labels


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: shape (B, T, 1) or (B, T)
        targets: shape (B, T) or (B, T, 1)
        """
        # reshape if needed
        if inputs.ndim == 3:
            inputs = inputs.squeeze(-1)  # (B, T)
        if targets.ndim == 3:
            targets = targets.squeeze(-1)  # (B, T)

        # apply sigmoid
        probas = torch.sigmoid(inputs)
        probas = torch.clamp(probas, 1e-6, 1 - 1e-6)

        # focal loss calculation
        loss_pos = -self.alpha * (1 - probas) ** self.gamma * targets * torch.log(probas)
        loss_neg = -(1 - self.alpha) * probas ** self.gamma * (1 - targets) * torch.log(1 - probas)

        loss = loss_pos + loss_neg  # (B, T)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # shape (B, T)


# In[ ]:




