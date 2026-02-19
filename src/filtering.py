#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, TensorDataset

class AutoEncoder(pl.LightningModule):
    def __init__(self, input_dim, latent_dim, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 4),
            nn.ReLU(),
            nn.Linear(latent_dim * 4, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim * 4),
            nn.ReLU(),
            nn.Linear(latent_dim * 4, input_dim) # Decoder output matches input dimension
        )

    def forward(self, x):
        # Input x is expected to be 2D: (batch_size, input_dim)
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        # The output x_hat is also 2D: (batch_size, input_dim)
        return x_hat

    def training_step(self, batch, batch_idx):
        x = batch[0] # Batch contains only the input tensor for reconstruction
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x) # Use MSE for reconstruction loss
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

def train_autoencoder(X_agg_scaled, latent_dim, batch_size=64, max_epochs=50):
    """
    訓練データでAutoEncoderを学習し、訓練データの潜在空間ベクトルとエンコーダーを返す関数。

    Args:
        X_agg_scaled (np.ndarray): スケーリング済みの集約特徴量NumPy配列 (n_samples, n_features)。
        latent_dim (int): 潜在空間の次元数。
        batch_size (int): 訓練時のバッチサイズ。
        max_epochs (int): 最大エポック数。

    Returns:
        tuple: 訓練データの潜在空間ベクトル (np.ndarray) と訓練済みエンコーダーモジュール (torch.nn.Module)。
    """
    # デバッグ用に、入力データの形状を確認
    print(f"Debug: train_autoencoder received X_agg_scaled with shape: {X_agg_scaled.shape}")

    # NumPy配列をPyTorch Tensorに変換
    X_tensor = torch.tensor(X_agg_scaled, dtype=torch.float32)

    # DataLoaderを作成 (ここではラベルは不要なので入力のみ)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # AutoEncoderモデルを初期化
    input_dim = X_agg_scaled.shape[1] # 入力次元数は集約特徴量の数
    print(f"Debug: AutoEncoder input_dim set to: {input_dim}") # 確認用
    model = AutoEncoder(input_dim, latent_dim)

    # PyTorch Lightning Trainerを初期化
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator='auto')

    # モデルの訓練
    trainer.fit(model, dataloader)

    # 訓練済みモデルからエンコーダー部分を抽出
    # デバイスをCPUに設定して潜在ベクトルを取得
    model.eval() # 評価モードに
    with torch.no_grad(): # 勾配計算を無効化
        # DataLoaderではなく、全ての訓練データを一度に処理して潜在ベクトルを取得
        # large datasetの場合はバッチ処理に分割が必要
        latent_train_tensor = model.encoder(X_tensor.to(model.device)) # データをモデルと同じデバイスに移動
        latent_train_np = latent_train_tensor.cpu().numpy() # NumPyに変換してCPUに戻す

    # エンコーダーモジュール自体を返す
    encoder = model.encoder

    return latent_train_np, encoder


def test_autoencoder(X_test_agg_scaled, encoder, batch_size=64):
    """
    訓練済みエンコーダーを使ってテストデータの潜在空間ベクトルを取得する関数。

    Args:
        X_test_agg_scaled (np.ndarray): スケーリング済みのテスト集約特徴量NumPy配列 (n_samples, n_features)。
        encoder (torch.nn.Module): 訓練済みのエンコーダーモジュール。
        batch_size (int): 処理時のバッチサイズ。

    Returns:
        np.ndarray: テストデータの潜在空間ベクトル (np.ndarray)。
    """
    # NumPy配列をPyTorch Tensorに変換
    X_test_tensor = torch.tensor(X_test_agg_scaled, dtype=torch.float32)

    # DataLoaderを作成
    dataset = TensorDataset(X_test_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False) # shuffleはFalse

    # エンコーダーを評価モードに設定し、デバイスに移動
    device = next(encoder.parameters()).device if next(encoder.parameters()).is_cuda else torch.device("cpu")
    encoder.eval()
    encoder.to(device)

    latent_test_list = []
    with torch.no_grad(): # 勾配計算を無効化
        for batch in dataloader:
            x_batch = batch[0].to(device) # データをデバイスに移動
            latent_batch = encoder(x_batch)
            latent_test_list.append(latent_batch.cpu().numpy()) # NumPyに変換してCPUに戻す

    # リストを結合してNumPy配列にする
    latent_test_np = np.concatenate(latent_test_list, axis=0)

    return latent_test_np


# In[ ]:


import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# Assuming the following functions are defined in preceding cells:
# sliding_window
# transform_sequence_data
# transform_window_list_to_numpy_and_index
# scale_features # Although scaling of aggregated features happens later, the function is general.
# prepare_data_for_clustering # We are essentially extracting parts of this for the new function.


def prepare_3d_data_processed(df, continuous_features, categorical_features, date_features,
                    sequence_length=60, nan_threshold=200):
    """
    戻り値を 4 つに拡張:
      - X_3d_numpy_filtered
      - original_indices_filtered  (start index: df_processed.index)
      - original_feature_names_list
      - df_processed               ★ これを追加（dropna 後のベースDF）
    """
    print("--- Starting 3D Data Preparation ---")

    initial_selected_columns = (
        ["close", "high", "low", "volume_ETH"]
        + ['MA_t_6', 'MA_t_24', 'MA_t_72', 'MA_t_168']
        + ['upper', 'lower']
        + continuous_features
        + categorical_features
        + date_features
    )

    existing_selected_columns = [col for col in initial_selected_columns if col in df.columns]
    if len(existing_selected_columns) != len(initial_selected_columns):
        missing_cols = list(set(initial_selected_columns) - set(existing_selected_columns))
        print(f"Warning: The following selected columns are not found in the input DataFrame and will be skipped: {missing_cols}")

    if not existing_selected_columns:
        print("Error: No valid columns selected from the input DataFrame. Returning empty results.")
        return np.array([]), pd.DatetimeIndex([]), [], pd.DataFrame()

    df_processed = df[existing_selected_columns].copy()

    existing_categorical_features_in_processed = [col for col in categorical_features if col in df_processed.columns]
    existing_categorical_features_in_processed = list(dict.fromkeys(existing_categorical_features_in_processed))
    if existing_categorical_features_in_processed:
        df_processed.loc[:, existing_categorical_features_in_processed] = (
            df_processed.loc[:, existing_categorical_features_in_processed].astype('category')
        )
        print(f"Converted columns to 'category' dtype: {existing_categorical_features_in_processed}")

    # NaN 多すぎカラムを削除
    nan_counts = df_processed.isnull().sum()
    columns_to_drop = nan_counts[nan_counts >= nan_threshold].index.tolist()
    if columns_to_drop:
        print(f"Dropping columns with >= {nan_threshold} NaNs: {columns_to_drop}")
        df_processed = df_processed.drop(columns=columns_to_drop)

    # NaN 行を削除（★ この index が今後の「正しい時間軸」になる）
    initial_rows = df_processed.shape[0]
    if isinstance(df_processed.index, pd.DatetimeIndex):
        df_processed = df_processed.dropna().copy()
    else:
        print("Warning: Input DataFrame does not have a DatetimeIndex. Using default index after dropna.")
        df_processed = df_processed.dropna().copy()

    print(f"Dropped {initial_rows - df_processed.shape[0]} rows with NaNs. Remaining rows: {df_processed.shape[0]}")

    if df_processed.shape[0] == 0:
        print("Error: No data remains after dropping NaNs. Returning empty results.")
        return np.array([]), pd.DatetimeIndex([]), [], pd.DataFrame()

    original_feature_names_list = df_processed.columns.tolist()
    print(f"Features remaining after NaN processing: {len(original_feature_names_list)}")

    # sliding window
    print(f"Creating sliding windows with length {sequence_length}...")
    X_window_list = sliding_window(df_processed, sequence_length)
    print(f"Created {len(X_window_list)} sliding windows.")

    if not X_window_list:
        print("Warning: No sliding windows created. Returning empty results.")
        return np.array([]), pd.DatetimeIndex([]), original_feature_names_list, df_processed

    print("Converting window list to 3D NumPy array and extracting original indices...")
    X_3d_numpy_all_sequences, original_end_indices_all_sequences = transform_window_list_to_numpy_and_index(X_window_list)

    print(f"Original X_3d_numpy shape: {X_3d_numpy_all_sequences.shape}")
    print(f"Original end indices length: {len(original_end_indices_all_sequences)}")

    # start index は dropna 後 df_processed の index をそのまま使う
    original_start_indices_all_sequences = df_processed.index[:len(X_window_list)]

    X_3d_numpy_filtered = X_3d_numpy_all_sequences
    original_indices_filtered = original_start_indices_all_sequences

    print(f"Final X_3d_numpy shape: {X_3d_numpy_filtered.shape}")
    print(f"Final original_indices (start times) length: {len(original_indices_filtered)}")

    # ★ df_processed を追加で返す
    return X_3d_numpy_filtered, original_indices_filtered, original_feature_names_list, df_processed


# In[ ]:


# Example usage (assuming original_df, X_3d_numpy, original_indices_filtered, feature_names_3d, strategy_labels are defined):
# # Define parameters for labeling
# horizon = 6 # Example horizon
# rolling_window_size = 60 # Example window size
# target_positive_rate = 0.1 # Example target rate (10% positive labels)
# ewma_alpha = 0.2 # Example EWMA alpha
# lower_threshold_multiplier = -1.0 # Example: lower threshold is -1 * upper threshold
# hysteresis_M = 3 # Example hysteresis window size

# # Ensure original_df is available and has a DatetimeIndex and 'close' column
# if 'original_df' not in locals() or not isinstance(original_df, pd.DataFrame) or 'close' not in original_df.columns or not isinstance(original_df.index, pd.DatetimeIndex):
#      print("Error: 'original_df' is not a valid DataFrame with a DatetimeIndex and 'close' column. Cannot generate labels.")
#      strategy_binary_labels = {}
# else:
#      # Call the function to generate strategy-specific binary labels
#      strategy_binary_labels = create_strategy_specific_binary_labels(
#          original_df=original_df,
#          X_3d_numpy=X_3d_numpy_filtered, # Use the filtered 3D data
#          original_indices_filtered=original_indices_filtered, # Use the filtered original indices
#          feature_names_3d=original_feature_names_list, # Use feature names from 3D data
#          strategy_labels=integrated_strategy_names, # Use the integrated strategy names
#          horizon=horizon,
#          rolling_window_size=rolling_window_size,
#          target_positive_rate=target_positive_rate,
#          ewma_alpha=ewma_alpha,
#          lower_threshold_multiplier=lower_threshold_multiplier,
#          hysteresis_M=hysteresis_M
#      )

#      # Print summary of generated labels per strategy
#      print("\n--- Summary of Generated Strategy-Specific Binary Labels ---")
#      if strategy_binary_labels:
#          for strategy, labels in strategy_binary_labels.items():
#              if isinstance(labels, np.ndarray):
#                  print(f"Strategy '{strategy}': {len(labels)} labels. Distribution: {np.unique(labels, return_counts=True)}")
#              else:
#                  print(f"Strategy '{strategy}': Processing failed. Details: {labels}")
#      else:
#          print("No strategy-specific binary labels were generated.")


# # The strategy_binary_labels dictionary now contains the final smoothed and re-quantilized binary labels
# # for each strategy, aligned with the sequences belonging to that strategy.
# # These labels can be used for training strategy-specific models.


# In[ ]:


# -*- coding: utf-8 -*-
from scipy.stats import skew, kurtosis
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, TensorDataset
import hdbscan # Import hdbscan
import os
from sklearn.cluster import KMeans # Import KMeans for testing
import numpy as np # Ensure numpy is imported
from sklearn.mixture import GaussianMixture # Import GaussianMixture

# Assuming AutoEncoder class is defined elsewhere (e.g., cell ZERvB1uf31Va) - This will be replaced
# Assuming train_autoencoder and test_autoencoder functions are defined elsewhere (e.g., cell ZERvB1uf31Va) - These will be replaced
# Assuming create_aggregated_features is defined elsewhere (e.g., cell JF5y50zNGmIN)
# Assuming CNNEncoder1D is defined elsewhere (newly added cell)
# Assuming ContrastiveLearningModule is defined elsewhere (newly added cell)
# Assuming train_contrastive_learning_model is defined elsewhere (newly added cell)
# Assuming extract_latent_vectors is defined elsewhere (newly added cell)
# Assuming data augmentation functions (add_gaussian_noise, random_scale, etc.) are defined elsewhere (newly added cell)
# Assuming DECModule is defined elsewhere (newly added cell)


def perform_clustering_on_subset(X_subset_3d, feature_names_3d, seq_len, original_indices_subset, latent_dim,
                                 clustering_method='hdbscan', # Add parameter to choose clustering method
                                 hdbscan_params=None, # Keep hdbscan_params but make it optional and default to None
                                 kmeans_params=None, # Add kmeans_params as an optional dictionary
                                 n_clusters=10, # Use a single parameter for desired number of clusters (for KMeans, GMM, and DEC)
                                 train_contrastive_learning_flag=True, # Flag to control CL training
                                 trained_encoder=None, # Accept a pre-trained encoder (for inference)
                                 encoder_save_path=None, # Path to save/load the trained encoder
                                 cl_learning_rate=1e-3, # CL training learning rate
                                 cl_temperature=0.07,  # CL training temperature
                                 cl_augmentation_strategies=None, # CL augmentation strategies
                                 batch_size_cl_train=64, # Batch size for CL training
                                 max_epochs_cl_train=100, # Max epochs for CL training
                                 batch_size_latent_extraction=128, # Batch size for latent extraction
                                 # Add DEC parameters
                                 use_dec_finetuning=False, # Flag to enable DEC fine-tuning
                                 dec_alpha=1.0,          # DEC Student-t distribution alpha
                                 dec_learning_rate=1e-3, # DEC training learning rate
                                 dec_finetune_encoder=False, # DEC training: finetune encoder or not
                                 max_epochs_dec_train=100, # Max epochs for DEC training
                                 dec_save_path=None       # Path to save/load DEC model state_dict
                                ):
    """
    Performs Contrastive Learning-based dimensionality reduction (with optional training),
    optional DEC fine-tuning, and clustering (HDBSCAN or KMeans) on a subset of 3D raw sequences.

    Args:
        X_subset_3d (np.ndarray): Subset of **3D** numpy array of raw sequences (n_subset_samples, seq_len, n_features_3d).
        feature_names_3d (list): List of feature names for the 3D data.
        seq_len (int): The length of each sequence in X_subset_3d.
        original_indices_subset (pd.DatetimeIndex or np.ndarray): Original indices corresponding to X_subset_3d
                                                                   (n_subset_samples,).
        latent_dim (int): The dimension of the latent space for the encoder.
        clustering_method (str): The clustering method to use ('hdbscan', 'kmeans', 'gmm').
        hdbscan_params (dict, optional): Dictionary of parameters for HDBSCAN. Required if clustering_method is 'hdbscan'.
        kmeans_params (dict, optional): Dictionary of parameters for KMeans. Required if clustering_method is 'kmeans'.
                                        If None, default KMeans parameters will be used.
        n_clusters (int): The desired number of clusters for KMeans, GMM, and DEC.
        train_contrastive_learning_flag (bool): If True, train the Contrastive Learning model on X_subset_3d.
                                               If False, use trained_encoder or load from encoder_save_path.
        trained_encoder (torch.nn.Module, optional): Pre-trained encoder module.
                                                    Used if train_contrastive_learning_flag is False and encoder_save_path is None.
        encoder_save_path (str, optional): Path to save the trained encoder's state_dict (if training)
                                           or load the encoder from (if not training and trained_encoder is None).
                                           Should be a file path (e.g., 'cnn_encoder_state_dict.pth').
        cl_learning_rate (float): Learning rate for Contrastive Learning training.
        cl_temperature (float): Temperature for Contrastive Learning InfoNCE loss.
        cl_augmentation_strategies (list, optional): List of augmentation functions and kwargs for CL training.
        batch_size_cl_train (int): Batch size for Contrastive Learning training if train_contrastive_learning_flag is True.
        max_epochs_cl_train (int): Max epochs for Contrastive Learning training if train_contrastive_learning_flag is True.
        batch_size_latent_extraction (int): Batch size for extracting latent vectors.
        use_dec_finetuning (bool): If True, perform DEC fine-tuning after CL.
        dec_alpha (float): The alpha parameter for the Student-t distribution in DEC.
        dec_learning_rate (float): Learning rate for DEC training.
        dec_finetune_encoder (bool): If True, finetune encoder during DEC training.
        max_epochs_dec_train (int): Max epochs for DEC training.
        dec_save_path (str, optional): Path to save/load DEC model state_dict.


    Returns:
        tuple: (final_representation, cluster_labels, original_indices_subset)
               - final_representation (np.ndarray or None): Latent vectors (if DEC not used) or DEC soft assignments (if DEC used) (n_subset_samples, latent_dim or n_clusters) or None.
               - cluster_labels (np.ndarray or None): Cluster labels (n_subset_samples,) or None if clustering fails/skipped.
               - original_indices_subset (pd.DatetimeIndex or np.ndarray): The original indices subset passed in.
               Returns (None, None, original_indices_subset) if critical processing fails.
    """
    if X_subset_3d is None or X_subset_3d.shape[0] == 0:
        print("Warning: X_subset_3d is empty. Skipping processing for this subset.")
        return None, None, original_indices_subset

    # Assert that the input is indeed 3D
    if X_subset_3d.ndim != 3:
         print(f"Error: perform_clustering_on_subset expected 3D input (n_samples, seq_len, n_features), but got {X_subset_3d.ndim}D input with shape {X_subset_3d.shape}.")
         print("Please ensure you are passing the 3D raw sequence data.")
         return None, None, original_indices_subset

    print(f"Starting pipeline for subset with shape: {X_subset_3d.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = None # Initialize encoder
    latent_representation = None # Initialize latent representation after CL


    # Determine input_channels and seq_len from the 3D data shape
    n_subset_samples, actual_seq_len, actual_n_features = X_subset_3d.shape
    input_channels_encoder = actual_n_features # Number of features is input channels for Conv1D
    seq_len_encoder = actual_seq_len         # Sequence length is input length for Conv1D


    # --- 1. Contrastive Learning: Train or Load and Extract Latent Vectors ---
    print("\n--- Step 1: Contrastive Learning (Train/Load & Extract) ---")

    if train_contrastive_learning_flag:
        print("Training Contrastive Learning model (Encoder)...")
        try:
            encoder, _ = train_contrastive_learning_model(
                X_data=X_subset_3d, # Pass the 3D subset data
                input_channels=input_channels_encoder,
                seq_len=seq_len_encoder,
                latent_dim=latent_dim,
                learning_rate=cl_learning_rate,
                temperature=cl_temperature,
                augmentation_strategies=cl_augmentation_strategies,
                batch_size=batch_size_cl_train,
                max_epochs=max_epochs_cl_train,
                encoder_save_path=encoder_save_path # Pass path to save encoder
            )
            if encoder is None:
                 print("Error: Contrastive Learning model training failed. Cannot proceed.")
                 return None, None, original_indices_subset

            print("CL training complete.")

        except Exception as e:
            print(f"Error during Contrastive Learning model training: {e}. Cannot proceed.")
            return None, None, original_indices_subset

    else: # train_contrastive_learning_flag is False
        print("Using existing encoder (loading or provided)...")
        encoder = trained_encoder
        if encoder is None and encoder_save_path:
            print(f"Attempting to load encoder state_dict from {encoder_save_path}...")
            try:
                encoder = CNNEncoder1D(input_channels=input_channels_encoder, seq_len=seq_len_encoder, latent_dim=latent_dim).to(device)
                encoder.load_state_dict(torch.load(encoder_save_path, map_location=device))
                encoder.to(device)
                print(f"Loaded encoder state_dict from {encoder_save_path}")
            except Exception as e:
                print(f"Error loading encoder state_dict from {encoder_save_path}: {e}. Cannot proceed without a valid encoder.")
                return None, None, original_indices_subset
        elif encoder is None and not encoder_save_path:
             print("Error: train_contrastive_learning_flag is False, but no trained_encoder or encoder_save_path provided. Cannot proceed without a valid encoder.")
             return None, None, original_indices_subset

        print("Encoder available.")


    # Extract latent vectors after CL (or from loaded encoder)
    print("Extracting latent vectors using the encoder...")
    try:
        latent_representation = extract_latent_vectors(
            encoder=encoder,
            X_data=X_subset_3d, # Pass the 3D subset data for extraction
            batch_size=batch_size_latent_extraction
        )
        if latent_representation is None or latent_representation.shape[0] == 0:
             print("Error: Latent representation could not be obtained or is empty after CL. Cannot proceed.")
             return None, None, original_indices_subset

        print("Latent vector extraction complete after CL.")
        print(f"Debug: Shape of latent representation after CL: {latent_representation.shape}") # Debug print

    except Exception as e:
        print(f"Error during latent vector extraction after CL: {e}. Cannot proceed.")
        return None, None, original_indices_subset


    # --- Handle NaNs and Infs in Latent Representation after CL ---
    if np.isnan(latent_representation).any() or np.isinf(latent_representation).any():
        print("Warning: Latent representation contains NaN or Inf values after CL. Replacing with 0 and clamping finite values.")
        latent_representation = np.nan_to_num(latent_representation, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
        print("Debug: Replaced NaN/Inf values in latent representation.")


    # --- 2. Optional DEC Fine-tuning ---
    final_representation_for_clustering = latent_representation # Default to CL latent vectors
    dec_cluster_assignments = None # Initialize DEC assignments

    if use_dec_finetuning:
        print("\n--- Step 2: Optional DEC Fine-tuning ---")
        # Ensure enough samples for DEC initialization (KMeans)
        if n_subset_samples < n_clusters:
             print(f"Warning: Not enough samples ({n_subset_samples}) for DEC initialization with {n_clusters} clusters. Skipping DEC.")
             # Proceed with CL latent vectors for clustering
             use_dec_finetuning = False # Disable DEC if not enough samples
        else:
            try:
                 # Create DataLoader for the data for DEC training
                 # DEC needs original data (X_subset_3d) as input
                 X_tensor_dec = torch.tensor(X_subset_3d, dtype=torch.float32).transpose(1, 2) # Shape (n_samples, n_features, seq_len)
                 dataset_dec = TensorDataset(X_tensor_dec)
                 dataloader_dec = DataLoader(dataset_dec, batch_size=batch_size_cl_train, shuffle=True, num_workers=os.cpu_count() // 2 or 1) # Use CL batch size for DEC training

                 # Initialize DEC Module
                 dec_model = DECModule(
                     encoder=encoder, # Use the trained encoder from CL
                     n_clusters=n_clusters,
                     alpha=dec_alpha,
                     learning_rate=dec_learning_rate,
                     finetune_encoder=dec_finetune_encoder # Control encoder finetuning during DEC
                 ).to(device) # Move DEC model to device


                 # Initialize DEC cluster centroids using KMeans on CL latent vectors
                 # Pass the DataLoader for the *entire* dataset
                 dec_model.initialize_centroids(DataLoader(TensorDataset(torch.tensor(X_subset_3d, dtype=torch.float32).transpose(1, 2)), batch_size=batch_size_latent_extraction, shuffle=False)) # Use extraction batch size for centroid init


                 # Load DEC state_dict if path is provided and exists, and not training DEC
                 if dec_save_path and not train_contrastive_learning_flag: # Only load if not training CL (implies not training DEC from scratch)
                      if os.path.exists(dec_save_path):
                           print(f"Attempting to load DEC model state_dict from {dec_save_path}...")
                           try:
                               dec_model.load_state_dict(torch.load(dec_save_path, map_location=device))
                               print(f"Loaded DEC model state_dict from {dec_save_path}")
                               # If loaded, assume training is skipped and use loaded model for assignments
                               print("Skipping DEC training, using loaded model for assignments.")
                               max_epochs_dec_train = 0 # Ensure training loop doesn't run


                           except Exception as e:
                               print(f"Error loading DEC model state_dict from {dec_save_path}: {e}. Proceeding with DEC training.")
                               # If loading fails, proceed with training

                 # Initialize DEC Trainer only if max_epochs > 0
                 if max_epochs_dec_train > 0:
                     print("Starting DEC fine-tuning...")
                     dec_trainer = pl.Trainer(
                         max_epochs=max_epochs_dec_train,
                         accelerator='auto',
                         # Add logging, checkpointing etc. as needed for DEC
                         # callbacks=[pl.callbacks.EarlyStopping(monitor='train_loss_dec', patience=10)] # Example early stopping
                     )
                     dec_trainer.fit(dec_model, dataloader_dec)
                     print("DEC fine-tuning finished.")

                 # Get final representation for clustering after DEC
                 # Option 1: Use DEC's soft assignments (q) as features for clustering (shape: n_samples, n_clusters)
                 # Option 2: Use the latent vectors (z) after DEC fine-tuning (shape: n_samples, latent_dim)
                 # Option 3: Use the hard assignments (argmax(q)) directly as labels (if n_clusters is fixed)

                 # Let's decide based on clustering_method. If clustering_method is KMeans/GMM,
                 # it expects features (vectors), not assignments. If clustering_method is HDBSCAN,
                 # it also expects features.
                 # So, it's better to use the *latent vectors* (z) after DEC fine-tuning as the input
                 # for the final clustering step, unless the goal is to use DEC's hard assignments directly.
                 # The request is to cluster on *potential vectors*. DEC fine-tunes the latent space.
                 # So, extract latent vectors using the fine-tuned encoder.

                 # Re-extract latent vectors using the potentially fine-tuned encoder
                 print("Extracting latent vectors using the potentially fine-tuned encoder...")
                 dec_model.encoder.eval() # Ensure encoder is in eval mode for extraction
                 final_latent_after_dec = extract_latent_vectors(
                     encoder=dec_model.encoder,
                     X_data=X_subset_3d, # Use original data with fine-tuned encoder
                     batch_size=batch_size_latent_extraction
                 )
                 if final_latent_after_dec is None or final_latent_after_dec.shape[0] == 0:
                      print("Error: Latent representation could not be obtained after DEC. Using CL latent vectors.")
                      final_representation_for_clustering = latent_representation # Fallback
                 else:
                      final_representation_for_clustering = final_latent_after_dec # Use DEC-tuned latent vectors
                      print("Using DEC-tuned latent vectors for final clustering.")
                      print(f"Debug: Shape of latent representation after DEC: {final_representation_for_clustering.shape}") # Debug print


                 # Optional: Get DEC hard assignments for potential use or comparison
                 dec_model.eval()
                 with torch.no_grad():
                      X_tensor_dec_all = torch.tensor(X_subset_3d, dtype=torch.float32).transpose(1, 2).to(device)
                      final_q, _ = dec_model(X_tensor_dec_all) # Get final soft assignments
                      dec_cluster_assignments = torch.argmax(final_q, dim=1).cpu().numpy() # Get hard assignments
                      print(f"Debug: Shape of DEC hard assignments: {dec_cluster_assignments.shape}") # Debug print


                 # Save DEC model state_dict if path is provided
                 if dec_save_path:
                     try:
                         # Ensure the directory exists
                         save_dir = os.path.dirname(dec_save_path)
                         if save_dir and not os.path.exists(save_dir):
                             os.makedirs(save_dir)
                             print(f"Created directory for saving DEC model: {save_dir}")

                         # Save the state_dict of the entire DEC model
                         torch.save(dec_model.state_dict(), dec_save_path)
                         print(f"Saved trained DEC model state_dict to {dec_save_path}")
                     except Exception as e:
                         print(f"Warning: Failed to save trained DEC model state_dict to {dec_save_path}: {e}")


            except Exception as e:
                print(f"Error during DEC fine-tuning: {e}. Skipping DEC.")
                # Proceed with CL latent vectors for clustering
                use_dec_finetuning = False # Ensure flag is false after failure
                final_representation_for_clustering = latent_representation # Ensure fallback


    # --- Handle NaNs and Infs in Final Representation before Clustering ---
    if np.isnan(final_representation_for_clustering).any() or np.isinf(final_representation_for_clustering).any():
        print("Warning: Final representation for clustering contains NaN or Inf values. Replacing with 0 and clamping finite values.")
        final_representation_for_clustering = np.nan_to_num(final_representation_for_clustering, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
        print("Debug: Replaced NaN/Inf values in final representation.")


    # --- 3. Clustering on Final Representation ---
    print("\n--- Step 3: Clustering on Final Representation ---")
    cluster_labels = None # Initialize cluster_labels

    # Determine the input for clustering based on whether DEC was used/successful
    # If DEC was used and produced valid assignments, we might use those,
    # or use the fine-tuned latent vectors. The request is to cluster on *potential vectors*.
    # So, use `final_representation_for_clustering` which contains either CL latent vectors
    # or DEC-tuned latent vectors.

    clustering_input = final_representation_for_clustering

    # Ensure enough samples for clustering based on the chosen method and its parameters
    min_samples_needed_for_method = 1 # Default minimum


    try:
        if clustering_method == 'hdbscan':
            print("Using HDBSCAN clustering on final representation...")
            # Default HDBSCAN parameters if none provided
            if hdbscan_params is None:
                hdbscan_params = {
                    'min_cluster_size': max(10, int(np.sqrt(n_subset_samples))), # Suggested starting point
                    'min_samples': None, # Often min_samples = min_cluster_size or slightly smaller
                    'cluster_selection_epsilon': 0.0,
                    'gen_min_span_tree': True, # Keep True for potential visualization later
                    'random_state': 42
                }
                print(f"Using default HDBSCAN parameters: {hdbscan_params}")
            else:
                 print(f"Using provided HDBSCAN parameters: {hdbscan_params}")


            # Ensure enough samples for HDBSCAN
            min_samples_needed = hdbscan_params.get('min_cluster_size', 10)
            if n_subset_samples < min_samples_needed: # Use n_subset_samples which is the total samples in the input subset
                 print(f"Warning: Not enough samples ({n_subset_samples}) for HDBSCAN with min_cluster_size={min_samples_needed}. Assigning all to -1.")
                 cluster_labels = np.full(n_subset_samples, -1, dtype=int)
            else:
                print(f"Debug: Shape of clustering input immediately before HDBSCAN fit: {clustering_input.shape}") # Debug Print
                clusterer = hdbscan.HDBSCAN(**hdbscan_params)
                # HDBSCAN fit_predict handles noise (-1 label)
                cluster_labels = clusterer.fit_predict(clustering_input)
                print(f"HDBSCAN clustering complete. Found {len(np.unique(cluster_labels)) - (1 if -1 in cluster_labels else 0)} clusters (excluding noise).")
                print(f"Cluster label distribution:\n{pd.Series(cluster_labels).value_counts().sort_index()}")

            # HDBSCAN stability (persistence) is typically evaluated during parameter tuning,
            # not as a single metric after fitting. Evaluation flag is less applicable here.


        elif clustering_method == 'kmeans':
            print(f"Using KMeans clustering with n_clusters={n_clusters}...") # Use the n_clusters parameter
            # Default KMeans parameters if none provided
            if kmeans_params is None:
                 kmeans_params = {
                     'n_clusters': n_clusters, # Use the n_clusters parameter
                     'random_state': 42,
                     'n_init': 10 # Or 'auto' in newer sklearn versions
                 }
                 print(f"Using default KMeans parameters: {kmeans_params}")
            else:
                 # Ensure n_clusters from argument takes precedence if provided in kmeans_params
                 if 'n_clusters' in kmeans_params:
                      print(f"Warning: n_clusters provided in both function argument ({n_clusters}) and kmeans_params ({kmeans_params['n_clusters']}). Using n_clusters from function argument.")
                 kmeans_params['n_clusters'] = n_clusters # Ensure consistency

                 # Ensure n_init is set if not provided
                 if 'n_init' not in kmeans_params:
                     kmeans_params['n_init'] = 10 # Default n_init


                 print(f"Using provided KMeans parameters: {kmeans_params}")


            # Ensure enough samples for KMeans
            if n_subset_samples < n_clusters: # Use n_subset_samples and n_clusters parameter
                 print(f"Warning: Not enough samples ({n_subset_samples}) for KMeans with n_clusters={n_clusters}. Assigning all to -1.")
                 cluster_labels = np.full(n_subset_samples, -1, dtype=int) # Or raise error
            else:
                kmeans = KMeans(**kmeans_params) # Use the combined parameters
                print(f"Debug: Shape of clustering input immediately before KMeans fit: {clustering_input.shape}") # Debug Print
                cluster_labels = kmeans.fit_predict(clustering_input)
                print(f"KMeans clustering complete. Found {len(np.unique(cluster_labels))} clusters.")
                print(f"Cluster label distribution:\n{pd.Series(cluster_labels).value_counts().sort_index()}")

            # Evaluate KMeans if requested
            # Evaluation metrics require at least 2 distinct labels and more than 1 sample
            if evaluate_clustering and len(np.unique(cluster_labels)) > 1 and n_subset_samples > 1:
                 print(f"Evaluating KMeans clustering using metric: {metric_for_evaluation}")
                 evaluation_metrics = {} # Initialize evaluation_metrics dictionary
                 try:
                     if metric_for_evaluation == 'silhouette':
                         # Silhouette score requires distance metric (default Euclidean for KMeans is fine)
                         score = silhouette_score(clustering_input, cluster_labels)
                         evaluation_metrics = {'silhouette_score': score}
                     elif metric_for_evaluation == 'davies_bouldin':
                         score = davies_bouldin_score(clustering_input, cluster_labels)
                         evaluation_metrics = {'davies_bouldin_score': score} # Lower is better
                     elif metric_for_evaluation == 'calinski_harabasz':
                         score = calinski_harabasz_score(clustering_input, cluster_labels)
                         evaluation_metrics = {'calinski_harabasz_score': score} # Higher is better
                     else:
                         print(f"Warning: Unknown evaluation metric '{metric_for_evaluation}'. Skipping evaluation.")

                 except Exception as e:
                     print(f"Error during KMeans evaluation: {e}. Skipping evaluation.")
                     evaluation_metrics = None


        elif clustering_method == 'gmm':
            print(f"Using Gaussian Mixture Model clustering with n_components={n_clusters}...") # Use n_clusters as n_components
            # Ensure enough samples for GMM (need at least n_components)
            if n_subset_samples < n_clusters: # Use n_subset_samples and n_clusters parameter
                 print(f"Warning: Not enough samples ({n_subset_samples}) for GMM with n_components={n_clusters}. Assigning all to -1.")
                 cluster_labels = np.full(n_subset_samples, -1, dtype=int) # Or raise error
            else:
                # GMM can be sensitive to scale, consider scaling latent_vectors if not already done
                # scaler_gmm = StandardScaler()
                # clustering_input_scaled = scaler_gmm.fit_transform(clustering_input)
                gmm = GaussianMixture(n_components=n_clusters, random_state=42) # Use n_clusters as n_components
                print(f"Debug: Shape of clustering input immediately before GMM fit: {clustering_input.shape}") # Debug Print
                cluster_labels = gmm.fit_predict(clustering_input)
                print(f"GMM clustering complete. Found {len(np.unique(cluster_labels))} components/clusters.")
                print(f"Cluster label distribution:\n{pd.Series(cluster_labels).value_counts().sort_index()}")

                # GMM also provides BIC/AIC for component selection, which is often done externally
                # bic = gmm.bic(clustering_input)
                # aic = gmm.aic(clustering_input)
                # print(f"GMM BIC: {bic}, AIC: {aic}")


            # Evaluate GMM if requested
            # Evaluation metrics require at least 2 distinct labels and more than 1 sample
            if evaluate_clustering and len(np.unique(cluster_labels)) > 1 and n_subset_samples > 1:
                 print(f"Evaluating GMM clustering using metric: {metric_for_evaluation}")
                 evaluation_metrics = {} # Initialize evaluation_metrics dictionary
                 try:
                     if metric_for_evaluation == 'silhouette':
                         score = silhouette_score(clustering_input, cluster_labels)
                         evaluation_metrics = {'silhouette_score': score}
                     elif metric_for_evaluation == 'davies_bouldin':
                         score = davies_bouldin_score(clustering_input, cluster_labels)
                         evaluation_metrics = {'davies_bouldin_score': score} # Lower is better
                     elif metric_for_evaluation == 'calinski_harabasz':
                         score = calinski_harabasz_score(clustering_input, cluster_labels)
                         evaluation_metrics = {'calinski_harabasz_score': score} # Higher is better
                     else:
                         print(f"Warning: Unknown evaluation metric '{metric_for_evaluation}'. Skipping evaluation.")

                 except Exception as e:
                     print(f"Error during GMM evaluation: {e}. Skipping evaluation.")
                     evaluation_metrics = None

        else:
            print(f"Error: Unknown clustering method '{clustering_method}'. Supported methods are 'hdbscan', 'kmeans', 'gmm'. Skipping clustering.")
            # Assign all to -1 for unknown method
            cluster_labels = np.full(n_subset_samples, -1, dtype=int)


    except Exception as e:
        print(f"An unexpected error occurred during Clustering: {e}")
        # Assign all to -1 in case of unexpected errors
        cluster_labels = np.full(n_subset_samples, -1, dtype=int)
        evaluation_metrics = None


    # Ensure cluster_labels is a NumPy array even if all samples are noise (-1)
    if cluster_labels is not None and not isinstance(cluster_labels, np.ndarray):
         cluster_labels = np.asarray(cluster_labels)


    return cluster_labels, evaluation_metrics # Return both labels and metrics


# In[ ]:


class SupervisedAutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        y_pred = self.classifier(z)
        return x_recon, y_pred, z  # z = latent vector


# In[ ]:


def train_supervised_autoencoder(model, X_train, y_train, num_epochs=50, batch_size=64, lr=1e-3):
    from torch.utils.data import DataLoader, TensorDataset
    from torch.optim import Adam
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                            torch.tensor(y_train, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x_recon, y_pred, _ = model(x)

            recon_loss = F.mse_loss(x_recon, x)
            cls_loss = F.cross_entropy(y_pred, y)
            loss = recon_loss + cls_loss  # 可変比重をかけてもよい

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

def extract_latent(model, X):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X, dtype=torch.float32).to(next(model.parameters()).device)
        _, _, z = model(x_tensor)
    return z.cpu().numpy()


# In[ ]:


def sliding_window(data: pd.DataFrame, sequence_length: int):
    """
    DataFrameをスライディングウィンドウでデータフレームのリストに変換
    """
    windows = []
    n_total = len(data)

    for i in range(n_total - sequence_length + 1):
        window = data.iloc[i : i + sequence_length].copy()
        windows.append(window)

    return windows


# In[ ]:


def transform_sequence_data(dataframes: list[pd.DataFrame], sub_window_ratio: float = 0.5) -> pd.DataFrame:
    """
    入力: 各サンプルをDataFrameとしたリスト
    出力: 固定長ベクトルを格納したDataFrame. Indexは元の時系列データの終了時点のインデックスとする。

    Args:
        dataframes (list[pd.DataFrame]): スライディングウィンドウで切り出されたデータフレームのリスト。
        sub_window_ratio (float): extract_aggregated_features に渡すサブウィンドウ分割の割合。

    Returns:
        pd.DataFrame: 抽出された集約特徴量を格納したDataFrame。
    """
    features_list = []
    indices = []
    for df_window in dataframes:
        # ここで sub_window_ratio を渡す
        features_list.append(extract_aggregated_features(df_window, sub_window_ratio=sub_window_ratio))
        # ウィンドウの最後の行のインデックスを新しいDataFrameのインデックスとして使用
        indices.append(df_window.index[-1])

    feature_df = pd.DataFrame(features_list, index=indices)
    return feature_df


# In[ ]:


def extract_aggregated_features(df: pd.DataFrame, sub_window_ratio: float = 0.5) -> pd.Series:
    """
    各サンプル（時系列データ）から多種統計量とサブウィンドウ統計量を抽出する。
    df: shape = (sequence_length, n_features)
    非数値型やboolean型のカラムは統計量の計算をスキップする。

    Args:
        df (pd.DataFrame): スライディングウィンドウで切り出された時系列データフレーム。
        sub_window_ratio (float): サブウィンドウ分割の割合。0.0より大きく1.0未満の値。
                                  例: 0.5なら前半50%と後半50%に分割。

    Returns:
        pd.Series: 抽出された集約特徴量を格納したSeries。
    """
    features = {}
    sequence_length = len(df)

    # サブウィンドウのインデックスを計算
    sub_window_size = int(sequence_length * sub_window_ratio)
    if sub_window_size <= 0 or sub_window_size >= sequence_length:
        # サブウィンドウサイズが無効な場合は、サブウィンドウ特徴量はスキップ
        sub_window_indices = None
        # print(f"Warning: Invalid sub_window_ratio ({sub_window_ratio}) for sequence_length ({sequence_length}). Skipping sub-window features.")
    else:
         sub_window_indices = {
             'first': (0, sub_window_size),
             'last': (sequence_length - sub_window_size, sequence_length)
         }


    for col in df.columns:
        series = df[col]

        # 数値型（booleanを除く）であることを確認
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            numeric_series = series.dropna() # NaNを除外して統計量を計算

            if not numeric_series.empty:
                # === 全ウィンドウでの統計量 ===
                features[f'{col}_mean'] = numeric_series.mean()
                features[f'{col}_std'] = numeric_series.std()
                features[f'{col}_min'] = numeric_series.min()
                features[f'{col}_max'] = numeric_series.max()
                features[f'{col}_last'] = series.iloc[-1] # NaNがあっても最後の値を取得
                features[f'{col}_diff'] = series.iloc[-1] - series.iloc[0] # 差分
                features[f'{col}_skew'] = numeric_series.skew() # 歪度
                features[f'{col}_kurtosis'] = numeric_series.kurtosis() # 尖度

                # 自己相関 (ラグ1) - 時系列が十分に長い場合のみ
                if len(numeric_series) > 1:
                     features[f'{col}_autocorr1'] = numeric_series.autocorr(lag=1)
                else:
                     features[f'{col}_autocorr1'] = np.nan


                # パーセンタイル
                features[f'{col}_p10'] = numeric_series.quantile(0.10)
                features[f'{col}_p25'] = numeric_series.quantile(0.25)
                features[f'{col}_p50'] = numeric_series.quantile(0.50)
                features[f'{col}_p75'] = numeric_series.quantile(0.75)
                features[f'{col}_p90'] = numeric_series.quantile(0.90)

                # === 新規追加：カテゴリ判別に役立つ可能性のある特徴量 ===

                # ウィンドウ内の価格変化率 (開始 vs 終了)
                if len(series) > 1 and series.iloc[0] != 0:
                    features[f'{col}_pct_change'] = (series.iloc[-1] - series.iloc[0]) / series.iloc[0]
                else:
                    features[f'{col}_pct_change'] = np.nan

                # ウィンドウ内の最大ドローダウン (Max Drawdown)
                # ウィンドウ内の最大ドローアップ (Max Drawup)
                if len(numeric_series) > 1:
                    cumulative_returns = numeric_series / numeric_series.iloc[0] - 1
                    features[f'{col}_max_drawdown'] = (cumulative_returns - cumulative_returns.cummax()).min()
                    features[f'{col}_max_drawup'] = (cumulative_returns - cumulative_returns.cummin()).max()
                else:
                    features[f'{col}_max_drawdown'] = np.nan
                    features[f'{col}_max_drawup'] = np.nan


                # ウィンドウ内の平均価格に対する最後の価格の比率
                if numeric_series.mean() != 0:
                     features[f'{col}_last_vs_mean_ratio'] = series.iloc[-1] / numeric_series.mean()
                else:
                     features[f'{col}_last_vs_mean_ratio'] = np.nan

                # ウィンドウ内のボラティリティの変化 (前半 vs 後半)
                if sub_window_indices and not sub_window_indices['first'][1] == 0 and not sub_window_indices['last'][0] == sequence_length:
                    sub_series_first = series.iloc[sub_window_indices['first'][0]:sub_window_indices['first'][1]].dropna()
                    sub_series_last = series.iloc[sub_window_indices['last'][0]:sub_window_indices['last'][1]].dropna()
                    if len(sub_series_first) > 1 and len(sub_series_last) > 1 and sub_series_first.std() != 0:
                        features[f'{col}_volatility_change_ratio'] = sub_series_last.std() / sub_series_first.std()
                    else:
                        features[f'{col}_volatility_change_ratio'] = np.nan
                else:
                     features[f'{col}_volatility_change_ratio'] = np.nan


                # サブウィンドウ統計量
                if sub_window_indices:
                    for window_name, (start, end) in sub_window_indices.items():
                        sub_series = series.iloc[start:end].dropna()
                        if not sub_series.empty:
                            features[f'{col}_sub_{window_name}_mean'] = sub_series.mean()
                            features[f'{col}_sub_{window_name}_std'] = sub_series.std()
                            features[f'{col}_sub_{window_name}_min'] = sub_series.min()
                            features[f'{col}_sub_{window_name}_max'] = sub_series.max()
                            features[f'{col}_sub_{window_name}_last'] = sub_series.iloc[-1] if not sub_series.empty else np.nan
                            features[f'{col}_sub_{window_name}_diff'] = sub_series.iloc[-1] - sub_series.iloc[0] if len(sub_series) > 1 else np.nan
                            # サブウィンドウ内の価格変化率
                            if len(sub_series) > 1 and sub_series.iloc[0] != 0:
                                features[f'{col}_sub_{window_name}_pct_change'] = (sub_series.iloc[-1] - sub_series.iloc[0]) / sub_series.iloc[0]
                            else:
                                features[f'{col}_sub_{window_name}_pct_change'] = np.nan

                        else:
                            features[f'{col}_sub_{window_name}_mean'] = np.nan
                            features[f'{col}_sub_{window_name}_std'] = np.nan
                            features[f'{col}_sub_{window_name}_min'] = np.nan
                            features[f'{col}_sub_{window_name}_max'] = np.nan
                            features[f'{col}_sub_{window_name}_last'] = np.nan
                            features[f'{col}_sub_{window_name}_diff'] = np.nan
                            features[f'{col}_sub_{window_name}_pct_change'] = np.nan # サブウィンドウ価格変化率もNaN


            else:
                # numeric_seriesが空の場合、全ての統計量をNaNにする
                stats_keys = [
                    'mean', 'std', 'min', 'max', 'last', 'diff', 'skew', 'kurtosis', 'autocorr1',
                    'p10', 'p25', 'p50', 'p75', 'p90', # 全ウィンドウ統計量
                    'pct_change', 'max_drawdown', 'max_drawup', 'last_vs_mean_ratio', 'volatility_change_ratio' # 新規追加分
                ]
                for key in stats_keys:
                    features[f'{col}_{key}'] = np.nan
                if sub_window_indices:
                     sub_stats_keys = ['mean', 'std', 'min', 'max', 'last', 'diff', 'pct_change'] # サブウィンドウ新規追加分
                     for window_name in sub_window_indices.keys():
                         for key in sub_stats_keys:
                             features[f'{col}_sub_{window_name}_{key}'] = np.nan


        elif pd.api.types.is_datetime64_any_dtype(series):
             # 日付/時刻型の場合はUnix timestampに変換して処理
             numeric_series = series.astype(np.int64) // 10**9
             numeric_series = numeric_series.dropna()

             if not numeric_series.empty:
                 features[f'{col}_mean'] = numeric_series.mean()
                 features[f'{col}_std'] = numeric_series.std()
                 features[f'{col}_min'] = numeric_series.min()
                 features[f'{col}_max'] = numeric_series.max()
                 features[f'{col}_last'] = (series.iloc[-1].timestamp() if pd.notna(series.iloc[-1]) else np.nan) if len(series) > 0 else np.nan # 最後のタイムスタンプ
                 features[f'{col}_diff'] = (series.iloc[-1] - series.iloc[0]).total_seconds() if len(series) > 1 and pd.notna(series.iloc[-1]) and pd.notna(series.iloc[0]) else np.nan # タイムスタンプの差分 (秒)

                 # 新規追加：タイムスタンプの差分（秒）を正規化？（データのスケールによる）
                 # ここでは単純な差分のみとする。必要ならスケーリング段階で処理。


                 # サブウィンドウ統計量 (タイムスタンプ)
                 if sub_window_indices:
                    for window_name, (start, end) in sub_window_indices.items():
                        sub_series = series.iloc[start:end].dropna()
                        if not sub_series.empty:
                            sub_numeric_series = sub_series.astype(np.int64) // 10**9
                            features[f'{col}_sub_{window_name}_mean'] = sub_numeric_series.mean()
                            features[f'{col}_sub_{window_name}_std'] = sub_numeric_series.std()
                            features[f'{col}_sub_{window_name}_min'] = sub_numeric_series.min()
                            features[f'{col}_sub_{window_name}_max'] = sub_numeric_series.max()
                            features[f'{col}_sub_{window_name}_last'] = (sub_series.iloc[-1].timestamp() if pd.notna(sub_series.iloc[-1]) else np.nan) if len(sub_series) > 0 else np.nan
                            features[f'{col}_sub_{window_name}_diff'] = (sub_series.iloc[-1] - sub_series.iloc[0]).total_seconds() if len(sub_series) > 1 and pd.notna(sub_series.iloc[-1]) and pd.notna(sub_series.iloc[0]) else np.nan
                        else:
                             features[f'{col}_sub_{window_name}_mean'] = np.nan
                             features[f'{col}_sub_{window_name}_std'] = np.nan
                             features[f'{col}_sub_{window_name}_min'] = np.nan
                             features[f'{col}_sub_{window_name}_max'] = np.nan
                             features[f'{col}_sub_{window_name}_last'] = np.nan
                             features[f'{col}_sub_{window_name}_diff'] = np.nan
             else:
                 # numeric_seriesが空の場合、全ての統計量をNaNにする
                 stats_keys = ['mean', 'std', 'min', 'max', 'last', 'diff']
                 for key in stats_keys:
                     features[f'{col}_{key}'] = np.nan
                 if sub_window_indices:
                      sub_stats_keys = ['mean', 'std', 'min', 'max', 'last', 'diff']
                      for window_name in sub_window_indices.keys():
                          for key in sub_stats_keys:
                              features[f'{col}_sub_{window_name}_{key}'] = np.nan


        else:
            # 非数値型やboolean型のカラムはスキップ
            pass


    return pd.Series(features)


# In[ ]:


# Assuming necessary imports are done in preceding cells
# import numpy as np
# import pandas as pd
# import torch
# import os
# from sklearn.preprocessing import StandardScaler
# import hdbscan
# import pickle

# # Assuming the following functions are defined in preceding cells:
# # prepare_data_for_clustering
# # perform_clustering_and_save (calls perform_clustering_on_subset internally for the initial clustering)
# # load_processed_data
# # perform_clustering_on_subset (modified to return 4 values)
# # extract_cluster_features
# # label_by_cluster
# # prepare_strategy_data
# # create_cluster_specific_labels
# # create_aggregated_features (already modified)

# # --- Configuration ---
# # Adjust these parameters as needed
# output_folder = '/content/drive/MyDrive/Colab Notebooks/cryptoprice_prediction_tft/clustering_results'
# sequence_length = 60
# horizon = 6 # Future horizon for labeling
# latent_dim = 128 # Latent dimension for AutoEncoder
# price_feature_index = 0 # Assuming 'close' is the first feature in the 3D array
# # Assuming MA features are at indices 1, 2, 3 in the 3D array (MA_t_6, MA_t_24, MA_t_72)
# ma_feature_indices_3d = (1, 2, 3)

# # HDBSCAN parameters - adjust based on data and desired clustering granularity
# hdbscan_params = {
#     'min_cluster_size': 100, # Minimum number of samples in a cluster
#     'min_samples': 10,      # More points will be considered as core points
#     'cluster_selection_epsilon': 0.2, # Distance threshold for merging clusters
#     'gen_min_span_tree': True # Required for plotting, but can be False for speed
# }

# # Labeling parameters - adjust based on desired strategy definitions
# labeling_params = {
#     'first_frac': 0.5, # Fraction of sequence for first segment trend check
#     'trend_thr_strong': 0.02, # Threshold for strong trend slope
#     'trend_thr_moderate': 0.005, # Threshold for moderate trend slope
#     'reversal_thr': 0.01, # Threshold for reversal price change
#     'range_thr': 0.002, # Threshold for range-bound slope
#     'volatility_thr_high': 50, # Threshold for high volatility (needs scaling adjustment)
#     'volatility_thr_low': 10,  # Threshold for low volatility (needs scaling adjustment)
#     'price_feature_index': price_feature_index, # Index of the price feature
#     # Parameters for strong_uptrend_cont logic (if using average MA values)
#     'uptrend_overall_score_threshold': 0.7,
#     'weights': {"slope_consistency": 1.0, "price_distribution": 1.0, "mean_avg_slope": 1.0, "mean_peak_count": 0.5},
#     'slope_consistency_mapping_start': 0.6,
#     'price_distribution_mapping_start': 0.7,
#     'strong_avg_slope_threshold': 0.02,
#     'moderate_avg_slope_threshold': 0.005,
#     'peak_count_threshold': 5,
#     # Add MA indices for strong_uptrend_cont labeling logic (if using future MA trend)
#     'ma_short_index': ma_feature_indices_3d[0],
#     'ma_mid_index': ma_feature_indices_3d[1],
#     'ma_long_index': ma_feature_indices_3d[2],
#     'uptrend_consistency_ratio': 0.8 # Consistency ratio for future MA trend check
# }

# # Binary Labeling parameters - adjust based on desired trade signals
# binary_labeling_params = {
#     'sequence_length': sequence_length,
#     'reversal_check_len': 3, # How many last points to check for immediate trend
#     'bb_upper_threshold_pct': 0.005, # Percentage distance from upper BB for label 0
#     'bb_lower_threshold_pct': 0.005,  # Percentage distance from lower BB for label 1
#     # Note: These BB thresholds should be adjusted based on the actual data scale and BB calculation
#     'horizon': horizon, # Pass horizon
#     'eval_window_L': 4, # Example: Evaluation window length
#     'k_of_l': 3, # Example: K for K-of-L rule
#     'buffer_epsilon': 0.001, # Example: Buffer epsilon
#     'slope_threshold': 0.0005, # Example: Slope threshold
#     'delta_slope': 3, # Example: Delta for slope calculation
#     'volatility_measure': 'atr', # Use ATR for volatility normalization
#     'volatility_window': 14, # Example: ATR window
#     'percentile_threshold': 70, # Example: 70th percentile for dynamic threshold
#     'past_window_for_percentile': 30, # Example: Past window for percentile calculation
#     'price_feature_index': price_feature_index # Pass price feature index
# }

# # --- Load Data ---
# # Assuming 'df' (the original preprocessed DataFrame) and 'top_50_features'
# # are available from previous steps in the notebook.
# # If not, load them here.
# # Example:
# # df = pd.read_pickle('/path/to/your/preprocessed_dataframe.pkl')
# # top_50_features = pd.read_pickle('/path/to/your/top_50_features.pkl')

# # --- Perform Initial Clustering (if not already done and saved) ---
# print("--- Checking for saved clustering results ---")
# X_3d_numpy, all_clusters, feature_names_3d, all_latent, original_indices_filtered = load_processed_data(output_folder)

# if X_3d_numpy is None or all_clusters is None or feature_names_3d is None or all_latent is None or original_indices_filtered is None:
#     print("Saved clustering results not found or incomplete. Performing clustering...")
#     # This function performs data prep, AE, HDBSCAN, and saves results
#     # It returns the results as well
#     X_3d_numpy, all_clusters, feature_names_3d, all_latent, original_indices_filtered = perform_clustering_and_save(
#         df=df, # Assuming df is available
#         top_50_features=top_50_features, # Assuming top_50_features is available
#         output_folder=output_folder,
#         sequence_length=sequence_length,
#         latent_dim=latent_dim,
#         hdbscan_params=hdbscan_params
#     )
#     if X_3d_numpy is None:
#          print("Error: Initial clustering failed. Cannot proceed.")
#          # Exit or handle error appropriately
# else:
#     print("Loaded saved clustering results.")
#     print(f"Loaded X_3d_numpy shape: {X_3d_numpy.shape}")
#     print(f"Loaded all_clusters shape: {all_clusters.shape}")
#     print(f"Loaded feature_names_3d length: {len(feature_names_3d)}")
#     print(f"Loaded all_latent shape: {all_latent.shape}")
#     print(f"Loaded original_indices_filtered length: {len(original_indices_filtered)}")


# # --- Perform Strategy Classification on ALL sequences based on initial clusters ---
# print("\n--- Classifying all sequences into strategies ---")
# # Use the loaded/generated all_clusters and X_3d_numpy
# # Need cluster_info for label_by_cluster. This should be derived from the clustering results.
# # We can compute cluster_info from all_clusters and X_3d_numpy
# print("Extracting cluster features for strategy labeling...")
# # Need to find the index of MA features for extract_cluster_features
# ma_indices_for_extract = {}
# try:
#      ma_indices_for_extract['MA_t_6'] = feature_names_3d.index('MA_t_6')
#      ma_indices_for_extract['MA_t_24'] = feature_names_3d.index('MA_t_24')
#      ma_indices_for_extract['MA_t_72'] = feature_names_3d.index('MA_t_72')
# except ValueError as e:
#      print(f"Error: MA feature not found for extract_cluster_features: {e}")
#      # Exit or handle error
#      ma_indices_for_extract = {} # Ensure it's empty if not found

# cluster_info_all = extract_cluster_features(
#     X_3d_numpy, # Use the full 3D data
#     all_clusters, # Use the full cluster labels
#     price_feature_index=price_feature_index,
#     ma_indices=ma_indices_for_extract # Pass the determined MA indices
# )

# print(f"Extracted info for {len(cluster_info_all)} clusters.")

# # Use the extracted cluster_info to label strategies
# label_strategy = label_by_cluster(
#     cluster_info_all, # Use cluster info for all data
#     X_3d_numpy, # Pass the 3D data for sample-level consistency check
#     all_clusters, # Pass the full cluster labels
#     **labeling_params # Pass labeling parameters
# )

# print("\n--- Strategy labels assigned to clusters ---")
# for cid, strategy in label_strategy.items():
#      num_samples = cluster_info_all.get(cid, {}).get("num_samples", 0)
#      print(f"Cluster {cid}: '{strategy}' ({num_samples} samples)")


# # --- Generate Binary Labels based on Strategies ---
# print("\n--- Generating binary labels based on strategies ---")
# # Generate binary labels for ALL sequences based on their assigned strategy
# # Need to find high and low feature indices for ATR if used
# high_feature_index = None
# low_feature_index = None
# try:
#      if binary_labeling_params.get('volatility_measure') == 'atr':
#           high_feature_index = feature_names_3d.index('high')
#           low_feature_index = feature_names_3d.index('low')
#           print(f"Found high feature index: {high_feature_index}, low feature index: {low_feature_index}")
# except ValueError as e:
#      print(f"Error: 'high' or 'low' feature not found in feature_names_3d for ATR calculation: {e}")
#      # Handle this error - maybe switch volatility_measure or skip labeling


# all_binary_labels = generate_binary_labels_from_strategy(
#     X_3d_numpy, # Use the full 3D data
#     np.array([label_strategy.get(cid, 'unknown') for cid in all_clusters]), # Get strategy label for each sequence
#     feature_names_3d, # Pass feature names for index lookup
#     sequence_length,
#     high_feature_index=high_feature_index, # Pass high index
#     low_feature_index=low_feature_index,   # Pass low index
#     **binary_labeling_params # Pass binary labeling parameters
# )

# if all_binary_labels is None:
#      print("Error: Binary label generation failed. Cannot proceed.")
#      # Exit or handle error
# else:
#      print(f"Generated {len(all_binary_labels)} binary labels.")
#      print(f"Binary label distribution: {pd.Series(all_binary_labels).value_counts()}")


# # --- Split Data (X, y, clusters, latent_vectors) into Train/Test ---
# # Perform the split on the original 3D data, binary labels, cluster IDs, and latent vectors
# # Use the original_indices_filtered to get the temporal split
# train_ratio = 0.8
# n_samples_total = X_3d_numpy.shape[0]
# split_index = int(n_samples_total * train_ratio)

# X_train_3d = X_3d_numpy[:split_index]
# X_test_3d = X_3d_numpy[split_index:]
# y_train_binary = all_binary_labels[:split_index]
# y_test_binary = all_binary_labels[split_index:]
# # Use the original cluster IDs for train/test split
# train_clusters_all = all_clusters[:split_index]
# test_clusters_all = all_clusters[split_index:]
# latent_train_all = all_latent[:split_index] # Latent vectors for train split
# latent_test_all = all_latent[split_index:] # Latent vectors for test split
# original_indices_train_all = original_indices_filtered[:split_index] # Original indices for train sequences
# original_indices_test_all = original_indices_filtered[split_index:] # Original indices for test sequences


# print("\n--- Data Split Shapes (3D data, Labels, Clusters, Latent) ---")
# print("X_train_3d shape:", X_train_3d.shape)
# print("y_train_binary shape:", y_train_binary.shape)
# print("train_clusters_all shape:", train_clusters_all.shape)
# print("latent_train_all shape:", latent_train_all.shape)
# print("original_indices_train_all shape:", original_indices_train_all.shape)
# print("X_test_3d shape:", X_test_3d.shape)
# print("y_test_binary shape:", y_test_binary.shape)
# print("test_clusters_all shape:", test_clusters_all.shape)
# print("latent_test_all shape:", latent_test_all.shape)
# print("original_indices_test_all shape:", original_indices_test_all.shape)


# # --- Map Test Clusters to Train Clusters (Optional but Recommended) ---
# # This step helps ensure consistency in cluster IDs between train and test sets
# # for subsequent strategy-specific modeling if needed.
# print("\n--- Mapping Test Clusters to Train Clusters ---")
# # Use the latent vectors and cluster IDs from the split
# # This function returns the *mapped* cluster IDs for the *entire* dataset size (train + test combined)
# # We need to re-split this mapped array back into train and test
# mapped_all_clusters = map_test_clusters_to_train(
#     latent_train_all,
#     train_clusters_all,
#     latent_test_all,
#     test_clusters_all
# )

# # Re-split the mapped cluster IDs
# mapped_train_clusters = mapped_all_clusters[:split_index]
# mapped_test_clusters = mapped_all_clusters[split_index:]

# print("Mapped train clusters shape:", mapped_train_clusters.shape)
# print("Mapped test clusters shape:", mapped_test_clusters.shape)
# print("Mapped train cluster distribution:\n", pd.Series(mapped_train_clusters).value_counts())
# print("Mapped test cluster distribution:\n", pd.Series(mapped_test_clusters).value_counts())


# # --- Scale Train/Test Data (3D data) ---
# # Identify numeric features in the 3D data using feature_names_3d
# # Assuming all features except the categorical and date features are numeric
# # Need the original list of all selected features before dropping NaN columns/rows
# # Let's assume for simplicity here that the first N features in feature_names_3d are numeric.
# # A more robust approach would be to check the dtypes in the original df before sliding window.
# # For now, let's assume features from the original 'top_50_features' list are numeric.
# # Need to get the indices of these numeric features within the final feature_names_3d list.

# # Reconstruct the full list of selected columns used before dropping NaNs
# # This was done in prepare_data_for_clustering
# # Let's assume the order is preserved and the first features are the numeric ones.
# # A safer way is to check dtypes in the original df *before* splitting and sliding window.
# # For now, we will approximate: identify numeric columns from the *first* sequence in X_3d_numpy.
# # This is not ideal as dtypes might vary or be object due to NaN handling before converting to NumPy.
# # A better way: pass the list of numeric feature names from the original df.

# # Let's assume feature_names_3d contains names in a consistent order,
# # and we can identify numeric ones by name or index range.
# # From prepare_data_for_clustering, the order was ['close', 'MA_t_6', 'MA_t_24', 'MA_t_72', 'MA_t_168', 'upper', 'lower'] + top_50_features + categorical + date
# # The numeric features are likely 'close', MAs, BBs, and top_50_features.
# # Categorical and Date features are at the end.

# # Identify numeric feature indices in feature_names_3d
# # Note: This assumes the order is preserved and numeric columns are at the beginning.
# # A more robust approach would be to check dtypes of the original df columns used.
# # Let's try to find indices of known numeric features by name.
# known_numeric_features = ['close', 'MA_t_6', 'MA_t_24', 'MA_t_72', 'MA_t_168', 'upper', 'lower'] + top_50_features # List of known numeric features

# numeric_feature_indices_3d = []
# for name in known_numeric_features:
#      try:
#           index = feature_names_3d.index(name)
#           numeric_feature_indices_3d.append(index)
#      except ValueError:
#           # print(f"Warning: Numeric feature '{name}' not found in feature_names_3d. Skipping scaling for this feature.")
#           pass # Skip if feature not found


# print(f"\nIdentified {len(numeric_feature_indices_3d)} numeric features for 3D data scaling.")
# # print("Numeric feature names for scaling:", [feature_names_3d[i] for i in numeric_feature_indices_3d])


# X_train_scaled_3d, X_test_scaled_3d, scaler_3d = scale_train_test_data(
#     X_train_3d, X_test_3d, numeric_feature_indices_3d
# )

# print("\n--- Scaled 3D Data Shapes ---")
# print("X_train_scaled_3d shape:", X_train_scaled_3d.shape)
# print("X_test_scaled_3d shape:", X_test_scaled_3d.shape)


# # --- Prepare Data by Strategy for Model Training ---
# # Use the *mapped* cluster IDs for splitting data by strategy for modeling
# # Use the scaled 3D data (X_train_scaled_3d, X_test_scaled_3d) and binary labels (y_train_binary, y_test_binary)
# # Use the mapped_train_clusters and mapped_test_clusters
# # Use the original_indices_train_all and original_indices_test_all

# # Create the integrated strategy names array aligned with the full original_indices_filtered
# # This requires the label_strategy dictionary which maps cluster ID to strategy name.
# # We use the mapped_all_clusters (which are aligned with original_indices_filtered)
# integrated_strategy_names = np.array([label_strategy.get(cid, 'unknown') for cid in mapped_all_clusters])
# print(f"\nIntegrated strategy names created. Shape: {integrated_strategy_names.shape}")


# train_data_by_strategy, test_data_by_strategy = prepare_strategy_data(
#     X_train_scaled_3d,
#     y_train_binary,
#     mapped_train_clusters, # Use mapped train clusters
#     X_test_scaled_3d,
#     y_test_binary,
#     mapped_test_clusters, # Use mapped test clusters
#     integrated_strategy_names, # Use the integrated strategy names array
#     original_indices_train_all, # Pass original start indices for train sequences
#     original_indices_test_all,  # Pass original start indices for test sequences
#     seq_length=sequence_length, # Pass sequence length
#     original_indices_filtered=original_indices_filtered # Pass the full original indices (start times)
# )

# print("\n--- Data Prepared by Strategy (Train) ---")
# for strategy, data in train_data_by_strategy.items():
#     print(f"Strategy '{strategy}': {len(data['X'])} sequences")

# print("\n--- Data Prepared by Strategy (Test) ---")
# for strategy, data in test_data_by_strategy.items():
#     print(f"Strategy '{strategy}': {len(data['X'])} sequences")


# # --- Optional: Further processing or Model Training per Strategy ---
# # Now you have train_data_by_strategy and test_data_by_strategy dictionaries.
# # You can iterate through these dictionaries and train/evaluate models specifically for each strategy.
# # For example:
# # for strategy, data in train_data_by_strategy.items():
# #     if len(data['X']) > 0:
# #         print(f"\nTraining model for strategy: {strategy}")
# #         # Data is already a list of 2D NumPy arrays
# #         # X_strat_train is a list of arrays with shape (seq_len, n_features)
# #         # y_strat_train is a list of single labels (or an array of labels)
# #         # original_indices_strat_train is a list of DatetimeIndex (seq_len,)

# #         # Stack X data if needed for batch processing in some models
# #         # X_strat_train_stacked = np.stack(data['X']) if data['X'] else np.array([])
# #         # y_strat_train_np = np.concatenate(data['y']) if data['y'] else np.array([]) # Assuming y is a list of single labels


# #         # Train your model (e.g., TFT) on data['X'], data['y'], data['original_indices']
# #         # ...

# # for strategy, data in test_data_by_strategy.items():
# #      if len(data['X']) > 0:
# #          print(f"\nEvaluating model for strategy: {strategy}")
# #          # X_strat_test_stacked = np.stack(data['X']) if data['X'] else np.array([])
# #          # y_strat_test_np = np.concatenate(data['y']) if data['y'] else np.array([])


# #          # Evaluate your trained model on data['X'], data['y'], data['original_indices']
# #          # ...

# # Example of how to access data for a specific strategy (e.g., 'strong_uptrend_cont'):
# # Note: Strategy names are now the ones defined in label_by_cluster and used in integrated_strategy_names
# uptrend_train_data = train_data_by_strategy.get('Uptrend') # Use the actual strategy name
# if uptrend_train_data and len(uptrend_train_data['X']) > 0:
#      print("\nExample: Data for 'Uptrend' training:")
#      print(f"Number of sequences: {len(uptrend_train_data['X'])}")
#      print(f"Shape of first sequence: {uptrend_train_data['X'][0].shape}")
#      print(f"Shape of first label: {np.shape(uptrend_train_data['y'][0])}") # Should be scalar if y_train_binary is 1D
#      print(f"Shape of first original_indices: {uptrend_train_data['original_indices'][0].shape}") # Should be (seq_len,)

#      # Example: Stack data for models expecting NumPy arrays
#      # X_uptrend_train_stacked = np.stack(uptrend_train_data['X'])
#      # y_uptrend_train_np = np.array(uptrend_train_data['y']) # Convert list of labels to NumPy array
#      # print("Stacked X shape:", X_uptrend_train_stacked.shape)
#      # print("NumPy y shape:", y_uptrend_train_np.shape)


# else:
#     print("\nNo training data found for 'Uptrend'.")

# # Example of how to access data for 'choppy' strategy:
# choppy_train_data = train_data_by_strategy.get('choppy')
# if choppy_train_data and len(choppy_train_data['X']) > 0:
#      print("\nExample: Data for 'choppy' training:")
#      print(f"Number of sequences: {len(choppy_train_data['X'])}")
#      print(f"Shape of first sequence: {choppy_train_data['X'][0].shape}")
#      print(f"Shape of first label: {np.shape(choppy_train_data['y'][0])}")
#      print(f"Shape of first original_indices: {choppy_train_data['original_indices'][0].shape}")

# else:
#      print("\nNo training data found for 'choppy'.")

# # Example of how to access data for 'choppy' test:
# choppy_test_data = test_data_by_strategy.get('choppy')
# if choppy_test_data and len(choppy_test_data['X']) > 0:
#      print("\nExample: Data for 'choppy' test:")
#      print(f"Number of sequences: {len(choppy_test_data['X'])}")
#      print(f"Shape of first sequence: {choppy_test_data['X'][0].shape}")
#      print(f"Shape of first label: {np.shape(choppy_test_data['y'][0])}")
#      print(f"Shape of first original_indices: {choppy_test_data['original_indices'][0].shape}")

# else:
#      print("\nNo test data found for 'choppy'.")


# # --- Finish Task ---
# print("\n--- Data Preparation and Strategy Assignment Complete ---")
# print("Data is now organized into 'train_data_by_strategy' and 'test_data_by_strategy' dictionaries.")
# print("You can now proceed with strategy-specific model training and evaluation.")


# In[ ]:


import numpy as np
import pandas as pd
# Assuming other necessary imports like from sklearn etc. are handled in filtering.py

# This code should be in filtering.py

# Helper functions (assuming they are defined elsewhere or within this cell initially)

# Assuming calculate_cosine_similarity, split_data, map_test_clusters_to_train, scale_train_test_data, prepare_strategy_data
# are defined in filtering.py before create_timeseries_dataframe

def create_timeseries_dataframe(features, targets, original_indices, original_cluster_ids, label_strategy, feature_names, seq_length=None):
    """
    Creates a pandas DataFrame suitable for PyTorch Forecasting datasets
    or similar time series models from concatenated numpy arrays and indices.

    Args:
        features (np.ndarray): Concatenated features array (total_timesteps, n_features).
        targets (np.ndarray): Concatenated targets array (total_timesteps,).
        original_indices (pd.DatetimeIndex or np.ndarray): DatetimeIndex or array of original indices (total_timesteps,).
                                                           Expected to be aligned with features and targets.
        original_cluster_ids (np.ndarray): Array of original cluster IDs for each time step (total_timesteps,).
                                           Expected to be aligned.
        label_strategy (dict): Dictionary mapping cluster_id to strategy_name.
        feature_names (list): List of names for the feature columns in `features`.
        seq_length (int): The length of each sequence. Required for calculating group_id and time_idx.

    Returns:
        pd.DataFrame: DataFrame with columns 'group_id', 'time_idx', 'target',
                      feature_names, 'original_cluster_id', 'strategy_name'.
    Raises:
        ValueError: If seq_length is not provided or is not positive.
        ValueError: If input arrays/indices do not have matching first dimensions.
    """
    if seq_length is None or not isinstance(seq_length, int) or seq_length <= 0:
         raise ValueError("seq_length parameter must be a positive integer.")

    total_timesteps = features.shape[0]

    if not (features.shape[0] == targets.shape[0] == len(original_indices) == original_cluster_ids.shape[0]):
         raise ValueError(f"Input arrays/indices must have matching first dimensions. Got features:{features.shape[0]}, targets:{targets.shape[0]}, original_indices:{len(original_indices)}, original_cluster_ids:{original_cluster_ids.shape[0]}")


    # Calculate group_id (sequence index) and time_idx within sequence
    # *** FIX: Use the correct sequence length for divmod ***
    # max_len = features.shape[1] # REMOVE THIS INCORRECT LINE

    # Use the seq_length parameter passed to the function
    sequence_length_for_divmod = seq_length

    # Check if total_timesteps is a multiple of seq_length - indicates if concatenation aligns with sequence boundaries
    # It's okay if not a perfect multiple if drop_last=False in DataLoader, but divmod needs the correct seq_length
    if total_timesteps % sequence_length_for_divmod != 0:
         print(f"Warning(create_timeseries_dataframe): Total timesteps ({total_timesteps}) is not a perfect multiple of sequence length ({sequence_length_for_divmod}).")


    group_id, time_idx = divmod(np.arange(total_timesteps), sequence_length_for_divmod)


    # Get strategy name for each time step based on the cluster ID
    # The cluster ID is repeated for each time step within a sequence in aligned_train/test_clusters_np
    # So we can directly map using the original_cluster_ids array
    strategy_names = np.array([label_strategy.get(cid, 'unknown') for cid in original_cluster_ids])


    # Create dictionary for DataFrame
    data = {
        'group_id': group_id,
        'time_idx': time_idx,
        'target': targets,
        'original_cluster_id': original_cluster_ids,
        'strategy_name': strategy_names
    }

    # Add feature columns
    # Ensure features[:, i] is treated correctly - it should be a 1D array
    for i, feature_name in enumerate(feature_names):
        # features[:, i] extracts the i-th feature across all time steps
        data[feature_name] = features[:, i]


    # Create DataFrame
    try:
        df = pd.DataFrame(data, index=original_indices) # Use original_indices as DataFrame index
        # Ensure index name is set if original_indices is DatetimeIndex
        if isinstance(original_indices, pd.DatetimeIndex):
             df.index.name = 'timestamp' # Or whatever the appropriate index name is
        elif isinstance(original_indices, pd.Index):
             df.index.name = original_indices.name # Use original index name
        else:
             df.index.name = 'index' # Default name


        # Ensure data types are appropriate (e.g., categorical features)
        # This part might need more sophisticated handling if specific dtypes are required
        # for categorical features before passing to the dataset.
        # For now, basic type inference by pandas is used.

    except Exception as e:
        print(f"Error creating DataFrame: {e}")
        # Re-raise the exception after printing, or handle it
        raise e # Re-raise to propagate the error


    return df


# In[ ]:


import pandas as pd
import numpy as np

def get_indices_by_moving_average_conditions(df: pd.DataFrame, short_window: int, mid_window: int, long_window: int) -> pd.Index:
    """
    短期、中期、長期の移動平均線が特定の順序で並び、かつそれぞれの傾きが正である期間のインデックスを取得する関数。

    Args:
        df (pd.DataFrame): 時系列価格データを含むDataFrame。必ず 'close' カラムを持つこと。
        short_window (int): 短期移動平均線の期間。
        mid_window (int): 中期移動平均線の期間。
        long_window (int): 長期移動平均線の期間。

    Returns:
        pd.Index: 条件を満たす期間のインデックス。
    """

    df_temp = df.copy() # 元のDataFrameを変更しないようにコピー（計算用）

    # 移動平均線の計算
    df_temp[f'MA_{short_window}'] = df_temp['close'].rolling(window=short_window).mean()
    df_temp[f'MA_{mid_window}'] = df_temp['close'].rolling(window=mid_window).mean()
    df_temp[f'MA_{long_window}'] = df_temp['close'].rolling(window=long_window).mean()

    # 移動平均線の傾き (単純な差分を使用)
    df_temp[f'MA_{short_window}_slope'] = df_temp[f'MA_{short_window}'].diff()
    df_temp[f'MA_{mid_window}_slope'] = df_temp[f'MA_{mid_window}'].diff()
    df_temp[f'MA_{long_window}_slope'] = df_temp[f'MA_{long_window}'].diff()


    # 条件1: 移動平均線が上から短期 > 中期 > 長期の順に並んでいる
    condition_order = (df_temp[f'MA_{short_window}'] > df_temp[f'MA_{mid_window}']) & \
                      (df_temp[f'MA_{mid_window}'] > df_temp[f'MA_{long_window}'])

    # 条件2: 各移動平均線の傾きが正である
    condition_slope = (df_temp[f'MA_{short_window}_slope'] > 0) & \
                      (df_temp[f'MA_{mid_window}_slope'] > 0) & \
                      (df_temp[f'MA_{long_window}_slope'] > 0)

    # 両方の条件を満たす行のブールインデックスを取得
    combined_condition = condition_order & condition_slope

    # 条件を満たす行のインデックスを取得
    filtered_indices = df_temp.index[combined_condition].copy()


    return filtered_indices

# 使用例:
# 仮のデータフレームを作成 (前回の例と同じ)
# data = {'close': np.random.rand(100) * 100 + np.linspace(0, 50, 100)}
# df_example = pd.DataFrame(data)

# 短期、中期、長期の期間を指定
# short = 5
# mid = 20
# long = 50

# 関数を呼び出してインデックスを取得
# indices_to_filter = get_indices_by_moving_average_conditions(df_example, short, mid, long)

# 元のデータフレームにインデックスを適用してフィルタリング
# filtered_data = df_example.loc[indices_to_filter]

# フィルタリングされたデータを表示
# print(filtered_data)


# In[ ]:


def filter_numpy_by_datetime_index(
    numpy_data: np.ndarray,
    numpy_datetime_index: pd.DatetimeIndex,
    filter_datetime_index: pd.DatetimeIndex
) -> np.ndarray:
    """
    NumPy配列とそれに対応するDatetimeIndexを受け取り、
    別のDatetimeIndexに含まれる時点に対応するNumPy配列のサンプルをフィルタリングする。

    Args:
        numpy_data (np.ndarray): フィルタリング対象のNumPy配列 (n_samples, seq_len, n_features)。
        numpy_datetime_index (pd.DatetimeIndex): numpy_dataの各サンプルに対応するDatetimeIndex (n_samples,)。
                                                 通常、各サンプルの終了時点のインデックス。
        filter_datetime_index (pd.DatetimeIndex): フィルタリング条件として使用するDatetimeIndex。
                                                  numpy_datetime_indexのサブセットである必要がある。

    Returns:
        np.ndarray: フィルタリングされたNumPy配列。
    """
    if not len(numpy_data) == len(numpy_datetime_index):
        raise ValueError("numpy_data and numpy_datetime_index must have the same number of samples.")

    # filter_datetime_indexに含まれるnumpy_datetime_indexのインデックス位置を探す
    # isin() と get_indexer() を組み合わせて高速に処理
    mask = numpy_datetime_index.isin(filter_datetime_index)
    # masked_indices = numpy_datetime_index.get_indexer(filter_datetime_index)
    # mask = masked_indices != -1 # filter_datetime_indexに存在しないものは-1になるため除外

    # マスクを使用してNumPy配列をフィルタリング
    filtered_numpy_data = numpy_data[mask]

    return filtered_numpy_data


# In[ ]:


def transform_window_list_to_numpy_and_index(
    window_list: list[pd.DataFrame]
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """
    スライディングウィンドウで生成されたDataFrameのリストを、
    (n_samples, seq_len, n_features)形状のNumPy配列と、
    各シーケンスの最後のデータのDatetimeIndexに変換する。

    Args:
        window_list (list[pd.DataFrame]): sliding_window関数で作成されたDataFrameのリスト。

    Returns:
        tuple[np.ndarray, pd.DatetimeIndex]:
            - NumPy配列 (n_samples, seq_len, n_features)
            - 各シーケンスの最後のデータのDatetimeIndex
    """
    if not window_list:
        return np.array([]), pd.DatetimeIndex([])

    n_samples = len(window_list)
    seq_len = len(window_list[0])
    n_features = window_list[0].shape[1]

    # 結果を格納するNumPy配列を初期化
    # データ型はウィンドウ内のデータの型に合わせるか、floatなどで統一
    # ここでは例としてfloat64を使用
    numpy_array = np.zeros((n_samples, seq_len, n_features), dtype=np.float64)

    # 各シーケンスの最後のDatetimeIndexを格納するリスト
    last_indices = []

    for i, window_df in enumerate(window_list):
        # データフレームをNumPy配列に変換
        # 非数値データが含まれる可能性があるため、適切に処理するか、事前に数値化しておく必要がある
        # ここでは、全てのカラムが数値型であると仮定して values を使用
        # 非数値カラムがある場合はエラーになるか、意図しない変換が行われる可能性あり
        try:
            numpy_array[i] = window_df.values
        except ValueError as e:
            print(f"Error converting DataFrame window to NumPy array at index {i}: {e}")
            print("Please ensure all columns in the DataFrame are numeric or handle non-numeric types.")
            # エラー処理（例: エラー発生時はNaNなどで埋める、または処理を中断）
            # ここでは例としてエラーメッセージを出力し、処理を続行（不適切なデータが入る可能性あり）
            # より頑健にするには、ここでraiseするか、NaN埋めなどの処理を追加
            pass


        # 各ウィンドウの最後のDatetimeIndexを取得
        # データフレームがDatetimeIndexを持っていると仮定
        last_indices.append(window_df.index[-1])

    # DatetimeIndexオブジェクトに変換
    datetime_index = pd.DatetimeIndex(last_indices)

    return numpy_array, datetime_index


# In[ ]:


###############  ルールベースのフィルタリングを行う場合  #############################
# Step 1: Prepare data (assumes sliding window sequences and binary labels)
# sequence_length = 60
# horizon = 12


# df.replace([np.inf, -np.inf], np.nan, inplace=True)


# # ルールベースのフィルタリング
# short = 5
# mid = 20
# long = 50

# indices_to_filter = get_indices_by_moving_average_conditions(df, short, mid, long)

# categorical_features = ["close_cusum", "dex_volume_cusum", "active_senders_cusum", "active_receivers_cusum", "address_count_sum_cusum", "contract_calls_cusum",
#                                      "whale_tx_count_cusum", "sign_entropy_12_cusum","sign_entropy_24_cusum",  "buy_sell_ratio_cusum", "MA_6_24_cross_flag", "MA_12_48_cross_flag", "MA_24_72_cross_flag",
#                                      "MA_slope_6_24_change_flag", "MA_slope_12_48_change_flag", "MA_slope_24_72_change_flag", "MA_slope_pct_change_6_24_change_flag",
#                                      "MA_slope_pct_change_12_48_change_flag", "MA_slope_pct_change_24_72_change_flag","volatility_change_flag"]
# date_features = ["hour", "day_of_week", "month", "hour_sin", "hour_cos", "day", "day_sin", "day_cos", "month_sin", "month_cos"]

# selected_columns = top_50_features + categorical_features + date_features + ["labels_12h"]

# df = df[selected_columns].copy()
# # 各カラムのNaNの数を計算
# nan_counts = df.isnull().sum()

# # NaNが50個以上のカラム名をリストとして取得
# columns_to_drop = nan_counts[nan_counts >= 200].index.tolist()

# # 対象カラムをDataFrameからドロップ
# df = df.drop(columns=columns_to_drop)
# df = df.dropna()


# y = df["labels_12h"]
# df = df.drop("labels_12h", axis=1)

# X = sliding_window(df, sequence_length)

# X, X_indices = transform_window_list_to_numpy_and_index(X)
# y = y.reindex(X_indices).values

# final_filter_indices = X_indices.intersection(indices_to_filter)

# mask = X_indices.isin(final_filter_indices)

# filtered_X = X[mask]

# filtered_y = y[mask]


# # データ全体の長さを取得
# n_samples = filtered_X.shape[0]

# # 分割割合を設定（例: 訓練データ80%、テストデータ20%）
# train_ratio = 0.8
# # split_index を計算
# split_index = int(n_samples * train_ratio)

# # 訓練データとテストデータに分割
# X_train = filtered_X[:split_index]
# X_test = filtered_X[split_index:]

# # yについても同様に分割（filtered_y は filtered_X とインデックスが揃っている前提）
# y_train = filtered_y[:split_index]
# y_test = filtered_y[split_index:]

# print("X_train shape:", X_train.shape)
# print("y_train shape:", y_train.shape)
# print("X_test shape:", X_test.shape)
# print("y_test shape:", y_test.shape)


# In[ ]:


# ----------------------------------------------------------------------------
# スケーリングの適用
# ----------------------------------------------------------------------------

# # StandardScaler を初期化
# scaler = StandardScaler()

# # 訓練データを一時的に2次元にreshape: (n_train_samples * seq_len, n_features)
# # reshape(-1, X_train.shape[-1]) は、最初の2つの次元をまとめて1つの次元にし、
# # 最後の次元 (n_features) はそのままにする、という意味です。
# X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])

# numeric_feature_indices = list(range(50))

# # 標準化したい数値特徴量の部分を抽出
# X_train_numeric = X_train_reshaped[:, numeric_feature_indices]

# # 訓練データでスケーラーを学習 (統計量を計算) し、同時に訓練データを変換
# X_train_numeric_scaled = scaler.fit_transform(X_train_numeric)

# # 標準化されなかった他の列（文字列などを含む可能性がある）
# X_train_non_numeric = np.delete(X_train_reshaped, numeric_feature_indices, axis=1)

# X_train_scaled_reshaped = np.empty_like(X_train_reshaped, dtype=object)

# col_index = 0
# numeric_scaled_col_index = 0
# non_numeric_col_index = 0

# for i in range(X_train_reshaped.shape[1]):
#     if i in numeric_feature_indices:
#         # 標準化された数値列を挿入
#         X_train_scaled_reshaped[:, i] = X_train_numeric_scaled[:, numeric_scaled_col_index]
#         numeric_scaled_col_index += 1
#     else:
#         # 標準化されなかった列を挿入
#         X_train_scaled_reshaped[:, i] = X_train_non_numeric[:, non_numeric_col_index]
#         non_numeric_col_index += 1
# # 結果の確認
# print("元のNumPy配列:")
# print(X_train_reshaped)
# print("\n標準化したい数値列のみ抽出:")
# print(X_train_numeric)
# print("\n標準化された数値列:")
# print(X_train_numeric_scaled)
# print("\n標準化後のNumPy配列 (元の列順):")
# print(X_train_scaled_reshaped)

# # 訓練データを元の3次元形状に戻す
# X_train_scaled = X_train_scaled_reshaped.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])

# # テストデータを一時的に2次元にreshape: (n_test_samples * seq_len, n_features)
# X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])

# # 標準化したい数値特徴量の部分を抽出
# X_test_numeric = X_test_reshaped[:, numeric_feature_indices]

# # 学習済みのスケーラーを使ってテストデータを変換 (fitはしない！)
# X_test_numeric_scaled = scaler.transform(X_test_numeric)

# X_test_non_numeric = np.delete(X_test_reshaped, numeric_feature_indices, axis=1)

# X_test_scaled_reshaped = np.empty_like(X_test_reshaped, dtype=object)

# col_index = 0
# numeric_scaled_col_index = 0
# non_numeric_col_index = 0
# for i in range(X_test_reshaped.shape[1]):
#     if i in numeric_feature_indices:
#         # 標準化された数値列を挿入
#         X_test_scaled_reshaped[:, i] = X_test_numeric_scaled[:, numeric_scaled_col_index]
#         numeric_scaled_col_index += 1
#     else:
#         # 標準化されなかった列を挿入
#         X_test_scaled_reshaped[:, i] = X_test_non_numeric[:, non_numeric_col_index]
#         non_numeric_col_index += 1

# # テストデータを元の3次元形状に戻す
# X_test_scaled = X_test_scaled_reshaped.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

# # ----------------------------------------------------------------------------
# # 結果の確認
# # ----------------------------------------------------------------------------

# print("\n--- Scaling Results ---")
# print("Original X_train shape:", X_train.shape)
# print("Scaled X_train shape:", X_train_scaled.shape)
# print("Original X_test shape:", X_test.shape)
# print("Scaled X_test shape:", X_test_scaled.shape)

# # スケーリング後のデータの統計量を確認 (平均が0、標準偏差が1に近づいているはず)
# print("\nScaled X_train statistics (first feature):")
# print("Mean:", X_train_scaled[:, :, 0].mean())
# print("Std:", X_train_scaled[:, :, 0].std())

# print("\nScaled X_test statistics (first feature):")
# # テストデータは訓練データの統計量で変換されているため、厳密に平均0/標準偏差1にはなりません
# print("Mean:", X_test_scaled[:, :, 0].mean())
# print("Std:", X_test_scaled[:, :, 0].std())

# # これ以降、X_train_scaled と X_test_scaled をモデルの入力として使用します。


# In[ ]:


###############  クラスタリングをもとにしたフィルタリングを行う場合  #############################
# import umap.umap_ as umap
# from sklearn.cluster import KMeans
# import hdbscan
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd


# # Step 1: Prepare data (assumes sliding window sequences and binary labels)
# sequence_length = 60
# horizon = 6

# df.replace([np.inf, -np.inf], np.nan, inplace=True)

# categorical_features = ["close_cusum", "dex_volume_cusum", "active_senders_cusum", "active_receivers_cusum", "address_count_sum_cusum", "contract_calls_cusum",
#                                      "whale_tx_count_cusum", "sign_entropy_12_cusum","sign_entropy_24_cusum",  "buy_sell_ratio_cusum", "MA_6_24_cross_flag", "MA_12_48_cross_flag", "MA_24_72_cross_flag",
#                                      "MA_slope_6_24_change_flag", "MA_slope_12_48_change_flag", "MA_slope_pct_change_6_24_change_flag",
#                                      "MA_slope_pct_change_12_48_change_flag", "MA_slope_pct_change_24_72_change_flag","volatility_change_flag"]
# date_features = ["hour", "day_of_week", "hour_sin", "hour_cos", "day", "day_sin", "day_cos"]

# selected_columns = ["close"] + ['MA_t_6', 'MA_t_24', 'MA_t_72', 'MA_t_168'] + top_50_features + categorical_features + date_features

# df = df[selected_columns]

# df[categorical_features] = df[categorical_features].astype('category')

# # 各カラムのNaNの数を計算
# nan_counts = df.isnull().sum()

# # NaNが50個以上のカラム名をリストとして取得
# columns_to_drop = nan_counts[nan_counts >= 200].index.tolist()

# # 対象カラムをDataFrameからドロップ
# df = df.drop(columns=columns_to_drop)
# df = df.dropna()

# original_df_columns = df.columns.tolist()
# feature_names = [col for col in original_df_columns]

# X = sliding_window(df, sequence_length)
# agg_X = transform_sequence_data(X)

# nan_counts = agg_X.isnull().sum()

# print(f"agg_X_nan: {nan_counts[nan_counts >= 100]}")

# agg_X = agg_X.dropna()
# print(f"Aggregated data shape after dropping NaNs: {agg_X.shape}")

# # y = df["labels_12h"]

# # df = df.drop("labels_12h", axis=1)

# # y_aligned = y.reindex(agg_X.index).dropna() # agg_X のインデックスに対応する y の値を取得し、NaNを除く

# # agg_X も y_aligned と同じインデックスを持つようにフィルタリング
# # agg_X = agg_X.reindex(y_aligned.index)


# # 分割割合を設定（例: 訓練データ80%、テストデータ20%）
# train_ratio = 0.8
# n_samples = len(agg_X) # agg_Xのサンプル数を取得
# # split_index を計算
# split_index = int(n_samples * train_ratio)

# # 訓練データを分割
# X_train_agg = agg_X.iloc[:split_index].copy()
# # # y_train = y_aligned.iloc[:split_index].copy() # 対応するラベルも同様に分割

# # テストデータを分割 (split_index から最後まで)
# X_test_agg = agg_X.iloc[split_index:].copy()
# # # y_test = y_aligned.iloc[split_index:].copy() # 対応するラベルも同様に分割


# # # ここで、X_train_agg と y_train_aligned, X_test_agg と y_test_aligned は
# # # それぞれインデックスが揃っており、対応する集約特徴量とラベルになっています。

# # print("\n--- Data Split Shapes ---")
# # print("X_train_agg shape:", X_train_agg.shape)
# # # print("y_train_aligned shape:", y_train.shape)
# # print("X_test_agg shape:", X_test_agg.shape)
# # # print("y_test_aligned shape:", y_test.shape)
# # # ----------------------------------------------------------------------------
# # # 数値特徴量と非数値特徴量の特定
# # # ----------------------------------------------------------------------------

# # DataFrameのカラムのデータ型を確認
# # 数値型 (int, float) のカラムを特定
# numeric_cols = X_train_agg.select_dtypes(include=np.number).columns.tolist()

# # 非数値型 (object, categoryなど) のカラムを特定
# non_numeric_cols = X_train_agg.select_dtypes(exclude=np.number).columns.tolist()

# print(f"数値特徴量 ({len(numeric_cols)}): {numeric_cols[:10]}...") # 例として最初の10個を表示
# print(f"非数値特徴量 ({len(non_numeric_cols)}): {non_numeric_cols}")

# # ----------------------------------------------------------------------------
# # スケーリングの適用
# # ----------------------------------------------------------------------------

# scaler = StandardScaler()

# # 訓練データの数値特徴量のみを抽出してスケーラーをfit+transform
# X_train_numeric = X_train_agg[numeric_cols]
# X_train_numeric_scaled = scaler.fit_transform(X_train_numeric)

# # 結果をDataFrameに戻す (カラム名を保持)
# X_train_numeric_scaled_df = pd.DataFrame(X_train_numeric_scaled,
#                                          index=X_train_agg.index,
#                                          columns=numeric_cols)

# # 訓練データの非数値特徴量を抽出 (DataFrameのまま)
# X_train_non_numeric_df = X_train_agg[non_numeric_cols]

# # スケーリングされた数値特徴量DataFrameと非数値特徴量DataFrameを元の列順で結合
# # 元のX_train_aggのカラム順を取得
# original_cols_order = X_train_agg.columns.tolist()

# # 結合したDataFrameを作成し、元の順序に並べ替える
# X_train_scaled = pd.concat([X_train_numeric_scaled_df, X_train_non_numeric_df], axis=1)
# X_train_scaled = X_train_scaled[original_cols_order] # 元の列順に戻す


# # テストデータの数値特徴量のみを抽出してスケーラーでtransform (fitはしない！)
# X_test_numeric = X_test_agg[numeric_cols]
# X_test_numeric_scaled = scaler.transform(X_test_numeric)

# # 結果をDataFrameに戻す (カラム名を保持)
# X_test_numeric_scaled_df = pd.DataFrame(X_test_numeric_scaled,
#                                         index=X_test_agg.index,
#                                         columns=numeric_cols)

# # テストデータの非数値特徴量を抽出 (DataFrameのまま)
# X_test_non_numeric_df = X_test_agg[non_numeric_cols]

# # スケーリングされた数値特徴量DataFrameと非数値特徴量DataFrameを元の列順で結合
# X_test_scaled = pd.concat([X_test_numeric_scaled_df, X_test_non_numeric_df], axis=1)
# X_test_scaled = X_test_scaled[original_cols_order] # 元の列順に戻す

# # X_train_scaled に NaN が含まれているか確認
# # 修正: pandas DataFrame/Series のisnull().any().any()を使用する
# if X_train_scaled.isnull().any().any():
#     print("X_train_scaled に NaN が含まれています。")
# else:
#     print("X_train_scaled に NaN は含まれていません。")

#     # X_test_scaled に NaN が含まれているか確認
# # 修正: pandas DataFrame/Series のisnull().any().any()を使用する
# if X_test_scaled.isnull().any().any():
#     print("X_test_scaled に NaN が含まれています。")
# else:
#     print("X_test_scaled に NaN は含まれていません。")

# # 潜在空間の次元数を設定
# latent_dim = 128 # ハイパーパラメータとして調整可能

# # Step 2: Extraction of latent vectors using an AutoEncoder
# # AutoEncoderの訓練 (2次元入力を渡す)
# print("\n--- Training AutoEncoder ---")
# latent_train, encoder = train_autoencoder(X_train_scaled.values, latent_dim=latent_dim, max_epochs=50) # エポック数は調整

# # テストデータの潜在空間表現を取得
# print("\n--- Getting Test Latent Vectors ---")
# latent_test = test_autoencoder(X_test_scaled.values, encoder)

# # 訓練データとテストデータの潜在ベクトルを結合
# all_latent = np.vstack((latent_train, latent_test))


# print("\n--- Latent Space Shapes ---")
# print("Latent train shape:", latent_train.shape)
# print("Latent test shape:", latent_test.shape)
# print("All latent shape:", all_latent.shape)


# # HDBSCANモデルの初期化と訓練データでの学習
# # min_cluster_size, min_samples, cluster_selection_epsilonなどのパラメータはデータに応じて調整が必要です。
# # 以下の値はあくまで例です。
# hdbscan_model = hdbscan.HDBSCAN(
#     min_cluster_size=100,
#     min_samples=10,
#     cluster_selection_epsilon=0.2,
#     gen_min_span_tree=True  # Trueにすると可視化に便利です
# )

# print("\n--- Clustering Training Data with HDBSCAN ---")
# # 訓練データの潜在ベクトルを使ってクラスタリング
# train_clusters = hdbscan_model.fit_predict(latent_train)

# # 訓練データのクラスタリング結果をDataFrameに追加 (元のインデックスを保持)
# X_train_scaled['hdbscan_cluster'] = train_clusters


# print(f"訓練データのクラスター数 (ノイズクラス含む): {len(set(train_clusters))}")
# print(f"訓練データのノイズポイント数 (-1): {np.sum(train_clusters == -1)}")

# # テストデータのクラスタリング
# # HDBSCANはpredictメソッドを持たず、学習済みのモデルを使って新しいデータ点をクラスタリングするには
# # `hdbscan.approximate_predict` または `hdbscan.membership_vectors` を使うか、
# # 訓練データとテストデータを結合して再度fitする必要があります。
# # ここでは簡便のため、訓練データとテストデータを結合してfitします。

# print("\n--- Clustering All Data (Train + Test) with HDBSCAN ---")
# # 結合したデータで再度HDBSCANを学習 (同じパラメータを使用)
# hdbscan_model_all = hdbscan.HDBSCAN(
#     min_cluster_size=100,
#     min_samples=10,
#     cluster_selection_epsilon=0.2,
#     gen_min_span_tree=True
# )

# all_clusters = hdbscan_model_all.fit_predict(all_latent)

# # 訓練データとテストデータのクラスタリング結果に分割
# train_clusters_all = all_clusters[:len(latent_train)]
# test_clusters_all = all_clusters[len(latent_train):]

# # クラスタリング結果を元のDataFrameに追加 (元のインデックスを保持)
# # X_train_aggとX_test_aggは既に元のDataFrameから分割されているため、インデックスを基準に結合し直す
# # まず結合されたDataFrameを作成
# df_combined = pd.concat([X_train_scaled, X_test_scaled])

# # 結合されたDataFrameのインデックス順にクラスター結果を適用
# df_combined['hdbscan_cluster_all'] = all_clusters

# # 訓練データとテストデータに再度分割し、クラスターカラムを追加
# X_train_scaled['hdbscan_cluster_all'] = df_combined.loc[X_train_scaled.index, 'hdbscan_cluster_all']
# X_test_scaled['hdbscan_cluster_all'] = df_combined.loc[X_test_scaled.index, 'hdbscan_cluster_all']

# # # y_train['hdbscan_cluster_all'] = df_combined.loc[y_train.index, 'hdbscan_cluster_all']
# # # y_test['hdbscan_cluster_all'] = df_combined.loc[y_test.index, 'hdbscan_cluster_all']


# print(f"全データのクラスター数 (ノイズクラス含む): {len(set(all_clusters))}")
# print(f"全データのノイズポイント数 (-1): {np.sum(all_clusters == -1)}")
# print(f"テストデータのクラスター数 (ノイズクラス含む): {len(set(test_clusters_all))}")
# print(f"テストデータのノイズポイント数 (-1): {np.sum(test_clusters_all == -1)}")

# # クラスタリング結果の確認 (オプション)
# # # 例えば、各クラスターのサイズや、クラスターごとのラベル分布などを確認できます。
# # print("\n--- Cluster Distribution (All Data) ---")
# # print(df_combined['hdbscan_cluster_all'].value_counts().sort_index())

# # print("\n--- Label Distribution per Cluster (Train Data) ---")
# # print(y_train.groupby('hdbscan_cluster_all')['labels_12h'].value_counts(normalize=True).unstack().fillna(0))

# # print("\n--- Label Distribution per Cluster (Test Data) ---")
# # print(y_test.groupby('hdbscan_cluster_all')['labels_12h'].value_counts(normalize=True).unstack().fillna(0))


# # クラスタリング結果を後続のモデルの入力として使用する場合
# # X_train_agg と X_test_agg に 'hdbscan_cluster_all' カラムが追加されました。
# # これをLightGBMなどのモデルの特徴量として利用できます。
# # 例:
# # features_for_lgbm_train = X_train_agg.drop(columns=['date', 'hdbscan_cluster']) # 他に不要なカラムがあれば除く
# # features_for_lgbm_train['hdbscan_cluster_all'] = X_train_agg['hdbscan_cluster_all']
# # ...

# # DataFrameのリストをNumPy配列に変換
# numpy_array_list = []
# for df_window in X:
#     # 各DataFrameをNumPy配列に変換
#     # NaNが含まれる場合は、ここでfillna()などで処理することを検討してください。
#     numpy_window = df_window.to_numpy()
#     numpy_array_list.append(numpy_window)

# # NumPy配列のリストをスタックして3次元配列にする
# if numpy_array_list:
#     # np.stack は新しい次元を追加してスタックします。
#     # axis=0 は、最初の次元 (サンプル次元) に沿ってスタックすることを意味します。
#     X_3d_numpy = np.stack(numpy_array_list, axis=0)

#     print("\n変換後の3次元NumPy配列:")
#     # print(X_3d_numpy) # 配列が大きい場合はコメントアウト推奨
#     print("形状:", X_3d_numpy.shape)
#     print("データ型:", X_3d_numpy.dtype)

#     # 形状の確認
#     samples = X_3d_numpy.shape[0]
#     timesteps = X_3d_numpy.shape[1]
#     features = X_3d_numpy.shape[2]
#     print(f"変換後の形状: (samples={samples}, timesteps={timesteps}, features={features})")

# else:
#     print("\nDataFrameのリストが空です。NumPy配列を作成できませんでした。")

# # 保存先のフォルダとファイル名を指定
# output_folder = '/content/drive/MyDrive/Colab Notebooks/cryptoprice_prediction_tft' # 保存したいフォルダを指定
# x_file_name = 'X_3d_numpy.pkl'
# clusters_file_name = 'all_clusters.pkl'
# feature_names_file_name = 'feature_names.pkl'
# all_latent_file_name = 'all_latent.pkl' # all_latentのファイル名

# x_file_path = os.path.join(output_folder, x_file_name)
# clusters_file_path = os.path.join(output_folder, clusters_file_name)
# feature_names_file_path = os.path.join(output_folder, feature_names_file_name)
# all_latent_file_path = os.path.join(output_folder, all_latent_file_name) # all_latentのファイルパス

# # X_3d_numpy, all_clusters, feature_names, all_latent を保存
# try:
#     # X_3d_numpy を保存
#     with open(x_file_path, 'wb') as f:
#         pickle.dump(X_3d_numpy, f)
#     print(f"NumPy配列 '{x_file_name}' を '{x_file_path}' に保存しました。")

#     # all_clusters を保存
#     with open(clusters_file_path, 'wb') as f:
#         pickle.dump(all_clusters, f)
#     print(f"NumPy配列 '{clusters_file_name}' を '{clusters_file_path}' に保存しました。")

#     # feature_names を保存
#     with open(feature_names_file_path, 'wb') as f:
#         pickle.dump(feature_names, f)
#     print(f"リスト '{feature_names_file_name}' を '{feature_names_file_path}' に保存しました。")

#     # all_latent を保存
#     with open(all_latent_file_path, 'wb') as f:
#         pickle.dump(all_latent, f)
#     print(f"NumPy配列 '{all_latent_file_name}' を '{all_latent_file_path}' に保存しました。")


# except Exception as e:
#     print(f"データの保存中にエラーが発生しました: {e}")


# In[ ]:


###########　クラスタごとの 平均ローソク足チャート を描く　###########################
def plot_cluster_mean_price(sequences, cluster_labels, seq_len):
    unique_clusters = np.unique(cluster_labels)
    for cluster in unique_clusters:
        cluster_data = sequences[cluster_labels == cluster]
        mean_series = cluster_data.mean(axis=0)  # (seq_len, n_features)
        plt.plot(mean_series[:, 0], label=f"Cluster {cluster}")  # 例: [0]=価格
    plt.legend()
    plt.title("Mean Price per Cluster")
    plt.show()
###########　クラスタごとの 移動平均方向（特徴量に含めている場合）###########################
def plot_cluster_ma_directions(cluster_data, cluster_labels, ma_indices=[1,2,3,4]):
    for cluster in np.unique(cluster_labels):
        data = cluster_data[cluster_labels == cluster]
        mean_ma = data.mean(axis=0)[:, ma_indices]
        for i, ma in enumerate(ma_indices):
            plt.plot(mean_ma[:, i], label=f"MA{i+1} (Cluster {cluster})")
        plt.title(f"Cluster {cluster} MA Directions")
        plt.legend()
        plt.show()


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from statsmodels.tsa.stattools import acf
from scipy.fft import fft
from scipy.signal import find_peaks

def compute_slope(series):
    """単純な線形回帰で傾きを返す"""
    x = np.arange(len(series))
    # NaNが含まれている場合はNaNを返す
    if np.isnan(series).any() or len(series) < 2:
        return np.nan
    # 全て同じ値の場合、傾きは0
    if np.all(series == series[0]):
        return 0.0
    return linregress(x, series).slope

def compute_autocorr(series, lag=1):
    """自己相関を計算"""
    # Explicitly convert to numpy array of float type, handling errors
    try:
        numeric_series = np.asarray(series, dtype=np.float64)
    except ValueError:
        print("Warning: Could not convert series to float64 in compute_autocorr. Returning NaN.")
        return np.nan # Return NaN if conversion fails

    # NaNが含まれている場合はNaNを返す
    if np.isnan(numeric_series).any() or len(numeric_series) <= lag:
        return np.nan
    # check if the series is all constants or contains less than lag+2 non-nan values
    if np.nanstd(numeric_series) < 1e-9 or np.sum(~np.isnan(numeric_series)) <= lag:
        return np.nan

    # Drop NaNs before calculating autocorrelation
    cleaned_series = numeric_series[~np.isnan(numeric_series)]

    if len(cleaned_series) <= lag:
         return np.nan # Not enough data after dropping NaNs

    return acf(cleaned_series, nlags=lag, fft=True)[lag]


def compute_fft_amplitude(series):
    """FFTの振幅の合計（DC成分除く）を計算"""
    # Explicitly convert to numpy array of float type, handling errors
    try:
        numeric_series = np.asarray(series, dtype=np.float64)
    except ValueError:
        print("Warning: Could not convert series to float64 in compute_fft_amplitude. Returning NaN.")
        return np.nan # Return NaN if conversion fails

    # NaNが含まれている場合はNaNを返す
    if np.isnan(numeric_series).any() or len(numeric_series) < 2:
        return np.nan
    # 全て同じ値の場合はNaNを返す (周波数成分がないため)
    if np.all(numeric_series == numeric_series[0]):
        return np.nan

    # NaNを除外してFFTを計算
    cleaned_series = numeric_series[~np.isnan(numeric_series)]

    if len(cleaned_series) < 2:
        return np.nan # データが足りない場合はNaN


    yf = fft(cleaned_series)
    # DC成分 (0 Hz) を除く
    amplitudes = np.abs(yf[1:len(yf)//2])
    return np.sum(amplitudes)

def count_peaks(series):
    """価格系列のピーク数をカウント"""
    # NaNを除外
    cleaned_series = np.asarray(series, dtype=np.float64)
    cleaned_series = cleaned_series[~np.isnan(cleaned_series)]
    if len(cleaned_series) < 3: # ピーク検出には最低3点必要
        return np.nan
    peaks, _ = find_peaks(cleaned_series)
    return len(peaks)

def compute_trend_consistency_in_samples(cluster_data, price_feature_index, first_frac):
    """
    クラスタ内の各サンプルについて前半・後半のスロープを計算し、両方が正である割合を返す
    """
    consistent_positive_slope_count = 0
    total_valid_samples = 0

    for sample_seq in cluster_data:
        price_sample = sample_seq[:, price_feature_index].astype(np.float64)
        if np.isnan(price_sample).any() or len(price_sample) < 2:
            continue

        total_valid_samples += 1
        sample_h = int(len(price_sample) * first_frac)
        sample_slope1 = compute_slope(price_sample[:sample_h])
        sample_slope2 = compute_slope(price_sample[sample_h:])

        if not np.isnan(sample_slope1) and not np.isnan(sample_slope2) and sample_slope1 > 0 and sample_slope2 > 0:
            consistent_positive_slope_count += 1

    if total_valid_samples > 0:
        return consistent_positive_slope_count / total_valid_samples
    else:
        return np.nan # 有効なサンプルがない場合はNaN


def extract_cluster_features(X, cluster_labels, price_feature_index=0, ma_indices={'MA_t_6': 1, 'MA_t_24': 2, 'MA_t_72': 3}):
    """
    各クラスタについて、平均トレンド傾き、ボラティリティ、変化点、
    自己相関、FFT振幅、ピーク数、**移動平均線のアベレージ**などの特徴量を計算
    """
    # ノイズクラスタ(-1)を除外
    unique_clusters = np.unique(cluster_labels[cluster_labels >= 0])
    cluster_info = {}

    for cid in unique_clusters:
        indices = np.where(cluster_labels == cid)[0]
        # クラスタに属するサンプルが少ない場合はスキップまたはNaNを返すなどを検討
        if len(indices) < 10: # 例: 最小サンプル数を10とする
             # print(f"Warning: Cluster {cid} has only {len(indices)} samples. Skipping feature extraction.")
             continue

        # オブジェクト配列から抽出した価格系列を数値型に変換し、変換できない値はNaNに置換
        price_series_all_samples_object = X[indices, :, price_feature_index] # shape (n_cluster_samples, seq_len)

        # 明示的に数値型に変換 (変換できない場合はNaN)
        price_series_all_samples_numeric = np.empty_like(price_series_all_samples_object, dtype=np.float64)
        for r in range(price_series_all_samples_object.shape[0]):
            for c in range(price_series_all_samples_object.shape[1]):
                try:
                    price_series_all_samples_numeric[r, c] = float(price_series_all_samples_object[r, c])
                except (ValueError, TypeError):
                    price_series_all_samples_numeric[r, c] = np.nan


        # 各サンプルごとに特徴量を計算し、その平均を取る
        # NaNを無視して計算する関数を使用するか、NaNを除去して計算
        # compute_slope, compute_autocorr, compute_fft_amplitude は内部でNaNを処理
        slopes = np.array([compute_slope(sample) for sample in price_series_all_samples_numeric])
        volatilities = np.array([np.nanstd(sample) for sample in price_series_all_samples_numeric]) # np.nanstdを使用
        autocorrs = np.array([compute_autocorr(sample) for sample in price_series_all_samples_numeric])
        fft_amplitudes = np.array([compute_fft_amplitude(sample) for sample in price_series_all_samples_numeric])
        # ピーク数を計算
        peak_counts = np.array([count_peaks(sample) for sample in price_series_all_samples_numeric])

        # 中央からの変化量
        mid = price_series_all_samples_numeric.shape[1] // 2
        changes = np.array([
            (np.nanmean(sample[mid:]) - np.nanmean(sample[:mid])) if len(sample[:mid]) > 0 and len(sample[mid:]) > 0 else np.nan
            for sample in price_series_all_samples_numeric
        ])

        # === 移動平均線のアベレージを計算 ===
        ma_averages = {}
        for ma_name, ma_idx in ma_indices.items():
            if ma_idx < X.shape[2]: # MAインデックスが特徴量数を超えていないか確認
                # クラスタに属する全サンプルの、指定されたMA系列を抽出
                ma_series_all_samples_object = X[indices, :, ma_idx] # shape (n_cluster_samples, seq_len)
                # 数値型に変換
                ma_series_all_samples_numeric = np.empty_like(ma_series_all_samples_object, dtype=np.float64)
                for r in range(ma_series_all_samples_object.shape[0]):
                     for c in range(ma_series_all_samples_object.shape[1]):
                         try:
                             ma_series_all_samples_numeric[r, c] = float(ma_series_all_samples_object[r, c])
                         except (ValueError, TypeError):
                             ma_series_all_samples_numeric[r, c] = np.nan

                # 各サンプルのMA系列の平均を計算し、さらにその平均を取る
                # または、全サンプル・全タイムステップでのMA値の平均を取る
                # ここでは、全サンプル・全タイムステップでのMA値の平均を取る
                mean_ma_value = np.nanmean(ma_series_all_samples_numeric)
                ma_averages[f'mean_{ma_name}'] = mean_ma_value
            else:
                ma_averages[f'mean_{ma_name}'] = np.nan # インデックスが無効な場合はNaN

        # 平均価格系列の計算 (可視化などに使用)
        avg_series = np.nanmean(price_series_all_samples_numeric, axis=0) # NaNを無視して平均を計算

        cluster_info[cid] = {
            "mean_slope": np.nanmean(slopes), # NaNを無視して平均
            "mean_volatility": np.nanmean(volatilities),
            "mean_change": np.nanmean(changes),
            "mean_autocorr1": np.nanmean(autocorrs),
            "mean_fft_amplitude": np.nanmean(fft_amplitudes),
            "mean_peak_count": np.nanmean(peak_counts), # 平均ピーク数を追加
            "avg_series": avg_series, # 各時刻の平均価格
            "num_samples": len(indices), # クラスタに含まれるサンプル数
            **ma_averages # 計算したMAアベレージを追加
        }

    return cluster_info


# In[ ]:


import numpy as np
import pandas as pd
from scipy.stats import linregress

def compute_slope(series):
    """単純な線形回帰で傾きを返す"""
    x = np.arange(len(series))
    # NaNが含まれている場合はNaNを返す
    if np.isnan(series).any() or len(series) < 2:
        return np.nan
    # 全て同じ値の場合、傾きは0
    if np.all(series == series[0]):
        return 0.0
    return linregress(x, series).slope


# Helper function to compute slope consistency ratio per cluster (moved from label_by_cluster)
def compute_trend_consistency_in_samples(cluster_data_3d, price_feature_index, first_frac):
    """
    クラスタ内の各サンプルについて前半・後半のスロープを計算し、両方が正である割合を返す (3D NumPy入力対応)

    Args:
        cluster_data_3d (np.ndarray): クラスタに属するサンプルを含む3D NumPy配列 (n_cluster_samples, seq_len, n_features)。
        price_feature_index (int): 価格特徴量のインデックス。
        first_frac (float): 前半部分の割合。

    Returns:
        float: 両方のスロープが正であるサンプルの割合。有効なサンプルがない場合は np.nan。
    """
    consistent_positive_slope_count = 0
    total_valid_samples = 0

    # cluster_data_3d は (n_cluster_samples, seq_len, n_features) の形状
    for sample_seq in cluster_data_3d: # 各 sample_seq は (seq_len, n_features)
        # 価格系列を抽出
        price_sample = sample_seq[:, price_feature_index].astype(np.float64)

        if np.isnan(price_sample).any() or len(price_sample) < 2:
            continue

        total_valid_samples += 1
        sample_h = int(len(price_sample) * first_frac)
        if sample_h == 0 or sample_h >= len(price_sample): # Ensure valid split points
             continue

        sample_slope1 = compute_slope(price_sample[:sample_h])
        sample_slope2 = compute_slope(price_sample[sample_h:])

        if not np.isnan(sample_slope1) and not np.isnan(sample_slope2) and sample_slope1 > 0 and sample_slope2 > 0:
            consistent_positive_slope_count += 1

    if total_valid_samples > 0:
        return consistent_positive_slope_count / total_valid_samples # Fix: Use total_valid_samples for denominator
    else:
        return np.nan # 有効なサンプルがない場合はNaN


def label_by_cluster(cluster_info, X_3d_numpy, cluster_labels, # X_3d_numpy and cluster_labels are needed for sample-level consistency check
                     first_frac=0.5,
                     trend_thr_strong=0.02, trend_thr_moderate=0.005,
                     reversal_thr=0.01, range_thr=0.002,
                     volatility_thr_high=50, volatility_thr_low=10,
                     price_feature_index=0, # Add price_feature_index
                     # Add parameters for strong_uptrend_cont criteria adjustment
                     uptrend_overall_score_threshold=0.7, # 緩和ポイント1: 総合スコアの閾値
                     weights={
                         "slope_consistency": 1.0,
                         "price_distribution": 1.0,
                         "mean_avg_slope": 1.0,
                         "mean_peak_count": 0.5
                     }, # 緩和ポイント2: 各基準の重み
                     slope_consistency_mapping_start=0.6, # 緩和ポイント3: スロープ一貫性スコアのマッピング開始点
                     price_distribution_mapping_start=0.7, # 緩和ポイント4: 価格分布スコアのマッピング開始点
                     strong_avg_slope_threshold=0.02, # 緩和ポイント5: クラスタ平均スロープの強い閾値
                     moderate_avg_slope_threshold=0.005, # 緩和ポイント5: クラスタ平均スロープの中程度閾値
                     peak_count_threshold=5, # 緩和ポイント6: ピーク数閾値
                     # Add MA indices for the new strong_uptrend_cont logic
                     ma_short_index=1,
                     ma_mid_index=2,
                     ma_long_index=3
                    ):
    """
    cluster_info[cid]['avg_series'] を使って
    前半・後半のスロープ、ボラティリティなどから
    より詳細なパターンを判定し、戦略名を返す (最低8パターン以上)
      - strong_uptrend_cont  : 強い上昇継続
      - moderate_uptrend_cont: 中程度の上昇継続
      - strong_downtrend_cont: 強い下降継続
      - moderate_downtrend_cont: 中程度の下降継続
      - uptrend_to_range     : 上昇からレンジへ
      - downtrend_to_range   : 下降からレンジへ
      - range_bound          : レンジ相場
      - reversal_up          : 下降から上昇への反転兆し
      - reversal_down          : 上昇から下降への反転兆し
      - high_volatility_trend: 高ボラティリティトレンド
      - low_volatility_range : 低ボラティリティレンジ
      - choppy               : 方向感のない乱高下
      - unknown              : その他
    """
    label_strategy = {}

    # Calculate global min/max prices from X_3d_numpy
    # Ensure X_3d_numpy is not None or empty
    if X_3d_numpy is not None and X_3d_numpy.shape[0] > 0:
         # Extract the price feature across all samples and time steps
         all_prices = X_3d_numpy[:, :, price_feature_index].flatten()
         # Ensure all_prices is numeric before calculating min/max
         all_prices_numeric = pd.to_numeric(all_prices, errors='coerce')
         global_min_price = np.nanmin(all_prices_numeric)
         global_max_price = np.nanmax(all_prices_numeric)
         global_price_range = global_max_price - global_min_price if global_max_price > global_min_price else 1.0 # Avoid division by zero
    else:
         global_min_price = 0.0
         global_max_price = 1.0
         global_price_range = 1.0 # Default range if no sequences are provided


    for cid, info in cluster_info.items():
        series = info.get("avg_series")
        volatility = info.get("mean_volatility", np.nan) # ボラティリティ情報を取得
        mean_peak_count = info.get("mean_peak_count", np.nan) # 平均ピーク数を取得

        # Get mean MA values from cluster_info
        mean_ma_6 = info.get('mean_MA_t_6', np.nan)
        mean_ma_24 = info.get('mean_MA_t_24', np.nan)
        mean_ma_72 = info.get('mean_MA_t_72', np.nan)

        # avg_seriesがNoneまたはデータ不足の場合はスキップ
        if series is None or len(series) < max(ma_short_index, ma_mid_index, ma_long_index) + 2: # MA計算に必要な最小長を確認
             label_strategy[cid] = "unknown"
             continue

        L = len(series)
        h = int(L * first_frac)

        # 前半・後半のスロープ計算 (NaNチェックを含む compute_slope を使用)
        slope1 = compute_slope(series[:h])
        slope2 = compute_slope(series[h:])

        # NaNが含まれる場合は unknown とする
        # Note: This check might be too strict if only avg_series slopes are used for some strategies.
        # Consider moving this check inside strategy-specific blocks if needed.
        # For now, keeping it here means any NaN in avg_series slopes makes it 'unknown'.
        # strong_uptrend_cont now uses mean MA values, so this check is less relevant for that strategy.
        # if np.isnan(slope1) or np.isnan(slope2):
        #      print(f"Debug: Cluster {cid} has NaN slope in avg_series. Labeling as unknown.")
        #      label_strategy[cid] = "unknown"
        #      continue


        # --- strong_uptrend_cont 判定ロジック (平均MA基準) ---
        # Check if mean MA values are valid and ordered correctly
        # This uses the *average* value of the MA lines across the entire sequence for the cluster
        if not np.isnan(mean_ma_6) and not np.isnan(mean_ma_24) and not np.isnan(mean_ma_72):
             # Check if mean MA values are in the desired order (short > mid > long)
             # AND if they are all above a certain price level (e.g., global min price or a threshold)
             # A simple check is just the order for now, assuming they are generally positive values.
             # More robust check could involve comparing to a baseline or checking mean MA slope.
             # Let's check the order and that the shortest MA average is significantly positive or above a threshold.
             # Using mean MA slope from cluster_info would be ideal, but extract_cluster_features calculates mean *value*, not slope.
             # We could calculate the slope of the *average* MA series in extract_cluster_features, or use the mean MA values here.
             # Let's use the mean MA values and check if they are ordered and increasing (mean MA_6 > mean MA_24 > mean MA_72).

             # The check for increasing trend should use the slope of the average MA series or the mean of sample MA slopes.
             # Since extract_cluster_features gives mean MA *values*, we can't directly check the slope of the average MA line over time here.
             # Reverting the strong_uptrend_cont logic to use the MA conditions from the average cluster series for consistency with other patterns,
             # but ensuring MA series themselves are used if available in `series`.

             # Check MA position order and positive slope of the *average* MA series for this cluster
             # Assuming MA series are available as separate features in the `series` (avg_series) object if extract_cluster_features was modified to include them.
             # If `series` only contains the average price, we cannot check slopes of average MA lines here.
             # Let's assume `series` (avg_series) *does* contain the average values of MA lines at each time step for the cluster.
             # This would mean `series` is shape (seq_len, n_features) where n_features includes price and average MAs.
             # However, `extract_cluster_features` currently returns `avg_series` as (seq_len,) i.e., only the average price.

             # Let's refine the strong_uptrend_cont logic to use the *mean* MA values calculated in `extract_cluster_features` (mean_MA_t_6, etc.)
             # These are single scalar values representing the average level of the MA line across all samples and time steps in the cluster.
             # This is not ideal for trend *continuation* as it's just an average level, not a slope.

             # A better approach is to calculate the slope of the *average* MA series for each cluster within `extract_cluster_features`
             # and store that. Or, calculate the mean of the *slopes* of MA series *within* each sample in `extract_cluster_features`.
             # Given the current structure, the best we can do with `mean_ma_6`, `mean_ma_24`, `mean_ma_72` is check their order.
             # We cannot check trend continuation using just these average levels.

             # Let's revert to using the MA conditions on the *average price series* (`series`) for strong_uptrend_cont,
             # as originally intended in the code snippet that was commented out,
             # and ensure `series` correctly contains the average MA lines over time if needed for this check.
             # If `series` only contains the average price, then checking MA conditions on it is wrong.

             # Let's assume `extract_cluster_features` was modified to return `avg_series` as (seq_len, n_features)
             # where features include the average price and average MA lines over time for the cluster.
             # Then we can access the average MA series from `series`.

             # Re-implementing strong_uptrend_cont based on MA conditions on the *average cluster series* (assuming it contains average MA lines)
             # Check if the average MA series are available in `series` (avg_series)
             # This requires `avg_series` to be 2D (seq_len, n_features) and include MA features.
             # Currently, `extract_cluster_features` returns `avg_series` as 1D (seq_len,).

             # Given the constraint that `extract_cluster_features` was modified to add mean MA *values* (scalars)
             # to `cluster_info['ma_averages']`, we cannot check MA order or slopes over time using these scalar values.
             # The request was to use the *information stored in ma_averages* for pattern detection.
             # This suggests using the mean MA *values* for the pattern detection.

             # Let's try a simplified strong_uptrend_cont logic using the mean MA values:
             # Check if the mean MA values are ordered (short > mid > long) and are significantly positive.
             # "Significantly positive" is subjective; let's compare them to a small positive threshold.
             ma_order_ok = (mean_ma_6 > mean_ma_24) and (mean_ma_24 > mean_ma_72)
             ma_positive_avg = (mean_ma_6 > 0) and (mean_ma_24 > 0) and (mean_ma_72 > 0) # Check if average MA levels are positive

             # This is a very weak indicator of trend continuation. A better approach would be to check the slope of the average price series
             # or the mean slope of MA series across samples.
             # However, adhering to the request to use `ma_averages` from `cluster_info`, this is one way.

             # Let's combine this with the average price series slope check for a more robust rule.
             # Strong uptrend if:
             # 1. Average MA levels are ordered (short > mid > long) AND
             # 2. Average MA levels are all positive AND
             # 3. Average price series has consistently positive slope (e.g., both slope1 and slope2 are positive and above a threshold)
             # 4. (Optional) Low volatility or consistent price increase.

             # Check if slopes of the average price series are positive and above a strong threshold
             avg_price_slopes_strong_positive = (not np.isnan(slope1) and slope1 > trend_thr_strong) and \
                                                (not np.isnan(slope2) and slope2 > trend_thr_strong)


             if not np.isnan(mean_ma_6) and not np.isnan(mean_ma_24) and not np.isnan(mean_ma_72):
                  ma_order_ok = (mean_ma_6 > mean_ma_24) and (mean_ma_24 > mean_ma_72)
                  ma_positive_avg = (mean_ma_6 > 0) and (mean_ma_24 > 0) and (mean_ma_72 > 0) # Check if average MA levels are positive

                  # Combine MA conditions with average price slope conditions
                  if ma_order_ok and ma_positive_avg and avg_price_slopes_strong_positive:
                       label_strategy[cid] = "strong_uptrend_cont"
                       # print(f"Debug: Cluster {cid} labeled as strong_uptrend_cont based on combined Avg MA and Avg Price Slope criteria.")
                       continue # strong_uptrend_cont と判定されたら他のパターンはチェックしない


        # --- Other pattern detection logic (only if not labeled as strong_uptrend_cont) ---
        # This part uses the slopes of the average price series (slope1, slope2) and volatility.
        # Ensure slope1 and slope2 are valid before using them for other patterns.
        if np.isnan(slope1) or np.isnan(slope2):
            # If slopes are NaN, these patterns cannot be determined based on slopes.
            # Fallback to unknown or other criteria. Let's label as 'unknown' for safety.
            label_strategy[cid] = "unknown"
            # print(f"Debug: Cluster {cid} has NaN slopes in avg_series. Labeling as unknown for other patterns.")
            continue # Move to next cluster


        # Now, use the valid slope1 and slope2 for other pattern checks
        if slope1 < -trend_thr_strong and slope2 < -trend_thr_strong:
            label_strategy[cid] = "strong_downtrend_cont"
        elif slope1 > trend_thr_moderate and slope2 > trend_thr_moderate:
            label_strategy[cid] = "moderate_uptrend_cont"
        elif slope1 < -trend_thr_moderate and slope2 < -trend_thr_moderate:
            label_strategy[cid] = "moderate_downtrend_cont"
        elif slope1 > trend_thr_moderate and abs(slope2) < range_thr:
             label_strategy[cid] = "uptrend_to_range"
        elif slope1 < -trend_thr_moderate and abs(slope2) < range_thr:
             label_strategy[cid] = "downtrend_to_range"
        elif abs(slope1) < range_thr and abs(slope2) < range_thr:
             label_strategy[cid] = "range_bound"
        elif slope1 < -reversal_thr and slope2 > reversal_thr:
             label_strategy[cid] = "reversal_up" # 下降から上昇への反転
        elif slope1 > reversal_thr and slope2 < -reversal_thr:
             label_strategy[cid] = "reversal_down" # 上昇から下降への反転
        # ボラティリティを考慮したパターン
        elif not np.isnan(volatility) and volatility > volatility_thr_high and (slope1 > trend_thr_moderate or slope1 < -trend_thr_moderate):
             label_strategy[cid] = "high_volatility_trend"
        elif not np.isnan(volatility) and volatility < volatility_thr_low and abs(slope1) < range_thr:
             label_strategy[cid] = "low_volatility_range"
        # その他、特定のパターンに当てはまらないもの
        else:
            label_strategy[cid] = "choppy" # 方向感のない乱高下など


    # ノイズクラスタ (-1) には 'unknown' を割り当てる
    if -1 in cluster_labels and -1 not in label_strategy:
         label_strategy[-1] = "unknown"


    # Print strategy distribution for verification
    print("\n--- Strategy Distribution per Cluster (label_by_cluster) ---")
    for cid, strategy in label_strategy.items():
        num_samples = cluster_info.get(cid, {}).get("num_samples", 0)
        print(f"Cluster {cid}: Strategy '{strategy}' ({num_samples} samples)")


    return label_strategy


# In[ ]:


import numpy as np
import pandas as pd

# Rename function and modify signature to process a single sample
def compute_trend_consistency_label_for_sample(
    price_sequence_with_future: np.ndarray, # Needs sequence + future window
    seq_length: int,
    horizon: int,
    sma_windows=(6, 12, 24),
    slope_thresh=0.0,
    consistency_ratio=0.8
) -> float:
    """
    単一の時系列サンプルとそれに続く将来期間のデータに基づき、
    トレンド継続性ラベル（1:上昇トレンド継続, 0:それ以外）を計算する関数。
    将来期間の移動平均線の傾きが特定の条件を満たす場合に1を返す。

    Args:
        price_sequence_with_future (np.ndarray): 現在のシーケンス (seq_length) と
                                                 それに続く将来期間 (horizon) を
                                                 結合した価格データのNumPy配列 (seq_length + horizon,).
        seq_length (int): 現在のシーケンスの長さ。
        horizon (int): ラベル判定に使用する将来期間の長さ。
        sma_windows (tuple): トレンド継続性判定に使用するSMAウィンドウサイズ。
        slope_thresh (float): トレンド継続性判定に使用する傾きの閾値。
        consistency_ratio (float): トレンド継続性判定に使用する一貫性の割合閾値。

    Returns:
        float: 1.0 (上昇トレンド継続) または 0.0 (それ以外)。
               データ不足等で判定不能な場合は np.nan。
    """
    # Ensure the input array has enough data for the sequence and the future window
    if len(price_sequence_with_future) < seq_length + horizon:
        # print(f"Debug(compute_trend_consistency): Insufficient data length ({len(price_sequence_with_future)}) for seq_length ({seq_length}) and horizon ({horizon}). Returning NaN.")
        return np.nan # Not enough data to evaluate the future window

    # Extract the future window part
    # The future window starts immediately after the current sequence ends
    future_prices = price_sequence_with_future[seq_length : seq_length + horizon]

    # Handle cases where future_prices is too short for SMA calculation
    max_sma_window = max(sma_windows) if sma_windows else 0
    if len(future_prices) < max_sma_window:
         # print(f"Debug(compute_trend_consistency): Future prices length ({len(future_prices)}) is less than max SMA window ({max_sma_window}). Cannot calculate all SMAs. Returning NaN.")
         return np.nan # Not enough data to calculate all SMAs

    df = pd.DataFrame({'price': future_prices})

    # Calculate moving average slopes for the future window
    slopes = []
    for w in sma_windows:
        if len(df) >= w: # Ensure window size is not larger than data length
             sma = df['price'].rolling(window=w).mean()
             # Calculate slope using diff for simplicity, consistent with original
             slope = sma.diff().dropna() # Drop NaN from diff result
             if not slope.empty:
                 slopes.append(slope)
             # else: print(f"Debug(compute_trend_consistency): SMA slope calculation resulted in empty series for window {w}.")
        # else: print(f"Debug(compute_trend_consistency): Data length ({len(df)}) less than SMA window ({w}). Cannot calculate slope.")


    # If no slopes were successfully calculated, cannot determine consistency
    if not slopes:
        # print(f"Debug(compute_trend_consistency): No slopes calculated. Returning NaN.")
        return np.nan

    # Check if slopes are consistently positive
    # We need to check for consistency across all slopes at *overlapping time points*.
    # The diff operation reduces the length of each slope series by (window - 1).
    # The comparison should only happen where all slope series have valid data.
    # The shortest slope series determines the number of points to check.
    min_slope_len = min(len(s) for s in slopes)

    if min_slope_len <= 0:
         # print(f"Debug(compute_trend_consistency): Minimum slope length is zero or less. Returning NaN.")
         return np.nan # Cannot check consistency

    # Check consistency for the overlapping valid points
    valid_count = 0
    total_points_to_check = min_slope_len

    # Assuming slopes[0] is the shortest valid slope series and its index aligns
    # with the start of valid data for other slopes. This is true if SMAs are calculated
    # from the same starting future_prices data.
    all_slopes_positive_at_point = True
    for i in range(total_points_to_check):
        current_point_positive = True
        for s in slopes:
             # Access the i-th valid slope point
             if i < len(s): # Ensure index is within bounds (redundant due to min_slope_len but safe)
                 if not (s.iloc[i] > slope_thresh): # Use iloc for position-based indexing
                     current_point_positive = False
                     break # No need to check other slopes for this point
             else:
                 # This case should not be reached due to min_slope_len
                 # print(f"Debug(compute_trend_consistency): Unexpected index access {i} out of bounds for a slope series of length {len(s)}.")
                 current_point_positive = False # Treat as not positive if index is out of bounds
                 break # Should not happen, but defensive check

        if current_point_positive:
            valid_count += 1
        # else: print(f"Debug(compute_trend_consistency): Point {i} is not consistently positive.")


    # Calculate consistency ratio
    # Use total_points_to_check as the base for the ratio
    ratio = valid_count / total_points_to_check if total_points_to_check > 0 else 0.0


    # Determine label based on consistency ratio
    if ratio >= consistency_ratio:
        # print(f"Debug(compute_trend_consistency): Ratio ({ratio}) >= consistency_ratio ({consistency_ratio}). Returning 1.0.")
        return 1.0 # Consistent positive trend
    else:
        # print(f"Debug(compute_trend_consistency): Ratio ({ratio}) < consistency_ratio ({consistency_ratio}). Returning 0.0.")
        return 0.0 # Trend not consistently positive


# Example usage (for a single sequence):
# Assuming you have a single sequence 'X_sample' (seq_len, n_features)
# and the corresponding future price data 'future_prices_sample' (horizon,)
# Combine them: price_data_for_labeling = np.concatenate((X_sample[:, price_feature_index], future_prices_sample))
# label = compute_trend_consistency_label_for_sample(price_data_for_labeling, seq_len=..., horizon=...)


# In[ ]:


import numpy as np
import pandas as pd
from scipy.stats import linregress

def compute_slope(series):
    """単純な線形回帰で傾きを返す"""
    x = np.arange(len(series))
    # NaNが含まれている場合はNaNを返す
    if np.isnan(series).any() or len(series) < 2:
        return np.nan
    # 全て同じ値の場合、傾きは0
    if np.all(series == series[0]):
        return 0.0
    return linregress(x, series).slope


# Rename function and modify signature to process a single sample
def compute_trend_consistency_label_for_sample(
    price_sequence_with_future: np.ndarray, # Needs sequence + future window
    seq_length: int, # This is the length of the *past* sequence, not directly used for future check but kept for context
    horizon: int,
    ma_indices=(1, 2, 3), # Use the specified MA indices (MA_t_6, MA_t_24, MA_t_72)
    slope_thresh=0.0,
    consistency_ratio=0.8
) -> float:
    """
    単一の時系列サンプルとそれに続く将来期間のデータに基づき、
    トレンド継続性ラベル（1:上昇トレンド継続, 0:それ以外）を計算する関数。
    将来期間の移動平均線の傾きが特定の条件を満たす場合に1を返す。

    Args:
        price_sequence_with_future (np.ndarray): 現在のシーケンス (seq_length) と
                                                 それに続く将来期間 (horizon) を
                                                 結合した価格データのNumPy配列 (seq_length + horizon, n_features).
                                                 ここで、n_features は元の全特徴量の数です。
        seq_length (int): 現在のシーケンスの長さ。
        horizon (int): ラベル判定に使用する将来期間の長さ。
        ma_indices (tuple): トレンド継続性判定に使用するMA系列のインデックス (MA_t_6, MA_t_24, MA_t_72)。
                            price_sequence_with_future の特徴量インデックスに対応します。
        slope_thresh (float): トレンド継続性判定に使用する傾きの閾値。
        consistency_ratio (float): トレンド継続性判定に使用する一貫性の割合閾値。

    Returns:
        float: 1.0 (上昇トレンド継続) または 0.0 (それ以外)。
               データ不足等で判定不能な場合は np.nan。
    """
    # Ensure the input array has enough data for the sequence and the future window, and has features
    if price_sequence_with_future is None or price_sequence_with_future.ndim != 2 or \
       price_sequence_with_future.shape[0] < seq_length + horizon or \
       price_sequence_with_future.shape[1] <= max(ma_indices): # Ensure MA indices are valid
        # print(f"Debug(compute_trend_consistency): Insufficient data shape {price_sequence_with_future.shape if price_sequence_with_future is not None else 'None'} for required length ({seq_length} + {horizon}) and MA indices {ma_indices}. Returning NaN.")
        return np.nan # Not enough data or invalid shape

    # Extract the future window part for the relevant MA features
    # The future window starts immediately after the current sequence ends
    # future_data will have shape (horizon, n_features)
    future_data = price_sequence_with_future[seq_length : seq_length + horizon, :]

    # Handle cases where future_data is too short for slope calculation
    # A slope requires at least 2 points.
    if future_data.shape[0] < 2:
         # print(f"Debug(compute_trend_consistency): Future data length ({future_data.shape[0]}) is less than 2. Cannot calculate slopes. Returning NaN.")
         return np.nan


    slopes = []
    # Calculate slopes for each specified MA series in the future window
    for ma_idx in ma_indices:
        if ma_idx < future_data.shape[1]: # Ensure index is within feature bounds
            ma_series_future = future_data[:, ma_idx].astype(np.float64) # Extract MA series and ensure numeric type
            # Compute slope for this specific MA series over the future horizon
            slope = compute_slope(ma_series_future)
            if not np.isnan(slope):
                slopes.append(slope)
            # else: print(f"Debug(compute_trend_consistency): Slope calculation resulted in NaN for MA index {ma_idx}.")
        # else: print(f"Debug(compute_trend_consistency): MA index {ma_idx} is out of bounds for future data shape {future_data.shape}.")


    # If no slopes were successfully calculated, cannot determine consistency
    if not slopes:
        # print(f"Debug(compute_trend_consistency): No valid slopes calculated for MA indices {ma_indices}. Returning NaN.")
        return np.nan

    # Check if all calculated slopes are positive
    # all_slopes_positive = all(s > slope_thresh for s in slopes) # This is the original simple check


    # The request mentioned "8割持続すれば". Implementing this "persistence" check
    # requires calculating slopes over sub-windows within the horizon or checking point-wise
    # conditions (MA order and positive slope) within the horizon.
    # Let's implement the point-wise check based on MA order and positive slope over a small delta.

    consistent_points_count = 0
    total_points_to_check = future_data.shape[0] # Check each point in the future window

    if total_points_to_check < 2: # Need at least 2 points to potentially check slope or order
        return np.nan

    # The input `price_sequence_with_future` is the *entire* sequence (past + future)
    # including all features. We need the raw price data from this combined sequence.
    # Assuming the raw price feature index is 0 (from the original request context)
    # raw_price_series_combined = price_sequence_with_future[:, 0].astype(np.float64) # Shape (seq_length + horizon,)

    # To check MA order and slopes point-wise in the future, we need the MA series values at each point.
    # The MA features in `price_sequence_with_future` are the MA values *at that specific timestamp*.
    # So we can directly use the MA columns from `future_data`.
    # `future_data` shape is (horizon, n_features).
    # `future_data[i, ma_short_index]` is the value of the short MA at the i-th time step in the future window.

    # Extract MA series over the future window
    # Shape: (horizon,) for each MA
    ma_short_future = future_data[:, ma_indices[0]].astype(np.float64)
    ma_medium_future = future_data[:, ma_indices[1]].astype(np.float64)
    ma_long_future = future_data[:, ma_indices[2]].astype(np.float64)

    # Calculate point-wise slopes (difference from previous point with delta=1) for MA series
    # Use diff with delta=1 for point-wise slope check
    # Note: Need to handle NaNs in MA series before diff. Using pandas Series to leverage diff's NaN handling.
    ma_short_slope_point_wise = pd.Series(ma_short_future).diff(periods=1).values
    ma_medium_slope_point_wise = pd.Series(ma_medium_future).diff(periods=1).values
    ma_long_slope_point_wise = pd.Series(ma_long_future).diff(periods=1).values


    # Start checking from the second point (index 1) in the future window, as diff with delta=1 produces NaN at index 0
    # Or, if a larger delta is used, start checking from delta index.
    # Let's use delta=1 for point-wise slope and start check from index 1.
    start_check_idx = 1 if total_points_to_check > 1 else 0 # If only one point, cannot check slope

    # If total_points_to_check is 1, we can only check MA order, not slope.
    # Let's define the condition for a single point as just the MA order.

    # Checkable points are from start_check_idx to total_points_to_check - 1
    num_checkable_points = total_points_to_check - start_check_idx

    if num_checkable_points <= 0 and total_points_to_check > 0: # If total_points_to_check is 1, num_checkable_points is 0, but we can still check MA order
         # Only return NaN if there's no data at all
         if total_points_to_check == 0:
             # print(f"Debug(compute_trend_consistency): No checkable points ({num_checkable_points}) in future window. Returning NaN.")
             return np.nan


    for i in range(total_points_to_check):
        # Check MA order condition at point i
        # MA_s[i] > MA_m[i] > MA_l[i] for Uptrend
        # Ensure MA values are not NaN at this point
        ma_short_val = ma_short_future[i]
        ma_medium_val = ma_medium_future[i]
        ma_long_val = ma_long_future[i]

        ma_positions_ordered_at_point = (not np.isnan([ma_short_val, ma_medium_val, ma_long_val]).any()) and \
                                        (ma_short_val > ma_medium_val) and \
                                        (ma_medium_val > ma_long_val)

        # Check positive slope for each MA at point i (using point-wise slopes)
        ma_slopes_positive_point_wise = True
        if i > 0: # Need a previous point for diff
             ma_short_slope_val = ma_short_slope_point_wise[i]
             ma_medium_slope_val = ma_medium_slope_point_wise[i]
             ma_long_slope_val = ma_long_slope_point_wise[i]

             ma_slopes_positive_point_wise = (not np.isnan([ma_short_slope_val, ma_medium_slope_val, ma_long_slope_val]).any()) and \
                                             (ma_short_slope_val > slope_thresh) and \
                                             (ma_medium_slope_val > slope_thresh) and \
                                             (ma_long_slope_val > slope_thresh)
        elif total_points_to_check == 1: # Special case: only one point in future window
             ma_slopes_positive_point_wise = True # Assume slope condition met if only one point and order holds (simplification)
        else: # i=0 and total_points_to_check > 1
             ma_slopes_positive_point_wise = False # Cannot check slope at i=0 using diff=1


        # A point is considered "consistent" if the MA order holds AND the point-wise slope condition holds (where applicable)
        # If total_points_to_check == 1, only MA order is checked.
        if total_points_to_check == 1:
             if ma_positions_ordered_at_point:
                  consistent_points_count += 1
        else: # total_points_to_check > 1
             if i > 0 and ma_positions_ordered_at_point and ma_slopes_positive_point_wise:
                  consistent_points_count += 1
             elif i == 0 and ma_positions_ordered_at_point: # For the first point, only check order
                  consistent_points_count += 1


    # Calculate consistency ratio based on total points checked
    # If total_points_to_check == 1, the ratio is 1 or 0.
    # If total_points_to_check > 1, the ratio is over points from index 0 to total_points_to_check - 1.
    ratio = consistent_points_count / total_points_to_check if total_points_to_check > 0 else 0.0


    # Determine label based on consistency ratio
    if ratio >= consistency_ratio:
        # print(f"Debug(compute_trend_consistency): Ratio ({ratio}) >= consistency_ratio ({consistency_ratio}). Returning 1.0.")
        return 1.0 # Consistent positive trend
    else:
        # print(f"Debug(compute_trend_consistency): Ratio ({ratio}) < consistency_ratio ({consistency_ratio}). Returning 0.0.")
        return 0.0 # Trend not consistently positive


# New function to check future MA conditions for Downtrend for a single sequence
def check_future_ma_conditions_downtrend(
    combined_sequence_data: np.ndarray, # Combined past + future data (seq_length + horizon, n_features)
    seq_length: int,
    horizon: int,
    ma_indices: tuple, # (short, medium, long) MA feature indices
    eval_window_L: int, # Evaluation window length (L)
    k_of_l: int,        # K parameter for K-of-L rule
    buffer_epsilon: float = 0.001, # Buffer epsilon (ε) for MA order
    slope_threshold: float = 0.0005, # Slope threshold (ε_s)
    delta_slope: int = 3 # Delta for MA slope calculation (MA[t] - MA[t-delta]). Used for K-of-L rule.
) -> float:
    """
    Checks if future MA conditions (order and slopes) are met for K-of-L points
    within the evaluation window [t+H, t+H+L) for a Downtrend.

    Args:
        combined_sequence_data (np.ndarray): Combined past (seq_length) and future (horizon + eval_window_L)
                                             data (seq_length + horizon + eval_window_L, n_features).
                                             This function expects data covering the period from the start of the
                                             past sequence up to the end of the evaluation window.
        seq_length (int): Length of the past sequence.
        horizon (int): Prediction horizon (H).
        ma_indices (tuple): Tuple of (short_ma_index, medium_ma_index, long_ma_index).
        eval_window_L (int): Evaluation window length (L).
        k_of_l (int): K parameter for K-of-L rule.
        buffer_epsilon (float): Buffer epsilon (ε) for MA order.
        slope_threshold (float): Slope threshold (ε_s).
        delta_slope (int): Delta for MA slope calculation (MA[t] - MA[t-delta]).

    Returns:
        float: 1.0 if conditions are met for K-of-L points, 0.0 otherwise, np.nan if data insufficient.
    """
    # Ensure enough data for the horizon and evaluation window
    required_total_length = seq_length + horizon + eval_window_L
    if combined_sequence_data is None or combined_sequence_data.ndim != 2 or \
       combined_sequence_data.shape[0] < required_total_length or \
       combined_sequence_data.shape[1] <= max(ma_indices): # Ensure MA indices are valid
        # print(f"Debug(check_future_ma_conditions_downtrend): Insufficient data shape {combined_sequence_data.shape if combined_sequence_data is not None else 'None'} for required length ({required_total_length}) and MA indices {ma_indices}. Returning NaN.")
        return np.nan # Not enough data

    # Extract the future evaluation window data [t+H, t+H+L)
    # This window starts at index seq_length + horizon relative to the start of the combined data.
    future_eval_window_data = combined_sequence_data[seq_length + horizon : required_total_length, :] # Shape (L, n_features)


    # Ensure the evaluation window has enough data points (L points)
    if future_eval_window_data.shape[0] != eval_window_L:
         # This should be caught by the initial length check, but as a safeguard
         # print(f"Debug(check_future_ma_conditions_downtrend): Extracted future eval window has incorrect length {future_eval_window_data.shape[0]}. Expected {eval_window_L}. Returning NaN.")
         return np.nan


    # Extract MA series for the future evaluation window
    short_ma_series = future_eval_window_data[:, ma_indices[0]].astype(np.float64)
    medium_ma_series = future_eval_window_data[:, ma_indices[1]].astype(np.float64)
    long_ma_series = future_eval_window_data[:, ma_indices[2]].astype(np.float64)

    # Calculate MA slopes within the future evaluation window using the specified delta
    # Need to ensure delta_slope is less than or equal to L-1 if L>1
    effective_delta_slope = min(delta_slope, eval_window_L - 1) if eval_window_L > 1 else 0
    if effective_delta_slope == 0 and eval_window_L > 1: # Need at least 2 points to calculate slope with delta=1
         effective_delta_slope = 1 # Default delta to 1 if eval window is large enough


    short_ma_slopes = calculate_ma_slope_diff(short_ma_series, delta=effective_delta_slope)
    medium_ma_slopes = calculate_ma_slope_diff(medium_ma_series, delta=effective_delta_slope)
    long_ma_slopes = calculate_ma_slope_diff(long_ma_series, delta=effective_delta_slope)


    # Check conditions for K-of-L points within the evaluation window [0, L-1] for Downtrend
    condition_met_count = 0
    # Start checking from index `effective_delta_slope` to L-1, as slopes calculated with delta need previous points
    start_check_idx = effective_delta_slope if eval_window_L > effective_delta_slope else (eval_window_L -1 if eval_window_L > 0 else 0)
    if eval_window_L == 1: start_check_idx = 0 # Special case for L=1, check at index 0 if possible


    num_checkable_points = eval_window_L - start_check_idx

    if num_checkable_points <= 0:
         # print(f"Debug(check_future_ma_conditions_downtrend): No checkable points ({num_checkable_points}) in future eval window for K-of-L check. Returning NaN.")
         return np.nan # Cannot check K-of-L rule

    effective_k_of_l = min(k_of_l, num_checkable_points)


    for u in range(start_check_idx, eval_window_L):
         # Check MA order condition at point u for Downtrend: Short < Medium < Long with buffer
         ma_order_downtrend = (not np.isnan([short_ma_series[u], medium_ma_series[u], long_ma_series[u]]).any()) and \
                              (short_ma_series[u] < medium_ma_series[u] * (1 - buffer_epsilon)) and \
                              (medium_ma_series[u] < long_ma_series[u] * (1 - buffer_epsilon))

         # Check slope condition at point u for Downtrend: slopes negative
         # Slope index corresponds to the end point of the diff calculation.
         # If delta=3, slope[u] is MA[u] - MA[u-3]. We check this at index u.
         # Ensure slope values are not NaN and index is within bounds
         if u < len(short_ma_slopes) and u < len(medium_ma_slopes) and u < len(long_ma_slopes): # Check index bounds
              slopes_negative = (not np.isnan([short_ma_slopes[u], medium_ma_slopes[u], long_ma_slopes[u]]).any()) and \
                                (short_ma_slopes[u] < -slope_threshold) and \
                                (medium_ma_slopes[u] < -slope_threshold) and \
                                (long_ma_slopes[u] < -slope_threshold)
         else:
              # print(f"Debug(check_future_ma_conditions_downtrend): Slope index {u} out of bounds for slope arrays. Skipping point.")
              slopes_negative = False # Cannot check slope


         # If both conditions (MA order and negative slopes) are met at this point
         if ma_order_downtrend and slopes_negative:
              condition_met_count += 1


    # Determine label based on K-of-L rule
    label = 1.0 if condition_met_count >= effective_k_of_l else 0.0

    # print(f"Debug(check_future_ma_conditions_downtrend): Checkable points: {num_checkable_points}, Condition met count: {condition_met_count}, K-of-L: {effective_k_of_l}. Label: {label}")

    return label


def generate_binary_labels_from_strategy(X_3d_numpy, strategy_labels, feature_names, sequence_length,
                                         horizon=6, # Prediction Horizon (H) - Example from request
                                         eval_window_L=4, # Evaluation Window Length (L) - Example from request
                                         k_of_l=3, # K parameter for K-of-L rule - Example from request
                                         buffer_epsilon=0.001, # Buffer epsilon (ε) - Example from request
                                         slope_threshold=0.0005, # Slope threshold (ε_s) - Example from request
                                         delta_slope=3, # Delta for MA slope calculation - Example from request
                                         price_feature_index=0, # Assuming price is the first feature
                                         # Add MA indices as parameters (need to map from feature_names)
                                         ma_short_name='MA_t_6',
                                         ma_medium_name='MA_t_24',
                                         ma_long_name='MA_t_72',
                                         # Keep other existing parameters, but they might not be used for the new rules
                                         reversal_check_len=3, # Existing parameter
                                         bb_upper_threshold_pct=0.05, # Existing parameter
                                         bb_lower_threshold_pct=0.05, # Existing parameter
                                         uptrend_threshold_pct=0.02, # Existing parameter (might be superseded)
                                         downtrend_threshold_pct=0.02, # Existing parameter (might be superseded)
                                         range_price_range_pct=0.01 # Existing parameter
                                        ):
    """
    Generates binary labels (0 or 1) for each sequence based on strategy and future conditions.
    Implements new K-of-L MA rule for Uptrend/Reversal_Up and Downtrend/Reversal_Down.

    Args:
        X_3d_numpy (np.ndarray): 3D numpy array of shape (n_sequences, sequence_length, n_features).
        strategy_labels (np.ndarray): 1D numpy array of shape (n_sequences,) containing strategy names.
        feature_names (list): List of feature names corresponding to the last dimension of X_3d_numpy.
        sequence_length (int): The length of each sequence.
        horizon (int): Prediction Horizon (H).
        eval_window_L (int): Evaluation Window Length (L).
        k_of_l (int): K parameter for K-of-L rule.
        buffer_epsilon (float): Buffer epsilon (ε) for MA order check.
        slope_threshold (float): Slope threshold (ε_s).
        delta_slope (int): Delta for MA slope calculation (MA[t] - MA[t-delta]).
        price_feature_index (int): The index of the price feature ('close').
        ma_short_name (str): Name of the short MA feature.
        ma_medium_name (str): Name of the medium MA feature.
        ma_long_name (str): Name of the long MA feature.
        reversal_check_len (int): (Existing)
        bb_upper_threshold_pct (float): (Existing)
        bb_lower_threshold_pct (float): (Existing)
        uptrend_threshold_pct (float): (Existing)
        downtrend_threshold_pct (float): (Existing)
        range_price_range_pct (float): (Existing)

    Returns:
        np.ndarray: 1D numpy array of binary labels (0 or 1) of shape (n_sequences,).
                    Returns None if required features or strategy labels are missing or inconsistent.
    """
    n_sequences = X_3d_numpy.shape[0]
    if n_sequences != len(strategy_labels):
        print(f"Error: Length mismatch between X_3d_numpy ({n_sequences}) and strategy_labels ({len(strategy_labels)}).")
        return np.full(n_sequences, np.nan, dtype=float) # Return NaN array on error

    # Ensure price feature index is valid
    if price_feature_index < 0 or price_feature_index >= X_3d_numpy.shape[2]:
        print(f"Error: Invalid price_feature_index ({price_feature_index}) for data shape {X_3d_numpy.shape}.")
        return np.full(n_sequences, np.nan, dtype=float) # Return NaN array on error

    # Map MA feature names to indices
    ma_short_idx = -1
    ma_medium_idx = -1
    ma_long_idx = -1
    try:
        ma_short_idx = feature_names.index(ma_short_name)
        ma_medium_idx = feature_names.index(ma_medium_name)
        ma_long_idx = feature_names.index(ma_long_name)
        ma_indices = (ma_short_idx, ma_medium_idx, ma_long_idx)

        # Check if MA indices are within feature bounds
        if max(ma_indices) >= X_3d_numpy.shape[2] or min(ma_indices) < 0:
             raise ValueError("MA feature index out of bounds.")

    except ValueError as e:
        print(f"Error: Could not find required MA feature names in feature_names list: {ma_short_name}, {ma_medium_name}, {ma_long_name}. Error: {e}")
        print(f"Available features: {feature_names}")
        return np.full(n_sequences, np.nan, dtype=float) # Cannot label without MA features

    labels = np.full(n_sequences, np.nan, dtype=float) # Initialize labels as NaN


    # Iterate through each sequence and assign a label based on its strategy and future conditions
    for i in range(n_sequences):
        strategy = strategy_labels[i]

        # Check if enough future data exists for the required window (H + L)
        # The data needed for sample i is from original index `i` up to
        # original index `i + sequence_length + horizon + eval_window_L - 1`.
        # This corresponds to the first row of sequences `i` to `i + sequence_length + horizon + eval_window_L - 1`.
        # The last required sequence starts at original index `i + sequence_length + horizon + eval_window_L - 1`.
        # This index must be less than n_sequences in X_3d_numpy.
        last_required_sequence_idx = i + sequence_length + horizon + eval_window_L - 1
        if last_required_sequence_idx >= n_sequences:
             # print(f"Debug: Not enough future sequences available in X_3d_numpy for sequence {i} with H={horizon}, L={eval_window_L}. Last required sequence index {last_required_sequence_idx} >= {n_sequences}. Skipping.")
             labels[i] = np.nan # Cannot label if future data is insufficient
             continue

        # Construct the combined data slice for the check functions
        # Data from original index `i` up to `i + sequence_length + horizon + eval_window_L - 1`.
        # This is the first row of sequences `i` to `i + sequence_length + horizon + eval_window_L - 1`.
        try:
             total_future_len = horizon + eval_window_L
             # combined_data_slice shape is (sequence_length + horizon + eval_window_L, n_features)
             combined_data_slice = X_3d_numpy[i : i + sequence_length + total_future_len, 0, :]
             # Ensure combined_data_slice is numeric
             combined_data_slice = combined_data_slice.astype(np.float64)


        except Exception as e:
             print(f"Error constructing combined data slice for sample {i}: {e}. Skipping.")
             labels[i] = np.nan
             continue


        # --- Strategy-based Labeling Logic ---
        if strategy == 'Uptrend' or strategy == 'Reversal_Up':
            # Use the new function to check future MA conditions for Uptrend
            labels[i] = check_future_ma_conditions(
                combined_data_slice, # Pass the combined data slice
                seq_length=sequence_length, # Use the parameter name
                horizon=horizon,
                ma_indices=ma_indices,
                eval_window_L=eval_window_L,
                k_of_l=k_of_l,
                buffer_epsilon=buffer_epsilon,
                slope_threshold=slope_threshold,
                delta_slope=delta_slope
            )
            # print(f"Debug: Sample {i}, Strategy '{strategy}': Label = {labels[i]}")

        elif strategy == 'Downtrend' or strategy == 'Reversal_Down':
             # Use the new function to check future MA conditions for Downtrend
             labels[i] = check_future_ma_conditions_downtrend(
                 combined_data_slice, # Pass the combined data slice
                 seq_length=sequence_length, # Use the parameter name
                 horizon=horizon,
                 ma_indices=ma_indices,
                 eval_window_L=eval_window_L,
                 k_of_l=k_of_l,
                 buffer_epsilon=buffer_epsilon,
                 slope_threshold=slope_threshold,
                 delta_slope=delta_slope
             )
             # print(f"Debug: Sample {i}, Strategy '{strategy}': Label = {labels[i]}")


        elif strategy == 'Range':
            # Range-bound strategies: Label 1 if future price stayed within a certain range over the horizon
            # This logic can still use a simpler price range check in the future window [t+H, t+H+L)
            # Need the price series within this future evaluation window
            future_eval_window_data_points = combined_data_slice[sequence_length + horizon : sequence_length + horizon + eval_window_L, :] # Shape (L, n_features)
            future_price_series = future_eval_window_data_points[:, price_feature_index]

            future_price_range = np.nanmax(future_price_series) - np.nanmin(future_price_series)
            # Use the price at the start of the evaluation window as a reference
            # Price at original index i + sequence_length + horizon
            reference_price = combined_data_slice[sequence_length + horizon, price_feature_index]


            if not np.isnan(future_price_range) and not np.isnan(reference_price) and reference_price > 0:
                 # Check if the range in the future evaluation window is within a percentage of the reference price
                 labels[i] = 1.0 if future_price_range / reference_price <= range_price_range_pct else 0.0
            else:
                 labels[i] = np.nan # Cannot determine label


        else:
            # For 'choppy', 'unknown', noise (-1) or any other strategies not explicitly handled
            # Default label to NaN or 0. Let's default to NaN for safety if no rule applies.
            labels[i] = np.nan # Or 0.0 depending on desired default for non-rule strategies


    # After iterating through all sequences, handle noise labels
    # Find indices where the original strategy label was -1 (noise)
    noise_indices = np.where(strategy_labels == -1)[0]
    if len(noise_indices) > 0:
         # Assign NaN or a specific label to noise sequences
         labels[noise_indices] = np.nan # Assign NaN to noise labels


    return labels


# In[ ]:


import umap.umap_ as umap
from sklearn.cluster import KMeans
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import pickle
import os

def prepare_data_for_clustering(df, selected_columns, categorical_features, nan_threshold=200, sequence_length=60):
    """
    クラスタリングのためのデータの準備（特徴量選択、NaN処理、カテゴリカル変換、集約特徴量化）を行う関数。

    Args:
        df (pd.DataFrame): 元のデータフレーム。
        selected_columns (list): 選択するカラム名のリスト。
        categorical_features (list): カテゴリカル特徴量のリスト。
        nan_threshold (int): NaNがこれ以上のカラムを削除する閾値。
        sequence_length (int): スライディングウィンドウのサイズ。

    Returns:
        pd.DataFrame: スケーリング済みの集約特徴量DataFrame。
        list: 集約特徴量化後の特徴量名のリスト。
        np.ndarray: 3次元形状の元の時系列データNumPy配列 (NaN処理・カラム選択後)。
        pd.DatetimeIndex: 3次元NumPy配列の各サンプルの終了時点のDatetimeIndex。
        list: 3次元NumPy配列の特徴量名のリスト (NaN削除後)。
    """
    df_processed = df[selected_columns].copy()

    # カテゴリカル特徴量の型変換
    existing_categorical_features = [col for col in categorical_features if col in df_processed.columns]
    if existing_categorical_features:
        df_processed[existing_categorical_features] = df_processed[existing_processed].astype('category')

    # NaNの数を計算し、閾値以上のカラムを削除
    nan_counts = df_processed.isnull().sum()
    columns_to_drop = nan_counts[nan_counts >= nan_threshold].index.tolist()
    if columns_to_drop:
        print(f"Dropping columns with >= {nan_threshold} NaNs: {columns_to_drop}")
        df_processed = df_processed.drop(columns=columns_to_drop)

    # 残った行のNaNを削除
    initial_rows = df_processed.shape[0]
    df_processed = df_processed.dropna().copy()
    print(f"Dropped {initial_rows - df_processed.shape[0]} rows with NaNs. Remaining rows: {df_processed.shape[0]}")

    # 最終的に使用される特徴量名をリストとして保持 (集約前)
    original_feature_names_list = df_processed.columns.tolist()

    # Step 2: Create sliding windows and aggregated features
    X_window_list = sliding_window(df_processed, sequence_length)

    # transform_sequence_data 関数で pandas が定義されていないというエラーが発生していたため、import pandas as pd を追加
    import pandas as pd
    agg_X = transform_sequence_data(X_window_list)

    print(f"Aggregated data shape before dropping NaNs: {agg_X.shape}")
    nan_counts_agg = agg_X.isnull().sum()
    print(f"Aggregated data NaN counts (>= 100):")
    print(nan_counts_agg[nan_counts_agg >= 100])

    # Aggregated dataのNaNを削除
    agg_X = agg_X.dropna().copy()
    print(f"Aggregated data shape after dropping NaNs: {agg_X.shape}")

    # Step 3: Identify numeric columns for scaling
    numeric_cols = agg_X.select_dtypes(include=np.number).columns.tolist()
    print(f"Identified {len(numeric_cols)} numeric columns for scaling.")

    # Step 4: Scale aggregated features
    agg_X_scaled, scaler = scale_features(agg_X, numeric_cols)
    print(f"Scaled aggregated data shape: {agg_X_scaled.shape}")

    # Step 5: Convert sliding window list to 3D NumPy array and get original indices
    # Ensure the 3D NumPy array aligns with the aggregated features index after dropna
    X_3d_numpy_all_sequences, original_indices_all_sequences = transform_window_list_to_numpy_and_index(X_window_list)

    # Filter X_3d_numpy_all_sequences and original_indices_all_sequences to match the index of agg_X_scaled (after dropna)
    valid_agg_indices = agg_X_scaled.index
    mask_valid_sequences = original_indices_all_sequences.isin(valid_agg_indices)

    X_3d_numpy_filtered = X_3d_numpy_all_sequences[mask_valid_sequences]
    original_indices_filtered = original_indices_all_sequences[mask_valid_sequences]

    print(f"\nOriginal X_3d_numpy shape: {X_3d_numpy_all_sequences.shape}")
    print(f"Original original_indices length: {len(original_indices_all_sequences)}")
    print(f"Aggregated data valid indices length (after dropna): {len(valid_agg_indices)}")
    print(f"Filtered X_3d_numpy shape (matching aggregated data): {X_3d_numpy_filtered.shape}")
    print(f"Filtered original_indices length (matching aggregated data): {len(original_indices_filtered)}")


    # 集約特徴量化後の特徴量名を返す
    aggregated_feature_names = agg_X_scaled.columns.tolist()


    return agg_X_scaled, aggregated_feature_names, X_3d_numpy_filtered, original_indices_filtered, original_feature_names_list


def perform_clustering_and_save(df, top_50_features, output_folder, sequence_length=60, latent_dim=128, hdbscan_params=None):
    """
    データの準備、AutoEncoderによる潜在空間抽出、HDBSCANによるクラスタリングを実行し、結果を保存する関数。

    Args:
        df (pd.DataFrame): 元のデータフレーム。
        top_50_features (list): 使用する上位50特徴量のリスト。
        output_folder (str): 結果を保存するフォルダパス。
        sequence_length (int): スライディングウィンドウのサイズ。
        latent_dim (int): AutoEncoderの潜在空間の次元数。
        hdbscan_params (dict, optional): HDBSCANのパラメータ辞書。指定しない場合はデフォルト値を使用。

    Returns:
        tuple:
            - X_3d_numpy (np.ndarray): 3次元形状の元の時系列データNumPy配列 (NaN処理・カラム選択・フィルタリング後)。
            - all_clusters (np.ndarray): 全データのクラスタID。
            - feature_names (list): 3次元NumPy配列の特徴量名のリスト。
            - all_latent (np.ndarray): 全データの潜在空間ベクトル。
            - original_indices_filtered (pd.DatetimeIndex): フィルタリング後のオリジナルインデックス。
    """
    # デフォルトのHDBSCANパラメータ
    if hdbscan_params is None:
        hdbscan_params = {
            'min_cluster_size': 100,
            'min_samples': 10,
            'cluster_selection_epsilon': 0.2,
            'gen_min_span_tree': True
        }

    # Step 1-5: Prepare data and get scaled aggregated features and filtered 3D NumPy array
    categorical_features = ["close_cusum", "dex_volume_cusum", "active_senders_cusum", "active_receivers_cusum", "address_count_sum_cusum", "contract_calls_cusum",
                                         "whale_tx_count_cusum", "sign_entropy_12_cusum","sign_entropy_24_cusum",  "buy_sell_ratio_cusum", "MA_6_24_cross_flag", "MA_12_48_cross_flag", "MA_24_72_cross_flag",
                                         "MA_slope_6_24_change_flag", "MA_slope_12_48_change_flag", "MA_slope_pct_change_6_24_change_flag",
                                         "MA_slope_pct_change_12_48_change_flag", "MA_slope_pct_change_24_72_change_flag","volatility_change_flag"]
    date_features = ["hour", "day_of_week", "hour_sin", "hour_cos", "day", "day_sin", "day_cos"]

    selected_columns = ["close"] + ['MA_t_6', 'MA_t_24', 'MA_t_72', 'MA_t_168'] + ['upper', 'lower'] + top_50_features + categorical_features + date_features

    # prepare_data_for_clustering から original_feature_names_list を取得するように変更
    agg_X_scaled, aggregated_feature_names, X_3d_numpy_filtered, original_indices_filtered, original_feature_names_list = prepare_data_for_clustering(
        df, selected_columns, categorical_features, nan_threshold=200, sequence_length=sequence_length
    )

    # Step 6: Perform clustering using AutoEncoder and HDBSCAN on aggregated features
    # perform_clustering_on_subset 関数に 2次元の agg_X_scaled を渡すように修正
    latent_all, all_clusters, _ = perform_clustering_on_subset(
        agg_X_scaled.values, # Pass the 2D aggregated data as NumPy array
        original_indices_filtered, # Pass the corresponding indices
        latent_dim=latent_dim,
        hdbscan_params=hdbscan_params,
        train_autoencoder_flag=True # Train AutoEncoder on the aggregated data
        # trained_autoencoder_encoder は None のまま
        # autoencoder_save_path は必要に応じて設定
    )


    # Step 7: Save processed data
    # feature_names は 3D NumPy配列の特徴量名リスト (集約前) を使用
    save_processed_data(output_folder, X_3d_numpy_filtered, all_clusters, original_feature_names_list, latent_all, original_indices_filtered)

    # クラスタリング後の処理に必要なデータを返す
    return X_3d_numpy_filtered, all_clusters, original_feature_names_list, latent_all, original_indices_filtered

# helper function definitions (scale_features, perform_clustering, save_processed_data, load_processed_data, perform_clustering_on_subset)
# These should be defined in the same cell or a preceding cell

# scale_features function (from previous response)
def scale_features(df_processed, numeric_cols):
    """
    数値特徴量のスケーリングを行う関数。

    Args:
        df_processed (pd.DataFrame): 準備されたデータフレーム。
        numeric_cols (list): 数値特徴量のカラム名のリスト。

    Returns:
        pd.DataFrame: スケーリングされたデータフレーム。
        StandardScaler: 学習済みのStandardScalerオブジェクト。
    """
    scaler = StandardScaler()

    # 数値特徴量のみを抽出し、スケーリング
    df_numeric = df_processed[numeric_cols]
    df_numeric_scaled = scaler.fit_transform(df_numeric)

    # スケーリング結果をDataFrameに戻す
    df_numeric_scaled_df = pd.DataFrame(df_numeric_scaled,
                                        index=df_processed.index,
                                        columns=numeric_cols)

    non_numeric_cols = [col for col in df_processed.columns if col not in numeric_cols]
    df_non_numeric = df_processed[non_numeric_cols]

    # スケーリングされた数値特徴量と非数値特徴量を元の列順で結合
    df_scaled = pd.concat([df_numeric_scaled_df, df_non_numeric], axis=1)
    df_scaled = df_scaled[df_processed.columns] # 元の列順に戻す

    return df_scaled, scaler

# perform_clustering function (This is not used anymore in perform_clustering_and_save, replaced by perform_clustering_on_subset)
# Keeping it here for completeness if it's used elsewhere, but the logic is superseded for the clustering pipeline.
def perform_clustering(X_agg_scaled, latent_dim, hdbscan_params):
    """
    AutoEncoderによる潜在空間抽出とHDBSCANによるクラスタリングを行う関数。
    (Note: This function is likely deprecated by perform_clustering_on_subset in the main pipeline)

    Args:
        X_agg_scaled (pd.DataFrame): スケーリング済みの集約特徴量DataFrame。
        latent_dim (int): AutoEncoderの潜在空間の次元数。
        hdbscan_params (dict): HDBSCANのパラメータ辞書。

    Returns:
        np.ndarray: 訓練データの潜在空間ベクトル。
        np.ndarray: 全データの潜在空間ベクトル。
        torch.nn.Module: 訓練済みAutoEncoderのエンコーダー部分。
        np.ndarray: 全データのクラスタID。
    """
    # 分割割合を設定（例: 訓練データ80%、テストデータ20%）
    train_ratio = 0.8
    n_samples = len(X_agg_scaled)
    split_index = int(n_samples * train_ratio)

    # 訓練データとテストデータに分割
    X_train_agg_scaled = X_agg_scaled.iloc[:split_index].copy()
    X_test_agg_scaled = X_agg_scaled.iloc[split_index:].copy()

    print("\n--- Data Split Shapes for Clustering ---")
    print("X_train_agg_scaled shape:", X_train_agg_scaled.shape)
    print("X_test_agg_scaled shape:", X_test_agg_scaled.shape)


    # AutoEncoderの訓練 (2次元入力を渡す)
    print("\n--- Training AutoEncoder ---")
    # AutoEncoderの訓練はNumPy配列を期待するため .values を使用
    latent_train, encoder = train_autoencoder(X_train_agg_scaled.values, latent_dim=latent_dim) # max_epochsはtrain_autoencoder内で設定

    # テストデータの潜在空間表現を取得
    print("\n--- Getting Test Latent Vectors ---")
    # AutoEncoderのテストもNumPy配列を期待するため .values を使用
    latent_test = test_autoencoder(X_test_agg_scaled.values, encoder)

    # 訓練データとテストデータの潜在ベクトルを結合
    all_latent = np.vstack((latent_train, latent_test))

    print("\n--- Latent Space Shapes ---")
    print("Latent train shape:", latent_train.shape)
    print("Latent test shape:", latent_test.shape)
    print("All latent shape:", all_latent.shape)

    # HDBSCANモデルの初期化と訓練データでの学習
    # 結合したデータでHDBSCANを学習
    print("\n--- Clustering All Data (Train + Test) with HDBSCAN ---")
    hdbscan_model_all = hdbscan.HDBSCAN(**hdbscan_params)

    all_clusters = hdbscan_model_all.fit_predict(all_latent)

    print(f"全データのクラスター数 (ノイズクラス含む): {len(set(all_clusters))}")
    print(f"全データのノイズポイント数 (-1): {np.sum(all_clusters == -1)}")

    return latent_train, all_latent, encoder, all_clusters


# save_processed_data function (from previous response)
def save_processed_data(output_folder, X_3d_numpy, all_clusters, feature_names, all_latent, original_indices_filtered): # <-- Added original_indices_filtered
    """
    処理済みデータとクラスタリング結果をファイルに保存する関数。

    Args:
        output_folder (str): 保存先のフォルダパス。
        X_3d_numpy (np.ndarray): 3次元形状のNumPy配列。
        all_clusters (np.ndarray): 全データのクラスタID。
        feature_names (list): 特徴量名のリスト (3D NumPy配列のもの)。
        all_latent (np.ndarray): 全データの潜在空間ベクトル (2D Aggregated Featuresのもの)。
        original_indices_filtered (pd.DatetimeIndex): フィルタリング後のオリジナルインデックス (2D Aggregated Featuresのもの)。
                                                     これは 3D NumPy配列の最初の次元と一致するはず。
    """
    # フォルダが存在しない場合は作成
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # ファイル名を指定
    x_file_name = 'X_3d_numpy.pkl'
    clusters_file_name = 'all_clusters.pkl'
    feature_names_file_name = 'feature_names.pkl' # For 3D data
    all_latent_file_name = 'all_latent.pkl'     # For 2D aggregated data
    original_indices_file_name = 'original_indices_filtered.pkl' # For 2D aggregated data (and aligns with 3D data samples)


    x_file_path = os.path.join(output_folder, x_file_name)
    clusters_file_path = os.path.join(output_folder, clusters_file_name)
    feature_names_file_path = os.path.join(output_folder, feature_names_file_name)
    all_latent_file_path = os.path.join(output_folder, all_latent_file_name)
    original_indices_file_path = os.path.join(output_folder, original_indices_file_name)


    # X_3d_numpy, all_clusters, feature_names, all_latent, original_indices_filtered を保存
    try:
        with open(x_file_path, 'wb') as f:
            pickle.dump(X_3d_numpy, f)
        print(f"NumPy配列 '{x_file_name}' を '{x_file_path}' に保存しました。")

        with open(clusters_file_path, 'wb') as f:
            pickle.dump(all_clusters, f)
        print(f"NumPy配列 '{clusters_file_name}' を '{clusters_file_path}' に保存しました。")

        with open(feature_names_file_path, 'wb') as f:
            pickle.dump(feature_names, f)
        print(f"リスト '{feature_names_file_name}' を '{feature_names_file_path}' に保存しました。")

        with open(all_latent_file_path, 'wb') as f:
            pickle.dump(all_latent, f)
        print(f"NumPy配列 '{all_latent_file_name}' を '{all_latent_file_path}' に保存しました。")

        with open(original_indices_file_path, 'wb') as f:
            pickle.dump(original_indices_filtered, f)
        print(f"DatetimeIndex '{original_indices_file_name}' を '{original_indices_file_path}' に保存しました。")


    except Exception as e:
        print(f"データの保存中にエラーが発生しました: {e}")


# load_processed_data function (from previous response)
def load_processed_data(output_folder):
    """
    保存された処理済みデータとクラスタリング結果をファイルから読み込む関数。

    Args:
        output_folder (str): 保存先のフォルダパス。

    Returns:
        tuple:
            - X_3d_numpy (np.ndarray): 3次元形状のNumPy配列。ファイルが存在しない場合はNone。
            - all_clusters (np.ndarray): 全データのクラスタID。ファイルが存在しない場合はNone。
            - feature_names (list): 特徴量名のリスト (3D NumPy配列のもの)。ファイルが存在しない場合はNone。
            - all_latent (np.ndarray): 全データの潜在空間ベクトル (2D Aggregated Featuresのもの)。ファイルが存在しない場合はNone。
            - original_indices_filtered (pd.DatetimeIndex): フィルタリング後のオリジナルインデックス (2D Aggregated Featuresのもの)。ファイルが存在しない場合はNone。
    """
    x_file_path = os.path.join(output_folder, 'X_3d_numpy.pkl')
    clusters_file_path = os.path.join(output_folder, 'all_clusters.pkl')
    feature_names_file_path = os.path.join(output_folder, 'feature_names.pkl')
    all_latent_file_path = os.path.join(output_folder, 'all_latent.pkl')
    original_indices_file_path = os.path.join(output_folder, 'original_indices_filtered.pkl')


    X_3d_numpy = None
    all_clusters = None
    feature_names = None
    all_latent = None
    original_indices_filtered = None

    # X_3d_numpy を読み込み
    if os.path.exists(x_file_path):
        try:
            with open(x_file_path, 'rb') as f:
                X_3d_numpy = pickle.load(f)
            print(f"NumPy配列 'X_3d_numpy.pkl' を '{x_file_path}' から読み込みました。")
            print(f"読み込んだX_3d_numpyの形状: {X_3d_numpy.shape}")
        except Exception as e:
            print(f"NumPy配列 'X_3d_numpy.pkl' の読み込み中にエラーが発生しました: {e}")
    else:
        print(f"ファイル '{x_file_path}' が見つかりませんでした。")

    # all_clusters を読み込み
    if os.path.exists(clusters_file_path):
        try:
            with open(clusters_file_path, 'rb') as f:
                all_clusters = pickle.load(f)
            print(f"NumPy配列 'all_clusters.pkl' を '{clusters_file_path}' から読み込みました。")
            print(f"読み込んだall_clustersの形状: {all_clusters.shape}")
        except Exception as e:
            print(f"NumPy配列 'all_clusters.pkl' の読み込み中にエラーが発生しました: {e}")
    else:
        print(f"ファイル '{clusters_file_path}' が見つかりませんでした。")

    # feature_names を読み込み (3Dデータ用)
    if os.path.exists(feature_names_file_path):
        try:
            with open(feature_names_file_path, 'rb') as f:
                feature_names = pickle.load(f)
            print(f"リスト 'feature_names.pkl' を '{feature_names_file_path}' から読み込みました。")
            print(f"読み込んだfeature_namesの長さ: {len(feature_names)}")
        except Exception as e:
            print(f"リスト 'feature_names.pkl' の読み込み中にエラーが発生しました: {e}")
    else:
        print(f"ファイル '{feature_names_file_path}' が見つかりませんでした。")

    # all_latent を読み込み (2D集約データ用)
    if os.path.exists(all_latent_file_path):
        try:
            with open(all_latent_file_path, 'rb') as f:
                all_latent = pickle.load(f)
            print(f"NumPy配列 'all_latent.pkl' を '{all_latent_file_path}' から読み込みました。")
            print(f"読み込んだall_latentの形状: {all_latent.shape}")
        except Exception as e:
            print(f"NumPy配列 'all_latent.pkl' の読み込み中にエラーが発生しました: {e}")
    else:
        print(f"ファイル '{all_latent_file_path}' が見つかりませんでした。")

    # original_indices_filtered を読み込み (2D集約データ用)
    if os.path.exists(original_indices_file_path):
         try:
             with open(original_indices_file_path, 'rb') as f:
                 original_indices_filtered = pickle.load(f)
             print(f"DatetimeIndex 'original_indices_filtered.pkl' を '{original_indices_file_path}' から読み込みました。")
             print(f"読み込んだoriginal_indices_filteredの長さ: {len(original_indices_filtered)}")
         except Exception as e:
             print(f"DatetimeIndex 'original_indices_filtered.pkl' の読み込み中にエラーが発生しました: {e}")
    else:
         print(f"ファイル '{original_indices_file_path}' が見つかりませんでした。")


    return X_3d_numpy, all_clusters, feature_names, all_latent, original_indices_filtered

# # perform_clustering_on_subset function (from previous response)
# # This function should now receive 2D aggregated data X_subset
# def perform_clustering_on_subset(X_subset, original_indices_subset, latent_dim, hdbscan_params,
#                                  train_autoencoder_flag=True, # Add flag to control training
#                                  trained_autoencoder_encoder=None, # Accept just the encoder
#                                  autoencoder_save_path=None, # Path to save/load the trained encoder
#                                  batch_size_ae_train=64, # Batch size for AE training
#                                  max_epochs_ae_train=50 # Max epochs for AE training
#                                 ):
#     """
#     Performs AutoEncoder-based dimensionality reduction (with optional training)
#     and HDBSCAN clustering on a subset of **2D aggregated features**.

#     Args:
#         X_subset (np.ndarray): Subset of **2D** numpy array of aggregated features (n_subset_samples, n_features).
#                                This should be the output of transform_sequence_data and scaling.
#         original_indices_subset (pd.DatetimeIndex or np.ndarray): Original indices corresponding to X_subset
#                                                                    (n_subset_samples,).
#         latent_dim (int): The dimension of the latent space for the AutoEncoder.
#         hdbscan_params (dict): Dictionary of parameters for HDBSCAN.
#         train_autoencoder_flag (bool): If True, train the AutoEncoder on X_subset.
#                                        If False, use trained_autoencoder_encoder or load from autoencoder_save_path.
#         trained_autoencoder_encoder (torch.nn.Module, optional): Pre-trained AutoEncoder encoder module.
#                                                                  Used if train_autoencoder_flag is False and autoencoder_save_path is None.
#         autoencoder_save_path (str, optional): Path to save the trained encoder (if training)
#                                                or load the encoder from (if not training and trained_autoencoder_encoder is None).
#                                                Should be a file path (e.g., 'encoder_state_dict.pth').
#         batch_size_ae_train (int): Batch size for AutoEncoder training if train_autoencoder_flag is True.
#         max_epochs_ae_train (int): Max epochs for AutoEncoder training if train_autoencoder_flag is True.


#     Returns:
#         tuple: (latent_representation, cluster_labels, original_indices_subset)
#                - latent_representation (np.ndarray): Latent vectors (n_subset_samples, latent_dim).
#                - cluster_labels (np.ndarray): HDBSCAN cluster labels (n_subset_samples,).
#                - original_indices_subset (pd.DatetimeIndex or np.ndarray): The original indices subset passed in.
#                Returns (None, None, None) if processing fails.
#     """
#     if X_subset is None or len(X_subset) == 0:
#         print("Warning: X_subset is empty. Skipping processing for this subset.")
#         return None, None, None

#     # Assert that the input is indeed 2D
#     if X_subset.ndim != 2:
#          print(f"Error: perform_clustering_on_subset expected 2D input (n_samples, n_features), but got {X_subset.ndim}D input with shape {X_subset.shape}.")
#          print("Please ensure you are passing the 2D aggregated features data.")
#          return None, None, None


#     print(f"Starting AutoEncoder processing and clustering for subset with shape: {X_subset.shape}")

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     encoder = None # Initialize encoder

#     # Get the input dimension (number of features) from the subset data
#     input_dim = X_subset.shape[-1]

#     # --- 1. AutoEncoder: Train or Load and Transform ---

#     if train_autoencoder_flag:
#         print("Training AutoEncoder on the subset data...")
#         try:
#             # Use the existing train_autoencoder function
#             # train_autoencoder expects NumPy array as input and calculates input_dim internally
#             latent_representation, encoder = train_autoencoder(
#                 X_subset, # Pass the 2D subset data for training
#                 latent_dim=latent_dim,
#                 batch_size=batch_size_ae_train,
#                 max_epochs=max_epochs_ae_train
#             )
#             print("AutoEncoder training complete.")

#             # Save the trained encoder if a path is provided
#             if autoencoder_save_path:
#                  try:
#                      # Ensure the directory exists
#                      save_dir = os.path.dirname(autoencoder_save_path)
#                      if save_dir and not os.path.exists(save_dir):
#                          os.makedirs(save_dir)
#                          print(f"Created directory: {save_dir}")

#                      # Save the state_dict of the encoder module
#                      torch.save(encoder.state_dict(), autoencoder_save_path)
#                      print(f"Saved trained encoder state_dict to {autoencoder_save_path}")
#                  except Exception as e:
#                      print(f"Warning: Failed to save trained encoder state_dict to {autoencoder_save_path}: {e}")

#         except Exception as e:
#             print(f"Error during AutoEncoder training: {e}")
#             return None, None, None # Return None if training fails

#     else: # train_autoencoder_flag is False
#         print("Using existing AutoEncoder encoder...")
#         encoder = trained_autoencoder_encoder
#         if encoder is None and autoencoder_save_path:
#             print(f"Attempting to load encoder from {autoencoder_save_path}...")
#             try:
#                 # Need to instantiate the AutoEncoder model first to load state_dict
#                 # Use the determined input_dim (which should be n_features from the 2D aggregated data)
#                 model = AutoEncoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
#                 model.load_state_dict(torch.load(autoencoder_save_path, map_location=device))
#                 encoder = model.encoder.to(device) # Get the encoder and move to device
#                 print(f"Loaded encoder state_dict from {autoencoder_save_path}")
#             except Exception as e:
#                 print(f"Error loading encoder state_dict from {autoencoder_save_path}: {e}")
#                 print("Cannot proceed without a valid encoder.")
#                 return None, None, None
#         elif encoder is None and not autoencoder_save_path:
#              print("Error: train_autoencoder_flag is False, but no trained_autoencoder_encoder or autoencoder_save_path provided.")
#              print("Cannot proceed without a valid encoder.")
#              return None, None, None

#         # If we are here, we have an encoder (either provided or loaded)
#         # Now, extract latent vectors for the subset using this encoder
#         print("Extracting latent vectors using the provided/loaded encoder...")
#         try:
#             # Use the existing test_autoencoder function (which uses an encoder)
#             # test_autoencoder expects 2D NumPy array as input
#             latent_representation = test_autoencoder(X_subset, encoder)
#             print("Latent vector extraction complete.")
#         except Exception as e:
#             print(f"Error during latent vector extraction with existing encoder: {e}")
#             return None, None, None # Return None if extraction fails


#     # At this point, latent_representation should be available
#     if latent_representation is None:
#          print("Error: Latent representation could not be obtained.")
#          return None, None, None


#     # --- 2. HDBSCAN Clustering ---
#     print("Starting HDBSCAN clustering on latent vectors...")
#     try:
#         # Ensure hdbscan_params is a dictionary
#         if not isinstance(hdbscan_params, dict):
#             print(f"Error: hdbscan_params must be a dictionary. Got {type(hdbscan_params)}.")
#             # Use default parameters as a fallback if invalid type
#             print("Using default HDBSCAN parameters.")
#             hdbscan_params_effective = {
#                 'min_cluster_size': 10,
#                 'min_samples': 5,
#                 'cluster_selection_epsilon': 0.1
#             }
#         else:
#              hdbscan_params_effective = hdbscan_params


#         # Check if latent_representation has enough samples for clustering
#         min_samples_needed = hdbscan_params_effective.get('min_cluster_size', 10) # Default min_cluster_size
#         if latent_representation.shape[0] < min_samples_needed:
#              print(f"Warning: Not enough samples ({latent_representation.shape[0]}) for HDBSCAN with min_cluster_size={min_samples_needed}. Skipping clustering.")
#              # Return all -1 labels if clustering is skipped due to insufficient data
#              cluster_labels = np.full(latent_representation.shape[0], -1, dtype=int)
#         else:
#             clusterer = hdbscan.HDBSCAN(**hdbscan_params_effective)
#             clusterer.fit(latent_representation)
#             cluster_labels = clusterer.labels_ # Shape (n_subset_samples,)
#             print(f"HDBSCAN clustering complete. Found {len(np.unique(cluster_labels)) - (1 if -1 in cluster_labels else 0)} clusters.")
#             print(f"Cluster label distribution:\n{pd.Series(cluster_labels).value_counts()}")


#     except Exception as e:
#         print(f"Error during HDBSCAN clustering: {e}")
#         return latent_representation, None, original_indices_subset # Return latent vectors and indices even if clustering failed


#     # --- 3. Return Results ---
#     return latent_representation, cluster_labels, original_indices_subset


# In[ ]:


import pickle
import os
import numpy as np # NumPy をインポート

def load_processed_data(output_folder):
    """
    保存された処理済みデータとクラスタリング結果をファイルから読み込む関数。

    Args:
        output_folder (str): 保存先のフォルダパス。

    Returns:
        tuple:
            - X_3d_numpy (np.ndarray): 3次元形状のNumPy配列。ファイルが存在しない場合はNone。
            - all_clusters (np.ndarray): 全データのクラスタID。ファイルが存在しない場合はNone。
            - feature_names (list): 特徴量名のリスト。ファイルが存在しない場合はNone。
            - all_latent (np.ndarray): 全データの潜在空間ベクトル。ファイルが存在しない場合はNone。
            - original_indices_filtered (pd.DatetimeIndex): フィルタリング後のオリジナルインデックス。ファイルが存在しない場合はNone。 # <-- Added
    """
    x_file_path = os.path.join(output_folder, 'X_3d_numpy.pkl')
    clusters_file_path = os.path.join(output_folder, 'all_clusters.pkl')
    feature_names_file_path = os.path.join(output_folder, 'feature_names.pkl')
    all_latent_file_path = os.path.join(output_folder, 'all_latent.pkl')
    original_indices_file_path = os.path.join(output_folder, 'original_indices_filtered.pkl') # <-- Added file path


    X_3d_numpy = None
    all_clusters = None
    feature_names = None
    all_latent = None
    original_indices_filtered = None # <-- Added initialization

    # X_3d_numpy を読み込み
    if os.path.exists(x_file_path):
        try:
            with open(x_file_path, 'rb') as f:
                X_3d_numpy = pickle.load(f)
            print(f"NumPy配列 'X_3d_numpy.pkl' を '{x_file_path}' から読み込みました。")
            print(f"読み込んだX_3d_numpyの形状: {X_3d_numpy.shape}")
        except Exception as e:
            print(f"NumPy配列 'X_3d_numpy.pkl' の読み込み中にエラーが発生しました: {e}")
    else:
        print(f"ファイル '{x_file_path}' が見つかりませんでした。")

    # all_clusters を読み込み
    if os.path.exists(clusters_file_path):
        try:
            with open(clusters_file_path, 'rb') as f:
                all_clusters = pickle.load(f)
            print(f"NumPy配列 'all_clusters.pkl' を '{clusters_file_path}' から読み込みました。")
            print(f"読み込んだall_clustersの形状: {all_clusters.shape}")
        except Exception as e:
            print(f"NumPy配列 'all_clusters.pkl' の読み込み中にエラーが発生しました: {e}")
    else:
        print(f"ファイル '{clusters_file_path}' が見つかりませんでした。")

    # feature_names を読み込み
    if os.path.exists(feature_names_file_path):
        try:
            with open(feature_names_file_path, 'rb') as f:
                feature_names = pickle.load(f)
            print(f"リスト 'feature_names.pkl' を '{feature_names_file_path}' から読み込みました。")
            # print(f"読み込んだfeature_namesの最初の5要素: {feature_names[:5]}") # オプション
            print(f"読み込んだfeature_namesの長さ: {len(feature_names)}")
        except Exception as e:
            print(f"リスト 'feature_names.pkl' の読み込み中にエラーが発生しました: {e}")
    else:
        print(f"ファイル '{feature_names_file_path}' が見つかりませんでした。")

    # all_latent を読み込み
    if os.path.exists(all_latent_file_path):
        try:
            with open(all_latent_file_path, 'rb') as f:
                all_latent = pickle.load(f)
            print(f"NumPy配列 'all_latent.pkl' を '{all_latent_file_path}' から読み込みました。")
            print(f"読み込んだall_latentの形状: {all_latent.shape}")
        except Exception as e:
            print(f"NumPy配列 'all_latent.pkl' の読み込み中にエラーが発生しました: {e}")
    else:
        print(f"ファイル '{all_latent_file_path}' が見つかりませんでした。")

    # <-- Added loading original_indices_filtered
    if os.path.exists(original_indices_file_path):
         try:
             with open(original_indices_file_path, 'rb') as f:
                 original_indices_filtered = pickle.load(f)
             print(f"DatetimeIndex 'original_indices_filtered.pkl' を '{original_indices_file_path}' から読み込みました。")
             print(f"読み込んだoriginal_indices_filteredの長さ: {len(original_indices_filtered)}")
         except Exception as e:
             print(f"DatetimeIndex 'original_indices_filtered.pkl' の読み込み中にエラーが発生しました: {e}")
    else:
         print(f"ファイル '{original_indices_file_path}' が見つかりませんでした。")
    # -->


    return X_3d_numpy, all_clusters, feature_names, all_latent, original_indices_filtered


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import torch
import os

# Helper functions (assuming they are defined elsewhere or within this cell initially)

def calculate_cosine_similarity(latent_vectors):
    """Calculate cosine similarity between latent vectors."""
    # Ensure data is float for cosine similarity calculation
    if latent_vectors.dtype != np.float32 and latent_vectors.dtype != np.float64:
        latent_vectors = latent_vectors.astype(np.float32)

    # Handle potential inf or NaN values which can cause issues with cosine similarity
    # Replace inf with a large number, NaN with mean or median (mean here)
    latent_vectors = np.nan_to_num(latent_vectors, nan=np.nanmean(latent_vectors) if np.nanmean(latent_vectors) is not np.nan else 0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)


    return cosine_similarity(latent_vectors)


def split_data(X, y, clusters, latent_vectors, train_ratio=0.8):
    """
    Splits data, labels, clusters, and latent vectors into train and test sets
    based on a temporal split.
    Assumes X is 3D (n_samples, seq_length, n_features),
    y is 1D (n_samples,), clusters is 1D (n_samples,),
    and latent_vectors is 2D (n_samples, latent_dim).
    n_samples corresponds to the number of sequences.
    """
    n_samples = X.shape[0]
    split_index = int(n_samples * train_ratio)

    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]
    train_clusters = clusters[:split_index]
    test_clusters = clusters[split_index:]
    latent_train = latent_vectors[:split_index]
    latent_test = latent_vectors[split_index:]

    print(f"Data split into train ({len(X_train)} sequences) and test ({len(X_test)} sequences).")

    return X_train, X_test, y_train, y_test, train_clusters, test_clusters, latent_train, latent_test

def map_test_clusters_to_train(latent_train, train_clusters, latent_test, test_clusters):
    """
    Maps test clusters to the most similar train clusters based on cosine similarity
    of latent vectors. Returns a new array of cluster IDs for the combined original data.
    """
    # Ensure unique train clusters are used
    unique_train_clusters = np.unique(train_clusters)
    mapped_all_clusters = np.zeros(len(latent_train) + len(latent_test), dtype=int)

    # Assign original train cluster IDs
    mapped_all_clusters[:len(latent_train)] = train_clusters

    # Map test clusters
    # Need to map based on the original test cluster IDs first, then find representative latent vectors
    unique_test_clusters = np.unique(test_clusters)
    if -1 in unique_test_clusters:
        # Optionally handle noise (-1) differently, maybe map to nearest non-noise train cluster
        # For simplicity here, we will try to map all non-noise test clusters.
        # Noise samples can also be excluded from the test set entirely later.
        unique_test_clusters = unique_test_clusters[unique_test_clusters != -1]


    if len(unique_test_clusters) > 0 and len(unique_train_clusters) > 0:
        # Find a representative latent vector for each unique test cluster
        representative_test_latent = []
        for cluster_id in unique_test_clusters:
            # Use the mean latent vector for samples in this cluster
            cluster_samples_indices = np.where(test_clusters == cluster_id)[0]
            if len(cluster_samples_indices) > 0:
                 representative_test_latent.append(np.mean(latent_test[cluster_samples_indices], axis=0))
            else:
                 representative_test_latent.append(np.zeros(latent_test.shape[-1])) # Handle empty test clusters


        representative_test_latent = np.array(representative_test_latent)


        # Find representative latent vector for each unique train cluster
        representative_train_latent = []
        # Need to map unique_train_clusters back to indices in latent_train and train_clusters
        for cluster_id in unique_train_clusters:
            cluster_samples_indices = np.where(train_clusters == cluster_id)[0]
            if len(cluster_samples_indices) > 0:
                 representative_train_latent.append(np.mean(latent_train[cluster_samples_indices], axis=0))
            else:
                 # This case should ideally not happen if train_clusters come from actual train data
                 representative_train_latent.append(np.zeros(latent_train.shape[-1])) # Handle empty train clusters


        representative_train_latent = np.array(representative_train_latent)


        # Calculate cosine similarity between test representatives and train representatives
        # Handle case where representative_train_latent or representative_test_latent is empty
        if representative_train_latent.shape[0] > 0 and representative_test_latent.shape[0] > 0:
             similarity_matrix = calculate_cosine_similarity(np.vstack([representative_test_latent, representative_train_latent]))
             # Extract the similarity between test reps and train reps
             similarity_test_train = similarity_matrix[:len(unique_test_clusters), len(unique_test_clusters):]


             # Find the best matching train cluster for each test cluster
             # For each test cluster representative, find the index of the train cluster representative
             # with the maximum similarity.
             best_train_cluster_indices = np.argmax(similarity_test_train, axis=1)


             # Create a mapping from original test cluster ID to new train cluster ID
             test_to_train_cluster_map = {
                 unique_test_clusters[i]: unique_train_clusters[best_train_cluster_indices[i]]
                 for i in range(len(unique_test_clusters))
             }
             print(f"Test cluster mapping: {test_to_train_cluster_map}")


             # Apply the mapping to the test samples in the overall mapped_all_clusters array
             # Iterate through the original test clusters
             for i, original_test_cluster_id in enumerate(test_clusters):
                 if original_test_cluster_id != -1 and original_test_cluster_id in test_to_train_cluster_map:
                      # The index in mapped_all_clusters for test data starts after train data
                      mapped_all_clusters[len(latent_train) + i] = test_to_train_cluster_map[original_test_cluster_id]
                 else:
                      # Keep noise cluster ID or assign a default if mapping failed
                      mapped_all_clusters[len(latent_train) + i] = original_test_cluster_id # Keep -1 for noise


        else:
             print("Warning: Cannot perform test cluster mapping. Either no unique train or test clusters or error in representatives.")
             # If mapping is not possible, keep original test cluster IDs (including noise)
             mapped_all_clusters[len(latent_train):] = test_clusters

    else:
         print("Warning: Cannot perform test cluster mapping. Either no unique train or test clusters.")
         # If mapping is not possible, keep original test cluster IDs (including noise)
         mapped_all_clusters[len(latent_train):] = test_clusters


    print(f"Shape of combined and mapped cluster IDs: {mapped_all_clusters.shape}")

    # Return the combined and mapped cluster IDs for the full original dataset size
    return mapped_all_clusters


# Redefining prepare_strategy_data to ensure correct extraction of 2D sequences and handle full sequences only
def prepare_strategy_data(
    X_train_scaled, y_train, train_clusters_all, original_indices_train_all,
    X_val_scaled, y_val, val_clusters_all, original_indices_val_all, # Add validation data
    X_test_scaled, y_test, test_clusters_all, original_indices_test_all,
    integrated_strategy_names, # This is the full array aligned with original_indices_filtered
    seq_length, # Pass seq_length
    original_indices_filtered # Add original_indices_filtered (Start times of ALL filtered sequences)
):
    """
    Prepares data split by strategy and filters out sequences not belonging to a defined strategy
    or where clustering failed (-1 cluster ID).
    Crucially, returns lists of 2D sequences (seq_length, num_features) and aligned targets/indices,
    only considering full sequences, for train, validation, and test sets.

    Args:
        X_train_scaled (np.ndarray): Scaled training features (num_train_sequences, seq_length, num_features).
        y_train (np.ndarray): Training labels (num_train_sequences,).
        train_clusters_all (np.ndarray): Integrated cluster IDs for all train sequences (num_train_sequences,).
        original_indices_train_all (pd.Index or np.ndarray): Original START indices for ALL train sequences (num_train_sequences,).
        X_val_scaled (np.ndarray): Scaled validation features (num_val_sequences, seq_length, num_features). # Added
        y_val (np.ndarray): Validation labels (num_val_sequences,). # Added
        val_clusters_all (np.ndarray): Integrated cluster IDs for all val sequences (num_val_sequences,). # Added
        original_indices_val_all (pd.Index or np.ndarray): Original START indices for ALL val sequences (num_val_sequences,). # Added
        X_test_scaled (np.ndarray): Test features (num_test_sequences, seq_length, num_features).
        y_test (np.ndarray): Test labels (num_test_sequences,).
        test_clusters_all (np.ndarray): Integrated cluster IDs for all test sequences (num_test_sequences,).
        original_indices_test_all (pd.Index or np.ndarray): Original START indices for ALL test sequences (num_test_sequences,).
        integrated_strategy_names (np.ndarray): Full array of strategy names for all sequences (num_total_sequences,),
                                                 aligned with original_indices_filtered (start times of sequences).
        seq_length (int): The length of each sequence.
        original_indices_filtered (pd.Index or np.ndarray): The full original START indices after filtering (num_total_sequences,).


    Returns:
        tuple: (train_data_by_strategy, val_data_by_strategy, test_data_by_strategy) dictionaries.
               Each dictionary maps strategy names to a dict {'X': list of 2D arrays,
                                                               'y': list of labels,
                                                               'original_indices': list of DatetimeIndex/arrays}.
               Returns empty dicts if essential data is missing or validation fails.
    """
    train_data_by_strategy = {}
    val_data_by_strategy = {} # Added
    test_data_by_strategy = {}

    # Assuming strategies_to_run is defined elsewhere or determine from integrated_strategy_names
    all_strategy_names = np.unique(integrated_strategy_names)

    print("--- Starting Data Preparation by Strategy ---")
    print(f"Found strategies in integrated_strategy_names: {all_strategy_names}")

    # Ensure original_indices_filtered is a pandas Index for efficient lookup
    original_indices_full_pd = None
    if isinstance(original_indices_filtered, (pd.Index, pd.DatetimeIndex)):
         original_indices_full_pd = original_indices_filtered
    elif isinstance(original_indices_filtered, np.ndarray):
         try:
              if pd.api.types.is_datetime64_any_dtype(original_indices_filtered):
                   original_indices_full_pd = pd.DatetimeIndex(original_indices_filtered)
              else:
                   original_indices_full_pd = pd.Index(original_indices_filtered)
              print("Converted original_indices_filtered to pandas Index.")
         except Exception as e:
              print(f"Error converting original_indices_filtered np.ndarray to pandas Index: {e}. Cannot prepare data by strategy.")
              return {}, {}, {} # Return empty dicts if conversion fails
    else:
         print(f"Error: original_indices_filtered is not a pandas Index/DatetimeIndex or np.ndarray. Got {type(original_indices_filtered)}. Cannot prepare data by strategy.")
         return {}, {}, {} # Return empty dicts if conversion fails

    # Create a mapping from original START index to strategy name for ALL filtered sequences
    if len(original_indices_full_pd) != len(integrated_strategy_names):
         print(f"Error: Mismatch between original_indices_filtered length ({len(original_indices_full_pd)}) and integrated_strategy_names length ({len(integrated_strategy_names)}). Data preparation issue likely. Cannot proceed.")
         return {}, {}, {} # Return empty dicts if lengths don't match

    strategy_map_from_start_index = {
        original_indices_full_pd[i]: integrated_strategy_names[i]
        for i in range(len(original_indices_full_pd))
    }


    # Determine the number of sequences in the train, val, and test sets from the input array shapes
    num_full_train_sequences = X_train_scaled.shape[0]
    num_full_val_sequences = X_val_scaled.shape[0] # Added
    num_full_test_sequences = X_test_scaled.shape[0]

    print(f"Number of full training sequences (from X_train_scaled shape): {num_full_train_sequences}")
    print(f"Number of full validation sequences (from X_val_scaled shape): {num_full_val_sequences}") # Added
    print(f"Number of full test sequences (from X_test_scaled shape): {num_full_test_sequences}")

    # Check if the number of train/val/test original start indices matches the number of sequences
    if len(original_indices_train_all) != num_full_train_sequences:
         print(f"Error: Mismatch between original_indices_train_all length ({len(original_indices_train_all)}) and number of full training sequences ({num_full_train_sequences}). Data split issue? Returning empty dicts.")
         return {}, {}, {}

    if len(original_indices_val_all) != num_full_val_sequences: # Added
         print(f"Error: Mismatch between original_indices_val_all length ({len(original_indices_val_all)}) and number of full validation sequences ({num_full_val_sequences}). Data split issue? Returning empty dicts.")
         return {}, {}, {}

    if len(original_indices_test_all) != num_full_test_sequences:
         print(f"Error: Mismatch between original_indices_test_all length ({len(original_indices_test_all)}) and number of full test sequences ({num_full_test_sequences}). Data split issue? Returning empty dicts.")
         return {}, {}, {}


    # Create a lookup map for positions in original_indices_full_pd (needed for extracting sequence indices)
    full_indices_position_map = {idx: pos for pos, idx in enumerate(original_indices_full_pd)}


    # --- Process Training Data ---
    print("\nProcessing Training Data (full sequences only)...")
    # Ensure original_indices_train_all is a pandas Index for efficient lookup
    if not isinstance(original_indices_train_all, (pd.Index, pd.DatetimeIndex)):
        try:
            if pd.api.types.is_datetime64_any_dtype(original_indices_train_all):
                 original_indices_train_all_pd = pd.DatetimeIndex(original_indices_train_all)
            else:
                 original_indices_train_all_pd = pd.Index(original_indices_train_all)
            print("Converted original_indices_train_all to pandas Index.")
        except Exception as e:
            print(f"Error converting original_indices_train_all to pandas Index: {e}. Cannot proceed with training data preparation.")
            return {}, {}, {} # Return empty dicts if conversion fails
    else:
        original_indices_train_all_pd = original_indices_train_all


    for strategy_name in all_strategy_names:
        # Skip default or failed categories
        if strategy_name in ['unprocessed', 'unprocessed_or_filtered', 'unknown']: # Also skip 'unknown' if not intended for training
             continue

        # Find indices of training sequences belonging to this strategy (excluding noise)
        strategy_train_sequence_indices = [
            i for i in range(num_full_train_sequences)
            if original_indices_train_all_pd[i] in strategy_map_from_start_index and
               strategy_map_from_start_index[original_indices_train_all_pd[i]] == strategy_name and
               train_clusters_all[i] != -1 # Exclude noise cluster sequences
        ]

        if not strategy_train_sequence_indices:
            # print(f"  No training sequences (excluding noise) found for strategy: {strategy_name}. Skipping.")
            continue

        print(f"  Processing training data for strategy: {strategy_name} ({len(strategy_train_sequence_indices)} sequences)")


        # Extract the actual sequences (2D arrays), labels, and original indices lists
        X_train_strat_list = [X_train_scaled[i] for i in strategy_train_sequence_indices]
        # FIX: Ensure y_train labels are stored as 1D NumPy arrays
        y_train_strat_list = [np.array([y_train[i]]) for i in strategy_train_sequence_indices] # y_train has one label per sequence, convert to 1D array


        original_indices_train_list_strat = []
        for i in strategy_train_sequence_indices:
             seq_start_index = original_indices_train_all_pd[i]
             if seq_start_index in full_indices_position_map:
                  start_pos_in_full = full_indices_position_map[seq_start_index]
                  try:
                       # Extract the block of indices for this sequence from the full list
                       # Check if there are enough indices in original_indices_full_pd
                       if start_pos_in_full + seq_length <= len(original_indices_full_pd):
                           original_indices_train_list_strat.append(original_indices_full_pd[start_pos_in_full : start_pos_in_full + seq_length])
                       else:
                           print(f"Warning: Not enough original indices in full list ({len(original_indices_full_pd)}) for training sequence starting at {seq_start_index} (position {start_pos_in_full}). Required {seq_length} indices. Skipping indices for this sequence.")
                           original_indices_train_list_strat.append(pd.Index([])) # Append empty index


                  except IndexError as e:
                       print(f"Error(prepare_strategy_data_train_indices): IndexError during slicing original_indices_full_pd for strategy '{strategy_name}', sequence index {i}, start_pos_in_full {start_pos_in_full}, seq_length {seq_length}. Size of original_indices_full_pd: {len(original_indices_full_pd)}. Error: {e}")
                       original_indices_train_list_strat.append(pd.Index([])) # Append empty index
                       continue # Continue to next sequence index


             else:
                  print(f"Warning: Start index {seq_start_index} for training sequence {i} not found in full original_indices_filtered. Skipping indices for this sequence.")
                  original_indices_train_list_strat.append(pd.Index([])) # Append empty index
                  continue # Continue to next sequence index


        if len(original_indices_train_list_strat) != len(X_train_strat_list):
             print(f"Warning: Mismatch in number of extracted training sequences ({len(X_train_strat_list)}) and original index lists ({len(original_indices_train_list_strat)}) for strategy '{strategy_name}'. This may indicate an issue with index extraction.")


        train_data_by_strategy[strategy_name] = {
            'X': X_train_strat_list,
            'y': y_train_strat_list,
            'original_indices': original_indices_train_list_strat # List of DatetimeIndex/arrays
        }

    # --- Process Validation Data --- # Added
    print("\nProcessing Validation Data (full sequences only)...")
    if not isinstance(original_indices_val_all, (pd.Index, pd.DatetimeIndex)):
        try:
            if pd.api.types.is_datetime64_any_dtype(original_indices_val_all):
                 original_indices_val_all_pd = pd.DatetimeIndex(original_indices_val_all)
            else:
                 original_indices_val_all_pd = pd.Index(original_indices_val_all)
            print("Converted original_indices_val_all to pandas Index.")
        except Exception as e:
            print(f"Error converting original_indices_val_all to pandas Index: {e}. Cannot proceed with validation data preparation.")
            return train_data_by_strategy, {}, {} # Return empty val/test dicts
    else:
        original_indices_val_all_pd = original_indices_val_all


    for strategy_name in all_strategy_names:
        if strategy_name in ['unprocessed', 'unprocessed_or_filtered', 'unknown']:
             continue
        if strategy_name not in train_data_by_strategy:
             continue

        strategy_val_sequence_indices = [
            i for i in range(num_full_val_sequences)
            if original_indices_val_all_pd[i] in strategy_map_from_start_index and
               strategy_map_from_start_index[original_indices_val_all_pd[i]] == strategy_name and
               val_clusters_all[i] != -1
        ]

        if not strategy_val_sequence_indices:
            continue

        print(f"  Processing validation data for strategy: {strategy_name} ({len(strategy_val_sequence_indices)} sequences)")

        X_val_strat_list = [X_val_scaled[i] for i in strategy_val_sequence_indices]
        # FIX: Ensure y_val labels are stored as 1D NumPy arrays
        y_val_strat_list = [np.array([y_val[i]]) for i in strategy_val_sequence_indices]


        original_indices_val_list_strat = []
        for i in strategy_val_sequence_indices:
             seq_start_index = original_indices_val_all_pd[i]
             if seq_start_index in full_indices_position_map:
                  start_pos_in_full = full_indices_position_map[seq_start_index]
                  try:
                       if start_pos_in_full + seq_length <= len(original_indices_full_pd):
                           original_indices_val_list_strat.append(original_indices_full_pd[start_pos_in_full : start_pos_in_full + seq_length])
                       else:
                            print(f"Warning: Not enough original indices in full list ({len(original_indices_full_pd)}) for validation sequence starting at {seq_start_index} (position {start_pos_in_full}). Required {seq_length} indices. Skipping indices for this sequence.")
                            original_indices_val_list_strat.append(pd.Index([]))

                  except IndexError as e:
                       print(f"Error(prepare_strategy_data_val_indices): IndexError during slicing original_indices_full_pd for strategy '{strategy_name}', sequence index {i}, start_pos_in_full {start_pos_in_full}, seq_length {seq_length}. Size of original_indices_full_pd: {len(original_indices_full_pd)}. Error: {e}")
                       original_indices_val_list_strat.append(pd.Index([]))
                       continue

             else:
                  print(f"Warning: Start index {seq_start_index} for validation sequence {i} not found in full original_indices_filtered. Skipping indices for this sequence.")
                  original_indices_val_list_strat.append(pd.Index([]))
                  continue

        if len(original_indices_val_list_strat) != len(X_val_strat_list):
             print(f"Warning: Mismatch in number of extracted validation sequences ({len(X_val_strat_list)}) and original index lists ({len(original_indices_val_list_strat)}) for strategy '{strategy_name}'.")

        val_data_by_strategy[strategy_name] = {
            'X': X_val_strat_list,
            'y': y_val_strat_list,
            'original_indices': original_indices_val_list_strat
        }


    # --- Process Test Data ---
    print("\nProcessing Test Data (full sequences only)...")
    if not isinstance(original_indices_test_all, (pd.Index, pd.DatetimeIndex)):
        try:
            if pd.api.types.is_datetime64_any_dtype(original_indices_test_all):
                 original_indices_test_all_pd = pd.DatetimeIndex(original_indices_test_all)
            else:
                 original_indices_test_all_pd = pd.Index(original_indices_test_all)
            print("Converted original_indices_test_all to pandas Index.")
        except Exception as e:
            print(f"Error converting original_indices_test_all to pandas Index: {e}. Cannot proceed with test data preparation.")
            return train_data_by_strategy, val_data_by_strategy, {} # Return empty test dict
    else:
        original_indices_test_all_pd = original_indices_test_all


    for strategy_name in all_strategy_names:
        # Skip default or failed categories
        if strategy_name in ['unprocessed', 'unprocessed_or_filtered', 'unknown']:
             continue
        # Only process test data for strategies that have training data
        if strategy_name not in train_data_by_strategy:
             continue

        strategy_test_sequence_indices = [
            i for i in range(num_full_test_sequences)
            if original_indices_test_all_pd[i] in strategy_map_from_start_index and
               strategy_map_from_start_index[original_indices_test_all_pd[i]] == strategy_name and
               test_clusters_all[i] != -1 # Exclude noise cluster sequences
        ]

        if not strategy_test_sequence_indices:
            continue

        print(f"  Processing test data for strategy: {strategy_name} ({len(strategy_test_sequence_indices)} sequences)")


        # Extract the actual sequences (2D arrays), labels, and original indices lists
        X_test_strat_list = [X_test_scaled[i] for i in strategy_test_sequence_indices]
        # FIX: Ensure y_test labels are stored as 1D NumPy arrays
        y_test_strat_list = [np.array([y_test[i]]) for i in strategy_test_sequence_indices]


        # Extract original indices for each full sequence (test set)
        original_indices_test_list_strat = []
        for i in strategy_test_sequence_indices:
             seq_start_index = original_indices_test_all_pd[i]
             if seq_start_index in full_indices_position_map:
                  start_pos_in_full = full_indices_position_map[seq_start_index]
                  try:
                       # Extract the block of indices for this sequence from the full list
                       # Check if there are enough indices in original_indices_full_pd
                       if start_pos_in_full + seq_length <= len(original_indices_full_pd):
                           original_indices_test_list_strat.append(original_indices_full_pd[start_pos_in_full : start_pos_in_full + seq_length])
                       else:
                            print(f"Warning: Not enough original indices in full list ({len(original_indices_full_pd)}) for test sequence starting at {seq_start_index} (position {start_pos_in_full}). Required {seq_length} indices. Skipping indices for this sequence.")
                            original_indices_test_list_strat.append(pd.Index([])) # Append empty index


                  except IndexError as e:
                       print(f"Error(prepare_strategy_data_test_indices): IndexError during slicing original_indices_full_pd for strategy '{strategy_name}', sequence index {i}, start_pos_in_full {start_pos_in_full}, seq_length {seq_length}. Size of original_indices_full_pd: {len(original_indices_full_pd)}. Error: {e}")
                       original_indices_test_list_strat.append(pd.Index([])) # Append empty index
                       continue # Continue to next sequence index

             else:
                  print(f"Warning: Start index {seq_start_index} for test sequence {i} not found in full original_indices_filtered. Skipping indices for this sequence.")
                  original_indices_test_list_strat.append(pd.Index([])) # Append empty index
                  continue # Continue to next sequence index


        if len(original_indices_test_list_strat) != len(X_test_strat_list):
             print(f"Warning: Mismatch in number of extracted test sequences ({len(X_test_strat_list)}) and original index lists ({len(original_indices_test_list_strat)}) for strategy '{strategy_name}'. This may indicate an issue with index extraction.")


        test_data_by_strategy[strategy_name] = {
            'X': X_test_strat_list,
            'y': y_test_strat_list,
            'original_indices': original_indices_test_list_strat # List of DatetimeIndex/arrays
        }


    print("\n--- Finished Data Preparation by Strategy ---")
    print("Summary of sequences per strategy:")
    print("Train:")
    for strategy, data in train_data_by_strategy.items():
        print(f"  '{strategy}': {len(data['X'])} sequences")
    print("Validation:") # Added
    for strategy, data in val_data_by_strategy.items(): # Added
        print(f"  '{strategy}': {len(data['X'])} sequences") # Added
    print("Test:")
    for strategy, data in test_data_by_strategy.items():
        print(f"  '{strategy}': {len(data['X'])} sequences")


    return train_data_by_strategy, val_data_by_strategy, test_data_by_strategy # Return all three dictionaries


# Redefining scale_train_val_test_data to handle train, validation, and test sets
def scale_train_val_test_data(X_train, X_val, X_test, numeric_feature_indices):
    """
    Scales specified numeric features in train, validation, and test sets using StandardScaler
    fitted only on the training data.
    Assumes X_train, X_val, and X_test are 3D (n_samples, seq_length, n_features).

    Args:
        X_train (np.ndarray): Training features (n_train_samples, seq_length, n_features).
        X_val (np.ndarray): Validation features (n_val_samples, seq_length, n_features).
        X_test (np.ndarray): Test features (n_test_samples, seq_length, n_features).
        numeric_feature_indices (list): List of indices of numeric features to scale.

    Returns:
        tuple: (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
               - X_train_scaled (np.ndarray): Scaled training features.
               - X_val_scaled (np.ndarray): Scaled validation features.
               - X_test_scaled (np.ndarray): Scaled test features.
               - scaler (StandardScaler): The fitted StandardScaler object.
    """
    print("--- Starting Scaling of Train, Validation, and Test Data ---")

    scaler = StandardScaler()

    # Reshape X_train, X_val, and X_test for scaling (flatten time and sample dimensions)
    # Shape becomes (n_samples * seq_length, n_features)
    n_samples_train, seq_length_train, n_features = X_train.shape
    n_samples_val, seq_length_val, _ = X_val.shape
    n_samples_test, seq_length_test, _ = X_test.shape

    X_train_reshaped = X_train.reshape(-1, n_features)
    X_val_reshaped = X_val.reshape(-1, n_features)
    X_test_reshaped = X_test.reshape(-1, n_features)

    # Apply scaling only to numeric features
    X_train_scaled_reshaped = X_train_reshaped.copy()
    X_val_scaled_reshaped = X_val_reshaped.copy()
    X_test_scaled_reshaped = X_test_reshaped.copy()

    # Fit scaler ONLY on training data
    if len(numeric_feature_indices) > 0:
        print(f"Scaling {len(numeric_feature_indices)} numeric features.")
        scaler.fit(X_train_reshaped[:, numeric_feature_indices])

        # Transform training, validation, and test data
        X_train_scaled_reshaped[:, numeric_feature_indices] = scaler.transform(X_train_reshaped[:, numeric_feature_indices])
        X_val_scaled_reshaped[:, numeric_feature_indices] = scaler.transform(X_val_reshaped[:, numeric_feature_indices])
        X_test_scaled_reshaped[:, numeric_feature_indices] = scaler.transform(X_test_reshaped[:, numeric_feature_indices])
    else:
        print("Warning: No numeric feature indices provided for scaling. Returning original data.")
        scaler = None # Set scaler to None if no scaling was done
        X_train_scaled = X_train
        X_val_scaled = X_val
        X_test_scaled = X_test
        return X_train_scaled, X_val_scaled, X_test_scaled, scaler


    # Reshape back to original 3D shape
    X_train_scaled = X_train_scaled_reshaped.reshape(n_samples_train, seq_length_train, n_features)
    X_val_scaled = X_val_scaled_reshaped.reshape(n_samples_val, seq_length_val, n_features)
    X_test_scaled = X_test_scaled_reshaped.reshape(n_samples_test, seq_length_test, n_features)

    print("--- Scaling Complete ---")
    print("X_train_scaled shape:", X_train_scaled.shape)
    print("X_val_scaled shape:", X_val_scaled.shape)
    print("X_test_scaled shape:", X_test_scaled.shape)


    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


# # ルールベースのカテゴリ作成⇒クラスタリングに使用する関数

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from statsmodels.tsa.stattools import acf
from scipy.fft import fft
from scipy.signal import find_peaks

def compute_slope(series):
    """単純な線形回帰で傾きを返す"""
    x = np.arange(len(series))
    # NaNが含まれている場合はNaNを返す
    if np.isnan(series).any() or len(series) < 2:
        return np.nan
    # 全て同じ値の場合、傾きは0
    if np.all(series == series[0]):
        return 0.0
    return linregress(x, series).slope

def compute_autocorr(series, lag=1):
    """自己相関を計算"""
    # Explicitly convert to numpy array of float type, handling errors
    try:
        numeric_series = np.asarray(series, dtype=np.float64)
    except ValueError:
        print("Warning: Could not convert series to float64 in compute_autocorr. Returning NaN.")
        return np.nan # Return NaN if conversion fails

    # NaNが含まれている場合はNaNを返す
    if np.isnan(numeric_series).any() or len(numeric_series) <= lag:
        return np.nan
    # check if the series is all constants or contains less than lag+2 non-nan values
    if np.nanstd(numeric_series) < 1e-9 or np.sum(~np.isnan(numeric_series)) <= lag:
        return np.nan

    # Drop NaNs before calculating autocorrelation
    cleaned_series = numeric_series[~np.isnan(numeric_series)]

    if len(cleaned_series) <= lag:
         return np.nan # Not enough data after dropping NaNs

    return acf(cleaned_series, nlags=lag, fft=True)[lag]


def compute_fft_amplitude(series):
    """FFTの振幅の合計（DC成分除く）を計算"""
    # Explicitly convert to numpy array of float type, handling errors
    try:
        numeric_series = np.asarray(series, dtype=np.float64)
    except ValueError:
        print("Warning: Could not convert series to float64 in compute_fft_amplitude. Returning NaN.")
        return np.nan # Return NaN if conversion fails

    # NaNが含まれている場合はNaNを返す
    if np.isnan(numeric_series).any() or len(numeric_series) < 2:
        return np.nan
    # 全て同じ値の場合はNaNを返す (周波数成分がないため)
    if np.all(numeric_series == numeric_series[0]):
        return np.nan

    # NaNを除外してFFTを計算
    cleaned_series = numeric_series[~np.isnan(numeric_series)]

    if len(cleaned_series) < 2:
        return np.nan # データが足りない場合はNaN


    yf = fft(cleaned_series)
    # DC成分 (0 Hz) を除く
    amplitudes = np.abs(yf[1:len(yf)//2])
    return np.sum(amplitudes)

def count_peaks(series):
    """価格系列のピーク数をカウント"""
    # NaNを除外
    cleaned_series = np.asarray(series, dtype=np.float64)
    cleaned_series = cleaned_series[~np.isnan(cleaned_series)]
    if len(cleaned_series) < 3: # ピーク検出には最低3点必要
        return np.nan
    peaks, _ = find_peaks(cleaned_series)
    return len(peaks)

def get_segment_indices(sequence_length, num_segments=3):
    """
    Get indices for the start, middle, and end segments of a sequence.
    Returns indices for representative points in each segment.
    Uses the last index of each segment.
    """
    if sequence_length == 0:
        return []
    if sequence_length < num_segments:
        # For short sequences, just take a few evenly spaced points
        return np.linspace(0, sequence_length - 1, sequence_length, dtype=int).tolist()


    segment_size = sequence_length // num_segments
    indices = []
    for i in range(num_segments):
        end_idx = min((i + 1) * segment_size - 1, sequence_length - 1)
        indices.append(end_idx)

    # Ensure the very last index is always included if not already
    if sequence_length - 1 not in indices:
        indices.append(sequence_length - 1)

    # Remove duplicates and sort
    indices = sorted(list(set(indices)))

    return indices


def classify_strategy(sequence, ma_short, ma_medium, ma_long, slope_short, slope_medium, slope_long, price_feature_index=0):
    """
    Classify a single sequence into one of 5 strategy categories
    ('Uptrend', 'Downtrend', 'Range', 'Reversal_Up', 'Reversal_Down')
    based on MA relationships, slopes, and other metrics.

    Args:
        sequence (np.ndarray): The original time series sequence (sequence_length, n_features).
        ma_short, ma_medium, ma_long (np.ndarray): Moving average arrays (sequence_length,).
        slope_short, slope_medium, slope_long (np.ndarray): Moving average slope arrays (sequence_length,).
        price_feature_index (int): The index of the price feature ('close').

    Returns:
        str: The classified strategy ('Uptrend', 'Downtrend', 'Range', 'Reversal_Up', 'Reversal_Down').
             Returns 'Range' if data is insufficient or classification fails.
    """
    seq_len = sequence.shape[0]
    # Define default fallback category
    fallback_category = 'Range' # Default fallback

    if seq_len == 0:
        return fallback_category

    # Ensure MA arrays have the correct length
    if not (len(ma_short) == len(ma_medium) == len(ma_long) == seq_len):
         # print(f"Warning: MA array lengths mismatch sequence length {seq_len}. Returning {fallback_category}.")
         return fallback_category


    # Get representative indices for segments
    num_segments_for_trend = min(3, seq_len if seq_len > 0 else 1)
    segment_indices = get_segment_indices(seq_len, num_segments=num_segments_for_trend)

    if not segment_indices:
         # print(f"Warning: Could not determine segment indices for seq_len {seq_len}. Returning {fallback_category}.")
         return fallback_category


    # Define tolerance for floating point comparisons and thresholds
    tolerance = 1e-6
    # Adjusted thresholds - these might need tuning
    strong_slope_threshold = 0.01 # Example: MA slope consistently > 0.01 for strong trend
    moderate_slope_threshold = 0.002 # Example: MA slope mostly > 0.002 for moderate trend
    range_ma_spread_ratio_threshold = 0.05 # Example: MA spread < 5% of price range for range
    reversal_price_change_threshold = 0.01 # Example: Price change > 1% for reversal

    # Check for NaNs in critical MA/slope data points
    # Only check indices that will be used
    critical_indices = sorted(list(set([0, seq_len - 1] + segment_indices)))
    if np.isnan(ma_short[critical_indices]).any() or \
       np.isnan(ma_medium[critical_indices]).any() or \
       np.isnan(ma_long[critical_indices]).any() or \
       np.isnan(slope_short[critical_indices]).any() or \
       np.isnan(slope_medium[critical_indices]).any() or \
       np.isnan(slope_long[critical_indices]).any():
         # print("Warning: NaN detected in critical MA/slope data points. Returning fallback category.")
         # Attempt to classify based on simpler price slope if MA/slope data is missing
         price_series = sequence[:, price_feature_index]
         if seq_len >= 2:
              price_slope = compute_slope(price_series)
              if not np.isnan(price_slope):
                   if price_slope > moderate_slope_threshold:
                       return 'Uptrend'
                   elif price_slope < -moderate_slope_threshold:
                       return 'Downtrend'
                   else:
                       return 'Range'
              else:
                   return fallback_category # Return fallback if price slope is also NaN
         else:
              return fallback_category # Return fallback if sequence too short for price slope


    # --- Trend Continuation Checks (Strong) ---
    # Strong Uptrend: Short > Medium > Long MA AND slopes positive throughout segments
    is_strong_uptrend = True
    for idx in segment_indices:
        if not (ma_short[idx] > ma_medium[idx] + tolerance and
                ma_medium[idx] > ma_long[idx] + tolerance and
                slope_short[idx] > strong_slope_threshold and
                slope_medium[idx] > strong_slope_threshold and
                slope_long[idx] > strong_slope_threshold):
            is_strong_uptrend = False
            break

    if is_strong_uptrend:
        return 'Uptrend'

    # Strong Downtrend: Short < Medium < Long MA AND slopes negative throughout segments
    is_strong_downtrend = True
    for idx in segment_indices:
        if not (ma_short[idx] < ma_medium[idx] - tolerance and
                ma_medium[idx] < ma_long[idx] - tolerance and
                slope_short[idx] < -strong_slope_threshold and
                slope_medium[idx] < -strong_slope_threshold and
                slope_long[idx] < -strong_slope_threshold):
            is_strong_downtrend = False
            break

    if is_strong_downtrend:
        return 'Downtrend'

    # --- Reversal Checks ---
    # Reversal Up: Downtrend-like start and Uptrend-like end based on MA order and price increase
    is_reversal_up = False
    if seq_len >= 2:
        start_idx = 0
        last_idx = seq_len - 1

        # Check for downtrend-like start (MA order Short < Medium < Long)
        start_downtrend_order = (ma_short[start_idx] < ma_medium[start_idx] - tolerance) and \
                                (ma_medium[start_idx] < ma_long[start_idx] - tolerance)

        # Check for uptrend-like end (MA order Short > Medium > Long)
        end_uptrend_order = (ma_short[last_idx] > ma_medium[last_idx] + tolerance) and \
                            (ma_medium[last_idx] > ma_long[last_idx] + tolerance)

        # Check if price significantly increased from start to end
        price_start = sequence[start_idx, price_feature_index]
        price_end = sequence[last_idx, price_feature_index]
        price_increased_significantly = False
        if not np.isnan([price_start, price_end]).any() and price_start > tolerance: # Avoid division by zero
             # Define a threshold for significant price increase (e.g., > reversal_price_change_threshold)
             if (price_end - price_start) / price_start > reversal_price_change_threshold:
                  price_increased_significantly = True

        if start_downtrend_order and end_uptrend_order and price_increased_significantly:
             is_reversal_up = True

    if is_reversal_up:
         return 'Reversal_Up'

    # Reversal Down: Uptrend-like start and Downtrend-like end based on MA order and price decrease
    is_reversal_down = False
    if seq_len >= 2 and not is_reversal_up: # Check only if not already classified as reversal up
        start_idx = 0
        last_idx = seq_len - 1

        # Check for uptrend-like start (MA order Short > Medium > Long)
        start_uptrend_order = (ma_short[start_idx] > ma_medium[start_idx] + tolerance) and \
                              (ma_medium[start_idx] > ma_long[start_idx] + tolerance)

        # Check for downtrend-like end (MA order Short < Medium < Long)
        end_downtrend_order = (ma_short[last_idx] < ma_medium[last_idx] - tolerance) and \
                              (ma_medium[last_idx] < ma_long[last_idx] - tolerance)

        # Check if price significantly decreased from start to end
        price_start = sequence[start_idx, price_feature_index]
        price_end = sequence[last_idx, price_feature_index]
        price_decreased_significantly = False
        if not np.isnan([price_start, price_end]).any() and price_start > tolerance: # Avoid division by zero
             # Define a threshold for significant price decrease (e.g., > reversal_price_change_threshold)
             if (price_start - price_end) / price_start > reversal_price_change_threshold:
                  price_decreased_significantly = True

        if start_uptrend_order and end_downtrend_order and price_decreased_significantly:
             is_reversal_down = True

    if is_reversal_down:
         return 'Reversal_Down'


    # --- Trend Continuation Checks (Moderate) ---
    # Moderate Uptrend: Short > Medium > Long MA AND slopes mostly positive towards the end
    is_moderate_uptrend = False
    if seq_len > 0:
        last_idx = seq_len - 1
        # Check MA order at the end
        if not np.isnan([ma_short[last_idx], ma_medium[last_idx], ma_long[last_idx]]).any() and \
           ma_short[last_idx] > ma_medium[last_idx] + tolerance and \
           ma_medium[last_idx] > ma_long[last_idx] + tolerance:

            # Check if slopes are positive in the last segment
            last_segment_start_idx = seq_len - (seq_len // num_segments_for_trend) if num_segments_for_trend > 0 else 0
            last_segment_indices = range(max(0, last_segment_start_idx), seq_len)

            positive_slopes_in_last_segment_count = 0
            total_points_in_last_segment = 0

            for idx in last_segment_indices:
                 if not np.isnan([slope_short[idx], slope_medium[idx], slope_long[idx]]).any():
                      total_points_in_last_segment += 1
                      if slope_short[idx] > -tolerance and slope_medium[idx] > -tolerance and slope_long[idx] > -tolerance:
                           positive_slopes_in_last_segment_count += 1

            # If at least 70% of points in the last segment have positive slopes
            if total_points_in_last_segment > 0 and positive_slopes_in_last_segment_count / total_points_in_last_segment >= 0.7: # Threshold 70%
                 is_moderate_uptrend = True

    if is_moderate_uptrend:
         return 'Uptrend'


    # Moderate Downtrend: Short < Medium < Long MA AND slopes mostly negative towards the end
    is_moderate_downtrend = False
    if seq_len > 0:
        last_idx = seq_len - 1
        # Check MA order at the end
        if not np.isnan([ma_short[last_idx], ma_medium[last_idx], ma_long[last_idx]]).any() and \
           ma_short[last_idx] < ma_medium[last_idx] - tolerance and \
           ma_medium[last_idx] < ma_long[last_idx] - tolerance:

            # Check if slopes are negative in the last segment
            last_segment_start_idx = seq_len - (seq_len // num_segments_for_trend) if num_segments_for_trend > 0 else 0
            last_segment_indices = range(max(0, last_segment_start_idx), seq_len)

            negative_slopes_in_last_segment_count = 0
            total_points_in_last_segment = 0

            for idx in last_segment_indices:
                 if not np.isnan([slope_short[idx], slope_medium[idx], slope_long[idx]]).any():
                      total_points_in_last_segment += 1
                      if slope_short[idx] < tolerance and slope_medium[idx] < tolerance and slope_long[idx] < tolerance:
                           negative_slopes_in_last_segment_count += 1

            # If at least 70% of points in the last segment have negative slopes
            if total_points_in_last_segment > 0 and negative_slopes_in_last_segment_count / total_points_in_last_segment >= 0.7: # Threshold 70%
                 is_moderate_downtrend = True

    if is_moderate_downtrend:
         return 'Downtrend'


    # --- Range-bound and Other Checks (Fallback) ---
    # If none of the above, classify as 'Range' or based on simple price trend
    # Check for Range-bound properties
    is_range_bound = True
    if seq_len > 0:
         price_min = np.nanmin(sequence[:, price_feature_index])
         price_max = np.nanmax(sequence[:, price_feature_index])
         price_range_seq = price_max - price_min if not np.isnan([price_min, price_max]).any() else np.nan

         if not np.isnan(price_range_seq) and price_range_seq > tolerance:
              for idx in range(seq_len):
                  if not np.isnan([ma_short[idx], ma_medium[idx], ma_long[idx]]).any():
                      ma_spread_at_idx = max(ma_short[idx], ma_medium[idx], ma_long[idx]) - min(ma_short[idx], ma_medium[idx], ma_long[idx])
                      relative_ma_spread_at_idx = ma_spread_at_idx / price_range_seq
                      if relative_ma_spread_at_idx > range_ma_spread_ratio_threshold:
                          is_range_bound = False
                          break
                  else:
                      is_range_bound = False
                      break

         else: # price_range_seq is NaN or zero
              # If price is constant, it's range-bound
              if not np.isnan(sequence[:, price_feature_index]).any() and np.all(sequence[:, price_feature_index] == sequence[0, price_feature_index]):
                   is_range_bound = True
              else:
                  is_range_bound = False

    if is_range_bound:
         return 'Range'


    # If still not classified, use simple overall price trend as a final fallback
    price_series = sequence[:, price_feature_index]
    if seq_len >= 2:
        price_slope = compute_slope(price_series)
        if not np.isnan(price_slope):
            if price_slope > -tolerance: # treat near-zero or positive slope as upward or range
                # Check if the ending MA order is somewhat upward-biased
                last_idx = seq_len - 1
                if not np.isnan([ma_short[last_idx], ma_medium[last_idx]]).any() and ma_short[last_idx] > ma_medium[last_idx] - tolerance:
                    return 'Uptrend'
                else:
                    return 'Range'
            else: # Negative slope
                 # Check if the ending MA order is somewhat downward-biased
                 last_idx = seq_len - 1
                 if not np.isnan([ma_short[last_idx], ma_medium[last_idx]]).any() and ma_short[last_idx] < ma_medium[last_idx] + tolerance:
                     return 'Downtrend'
                 else:
                     return 'Range'
        else:
            return fallback_category # Return fallback if price slope is NaN
    else:
        return fallback_category # Return fallback if sequence too short


# Main function to classify all sequences
def classify_all_sequences(X_3d_numpy, feature_names):
    """
    Classifies all sequences in the 3D numpy array into strategy categories.

    Args:
        X_3d_numpy (np.ndarray): 3D numpy array of shape (n_sequences, sequence_length, n_features).
        feature_names (list): List of feature names corresponding to the last dimension of X_3d_numpy.

    Returns:
        np.ndarray: 1D numpy array of strategy labels (strings) of shape (n_sequences,).
                    Returns None if feature names are missing or inconsistent.
    """
    n_sequences, seq_len, n_features = X_3d_numpy.shape
    # Initialize with a default category from the allowed 5 categories
    # Using 'Range' as a relatively neutral default.
    strategy_labels = np.full(n_sequences, 'Range', dtype=object)

    # Ensure required feature names (MA_t_6, MA_t_24, MA_t_72, close) are in the list and get their indices
    # Update MA names based on common usage if needed, or rely on provided names.
    # Assuming the names are exactly 'MA_t_6', 'MA_t_24', 'MA_t_72', 'close' as used previously.
    required_features = ['MA_t_6', 'MA_t_24', 'MA_t_72', 'close']
    feature_indices = {}
    try:
        # Attempt to find MA feature names, use common alternatives if direct match fails
        ma_short_candidates = ['MA_t_6', 'MA_6', 'SMA_6']
        ma_mid_candidates = ['MA_t_24', 'MA_24', 'SMA_24']
        ma_long_candidates = ['MA_t_72', 'MA_72', 'SMA_72']
        price_candidates = ['close', 'Close', 'PRICE']

        found_ma_short = False
        for name in ma_short_candidates:
            if name in feature_names:
                feature_indices['MA_short'] = feature_names.index(name)
                found_ma_short = True
                break

        found_ma_mid = False
        for name in ma_mid_candidates:
            if name in feature_names:
                feature_indices['MA_medium'] = feature_names.index(name)
                found_ma_mid = True
                break

        found_ma_long = False
        for name in ma_long_candidates:
            if name in feature_names:
                feature_indices['MA_long'] = feature_names.index(name)
                found_ma_long = True
                break

        found_price = False
        for name in price_candidates:
            if name in feature_names:
                feature_indices['price'] = feature_names.index(name)
                found_price = True
                break


        if not (found_ma_short and found_ma_mid and found_ma_long and found_price):
             missing = []
             if not found_ma_short: missing.append(f"Short MA ({ma_short_candidates})")
             if not found_ma_mid: missing.append(f"Medium MA ({ma_mid_candidates})")
             if not found_ma_long: missing.append(f"Long MA ({ma_long_candidates})")
             if not found_price: missing.append(f"Price ({price_candidates})")
             print(f"Error: Required features for classification not found in feature_names: {', '.join(missing)}")
             print(f"Available features: {feature_names}")
             # Return an array of the default fallback category if critical features are missing
             return np.full(n_sequences, 'Range', dtype=object) # Return default category array


    except ValueError as e:
        # This block should ideally not be reached if the checks above are thorough
        print(f"Unexpected ValueError during feature index lookup: {e}")
        print(f"Available features: {feature_names}")
        return np.full(n_sequences, 'Range', dtype=object) # Return default category array


    # Use the determined indices for MA features and price feature
    ma_short_index = feature_indices.get('MA_short')
    ma_medium_index = feature_indices.get('MA_medium')
    ma_long_index = feature_indices.get('MA_long')
    price_feature_index = feature_indices.get('price')

    # Re-check if indices were found (should be true based on the try-except block)
    if ma_short_index is None or ma_medium_index is None or ma_long_index is None or price_feature_index is None:
         print("Error: Critical feature indices are None after lookup. This indicates an issue with the lookup logic.")
         return np.full(n_sequences, 'Range', dtype=object) # Return default category array


    for i in range(n_sequences):
        sequence = X_3d_numpy[i, :, :] # Shape: (seq_len, n_features)

        # Ensure indices are within the actual number of features in the sequence data
        if ma_short_index >= n_features or ma_medium_index >= n_features or ma_long_index >= n_features or price_feature_index >= n_features:
            print(f"Warning: Feature index out of bounds for sequence {i}. Index {max(ma_short_index, ma_medium_index, ma_long_index, price_feature_index)} is >= n_features {n_features}. Skipping classification for this sequence.")
            # Keep the default 'Range' label for this sequence
            continue


        # Extract MA sequences using the correct indices and handle potential non-numeric data
        try:
            ma_short_seq = sequence[:, ma_short_index].astype(np.float64)
            ma_medium_seq = sequence[:, ma_medium_index].astype(np.float64)
            ma_long_seq = sequence[:, ma_long_index].astype(np.float64)
            price_seq_for_slope = sequence[:, price_feature_index].astype(np.float64) # Use for price slope calculation


        except ValueError as e:
             print(f"Warning: Failed to convert sequence data to float64 for sequence {i} at indices {ma_short_index, ma_medium_index, ma_long_index, price_feature_index}: {e}. Skipping classification for this sequence.")
             # Keep the default 'Range' label for this sequence
             continue


        # Calculate slopes for these extracted MA sequences
        # compute_slope function handles NaNs internally
        slope_short = np.diff(ma_short_seq, prepend=ma_short_seq[0] if ma_short_seq.shape[0] > 0 and not np.isnan(ma_short_seq[0]) else np.nan)
        slope_medium = np.diff(ma_medium_seq, prepend=ma_medium_seq[0] if ma_medium_seq.shape[0] > 0 and not np.isnan(ma_medium_seq[0]) else np.nan)
        slope_long = np.diff(ma_long_seq, prepend=ma_long_seq[0] if ma_long_seq.shape[0] > 0 and not np.isnan(ma_long_seq[0]) else np.nan)

        # Ensure slopes arrays have the same length as sequence
        # np.diff with prepend should do this, but double-check
        if not (len(slope_short) == len(slope_medium) == len(slope_long) == seq_len):
             print(f"Warning: Slope array lengths mismatch seq_len {seq_len} for sequence {i}. Skipping classification.")
             # Keep the default 'Range' label for this sequence
             continue


        # Classify the current sequence
        strategy_labels[i] = classify_strategy(
            sequence, # Pass the original sequence for other features like price
            ma_short_seq, ma_medium_seq, ma_long_seq,
            slope_short, slope_medium, slope_long,
            price_feature_index=price_feature_index # Pass the price feature index
        )

    return strategy_labels


# In[ ]:


import numpy as np
import pandas as pd
from scipy.stats import linregress

# Helper function to compute slope (already defined)
def compute_slope(series):
    """単純な線形回帰で傾きを返す"""
    x = np.arange(len(series))
    # NaNが含まれている場合はNaNを返す
    if np.isnan(series).any() or len(series) < 2:
        return np.nan
    # 全て同じ値の場合、傾きは0
    if np.all(series == series[0]):
        return 0.0
    return linregress(x, series).slope

# Helper function to calculate EMA (Exponential Moving Average)
def calculate_ema(series, window):
    """計算可能な EMA を返す"""
    # NaNが含まれている場合は計算しないか、fillna(method='ffill') などで補間を検討
    # ここでは pandas の ewm を使用。min_periods=0 で最初から計算
    return pd.Series(series).ewm(span=window, adjust=False, min_periods=0).mean().values

# Helper function to calculate MA slope using difference
def calculate_ma_slope_diff(series, delta=3):
     """MA系列の傾きを差分で計算"""
     # Ensure delta is valid
     if delta <= 0 or delta >= len(series):
          # print(f"Warning(calculate_ma_slope_diff): Invalid delta {delta} for series length {len(series)}. Returning NaN.")
          return np.full_like(series, np.nan) # Return array of NaNs

     # Handle potential NaNs in the series before differencing
     # Option 1: Interpolate (e.g., ffill/bfill) before diff
     # Option 2: Use pandas diff which handles NaNs (result will be NaN if current or previous is NaN)
     # Let's use pandas diff for simplicity, results in NaN if diff cannot be calculated
     ma_slope = pd.Series(series).diff(periods=delta).values

     return ma_slope

# Helper function to calculate ATR (Average True Range)
def calculate_atr(df, window=14):
    """Calculates Average True Range (ATR).
    Requires 'high', 'low', 'close' columns in the input DataFrame.
    Returns a Series of ATR values aligned with the DataFrame index.
    """
    if not all(col in df.columns for col in ['high', 'low', 'close']):
        print("Warning: Input DataFrame for ATR calculation must contain 'high', 'low', and 'close' columns. Returning NaN series.")
        return pd.Series(np.nan, index=df.index)

    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())

    # True Range (TR) is the maximum of the three
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # ATR is the rolling mean of TR
    atr = tr.rolling(window=window).mean()

    return atr

# Helper function to calculate Rolling Standard Deviation
def calculate_rolling_std(series, window):
    """Calculates Rolling Standard Deviation."""
    return pd.Series(series).rolling(window=window).std().values


# New function to check future MA conditions for a single sequence
def check_future_ma_conditions(
    combined_sequence_data: np.ndarray, # Combined past + future data (seq_length + horizon + eval_window_L, n_features)
    seq_length: int,
    horizon: int,
    ma_indices: tuple, # (short, medium, long) MA feature indices
    eval_window_L: int, # Evaluation window length (L)
    k_of_l: int = 2,        # K parameter for K-of-L rule - MODIFIED from 3
    buffer_epsilon: float = 0.001, # Buffer epsilon (ε) for MA order (absolute)
    slope_threshold: float = 0.0005, # Slope threshold (ε_s) (absolute)
    delta_slope: int = 3, # Delta for MA slope calculation
    volatility_measure: str = 'atr', # 'atr' or 'rolling_std' for normalization
    volatility_window: int = 14, # Window size for volatility measure
    price_feature_index: int = 0, # Index of the price feature ('close')
    high_feature_index: int = None, # Index of the 'high' feature for ATR
    low_feature_index: int = None, # Index of the 'low' feature for ATR
    percentile_threshold: float = None, # Percentile threshold for dynamic slope/difference (e.g., 70 for 70th percentile)
    past_window_for_percentile: int = None # Past window length for percentile calculation
) -> float:
    """
    Checks if future MA conditions (order and slopes), normalized by volatility,
    are met for K-of-L points within the evaluation window [t+H, t+H+L).
    Incorporates optional dynamic thresholds based on percentiles.

    Args:
        combined_sequence_data (np.ndarray): Combined past (seq_length) and future (horizon + eval_window_L)
                                             data (seq_length + horizon + eval_window_L, n_features).
        seq_length (int): Length of the past sequence.
        horizon (int): Prediction horizon (H).
        ma_indices (tuple): Tuple of (short_ma_index, medium_ma_index, long_ma_index).
        eval_window_L (int): Evaluation window length (L).
        k_of_l (int): K parameter for K-of-L rule.
        buffer_epsilon (float): Buffer epsilon (ε) for MA order (absolute).
        slope_threshold (float): Slope threshold (ε_s) (absolute).
        delta_slope (int): Delta for MA slope calculation.
        volatility_measure (str): 'atr' or 'rolling_std' for normalization.
        volatility_window (int): Window size for volatility measure.
        price_feature_index (int): Index of the price feature ('close').
        high_feature_index (int): Index of the 'high' feature for ATR. Required if volatility_measure is 'atr'.
        low_feature_index (int): Index of the 'low' feature for ATR. Required if volatility_measure is 'atr'.
        percentile_threshold (float, optional): If not None, use this percentile of recent volatility-normalized differences/slopes
                                                as a dynamic threshold instead of the fixed slope_threshold.
                                                Value between 0 and 100.
        past_window_for_percentile (int, optional): Past window length to calculate percentiles from.
                                                    If None, use the entire past sequence (seq_length).


    Returns:
        float: 1.0 if conditions are met for K-of-L points, 0.0 otherwise, np.nan if data insufficient or volatility is zero.
    """
    # Ensure enough data for the horizon and evaluation window
    required_total_length = seq_length + horizon + eval_window_L
    if combined_sequence_data is None or combined_sequence_data.ndim != 2 or \
       combined_sequence_data.shape[0] < required_total_length or \
       combined_sequence_data.shape[1] <= max(ma_indices): # Ensure MA indices are valid
        # print(f"Debug(check_future_ma_conditions): Insufficient data shape {combined_sequence_data.shape if combined_sequence_data is not None else 'None'} for required length ({required_total_length}) and MA indices {ma_indices}. Returning NaN.")
        return np.nan # Not enough data

    # Extract the future evaluation window data [t+H, t+H+L)
    future_eval_window_data = combined_sequence_data[seq_length + horizon : required_total_length, :] # Shape (L, n_features)

    # Ensure the evaluation window has enough data points (L points)
    if future_eval_window_data.shape[0] != eval_window_L:
         # This should be caught by the initial length check, but as a safeguard
         # print(f"Debug(check_future_ma_conditions): Extracted future eval window has incorrect length {future_eval_window_data.shape[0]}. Expected {eval_window_L}. Returning NaN.")
         return np.nan

    # Extract MA series for the future evaluation window
    short_ma_series = future_eval_window_data[:, ma_indices[0]].astype(np.float64)
    medium_ma_series = future_eval_window_data[:, ma_indices[1]].astype(np.float64)
    long_ma_series = future_eval_window_data[:, ma_indices[2]].astype(np.float64)

    # Calculate MA slopes within the future evaluation window
    effective_delta_slope = min(delta_slope, eval_window_L - 1) if eval_window_L > 1 else 0
    if effective_delta_slope == 0 and eval_window_L > 1:
         effective_delta_slope = 1


    short_ma_slopes = calculate_ma_slope_diff(short_ma_series, delta=effective_delta_slope)
    medium_ma_slopes = calculate_ma_slope_diff(medium_ma_series, delta=effective_delta_slope)
    long_ma_slopes = calculate_ma_slope_diff(long_ma_series, delta=effective_delta_slope)

    # --- Calculate Volatility for Normalization ---
    # Need data spanning the sequence + horizon + eval_window_L to calculate volatility up to the evaluation window
    volatility_data_span = combined_sequence_data[:required_total_length, :] # Data from start of seq to end of eval window

    if volatility_measure.lower() == 'atr': # Check for 'atr' case-insensitively
        if high_feature_index is None or low_feature_index is None:
             print("Error: high_feature_index and low_feature_index must be provided for ATR calculation.")
             return np.nan
        if high_feature_index >= volatility_data_span.shape[1] or low_feature_index >= volatility_data_span.shape[1] or price_feature_index >= volatility_data_span.shape[1]:
             print("Error: Feature index out of bounds for ATR calculation.")
             return np.nan

        # Create a temporary DataFrame for ATR calculation
        temp_df_for_atr = pd.DataFrame({
            'high': volatility_data_span[:, high_feature_index].astype(np.float64),
            'low': volatility_data_span[:, low_feature_index].astype(np.float64),
            'close': volatility_data_span[:, price_feature_index].astype(np.float64)
        })
        # Calculate ATR over the relevant window
        atr_series = calculate_atr(temp_df_for_atr, window=volatility_window)
        # Get the ATR value at the start of the evaluation window (index seq_length + horizon relative to start)
        # This index corresponds to index 0 of the future_eval_window_data
        volatility_value_at_eval_start = atr_series.iloc[seq_length + horizon]

    elif volatility_measure == 'rolling_std':
         # Calculate rolling std of the price series over the relevant window
         price_series_for_std = volatility_data_span[:, price_feature_index].astype(np.float64)
         rolling_std_series = calculate_rolling_std(price_series_for_std, window=volatility_window)
         volatility_value_at_eval_start = rolling_std_series[seq_length + horizon]

    else:
        print(f"Error: Unknown volatility_measure '{volatility_measure}'. Supported: 'atr', 'rolling_std'.")
        return np.nan

    # Handle zero or NaN volatility
    if np.isnan(volatility_value_at_eval_start) or volatility_value_at_eval_start < 1e-8: # Add a small epsilon
        # print(f"Warning: Volatility value at evaluation window start is zero or NaN ({volatility_value_at_eval_start}). Cannot normalize. Returning NaN label.")
        return np.nan # Cannot normalize if volatility is zero or NaN


    # --- Calculate Dynamic Thresholds based on Percentiles (Optional) ---
    dynamic_slope_threshold = slope_threshold # Default to fixed threshold
    if percentile_threshold is not None and past_window_for_percentile is not None:
        if seq_length < past_window_for_percentile:
             print(f"Warning: Past window for percentile ({past_window_for_percentile}) is larger than sequence length ({seq_length}). Skipping dynamic threshold calculation.")
             # Use fixed threshold
        else:
            # Extract data from the past window (last past_window_for_percentile points of the sequence)
            past_window_data = combined_sequence_data[seq_length - past_window_for_percentile : seq_length, :] # Shape (past_window, n_features)

            past_short_ma_series = past_window_data[:, ma_indices[0]].astype(np.float64)
            past_medium_ma_series = past_window_data[:, ma_indices[1]].astype(np.float64)
            past_long_ma_series = past_window_data[:, ma_indices[2]].astype(np.float64)

            effective_delta_slope_past = min(delta_slope, past_window_for_percentile - 1) if past_window_for_percentile > 1 else 0
            if effective_delta_slope_past == 0 and past_window_for_percentile > 1:
                 effective_delta_slope_past = 1


            past_short_ma_slopes = calculate_ma_slope_diff(past_short_ma_series, delta=effective_delta_slope_past)
            past_medium_ma_slopes = calculate_ma_slope_diff(past_medium_ma_series, delta=effective_delta_slope_past)
            past_long_ma_slopes = calculate_ma_slope_diff(past_long_ma_series, delta=effective_delta_slope_past)

            volatility_data_span_past = combined_sequence_data[:seq_length, :] # Data from start of seq to end of past window

            if volatility_measure.lower() == 'atr': # Check for 'atr' case-insensitively
                 if high_feature_index is None or low_feature_index is None:
                      print("Error: high/low indices needed for past ATR calculation.")
                      # Use fixed threshold
                 else:
                      temp_df_for_atr_past = pd.DataFrame({
                          'high': volatility_data_span_past[:, high_feature_index].astype(np.float64),
                          'low': volatility_data_span_past[:, low_feature_index].astype(np.float64),
                          'close': volatility_data_span_past[:, price_feature_index].astype(np.float64)
                      })
                      atr_series_past = calculate_atr(temp_df_for_atr_past, window=volatility_window)
                      volatility_value_at_past_end = atr_series_past.iloc[seq_length - 1] # Volatility at the end of the past window

            elif volatility_measure == 'rolling_std':
                 price_series_for_std_past = volatility_data_span_past[:, price_feature_index].astype(np.float64)
                 rolling_std_series_past = calculate_rolling_std(price_series_for_std_past, window=volatility_window)
                 volatility_value_at_past_end = rolling_std_series_past[seq_length - 1]

            else:
                 volatility_value_at_past_end = np.nan # Should be caught earlier


            normalized_past_slopes = []
            start_check_idx_past = effective_delta_slope_past if past_window_for_percentile > effective_delta_slope_past else (past_window_for_percentile -1 if past_window_for_percentile > 0 else 0)

            if not np.isnan(volatility_value_at_past_end) and volatility_value_at_past_end > 1e-8:
                 for u_past in range(start_check_idx_past, past_window_for_percentile):
                      if u_past < len(past_short_ma_slopes) and u_past < len(past_medium_ma_slopes) and u_past < len(past_long_ma_slopes):
                           normalized_past_slopes.append(np.abs(past_short_ma_slopes[u_past] / volatility_value_at_past_end))
                           normalized_past_slopes.append(np.abs(past_medium_ma_slopes[u_past] / volatility_value_at_past_end))
                           normalized_past_slopes.append(np.abs(past_long_ma_slopes[u_past] / volatility_value_at_past_end))

                 if normalized_past_slopes:
                      dynamic_slope_threshold = np.percentile(normalized_past_slopes, percentile_threshold)
                      # print(f"Debug: Calculated dynamic slope threshold ({percentile_threshold}th percentile) = {dynamic_slope_threshold}")
                 else:
                      # print("Warning: No valid normalized past slopes for percentile calculation. Using fixed threshold.")
                      dynamic_slope_threshold = slope_threshold # Fallback
            else:
                 # print(f"Warning: Volatility value at past window end is zero or NaN ({volatility_value_at_past_end}). Cannot normalize past slopes for percentile. Using fixed threshold.")
                 dynamic_slope_threshold = slope_threshold # Fallback


    # --- Check conditions for K-of-L points within the evaluation window [0, L-1] ---
    condition_met_count = 0
    # Start checking from index effective_delta_slope to L-1, as slopes need previous points
    start_check_idx = effective_delta_slope if eval_window_L > effective_delta_slope else (eval_window_L -1 if eval_window_L > 0 else 0)
    if eval_window_L == 1: start_check_idx = 0


    for u in range(start_check_idx, eval_window_L):
         # Check MA order condition at point u for Uptrend: Short > Medium > Long with buffer
         # Normalize buffer epsilon by volatility
         normalized_buffer_epsilon = buffer_epsilon / volatility_value_at_eval_start if volatility_value_at_eval_start > 1e-8 else buffer_epsilon # Fallback if volatility is zero


         ma_order_uptrend = (not np.isnan([short_ma_series[u], medium_ma_series[u], long_ma_series[u]]).any()) and \
                             (short_ma_series[u] > medium_ma_series[u] + normalized_buffer_epsilon) and \
                             (medium_ma_series[u] > long_ma_series[u] + normalized_buffer_epsilon)

         # Check slope condition at point u for Uptrend: slopes positive and above threshold
         # Normalize slopes by volatility
         if u < len(short_ma_slopes) and u < len(medium_ma_slopes) and u < len(long_ma_slopes):
              # Ensure slope values are not NaN before normalization
              if not np.isnan([short_ma_slopes[u], medium_ma_slopes[u], long_ma_slopes[u]]).any():
                   normalized_short_ma_slope = short_ma_slopes[u] / volatility_value_at_eval_start if volatility_value_at_eval_start > 1e-8 else np.nan
                   normalized_medium_ma_slope = medium_ma_slopes[u] / volatility_value_at_eval_start if volatility_value_at_eval_start > 1e-8 else np.nan
                   normalized_long_ma_slope = long_ma_slopes[u] / volatility_value_at_eval_start if volatility_value_at_eval_start > 1e-8 else np.nan

                   # Check if normalized slopes are positive and above the dynamic/fixed threshold
                   slopes_positive_and_above_threshold = (not np.isnan([normalized_short_ma_slope, normalized_medium_ma_slope, normalized_long_ma_slope]).any()) and \
                                                         (normalized_short_ma_slope > dynamic_slope_threshold) and \
                                                         (normalized_medium_ma_slope > dynamic_slope_threshold) and \
                                                         (normalized_long_ma_slope > dynamic_slope_threshold)
              else:
                  slopes_positive_and_above_threshold = False # Slopes are NaN
         else:
              # print(f"Debug(check_future_ma_conditions): Slope index {u} out of bounds for slope arrays. Skipping point.")
              slopes_positive_and_above_threshold = False


         # If both conditions (MA order and positive normalized slopes above threshold) are met at this point
         if ma_order_uptrend and slopes_positive_and_above_threshold:
              condition_met_count += 1


    # Determine label based on K-of-L rule
    # Checkable points are from start_check_idx to eval_window_L - 1
    num_checkable_points = eval_window_L - start_check_idx

    if num_checkable_points <= 0:
         # print(f"Debug(check_future_ma_conditions): No checkable points ({num_checkable_points}) in future eval window. Returning NaN.")
         return np.nan # Cannot check K-of-L

    # Ensure k_of_l is not greater than the number of checkable points
    effective_k_of_l = min(k_of_l, num_checkable_points)

    label = 1.0 if condition_met_count >= effective_k_of_l else 0.0

    # print(f"Debug(check_future_ma_conditions): Checkable points: {num_checkable_points}, Condition met count: {condition_met_count}, K-of-L: {effective_k_of_l}. Label: {label}")

    return label

# New function to check future MA conditions for Downtrend for a single sequence
def check_future_ma_conditions_downtrend(
    combined_sequence_data: np.ndarray, # Combined past + future data (seq_length + horizon + eval_window_L, n_features)
    seq_length: int,
    horizon: int,
    ma_indices: tuple, # (short, medium, long) MA feature indices
    eval_window_L: int, # Evaluation window length (L)
    k_of_l: int = 2,        # K parameter for K-of-L rule - MODIFIED from 3
    buffer_epsilon: float = 0.001, # Buffer epsilon (ε) for MA order (absolute)
    slope_threshold: float = 0.0005, # Slope threshold (ε_s) (absolute)
    delta_slope: int = 3, # Delta for MA slope calculation
    volatility_measure: str = 'atr', # 'atr' or 'rolling_std' for normalization
    volatility_window: int = 14, # Window size for volatility measure
    price_feature_index: int = 0, # Index of the price feature ('close')
    high_feature_index: int = None, # Index of the 'high' feature for ATR
    low_feature_index: int = None, # Index of the 'low' feature for ATR
    percentile_threshold: float = None, # Percentile threshold for dynamic slope/difference
    past_window_for_percentile: int = None # Past window length for percentile calculation
) -> float:
    """
    Checks if future MA conditions (order and slopes), normalized by volatility,
    are met for K-of-L points within the evaluation window [t+H, t+H+L) for a Downtrend.
    Incorporates optional dynamic thresholds based on percentiles.

    Args:
        combined_sequence_data (np.ndarray): Combined past (seq_length) and future (horizon + eval_window_L)
                                             data (seq_length + horizon + eval_window_L, n_features).
        seq_length (int): Length of the past sequence.
        horizon (int): Prediction horizon (H).
        ma_indices (tuple): Tuple of (short_ma_index, medium_ma_index, long_ma_index).
        eval_window_L (int): Evaluation window length (L).
        k_of_l (int): K parameter for K-of-L rule.
        buffer_epsilon (float): Buffer epsilon (ε) for MA order (absolute).
        slope_threshold (float): Slope threshold (ε_s) (absolute).
        delta_slope (int): Delta for MA slope calculation.
        volatility_measure (str): 'atr' or 'rolling_std' for normalization.
        volatility_window (int): Window size for volatility measure.
        price_feature_index (int): Index of the price feature ('close').
        high_feature_index (int): Index of the 'high' feature for ATR. Required if volatility_measure is 'atr'.
        low_feature_index (int): Index of the 'low' feature for ATR. Required if volatility_measure is 'atr'.
        percentile_threshold (float, optional): If not None, use this percentile of recent volatility-normalized differences/slopes
                                                as a dynamic threshold instead of the fixed slope_threshold.
                                                Value between 0 and 100.
        past_window_for_percentile (int, optional): Past window length to calculate percentiles from.
                                                    If None, use the entire past sequence (seq_length).


    Returns:
        float: 1.0 if conditions are met for K-of-L points, 0.0 otherwise, np.nan if data insufficient or volatility is zero.
    """
    # Ensure enough data for the horizon and evaluation window
    required_total_length = seq_length + horizon + eval_window_L
    if combined_sequence_data is None or combined_sequence_data.ndim != 2 or \
       combined_sequence_data.shape[0] < required_total_length or \
       combined_sequence_data.shape[1] <= max(ma_indices): # Ensure MA indices are valid
        # print(f"Debug(check_future_ma_conditions_downtrend): Insufficient data shape {combined_sequence_data.shape if combined_sequence_data is not None else 'None'} for required length ({required_total_length}) and MA indices {ma_indices}. Returning NaN.")
        return np.nan # Not enough data

    # Extract the future evaluation window data [t+H, t+H+L)
    future_eval_window_data = combined_sequence_data[seq_length + horizon : required_total_length, :] # Shape (L, n_features)

    # Ensure the evaluation window has enough data points (L points)
    if future_eval_window_data.shape[0] != eval_window_L:
         # This should be caught by the initial length check, but as a safeguard
         # print(f"Debug(check_future_ma_conditions_downtrend): Extracted future eval window has incorrect length {future_eval_window_data.shape[0]}. Expected {eval_window_L}. Returning NaN.")
         return np.nan


    # Extract MA series for the future evaluation window
    short_ma_series = future_eval_window_data[:, ma_indices[0]].astype(np.float64)
    medium_ma_series = future_eval_window_data[:, ma_indices[1]].astype(np.float64)
    long_ma_series = future_eval_window_data[:, ma_indices[2]].astype(np.float64)

    # Calculate MA slopes within the future evaluation window
    effective_delta_slope = min(delta_slope, eval_window_L - 1) if eval_window_L > 1 else 0
    if effective_delta_slope == 0 and eval_window_L > 1:
         effective_delta_slope = 1


    short_ma_slopes = calculate_ma_slope_diff(short_ma_series, delta=effective_delta_slope)
    medium_ma_slopes = calculate_ma_slope_diff(medium_ma_series, delta=effective_delta_slope)
    long_ma_slopes = calculate_ma_slope_diff(long_ma_series, delta=effective_delta_slope)

    # --- Calculate Volatility for Normalization ---
    volatility_data_span = combined_sequence_data[:required_total_length, :] # Data from start of seq to end of eval window

    if volatility_measure.lower() == 'atr': # Check for 'atr' case-insensitively
        if high_feature_index is None or low_feature_index is None:
             print("Error: high_feature_index and low_feature_index must be provided for ATR calculation.")
             return np.nan
        if high_feature_index >= volatility_data_span.shape[1] or low_feature_index >= volatility_data_span.shape[1] or price_feature_index >= volatility_data_span.shape[1]:
             print("Error: Feature index out of bounds for ATR calculation.")
             return np.nan

        temp_df_for_atr = pd.DataFrame({
            'high': volatility_data_span[:, high_feature_index].astype(np.float64),
            'low': volatility_data_span[:, low_feature_index].astype(np.float64),
            'close': volatility_data_span[:, price_feature_index].astype(np.float64)
        })
        atr_series = calculate_atr(temp_df_for_atr, window=volatility_window)
        volatility_value_at_eval_start = atr_series.iloc[seq_length + horizon]

    elif volatility_measure == 'rolling_std':
         price_series_for_std = volatility_data_span[:, price_feature_index].astype(np.float64)
         rolling_std_series = calculate_rolling_std(price_series_for_std, window=volatility_window)
         volatility_value_at_eval_start = rolling_std_series[seq_length + horizon]

    else:
        print(f"Error: Unknown volatility_measure '{volatility_measure}'. Supported: 'atr', 'rolling_std'.")
        return np.nan

    # Handle zero or NaN volatility
    if np.isnan(volatility_value_at_eval_start) or volatility_value_at_eval_start < 1e-8: # Add a small epsilon
        # print(f"Warning: Volatility value at evaluation window start is zero or NaN ({volatility_value_at_eval_start}). Cannot normalize. Returning NaN label.")
        return np.nan # Cannot normalize if volatility is zero or NaN


    # --- Calculate Dynamic Thresholds based on Percentiles (Optional) ---
    dynamic_slope_threshold = slope_threshold # Default to fixed threshold
    if percentile_threshold is not None and past_window_for_percentile is not None:
        if seq_length < past_window_for_percentile:
             print(f"Warning: Past window for percentile ({past_window_for_percentile}) is larger than sequence length ({seq_length}). Skipping dynamic threshold calculation.")
             # Use fixed threshold
        else:
            past_window_data = combined_sequence_data[seq_length - past_window_for_percentile : seq_length, :] # Shape (past_window, n_features)

            past_short_ma_series = past_window_data[:, ma_indices[0]].astype(np.float64)
            past_medium_ma_series = past_window_data[:, ma_indices[1]].astype(np.float64)
            past_long_ma_series = past_window_data[:, ma_indices[2]].astype(np.float64)

            effective_delta_slope_past = min(delta_slope, past_window_for_percentile - 1) if past_window_for_percentile > 1 else 0
            if effective_delta_slope_past == 0 and past_window_for_percentile > 1:
                 effective_delta_slope_past = 1


            past_short_ma_slopes = calculate_ma_slope_diff(past_short_ma_series, delta=effective_delta_slope_past)
            past_medium_ma_slopes = calculate_ma_slope_diff(past_medium_ma_series, delta=effective_delta_slope_past)
            past_long_ma_slopes = calculate_ma_slope_diff(past_long_ma_series, delta=effective_delta_slope_past)

            volatility_data_span_past = combined_sequence_data[:seq_length, :] # Data from start of seq to end of past window

            if volatility_measure.lower() == 'atr': # Check for 'atr' case-insensitively
                 if high_feature_index is None or low_feature_index is None:
                      print("Error: high/low indices needed for past ATR calculation.")
                      # Use fixed threshold
                 else:
                      temp_df_for_atr_past = pd.DataFrame({
                          'high': volatility_data_span_past[:, high_feature_index].astype(np.float64),
                          'low': volatility_data_span_past[:, low_feature_index].astype(np.float64),
                          'close': volatility_data_span_past[:, price_feature_index].astype(np.float64)
                      })
                      atr_series_past = calculate_atr(temp_df_for_atr_past, window=volatility_window)
                      volatility_value_at_past_end = atr_series_past.iloc[seq_length - 1] # Volatility at the end of the past window

            elif volatility_measure == 'rolling_std':
                 price_series_for_std_past = volatility_data_span_past[:, price_feature_index].astype(np.float64)
                 rolling_std_series_past = calculate_rolling_std(price_series_for_std_past, window=volatility_window)
                 volatility_value_at_past_end = rolling_std_series_past[seq_length - 1]

            else:
                 volatility_value_at_past_end = np.nan # Should be caught earlier


            normalized_past_slopes = []
            start_check_idx_past = effective_delta_slope_past if past_window_for_percentile > effective_delta_slope_past else (past_window_for_percentile -1 if past_window_for_percentile > 0 else 0)

            if not np.isnan(volatility_value_at_past_end) and volatility_value_at_past_end > 1e-8:
                 for u_past in range(start_check_idx_past, past_window_for_percentile):
                      if u_past < len(past_short_ma_slopes) and u_past < len(past_medium_ma_slopes) and u_past < len(past_long_ma_slopes):
                           normalized_past_slopes.append(np.abs(past_short_ma_slopes[u_past] / volatility_value_at_past_end))
                           normalized_past_slopes.append(np.abs(past_medium_ma_slopes[u_past] / volatility_value_at_past_end))
                           normalized_past_slopes.append(np.abs(past_long_ma_slopes[u_past] / volatility_value_at_past_end))

                 if normalized_past_slopes:
                      dynamic_slope_threshold = np.percentile(normalized_past_slopes, percentile_threshold)
                      # print(f"Debug: Calculated dynamic slope threshold ({percentile_threshold}th percentile) = {dynamic_slope_threshold}")
                 else:
                      # print("Warning: No valid normalized past slopes for percentile calculation. Using fixed threshold.")
                      dynamic_slope_threshold = slope_threshold # Fallback
            else:
                 # print(f"Warning: Volatility value at past window end is zero or NaN ({volatility_value_at_past_end}). Cannot normalize past slopes for percentile. Using fixed threshold.")
                 dynamic_slope_threshold = slope_threshold # Fallback


    # --- Check conditions for K-of-L points within the evaluation window [0, L-1] for Downtrend ---
    condition_met_count = 0
    start_check_idx = effective_delta_slope if eval_window_L > effective_delta_slope else (eval_window_L -1 if eval_window_L > 0 else 0)
    if eval_window_L == 1: start_check_idx = 0


    for u in range(start_check_idx, eval_window_L):
         # Check MA order condition at point u for Downtrend: Short < Medium < Long with buffer
         normalized_buffer_epsilon = buffer_epsilon / volatility_value_at_eval_start if volatility_value_at_eval_start > 1e-8 else buffer_epsilon # Fallback if volatility is zero

         ma_order_downtrend = (not np.isnan([short_ma_series[u], medium_ma_series[u], long_ma_series[u]]).any()) and \
                              (short_ma_series[u] < medium_ma_series[u] - normalized_buffer_epsilon) and \
                              (medium_ma_series[u] < long_ma_series[u] - normalized_buffer_epsilon)

         # Check slope condition at point u for Downtrend: slopes negative and below -threshold
         # Normalize slopes by volatility
         if u < len(short_ma_slopes) and u < len(medium_ma_slopes) and u < len(long_ma_slopes):
              if not np.isnan([short_ma_slopes[u], medium_ma_slopes[u], long_ma_slopes[u]]).any():
                   normalized_short_ma_slope = short_ma_slopes[u] / volatility_value_at_eval_start if volatility_value_at_eval_start > 1e-8 else np.nan
                   normalized_medium_ma_slope = medium_ma_slopes[u] / volatility_value_at_eval_start if volatility_value_at_eval_start > 1e-8 else np.nan
                   normalized_long_ma_slope = long_ma_slopes[u] / volatility_value_at_eval_start if volatility_value_at_eval_start > 1e-8 else np.nan

                   # Check if normalized slopes are negative and below -dynamic/fixed threshold
                   slopes_negative_and_below_threshold = (not np.isnan([normalized_short_ma_slope, normalized_medium_ma_slope, normalized_long_ma_slope]).any()) and \
                                                         (normalized_short_ma_slope < -dynamic_slope_threshold) and \
                                                         (normalized_medium_ma_slope < -dynamic_slope_threshold) and \
                                                         (normalized_long_ma_slope < -dynamic_slope_threshold)
              else:
                   slopes_negative_and_below_threshold = False # Slopes are NaN
         else:
              # print(f"Debug(check_future_ma_conditions_downtrend): Slope index {u} out of bounds for slope arrays. Skipping point.")
              slopes_negative_and_below_threshold = False


         # If both conditions (MA order and negative normalized slopes below threshold) are met at this point
         if ma_order_downtrend and slopes_negative_and_below_threshold:
              condition_met_count += 1


    # Determine label based on K-of-L rule
    num_checkable_points = eval_window_L - start_check_idx

    if num_checkable_points <= 0:
         # print(f"Debug(check_future_ma_conditions_downtrend): No checkable points ({num_checkable_points}) in future eval window. Returning NaN.")
         return np.nan

    effective_k_of_l = min(k_of_l, num_checkable_points)


    label = 1.0 if condition_met_count >= effective_k_of_l else 0.0

    # print(f"Debug(check_future_ma_conditions_downtrend): Checkable points: {num_checkable_points}, Condition met count: {condition_met_count}, K-of-L: {effective_k_of_l}. Label: {label}")

    return label


def generate_binary_labels_from_strategy(X_3d_numpy, strategy_labels, feature_names, sequence_length,
                                         horizon=6, # Prediction Horizon (H)
                                         eval_window_L=4, # Evaluation Window Length (L)
                                         k_of_l=2, # K parameter for K-of-L rule - MODIFIED from 3
                                         buffer_epsilon=0.001, # Buffer epsilon (ε) for MA order (absolute)
                                         slope_threshold=0.0005, # Slope threshold (ε_s) (absolute)
                                         delta_slope=3, # Delta for MA slope calculation
                                         price_feature_index=0, # Assuming price is the first feature
                                         high_feature_index=None, # Index of the 'high' feature for ATR
                                         low_feature_index=None, # Index of the 'low' feature for ATR
                                         volatility_measure: str = 'atr', # 'atr' or 'rolling_std' for normalization
                                         volatility_window: int = 14, # Window size for volatility measure
                                         percentile_threshold: float = None, # Percentile threshold for dynamic slope/difference
                                         past_window_for_percentile: int = None, # Past window length for percentile calculation
                                         # Add MA indices as parameters (need to map from feature_names)
                                         ma_short_name='MA_t_6',
                                         ma_medium_name='MA_t_24',
                                         ma_long_name='MA_t_72',
                                         # Keep other existing parameters, but they might not be used for the new rules
                                         reversal_check_len=3, # Existing parameter
                                         bb_upper_threshold_pct=0.05, # Existing parameter
                                         bb_lower_threshold_pct=0.05, # Existing parameter
                                         uptrend_threshold_pct=0.02, # Existing parameter (might be superseded)
                                         downtrend_threshold_pct=0.02, # Existing parameter (might be superseded)
                                         range_price_range_pct=0.01, # Existing parameter
                                         # Parameters for new scoring, dynamic threshold, and hysteresis logic
                                         score_feature_weights: dict = None, # Weights for calculate_score (if not using default EMA diff)
                                         score_ema_short_window: int = 6, # Short EMA window for default scoring
                                         score_ema_medium_window: int = 24, # Medium EMA window for default scoring
                                         dynamic_thresh_rolling_window: int = 60, # Rolling window for dynamic threshold (e.g., 60 for past sequence length)
                                         dynamic_thresh_percentile: float = 70, # Percentile for dynamic threshold
                                         dynamic_thresh_ewma_alpha: float = 0.1, # EWMA alpha for dynamic threshold smoothing
                                         hysteresis_window_size: int = 3 # Window size for hysteresis
                                        ):
    """
    Generates binary labels (0 or 1) for each sequence based on strategy,
    a calculated score, dynamic threshold, and hysteresis logic.

    Args:
        X_3d_numpy (np.ndarray): 3D numpy array of shape (n_sequences, sequence_length, n_features).
        strategy_labels (np.ndarray): 1D numpy array of shape (n_sequences,) containing strategy names.
        feature_names (list): List of feature names corresponding to the last dimension of X_3d_numpy.
        sequence_length (int): The length of each sequence.
        horizon (int): Prediction Horizon (H).
        eval_window_L (int): Evaluation Window Length (L).
        k_of_l (int): K parameter for K-of-L rule.
        buffer_epsilon (float): Buffer epsilon (ε) for MA order (absolute).
        slope_threshold (float): Slope threshold (ε_s) (absolute).
        delta_slope (int): Delta for MA slope calculation (MA[t] - MA[t-delta]).
        price_feature_index (int): The index of the price feature ('close').
        high_feature_index (int): Index of the 'high' feature for ATR. Required if volatility_measure is 'atr' or 'ATR'.
        low_feature_index (int): Index of the 'low' feature for ATR. Required if volatility_measure is 'atr' or 'ATR'.
        volatility_measure (str): 'atr', 'ATR' or 'rolling_std' for normalization.
        volatility_window (int): Window size for volatility measure.
        percentile_threshold (float, optional): Percentile threshold for dynamic slope/difference in K-of-L check.
        past_window_for_percentile (int, optional): Past window length for percentile calculation in K-of-L check.
        ma_short_name (str): Name of the short MA feature.
        ma_medium_name (str): Name of the medium MA feature.
        ma_long_name (str): Name of the long MA feature.
        reversal_check_len (int): (Existing)
        bb_upper_threshold_pct (float): (Existing)
        bb_lower_threshold_pct (float): (Existing)
        uptrend_threshold_pct (float): (Existing)
        downtrend_threshold_pct (float): (Existing)
        range_price_range_pct (float): (Existing)
        score_feature_weights (dict, optional): Weights for calculate_score (if not using default EMA diff).
        score_ema_short_window (int): Short EMA window for default scoring.
        score_ema_medium_window (int): Medium EMA window for default scoring.
        dynamic_thresh_rolling_window (int): Rolling window for dynamic threshold calculation.
        dynamic_thresh_percentile (float): Percentile for dynamic threshold.
        dynamic_thresh_ewma_alpha (float): EWMA alpha for dynamic threshold smoothing.
        hysteresis_window_size (int): Window size for hysteresis logic.

    Returns:
        np.ndarray: 1D numpy array of binary labels (0, 1, or np.nan) of shape (n_sequences,).
                    Returns None if required features or strategy labels are missing or inconsistent.
    """
    print(f"--- generate_binary_labels_from_strategy called ---")
    print(f"X_3d_numpy shape: {X_3d_numpy.shape}")
    print(f"strategy_labels shape: {strategy_labels.shape}")
    print(f"feature_names length: {len(feature_names)}")
    print(f"sequence_length: {sequence_length}")

    n_sequences = X_3d_numpy.shape[0]
    if n_sequences != len(strategy_labels):
        print(f"Error: Length mismatch between X_3d_numpy ({n_sequences}) and strategy_labels ({len(strategy_labels)}).")
        return np.full(n_sequences, np.nan, dtype=float) # Return NaN array on error

    # Ensure price, high, low feature indices are valid if needed for scoring/volatility
    # These indices are needed for calculate_score and potentially check_future_ma_conditions
    price_feature_idx = -1
    high_feature_idx = -1
    low_feature_idx = -1

    try:
        price_feature_idx = feature_names.index(price_feature_name)
        if volatility_measure.lower() == 'atr':
             high_feature_idx = feature_names.index(high_feature_name)
             low_feature_idx = feature_names.index(low_feature_name)

        # Check if indices are within bounds
        if price_feature_idx < 0 or price_feature_idx >= X_3d_numpy.shape[2]:
             raise ValueError(f"Price feature index out of bounds: {price_feature_idx}")
        if volatility_measure.lower() == 'atr':
             if high_feature_idx < 0 or high_feature_idx >= X_3d_numpy.shape[2] or \
                low_feature_idx < 0 or low_feature_idx >= X_3d_numpy.shape[2]:
                 raise ValueError(f"High/Low feature index out of bounds: {high_feature_idx}, {low_feature_idx}")

    except ValueError as e:
         print(f"Error: Could not find required price/high/low feature names or index out of bounds: {e}")
         print(f"Available features: {feature_names}")
         return np.full(n_sequences, np.nan, dtype=float) # Cannot label without these features


    # Map MA feature names to indices (needed for K-of-L check if used)
    ma_short_idx = -1
    ma_medium_idx = -1
    ma_long_idx = -1
    try:
        ma_short_idx = feature_names.index(ma_short_name)
        ma_medium_idx = feature_names.index(ma_medium_name)
        ma_long_idx = feature_names.index(ma_long_name)
        ma_indices = (ma_short_idx, ma_medium_idx, ma_long_idx)

        # Check if MA indices are within feature bounds
        if max(ma_indices) >= X_3d_numpy.shape[2] or min(ma_indices) < 0:
             raise ValueError("MA feature index out of bounds.")

    except ValueError as e:
        # MA features are only needed for the K-of-L check in Uptrend/Downtrend/Reversal strategies.
        # If they are missing, these specific strategies cannot use the K-of-L rule.
        # We can print a warning and proceed, letting those specific strategy checks fail or fallback.
        print(f"Warning: Could not find required MA feature names ({ma_short_name}, {ma_medium_name}, {ma_long_name}) in feature_names list for K-of-L check. Error: {e}")
        print(f"Available features: {feature_names}")
        ma_indices = (-1, -1, -1) # Set to invalid indices


    # Initialize arrays to store scores, dynamic thresholds, candidate labels, and final labels
    scores_array = np.full(n_sequences, np.nan, dtype=float)
    dynamic_thresholds_array = np.full(n_sequences, np.nan, dtype=float)
    candidate_labels_array = np.full(n_sequences, np.nan, dtype=float)
    final_labels = np.full(n_sequences, np.nan, dtype=float) # Initialize final labels with NaN


    # --- Calculate Scores for Each Sequence ---
    print("\nCalculating scores for each sequence...")
    for i in range(n_sequences):
        sequence_data = X_3d_numpy[i, :, :] # Shape: (seq_len, n_features)

        # Calculate score for the current sequence
        try:
             scores_over_time = calculate_score(
                 sequence_data=sequence_data,
                 feature_names=feature_names,
                 ema_short_window=score_ema_short_window,
                 ema_medium_window=score_ema_medium_window,
                 volatility_window=volatility_window, # Use same volatility window as K-of-L check
                 volatility_measure=volatility_measure, # Use same volatility measure as K-of-L check
                 feature_weights=score_feature_weights,
                 price_feature_name=price_feature_name,
                 high_feature_name=high_feature_name,
                 low_feature_name=low_feature_name
             )
             # The score for the sequence is the score at the last time step (index seq_len - 1)
             if scores_over_time is not None and len(scores_over_time) == sequence_length:
                  scores_array[i] = scores_over_time[-1] # Score at the end of the sequence
             else:
                  scores_array[i] = np.nan # Score calculation failed or returned unexpected shape

        except Exception as e:
             print(f"Error calculating score for sequence {i}: {e}. Assigning NaN.")
             scores_array[i] = np.nan # Assign NaN if score calculation fails


    # --- Calculate Dynamic Thresholds ---
    # Calculate dynamic thresholds over the entire sequence of scores
    print("\nCalculating dynamic thresholds...")
    # The dynamic threshold is calculated based on the *history* of scores.
    # So, for each sequence `i`, the dynamic threshold `tau_i` is calculated
    # using the scores from sequence 0 up to sequence i.
    # This requires the scores_array to be computed first for all sequences.

    # Calculate dynamic thresholds based on the scores_array
    # The dynamic threshold at index `i` should be based on scores_array[:i+1]
    # using a rolling window over the sequence *index*, not time steps within a sequence.
    # Let's re-interpret the dynamic threshold calculation: it should be based on
    # the scores of *previous sequences* in the dataset, not time steps within a single sequence.
    # So, `calculate_dynamic_threshold` should be applied to `scores_array` itself.

    try:
         dynamic_thresholds_array = calculate_dynamic_threshold(
             scores=scores_array,
             rolling_window=dynamic_thresh_rolling_window,
             percentile=dynamic_thresh_percentile,
             ewma_alpha=dynamic_thresh_ewma_alpha
         )
         if dynamic_thresholds_array is None or len(dynamic_thresholds_array) != n_sequences:
              print("Error: Dynamic threshold calculation failed or returned incorrect length. Assigning NaN thresholds.")
              dynamic_thresholds_array = np.full(n_sequences, np.nan, dtype=float) # Reset to NaN array

    except Exception as e:
         print(f"Error calculating dynamic thresholds over sequences: {e}. Assigning NaN.")
         dynamic_thresholds_array = np.full(n_sequences, np.nan, dtype=float) # Assign NaN if calculation fails


    # --- Generate Candidate Labels based on Score vs Dynamic Threshold ---
    print("\nGenerating candidate labels...")
    # Candidate label is 1 if score > dynamic_threshold, 0 otherwise.
    # Handle NaNs in scores or dynamic_thresholds.
    for i in range(n_sequences):
        score = scores_array[i]
        threshold = dynamic_thresholds_array[i]

        if not np.isnan(score) and not np.isnan(threshold):
             candidate_labels_array[i] = 1.0 if score > threshold else 0.0
        else:
             candidate_labels_array[i] = np.nan # Candidate label is NaN if score or threshold is NaN


    # --- Apply Hysteresis to Candidate Labels ---
    print("\nApplying hysteresis to candidate labels...")
    try:
         final_labels = apply_hysteresis(
             candidate_labels=candidate_labels_array,
             window_size=hysteresis_window_size,
             on_value=1,
             off_value=0,
             ignore_value=np.nan
         )
         if final_labels is None or len(final_labels) != n_sequences:
              print("Error: Hysteresis application failed or returned incorrect length. Assigning NaN final labels.")
              final_labels = np.full(n_sequences, np.nan, dtype=float) # Reset to NaN array

    except Exception as e:
         print(f"Error applying hysteresis: {e}. Assigning NaN final labels.")
         final_labels = np.full(n_sequences, np.nan, dtype=float) # Assign NaN if application fails


    # --- Optional: Overwrite labels for specific strategies using K-of-L rule ---
    # This part seems to be from a previous approach.
    # The request is to use the new score/threshold/hysteresis logic for binary labels.
    # Let's keep the K-of-L logic commented out or decide how it should integrate.
    # For now, focusing on the requested score/threshold/hysteresis approach.

    # # Iterate through each sequence and assign a label based on its strategy and future conditions
    # # This part used the K-of-L logic based on future MA conditions.
    # # Let's keep it here but comment it out, as the new request focuses on the score/threshold method.
    # # If the goal is to combine methods (e.g., use K-of-L only for Uptrend/Downtrend strategies
    # # and score/threshold for others), the logic needs to be refined.

    # print("\nApplying strategy-specific K-of-L rules (if applicable)...")
    # for i in range(n_sequences):
    #     strategy = strategy_labels[i]

    #     # Skip noise cluster (-1)
    #     if strategy == -1:
    #          final_labels[i] = np.nan # Assign NaN for noise
    #          continue

    #     # Check if enough future data exists for the required window (H + L) for K-of-L check
    #     last_required_sequence_idx_for_kofl = i + sequence_length + horizon + eval_window_L - 1
    #     if last_required_sequence_idx_for_kofl >= n_sequences:
    #          # print(f"Debug: Not enough future sequences available in X_3d_numpy for sequence {i} for K-of-L check. Skipping K-of-L check.")
    #          # Keep the label from hysteresis, or set to NaN if hysteresis also failed
    #          continue # Keep the label from hysteresis

    #     # Construct the combined data slice (past sequence + future evaluation window data points)
    #     try:
    #          total_future_len_kofl = horizon + eval_window_L
    #          combined_data_slice_kofl = X_3d_numpy[i : i + sequence_length + total_future_len_kofl, 0, :] # Shape (sequence_length + horizon + eval_window_L, n_features)
    #          combined_data_slice_kofl = combined_data_slice_kofl.astype(np.float64)

    #     except Exception as e:
    #          print(f"Error constructing combined data slice for K-of-L check for sample {i}: {e}. Skipping K-of-L check.")
    #          continue # Keep the label from hysteresis


    #     # --- Strategy-based K-of-L Logic ---
    #     # This logic will overwrite the hysteresis-based label if a clear K-of-L pattern is detected.
    #     # This might be an alternative labeling approach, not necessarily an improvement over hysteresis.
    #     # Let's implement it as an alternative labeling mechanism per strategy,
    #     # but the primary request is the score/threshold/hysteresis method.

    #     # If MA indices were not found, K-of-L check cannot be performed
    #     if -1 in ma_indices:
    #          # print("Warning: MA features not available. Skipping K-of-L check for strategy-based labeling.")
    #          continue # Skip K-of-L check if MA indices are invalid


    #     if strategy in ['Uptrend', 'Reversal_Up']:
    #         # Use the modified function to check future MA conditions for Uptrend
    #         kofl_label = check_future_ma_conditions(
    #             combined_data_slice_kofl,
    #             seq_length=sequence_length,
    #             horizon=horizon,
    #             ma_indices=ma_indices,
    #             eval_window_L=eval_window_L,
    #             k_of_l=k_of_l,
    #             buffer_epsilon=buffer_epsilon,
    #             slope_threshold=slope_threshold,
    #             delta_slope=delta_slope,
    #             volatility_measure=volatility_measure,
    #             volatility_window=volatility_window,
    #             price_feature_index=price_feature_idx, # Use the found price index
    #             high_feature_index=high_feature_idx, # Use the found high index
    #             low_feature_index=low_feature_idx,   # Use the found low index
    #             percentile_threshold=percentile_threshold,
    #             past_window_for_percentile=past_window_for_percentile
    #         )
    #         # If K-of-L check provides a valid label (0 or 1), use it to overwrite the hysteresis label
    #         if not np.isnan(kofl_label):
    #              final_labels[i] = kofl_label
    #              # print(f"Debug: Sample {i}, Strategy '{strategy}': K-of-L label = {kofl_label}. Overwriting hysteresis label.")
    #          # else: Keep the hysteresis label or NaN if K-of-L was NaN


    #     elif strategy in ['Downtrend', 'Reversal_Down']:
    #          # Use the modified function to check future MA conditions for Downtrend
    #          kofl_label = check_future_ma_conditions_downtrend(
    #              combined_data_slice_kofl,
    #              seq_length=sequence_length,
    #              horizon=horizon,
    #              ma_indices=ma_indices,
    #              eval_window_L=eval_window_L,
    #              k_of_l=k_of_l,
    #              buffer_epsilon=buffer_epsilon,
    #              slope_threshold=slope_threshold,
    #              delta_slope=delta_slope,
    #              volatility_measure=volatility_measure,
    #              volatility_window=volatility_window,
    #              price_feature_index=price_feature_idx, # Use the found price index
    #              high_feature_index=high_feature_idx, # Use the found high index
    #              low_feature_index=low_feature_idx,   # Use the found low index
    #              percentile_threshold=percentile_threshold,
    #              past_window_for_percentile=past_window_for_percentile
    #          )
    #          # If K-of-L check provides a valid label (0 or 1), use it
    #          if not np.isnan(kofl_label):
    #               final_labels[i] = kofl_label
    #               # print(f"Debug: Sample {i}, Strategy '{strategy}': K-of-L label = {kofl_label}. Overwriting hysteresis label.")
    #           # else: Keep the hysteresis label or NaN if K-of-L was NaN

        # For 'Range', 'choppy', 'unknown', etc., keep the label from hysteresis or NaN.
        # The K-of-L logic was specifically for trend continuation and reversals based on MA.


    # Noise cluster (-1) labels should be NaN
    noise_indices = np.where(strategy_labels == -1)[0]
    if len(noise_indices) > 0:
         final_labels[noise_indices] = np.nan # Assign NaN for noise


    print("\nFinished generating binary labels.")
    print(f"Generated {len(final_labels)} binary labels.")
    print(f"Label distribution: {np.unique(final_labels, return_counts=True)}")

    return final_labels


# In[ ]:


from scipy.stats import skew, kurtosis
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, normalize # Import normalize for L2 normalization
from sklearn.decomposition import PCA # Import PCA
from torch.utils.data import Dataset, DataLoader, TensorDataset
import hdbscan # Import hdbscan
import os
from sklearn.cluster import KMeans # Import KMeans for testing
from sklearn.cluster import MiniBatchKMeans # Import MiniBatchKMeans for Spherical KMeans init
from sklearn.mixture import GaussianMixture # Import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score # Import evaluation metrics
from sklearn.manifold import TSNE # Import TSNE for visualization


# Assuming AutoEncoder class is defined elsewhere (e.g., cell ZERvB1uf31Va) - This will be replaced
# Assuming train_autoencoder and test_autoencoder functions are defined elsewhere (e.g., cell ZERvB1uf31Va) - These will be replaced
# Assuming create_aggregated_features is defined elsewhere (e.g., cell JF5y50zNGmIN)
# Assuming CNNEncoder1D is defined elsewhere (newly added cell)
# Assuming ContrastiveLearningModule is defined elsewhere (newly added cell)
# Assuming train_contrastive_learning_model is defined elsewhere (newly added cell)
# Assuming extract_latent_vectors is defined elsewhere (newly added cell)
# Assuming data augmentation functions (add_gaussian_noise, random_scale, etc.) are defined elsewhere (newly added cell)
# Assuming DECModule is defined elsewhere (newly added cell)
# Assuming SupervisedPretrainingModule and train_supervised_pretraining_model are defined

# Define Spherical KMeans (using sklearn's KMeans with normalized data)
class SphericalKMeans(KMeans):
    """KMeans on normalized data."""
    def fit(self, X, y=None, sample_weight=None):
        X_normalized = normalize(X, norm='l2')
        return super().fit(X_normalized, y=y, sample_weight=sample_weight)

    def fit_predict(self, X, y=None, sample_weight=None):
        X_normalized = normalize(X, norm='l2')
        return super().fit_predict(X_normalized, y=y, sample_weight=sample_weight)

    def predict(self, X):
        X_normalized = normalize(X, norm='l2')
        return super().predict(X_normalized)


def perform_clustering_on_subset(X_subset_3d, feature_names_3d, seq_len, original_indices_subset,
                                 latent_dim, num_classes=None, y_labels_subset=None, # Add num_classes and y_labels_subset for supervised pretraining
                                 clustering_method='hdbscan', # Add parameter to choose clustering method
                                 hdbscan_params=None, # Keep hdbscan_params but make it optional and default to None
                                 kmeans_params=None, # Add kmeans_params as an optional dictionary
                                 n_clusters=10, # Use a single parameter for desired number of clusters (for KMeans, GMM, and DEC)
                                 pretraining_method='contrastive', # 'contrastive' or 'supervised'
                                 train_pretraining_flag=True, # Flag to control pretraining (CL or Supervised)
                                 trained_encoder=None, # Accept a pre-trained encoder (for inference)
                                 encoder_save_path=None, # Path to save/load the trained encoder
                                 # Parameters for Contrastive Learning pretraining
                                 cl_learning_rate=1e-3,
                                 cl_temperature=0.07,
                                 cl_augmentation_strategies=None,
                                 batch_size_cl_train=64,
                                 max_epochs_cl_train=100,
                                 # Parameters for Supervised pretraining
                                 supervised_learning_rate=1e-3,
                                 loss_fn_name='CrossEntropy',
                                 focal_loss_params=None,
                                 regularization_fn_name=None, # 'SupCon' etc.
                                 regularization_params=None,
                                 regularization_weight=0.1,
                                 supervised_augmentation_strategies=None, # Augmentations for SupCon in supervised pretraining
                                 batch_size_supervised_train=64,
                                 max_epochs_supervised_train=100,
                                 supervised_model_save_path=None, # Path to save the entire supervised model
                                 # Add DEC parameters
                                 use_dec_finetuning=False, # Flag to enable DEC fine-tuning
                                 dec_alpha=1.0,          # DEC Student-t distribution alpha
                                 dec_learning_rate=1e-3, # DEC training learning rate
                                 dec_finetune_encoder=False, # DEC training: finetune encoder or not
                                 max_epochs_dec_train=100, # Max epochs for DEC training
                                 dec_save_path=None,       # Path to save/load DEC model state_dict
                                 # Add parameters for normalization and PCA
                                 apply_l2_normalization=True, # Flag to apply L2 normalization to latent vectors
                                 use_pca=True,             # Flag to apply PCA after normalization
                                 n_components_pca=50,      # Number of components for PCA
                                 evaluate_clustering: bool = False, # Flag to perform clustering evaluation
                                 metric_for_evaluation: str = 'silhouette', # 'silhouette', 'davies_bouldin', 'calinski_harabasz'
                                 batch_size_latent_extraction=64
                                ):
    """
    Performs dimensionality reduction (Contrastive Learning or Supervised pretraining,
    with optional DEC fine-tuning), optional L2 normalization, PCA, and clustering
    (HDBSCAN, KMeans, GMM, or Spherical KMeans) on a subset of 3D raw sequences.

    Args:
        X_subset_3d (np.ndarray): Subset of **3D** numpy array of raw sequences (n_subset_samples, seq_len, n_features_3d).
        feature_names_3d (list): List of feature names for the 3D data.
        seq_len (int): The length of each sequence in X_subset_3d.
        original_indices_subset (pd.DatetimeIndex or np.ndarray): Original indices corresponding to X_subset_3d
                                                                   (n_subset_samples,).
        latent_dim (int): The dimension of the latent space for the encoder.
        num_classes (int, optional): The number of output classes for supervised pretraining. Required if pretraining_method is 'supervised'.
        y_labels_subset (np.ndarray, optional): Labels corresponding to X_subset_3d. Required if pretraining_method is 'supervised'.
        clustering_method (str): The clustering method to use ('hdbscan', 'kmeans', 'gmm', 'spherical_kmeans').
        hdbscan_params (dict, optional): Dictionary of parameters for HDBSCAN. Required if clustering_method is 'hdbscan'.
        kmeans_params (dict, optional): Dictionary of parameters for KMeans/Spherical KMeans. Required if clustering_method is 'kmeans' or 'spherical_kmeans'.
                                        If None, default parameters will be used.
        n_clusters (int): The desired number of clusters for KMeans, GMM, and DEC.
        pretraining_method (str): The pretraining method to use ('contrastive' or 'supervised').
        train_pretraining_flag (bool): If True, train the specified pretraining model on X_subset_3d.
                                       If False, use trained_encoder or load from encoder_save_path.
        trained_encoder (torch.nn.Module, optional): Pre-trained encoder module.
                                                    Used if train_pretraining_flag is False and encoder_save_path is None.
        encoder_save_path (str, optional): Path to save the trained encoder's state_dict (if training)
                                           or load the encoder from (if not training and trained_encoder is None).
                                           Should be a file path (e.e.g., 'cnn_encoder_state_dict.pth').
        # Parameters for Contrastive Learning pretraining
        cl_learning_rate (float): Learning rate for Contrastive Learning training.
        cl_temperature (float): Temperature for the InfoNCE loss.
        cl_augmentation_strategies (list, optional): List of augmentation functions and kwargs for CL training.
        batch_size_cl_train (int): Batch size for Contrastive Learning training if train_pretraining_flag is True and pretraining_method is 'contrastive'.
        max_epochs_cl_train (int): Max epochs for Contrastive Learning training if train_pretraining_flag is True and pretraining_method is 'contrastive'.
        # Parameters for Supervised pretraining
        supervised_learning_rate (float): Learning rate for Supervised pretraining.
        loss_fn_name (str): Name of the classification loss function ('CrossEntropy' or 'FocalLoss').
        focal_loss_params (dict, optional): Parameters for FocalLoss if used.
        regularization_fn_name (str, optional): Name of the regularization loss function ('SupCon').
        regularization_params (dict, optional): Parameters for regularization loss.
        regularization_weight (float): Weight for regularization loss.
        supervised_augmentation_strategies (list, optional): Augmentation strategies for SupCon if used in supervised pretraining.
        batch_size_supervised_train (int): Batch size for Supervised pretraining if train_pretraining_flag is True and pretraining_method is 'supervised'.
        max_epochs_supervised_train (int): Max epochs for Supervised pretraining if train_pretraining_flag is True and pretraining_method is 'supervised'.
        supervised_model_save_path (str, optional): Path to save the entire supervised model's state_dict if pretraining_method is 'supervised'.
        # Add DEC parameters
        use_dec_finetuning (bool): If True, perform DEC fine-tuning after CL or Supervised pretraining.
        dec_alpha (float): The alpha parameter for the Student-t distribution in DEC.
        dec_learning_rate (float): Learning rate for DEC training.
        dec_finetune_encoder (bool): If True, finetune encoder during DEC training.
        max_epochs_dec_train (int): Max epochs for DEC training.
        dec_save_path (str, optional): Path to save/load DEC model state_dict.
        # Add parameters for normalization and PCA
        apply_l2_normalization (bool): If True, apply L2 normalization to latent vectors before clustering/PCA.
        use_pca (bool): If True, apply PCA for dimensionality reduction after normalization (if applied).
        n_components_pca (int): Number of components for PCA.
        evaluate_clustering (bool): If True, evaluate the clustering result using specified metric.
        metric_for_evaluation (str): Metric to use for evaluation if evaluate_clustering is True.
        batch_size_latent_extraction (int): Batch size for extracting latent vectors.


    Returns:
        tuple: (final_representation, cluster_labels, original_indices_subset)
               - final_representation (np.ndarray or None): Latent vectors (after normalization/PCA if applied) or DEC soft assignments (if DEC used) (n_subset_samples, latent_dim or n_clusters) or None.
               - cluster_labels (np.ndarray or None): Cluster labels (n_subset_samples,) or None if clustering fails/skipped.
               - original_indices_subset (pd.DatetimeIndex or np.ndarray): The original indices subset passed in.
               Returns (None, None, original_indices_subset) if critical processing fails.
    """
    if X_subset_3d is None or X_subset_3d.shape[0] == 0:
        print("Warning: X_subset_3d is empty. Skipping processing for this subset.")
        return None, None, original_indices_subset

    # Assert that the input is indeed 3D
    if X_subset_3d.ndim != 3:
         print(f"Error: perform_clustering_on_subset expected 3D input (n_samples, seq_len, n_features), but got {X_subset_3d.ndim}D input with shape {X_subset_3d.shape}.")
         print("Please ensure you are passing the 3D raw sequence data.")
         return None, None, original_indices_subset

    print(f"Starting pipeline for subset with shape: {X_subset_3d.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = None # Initialize encoder
    latent_representation = None # Initialize latent representation

    # Determine input_channels and seq_len from the 3D data shape
    n_subset_samples, actual_seq_len, actual_n_features = X_subset_3d.shape
    input_channels_encoder = actual_n_features # Number of features is input channels for Conv1D
    seq_len_encoder = actual_seq_len         # Sequence length is input length for Conv1D

    # Ensure num_classes and y_labels_subset are provided if supervised pretraining is used
    if pretraining_method == 'supervised':
        if num_classes is None or y_labels_subset is None:
            print("Error: num_classes and y_labels_subset must be provided for supervised pretraining.")
            return None, None, original_indices_subset
        if len(y_labels_subset) != n_subset_samples:
            print(f"Error: Length mismatch between X_subset_3d ({n_subset_samples}) and y_labels_subset ({len(y_labels_subset)}) for supervised pretraining.")
            return None, None, original_indices_subset


    # --- 1. Pretraining (Contrastive or Supervised): Train or Load ---
    print(f"\n--- Step 1: Pretraining ({pretraining_method}) ---")

    if train_pretraining_flag:
        if pretraining_method == 'contrastive':
            print("Training Contrastive Learning model...")
            try:
                encoder, _ = train_contrastive_learning_model(
                    X_data=X_subset_3d,
                    input_channels=input_channels_encoder,
                    seq_len=seq_len_encoder,
                    latent_dim=latent_dim,
                    learning_rate=cl_learning_rate,
                    temperature=cl_temperature,
                    augmentation_strategies=cl_augmentation_strategies,
                    batch_size=batch_size_cl_train,
                    max_epochs=max_epochs_cl_train,
                    encoder_save_path=encoder_save_path # Pass path to save encoder
                )
                if encoder is None:
                     print("Error: Contrastive Learning model training failed. Cannot proceed.")
                     return None, None, original_indices_subset

                print("CL training complete.")

            except Exception as e:
                print(f"Error during Contrastive Learning model training: {e}. Cannot proceed.")
                return None, None, original_indices_subset

        elif pretraining_method == 'supervised':
            print("Training Supervised Pretraining model...")
            try:
                 # train_supervised_pretraining_model returns the full module, not just the encoder
                 supervised_module, _ = train_supervised_pretraining_model(
                     X_data=X_subset_3d,
                     y_labels=y_labels_subset,
                     input_channels=input_channels_encoder,
                     seq_len=seq_len_encoder,
                     latent_dim=latent_dim,
                     num_classes=num_classes,
                     learning_rate=supervised_learning_rate,
                     loss_fn_name=loss_fn_name,
                     focal_loss_params=focal_loss_params,
                     regularization_fn_name=regularization_fn_name,
                     regularization_params=regularization_params,
                     regularization_weight=regularization_weight,
                     augmentation_strategies=supervised_augmentation_strategies, # Augmentations for SupCon in supervised
                     batch_size=batch_size_supervised_train,
                     max_epochs=max_epochs_supervised_train,
                     encoder_save_path=encoder_save_path, # Pass path to save encoder
                     supervised_model_save_path=supervised_model_save_path # Pass path to save entire supervised model
                 )
                 if supervised_module is None:
                     print("Error: Supervised pretraining failed. Cannot proceed.")
                     return None, None, original_indices_subset

                 # Get the trained encoder from the supervised module
                 encoder = supervised_module.get_encoder()
                 if encoder is None:
                     print("Error: Could not get encoder from trained supervised module. Cannot proceed.")
                     return None, None, original_indices_subset

                 print("Supervised pretraining complete.")

            except Exception as e:
                print(f"Error during Supervised pretraining: {e}. Cannot proceed.")
                return None, None, original_indices_subset

        else:
             print(f"Error: Unknown pretraining method '{pretraining_method}'. Supported: 'contrastive', 'supervised'. Cannot proceed.")
             return None, None, original_indices_subset


    else: # train_pretraining_flag is False (load existing encoder)
        print("Using existing encoder (loading or provided)...")
        encoder = trained_encoder
        if encoder is None and encoder_save_path:
            print(f"Attempting to load encoder state_dict from {encoder_save_path}...")
            try:
                # Need to instantiate CNNEncoder1D with correct parameters
                # Assuming CNNEncoder1D was defined with input_channels, seq_len, and output_dim=latent_dim
                encoder = CNNEncoder1D(input_channels=input_channels_encoder, output_dim=latent_dim) # seq_len is not needed for this CNNEncoder1D version
                encoder.load_state_dict(torch.load(encoder_save_path, map_location=device))
                encoder.to(device)
                print(f"Loaded encoder state_dict from {encoder_save_path}")
            except Exception as e:
                print(f"Error loading encoder state_dict from {encoder_save_path}: {e}. Cannot proceed without a valid encoder.")
                return None, None, original_indices_subset
        elif encoder is None and not encoder_save_path:
             print("Error: train_pretraining_flag is False, but no trained_encoder or encoder_save_path provided. Cannot proceed without a valid encoder.")
             return None, None, original_indices_subset

        print("Encoder available.")

    # --- 2. Extract Latent Vectors ---
    # Use the extract_latent_vectors function with the obtained encoder
    print("\n--- Step 2: Extract Latent Vectors ---")
    try:
         # Create DataLoader for the data for extraction
         # X_subset_3d is (n_samples, seq_len, n_features)
         # extract_latent_vectors expects (n_samples, seq_len, n_features) NumPy array
         # and handles the transpose to (n_samples, n_features, seq_len) internally for the encoder
         # Or, it expects a DataLoader yielding batches in the format the encoder expects.
         # Let's use the function `extract_latent_vectors` which takes the NumPy array
         # and handles DataLoader creation and processing internally.

         # Ensure the encoder is in evaluation mode before extraction
         encoder.eval()

         latent_representation = extract_latent_vectors(
             encoder=encoder,
             X_data=X_subset_3d, # Pass the 3D subset data for extraction
             batch_size=batch_size_latent_extraction
         )

         if latent_representation is None or latent_representation.shape[0] == 0:
              print("Error: Latent representation could not be obtained or is empty after extraction. Cannot proceed.")
              return None, None, original_indices_subset

         print("Latent vector extraction complete.")
         print(f"Debug: Shape of latent representation after extraction: {latent_representation.shape}") # Debug print


    except Exception as e:
         print(f"Error during latent vector extraction: {e}. Cannot proceed.")
         return None, None, original_indices_subset


    # --- Handle NaNs and Infs in Latent Representation ---
    if np.isnan(latent_representation).any() or np.isinf(latent_representation).any():
        print("Warning: Latent representation contains NaN or Inf values after extraction. Replacing with 0 and clamping finite values.")
        latent_representation = np.nan_to_num(latent_representation, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
        print("Debug: Replaced NaN/Inf values in latent representation.")


    # --- 3. Optional DEC Fine-tuning ---
    # DEC fine-tuning is applied to the latent space obtained from the pre-trained encoder.
    # After DEC, we can either use the DEC soft assignments as the final representation
    # or re-extract latent vectors using the potentially fine-tuned encoder.
    final_representation_for_clustering = latent_representation # Default to pretraining latent vectors
    dec_cluster_assignments = None # Initialize DEC assignments

    if use_dec_finetuning:
        print("\n--- Step 3: Optional DEC Fine-tuning ---")
        # Ensure enough samples for DEC initialization (KMeans)
        if n_subset_samples < n_clusters:
             print(f"Warning: Not enough samples ({n_subset_samples}) for DEC initialization with {n_clusters} clusters. Skipping DEC.")
             # Proceed with pretraining latent vectors for clustering
             use_dec_finetuning = False # Disable DEC if not enough samples
        else:
            try:
                 # Create DataLoader for the data for DEC training
                 # DEC needs original data (X_subset_3d) as input
                 # Transpose X_subset_3d to (n_samples, n_features, seq_len) for the DataLoader
                 X_tensor_dec = torch.tensor(X_subset_3d, dtype=torch.float32).transpose(1, 2) # Shape (n_samples, n_features, seq_len)
                 dataset_dec = TensorDataset(X_tensor_dec)
                 dataloader_dec = DataLoader(dataset_dec, batch_size=batch_size_latent_extraction, shuffle=True, num_workers=os.cpu_count() // 2 or 1) # Use extraction batch size for consistency

                 # Initialize DEC Module
                 # Pass the trained encoder
                 dec_model = DECModule(
                     encoder=encoder, # Use the trained encoder
                     n_clusters=n_clusters,
                     alpha=dec_alpha,
                     learning_rate=dec_learning_rate,
                     finetune_encoder=dec_finetune_encoder # Control encoder finetuning during DEC
                 ).to(device) # Move DEC model to device


                 # Initialize DEC cluster centroids using KMeans on the extracted latent vectors
                 # Pass the DataLoader for the *entire* dataset (dataloader_dec)
                 dec_model.initialize_centroids(dataloader_dec)


                 # Load DEC state_dict if path is provided and exists
                 if dec_save_path:
                      if os.path.exists(dec_save_path):
                           print(f"Attempting to load DEC model state_dict from {dec_save_path}...")
                           try:
                               dec_model.load_state_dict(torch.load(dec_save_path, map_location=device))
                               print(f"Loaded DEC model state_dict from {dec_save_path}")
                               # If loaded, assume training is skipped and use loaded model for assignments
                               print("Skipping DEC training, using loaded model for assignments.")
                               max_epochs_dec_train = 0 # Ensure training loop doesn't run


                           except Exception as e:
                               print(f"Error loading DEC model state_dict from {dec_save_path}: {e}. Proceeding with DEC training.")
                               # If loading fails, proceed with training

                 # Initialize DEC Trainer only if max_epochs > 0
                 if max_epochs_dec_train > 0:
                     print("Starting DEC fine-tuning...")
                     dec_trainer = pl.Trainer(
                         max_epochs=max_epochs_dec_train,
                         accelerator='auto',
                         # Add logging, checkpointing etc. as needed for DEC
                         # callbacks=[pl.callbacks.EarlyStopping(monitor='train_loss_dec', patience=10)] # Example early stopping
                     )
                     dec_trainer.fit(dec_model, dataloader_dec)
                     print("DEC fine-tuning finished.")

                 # Get final representation for clustering after DEC
                 # Re-extract latent vectors using the potentially fine-tuned encoder
                 # Ensure the encoder is in eval mode after DEC training (if it was finetuned)
                 dec_model.encoder.eval()
                 # Need a dataloader for extraction with the potentially fine-tuned encoder
                 # Use the same dataset_dec but with shuffle=False for extraction
                 extraction_dataloader_dec = DataLoader(dataset_dec, batch_size=batch_size_latent_extraction, shuffle=False)

                 final_latent_after_dec = []
                 with torch.no_grad():
                      for batch in extraction_dataloader_dec:
                           x_batch = batch[0].to(device)
                           z_batch = dec_model.encoder(x_batch)
                           final_latent_after_dec.append(z_batch.cpu().numpy())

                 final_latent_after_dec_np = np.concatenate(final_latent_after_dec, axis=0)


                 if final_latent_after_dec_np is None or final_latent_after_dec_np.shape[0] == 0:
                      print("Error: Latent representation could not be obtained after DEC. Using pretraining latent vectors.")
                      final_representation_for_clustering = latent_representation # Fallback
                 else:
                      final_representation_for_clustering = final_latent_after_dec_np # Use DEC-tuned latent vectors
                      print("Using DEC-tuned latent vectors for final clustering.")
                      print(f"Debug: Shape of latent representation after DEC: {final_representation_for_clustering.shape}") # Debug print


                 # Optional: Get DEC hard assignments for potential use or comparison
                 dec_model.eval()
                 with torch.no_grad():
                      # Use the same extraction dataloader
                      dec_cluster_assignments_list = []
                      for batch in extraction_dataloader_dec:
                           x_batch = batch[0].to(device)
                           final_q, _ = dec_model(x_batch) # Get final soft assignments
                           dec_cluster_assignments_list.append(torch.argmax(final_q, dim=1).cpu().numpy()) # Get hard assignments
                      dec_cluster_assignments = np.concatenate(dec_cluster_assignments_list, axis=0)
                      print(f"Debug: Shape of DEC hard assignments: {dec_cluster_assignments.shape}") # Debug print


                 # Save DEC model state_dict if path is provided
                 if dec_save_path and max_epochs_dec_train > 0: # Only save if training actually happened
                     try:
                         # Ensure the directory exists
                         save_dir = os.path.dirname(dec_save_path)
                         if save_dir and not os.path.exists(save_dir):
                             os.makedirs(save_dir)
                             print(f"Created directory for saving DEC model: {save_dir}")

                         # Save the state_dict of the entire DEC model
                         torch.save(dec_model.state_dict(), dec_save_path)
                         print(f"Saved trained DEC model state_dict to {dec_save_path}")
                     except Exception as e:
                         print(f"Warning: Failed to save trained DEC model state_dict to {dec_save_path}: {e}")


            except Exception as e:
                print(f"Error during DEC fine-tuning: {e}. Skipping DEC.")
                # Proceed with pretraining latent vectors for clustering
                use_dec_finetuning = False # Ensure flag is false after failure
                final_representation_for_clustering = latent_representation # Ensure fallback


    # --- Handle NaNs and Infs in Final Representation before Normalization/PCA ---
    if np.isnan(final_representation_for_clustering).any() or np.isinf(final_representation_for_clustering).any():
        print("Warning: Final representation contains NaN or Inf values. Replacing with 0 and clamping finite values.")
        final_representation_for_clustering = np.nan_to_num(final_representation_for_clustering, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
        print("Debug: Replaced NaN/Inf values in final representation.")


    # --- 4. Apply L2 Normalization ---
    representation_after_normalization = final_representation_for_clustering
    if apply_l2_normalization:
        print("\n--- Step 4: Applying L2 Normalization ---")
        # Handle potential zero vectors before normalization
        norm = np.linalg.norm(representation_after_normalization, axis=1, keepdims=True)
        # Replace zero norms with 1 to avoid division by zero; these vectors will remain unchanged
        norm = np.where(norm == 0, 1, norm)
        representation_after_normalization = representation_after_normalization / norm
        print(f"Applied L2 normalization. Shape: {representation_after_normalization.shape}")
        # Check for NaNs/Infs again after normalization
        if np.isnan(representation_after_normalization).any() or np.isinf(representation_after_normalization).any():
             print("Warning: NaN or Inf values detected after L2 normalization. Replacing with 0 and clamping finite values.")
             representation_after_normalization = np.nan_to_num(representation_after_normalization, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
             print("Debug: Replaced NaN/Inf values after normalization.")


    # --- 5. Apply PCA for Dimensionality Reduction ---
    representation_after_pca = representation_after_normalization
    pca_model = None # Initialize PCA model
    # Check if PCA should be applied and if there are enough samples
    if use_pca and n_components_pca < representation_after_normalization.shape[1]:
        print(f"\n--- Step 5: Applying PCA with {n_components_pca} components ---")
        # Ensure enough samples for PCA
        if representation_after_pca.shape[0] < n_components_pca:
             print(f"Warning: Not enough samples ({representation_after_pca.shape[0]}) for PCA with {n_components_pca} components. Skipping PCA and using representation after normalization.")
             use_pca = False # Disable PCA if not enough samples
             representation_after_pca = representation_after_normalization # Revert to pre-PCA data
        else:
            try:
                 pca_model = PCA(n_components=n_components_pca, random_state=42)
                 representation_after_pca = pca_model.fit_transform(representation_after_normalization)
                 print(f"PCA applied. Shape: {representation_after_pca.shape}")
                 print(f"Explained variance ratio (sum): {pca_model.explained_variance_ratio_.sum():.4f}")

                 # Check for NaNs/Infs again after PCA
                 if np.isnan(representation_after_pca).any() or np.isinf(representation_after_pca).any():
                      print("Warning: NaN or Inf values detected after PCA. Replacing with 0 and clamping finite values.")
                      representation_after_pca = np.nan_to_num(representation_after_pca, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
                      print("Debug: Replaced NaN/Inf values after PCA.")

            except Exception as e:
                 print(f"Error during PCA: {e}. Skipping PCA.")
                 use_pca = False # Disable PCA after failure
                 representation_after_pca = representation_after_normalization # Revert to pre-PCA data
    elif use_pca and n_components_pca >= representation_after_normalization.shape[1]:
         print(f"\n--- Step 5: Skipping PCA ---")
         print(f"n_components_pca ({n_components_pca}) is >= the number of features ({representation_after_normalization.shape[1]}) after normalization. No dimensionality reduction needed by PCA.")
         use_pca = False # Disable PCA if n_components is not less than current features
         representation_after_pca = representation_after_normalization # Use representation after normalization


    # The representation used for final clustering is now representation_after_pca
    clustering_input = representation_after_pca


    # --- 6. Clustering on Final Representation ---
    print("\n--- Step 6: Clustering on Final Representation ---")
    cluster_labels = None # Initialize cluster_labels
    evaluation_metrics = None # Initialize evaluation metrics


    # Ensure enough samples for clustering based on the chosen method and its parameters
    min_samples_needed_for_method = 1 # Default minimum

    # Determine effective n_clusters/n_components for KMeans/GMM/Spherical KMeans
    # Use the n_clusters parameter passed to the function
    effective_n_clusters = n_clusters

    try:
        if clustering_method == 'hdbscan':
            print("Using HDBSCAN clustering...")
            # Default HDBSCAN parameters if none provided
            if hdbscan_params is None:
                hdbscan_params = {
                    'min_cluster_size': max(10, int(np.sqrt(clustering_input.shape[0]))), # Suggested starting point
                    'min_samples': None, # Often min_samples = min_cluster_size or slightly smaller
                    'cluster_selection_epsilon': 0.0, # Default 0.0
                    'gen_min_span_tree': False, # Can set to True for plotting, but False is faster
                    'random_state': 42 # HDBSCAN uses random state for some aspects
                }
                # If min_samples is None, HDBSCAN defaults it to min_cluster_size
                print(f"Using default HDBSCAN parameters: {hdbscan_params}")
            else:
                 print(f"Using provided HDBSCAN parameters: {hdbscan_params}")


            min_samples_needed_for_method = hdbscan_params.get('min_cluster_size', 10)

            if clustering_input.shape[0] < min_samples_needed_for_method:
                 print(f"Warning: Not enough samples ({clustering_input.shape[0]}) for HDBSCAN with min_cluster_size={min_samples_needed_for_method}. Assigning all to -1.")
                 cluster_labels = np.full(n_subset_samples, -1, dtype=int)
            else:
                print(f"Debug: Shape of clustering input immediately before HDBSCAN fit: {clustering_input.shape}") # Debug Print
                clusterer = hdbscan.HDBSCAN(**hdbscan_params)
                cluster_labels = clusterer.fit_predict(clustering_input)
                print(f"HDBSCAN clustering complete. Found {len(np.unique(cluster_labels)) - (1 if -1 in cluster_labels else 0)} clusters (excluding noise).")
                print(f"Cluster label distribution:\n{pd.Series(cluster_labels).value_counts().sort_index()}")

            # HDBSCAN stability (persistence) is typically evaluated during parameter tuning,
            # not as a single metric after fitting. Evaluation flag is less applicable here.


        elif clustering_method == 'kmeans':
            print(f"Using KMeans clustering with n_clusters={effective_n_clusters}...")
            # Default KMeans parameters if none provided
            if kmeans_params is None:
                 kmeans_params = {
                     'n_clusters': effective_n_clusters, # Use the effective n_clusters parameter
                     'random_state': 42,
                     'n_init': 10 # Or 'auto' in newer sklearn versions
                 }
                 print(f"Using default KMeans parameters: {kmeans_params}")
            else:
                 # Ensure n_clusters from argument takes precedence if provided in kmeans_params
                 if 'n_clusters' in kmeans_params:
                      print(f"Warning: n_clusters provided in both function argument ({effective_n_clusters}) and kmeans_params ({kmeans_params['n_clusters']}). Using n_clusters from function argument.")
                 kmeans_params['n_clusters'] = effective_n_clusters # Ensure consistency

                 # Ensure n_init is set if not provided
                 if 'n_init' not in kmeans_params:
                     kmeans_params['n_init'] = 10 # Default n_init


                 print(f"Using provided KMeans parameters: {kmeans_params}")

            min_samples_needed_for_method = kmeans_params['n_clusters']

            if clustering_input.shape[0] < min_samples_needed_for_method:
                 print(f"Warning: Not enough samples ({clustering_input.shape[0]}) for KMeans with n_clusters={kmeans_params['n_clusters']}. Assigning all to -1.")
                 cluster_labels = np.full(n_subset_samples, -1, dtype=int) # Or raise error
            else:
                kmeans = KMeans(**kmeans_params) # Use the combined parameters
                print(f"Debug: Shape of clustering input immediately before KMeans fit: {clustering_input.shape}") # Debug Print
                cluster_labels = kmeans.fit_predict(clustering_input)
                print(f"KMeans clustering complete. Found {len(np.unique(cluster_labels))} clusters.")
                print(f"Cluster label distribution:\n{pd.Series(cluster_labels).value_counts().sort_index()}")

            # Evaluate KMeans if requested
            if evaluate_clustering and len(np.unique(cluster_labels)) > 1 and clustering_input.shape[0] > 1: # Need at least 2 distinct labels and > 1 sample
                 print(f"Evaluating KMeans clustering using metric: {metric_for_evaluation}")
                 try:
                     if metric_for_evaluation == 'silhouette':
                         # Silhouette score requires distance metric (default Euclidean for KMeans is fine)
                         score = silhouette_score(clustering_input, cluster_labels)
                         evaluation_metrics = {'silhouette_score': score}
                     elif metric_for_evaluation == 'davies_bouldin':
                         score = davies_bouldin_score(clustering_input, cluster_labels)
                         evaluation_metrics = {'davies_bouldin_score': score} # Lower is better
                     elif metric_for_evaluation == 'calinski_harabasz':
                         score = calinski_harabasz_score(clustering_input, cluster_labels)
                         evaluation_metrics = {'calinski_harabasz_score': score} # Higher is better
                     else:
                         print(f"Warning: Unknown evaluation metric '{metric_for_evaluation}'. Skipping evaluation.")

                 except Exception as e:
                     print(f"Error during KMeans evaluation: {e}. Skipping evaluation.")
                     evaluation_metrics = None


        elif clustering_method == 'gmm':
            print(f"Using Gaussian Mixture Model clustering with n_components={effective_n_clusters}...") # Use effective_n_clusters as n_components
            min_samples_needed_for_method = effective_n_clusters
            if clustering_input.shape[0] < min_samples_needed_for_method:
                 print(f"Warning: Not enough samples ({clustering_input.shape[0]}) for GMM with n_components={effective_n_clusters}. Assigning all to -1.")
                 cluster_labels = np.full(n_subset_samples, -1, dtype=int) # Or raise error
            else:
                # GMM can be sensitive to scale, consider scaling latent_vectors if not already done
                # scaler_gmm = StandardScaler()
                # clustering_input_scaled = scaler_gmm.fit_transform(clustering_input)
                gmm = GaussianMixture(n_components=effective_n_clusters, random_state=42) # Use effective_n_clusters as n_components
                print(f"Debug: Shape of clustering input immediately before GMM fit: {clustering_input.shape}") # Debug Print
                cluster_labels = gmm.fit_predict(clustering_input)
                print(f"GMM clustering complete. Found {len(np.unique(cluster_labels))} components/clusters.")
                print(f"Cluster label distribution:\n{pd.Series(cluster_labels).value_counts().sort_index()}")

                # GMM also provides BIC/AIC for component selection, which is often done externally
                # bic = gmm.bic(clustering_input)
                # aic = gmm.aic(clustering_input)
                # print(f"GMM BIC: {bic}, AIC: {aic}")


            # Evaluate GMM if requested (similar to KMeans evaluation logic above)
            if evaluate_clustering and len(np.unique(cluster_labels)) > 1 and clustering_input.shape[0] > 1: # Need at least 2 clusters
                 print(f"Evaluating GMM clustering using metric: {metric_for_evaluation}")
                 try:
                     if metric_for_evaluation == 'silhouette':
                         score = silhouette_score(clustering_input, cluster_labels)
                         evaluation_metrics = {'silhouette_score': score}
                     elif metric_for_evaluation == 'davies_bouldin':
                         score = davies_bouldin_score(clustering_input, cluster_labels)
                         evaluation_metrics = {'davies_bouldin_score': score}
                     elif metric_for_evaluation == 'calinski_harabasz':
                         score = calinski_harabasz_score(clustering_input, cluster_labels)
                         evaluation_metrics = {'calinski_harabasz_score': score}
                     else:
                         print(f"Warning: Unknown evaluation metric '{metric_for_evaluation}'. Skipping evaluation.")

                 except Exception as e:
                     print(f"Error during GMM evaluation: {e}. Skipping evaluation.")
                     evaluation_metrics = None

        elif clustering_method == 'spherical_kmeans':
            print(f"Using Spherical KMeans clustering with n_clusters={effective_n_clusters}...")
            # Default KMeans parameters for Spherical KMeans if none provided
            if kmeans_params is None:
                 kmeans_params = {
                     'n_clusters': effective_n_clusters,
                     'random_state': 42,
                     'n_init': 10
                 }
                 print(f"Using default Spherical KMeans parameters: {kmeans_params}")
            else:
                 # Ensure n_clusters from argument takes precedence if provided in kmeans_params
                 if 'n_clusters' in kmeans_params:
                      print(f"Warning: n_clusters provided in both function argument ({effective_n_clusters}) and kmeans_params ({kmeans_params['n_clusters']}). Using n_clusters from function argument.")
                 kmeans_params['n_clusters'] = effective_n_clusters # Ensure consistency

                 # Ensure n_init is set if not provided
                 if 'n_init' not in kmeans_params:
                     kmeans_params['n_init'] = 10 # Default n_init

                 print(f"Using provided Spherical KMeans parameters: {kmeans_params}")


            min_samples_needed_for_method = kmeans_params['n_clusters']

            if clustering_input.shape[0] < min_samples_needed_for_method:
                 print(f"Warning: Not enough samples ({clustering_input.shape[0]}) for Spherical KMeans with n_clusters={kmeans_params['n_clusters']}. Assigning all to -1.")
                 cluster_labels = np.full(n_subset_samples, -1, dtype=int) # Or raise error
            else:
                # Spherical KMeans wrapper handles normalization internally
                spherical_kmeans = SphericalKMeans(**kmeans_params)
                print(f"Debug: Shape of clustering input immediately before Spherical KMeans fit: {clustering_input.shape}") # Debug Print
                cluster_labels = spherical_kmeans.fit_predict(clustering_input)
                print(f"Spherical KMeans clustering complete. Found {len(np.unique(cluster_labels))} clusters.")
                print(f"Cluster label distribution:\n{pd.Series(cluster_labels).value_counts().sort_index()}")

            # Evaluate Spherical KMeans (evaluation metrics like silhouette still work on normalized data)
            if evaluate_clustering and len(np.unique(cluster_labels)) > 1 and clustering_input.shape[0] > 1: # Need at least 2 distinct labels and > 1 sample
                 print(f"Evaluating Spherical KMeans clustering using metric: {metric_for_evaluation}")
                 # Need to normalize the input again for silhouette calculation
                 clustering_input_normalized = normalize(clustering_input, norm='l2')
                 try:
                     if metric_for_evaluation == 'silhouette':
                         score = silhouette_score(clustering_input_normalized, cluster_labels)
                         evaluation_metrics = {'silhouette_score': score}
                     elif metric_for_evaluation == 'davies_bouldin':
                         score = davies_bouldin_score(clustering_input_normalized, cluster_labels)
                         evaluation_metrics = {'davies_bouldin_score': score} # Lower is better
                     elif metric_for_evaluation == 'calinski_harabasz':
                         score = calinski_harabasz_score(clustering_input_normalized, cluster_labels)
                         evaluation_metrics = {'calinski_harabasz_score': score} # Higher is better
                     else:
                         print(f"Warning: Unknown evaluation metric '{metric_for_evaluation}'. Skipping evaluation.")

                 except Exception as e:
                     print(f"Error during Spherical KMeans evaluation: {e}. Skipping evaluation.")
                     evaluation_metrics = None


        else:
            print(f"Error: Unknown clustering method '{clustering_method}'. Supported methods are 'hdbscan', 'kmeans', 'gmm', 'spherical_kmeans'. Assigning all to -1.")
            # Assign all to -1 for unknown method
            cluster_labels = np.full(n_samples, -1, dtype=int)


    except Exception as e:
        print(f"An unexpected error occurred during Clustering: {e}")
        # Assign all to -1 in case of unexpected errors
        cluster_labels = np.full(n_samples, -1, dtype=int)
        evaluation_metrics = None


    # Ensure cluster_labels is a NumPy array even if all samples are noise (-1)
    if cluster_labels is not None and not isinstance(cluster_labels, np.ndarray):
         cluster_labels = np.asarray(cluster_labels)


    # --- 6. Return Results ---
    # Return the final representation used for clustering (after normalization/PCA),
    # the cluster labels, and original indices
    return clustering_input, cluster_labels, original_indices_subset # Return representation after PCA/Norm


# In[ ]:


from scipy.stats import skew, kurtosis

def create_aggregated_features(X_subset, feature_names, seq_len):
    """
    各時間系列シーケンスから集約特徴量を作成します。

    Args:
        X_subset (np.ndarray): 3D NumPy array of shape (n_samples, seq_len, n_features).
        feature_names (list): List of names for the features in the last dimension of X_subset.
        seq_len (int): The length of each sequence.

    Returns:
        np.ndarray: 2D NumPy array of shape (n_samples, n_aggregated_features) containing
                    aggregated features for each sequence.
                    Returns an empty array if X_subset is empty or invalid.
    """
    if X_subset is None or X_subset.shape[0] == 0 or X_subset.ndim != 3 or X_subset.shape[1] != seq_len:
        # Check only relevant shape dimensions for the function's operation
        print(f"Warning: Invalid X_subset shape {X_subset.shape if X_subset is not None else 'None'} for aggregation. Expected (n_samples, {seq_len}, n_features). Returning empty array.")
        return np.array([])

    n_samples, actual_seq_len, actual_n_features = X_subset.shape
    # Use the actual sequence length from the input data
    seq_len = actual_seq_len

    aggregated_features_list = []
    aggregated_feature_names = []

    print(f"Creating aggregated features for {n_samples} sequences with {actual_n_features} features...")
    # Ensure feature_names is not None and has some names, even if the count doesn't match exactly
    if feature_names is None or len(feature_names) == 0:
        print("Warning: feature_names list is empty or None. Using generic feature names.")
        # Create generic names if feature_names is missing or empty
        feature_names = [f'feature_{j}' for j in range(actual_n_features)]
    elif len(feature_names) != actual_n_features:
         print(f"Warning: Length of feature_names ({len(feature_names)}) does not match actual number of features in data ({actual_n_features}). Aggregating based on data shape and using available feature names.")


    # Iterate over each sample (sequence)
    for i in range(n_samples):
        sequence_data = X_subset[i] # Shape: (seq_len, actual_n_features)
        sample_aggregated_features = []
        current_sample_aggregated_feature_names = [] # Collect names for this sample's features (should be consistent across samples)

        # Iterate over each feature within the sequence based on the actual number of features
        for j in range(actual_n_features):
            # Safely get the feature name
            feature_name = feature_names[j] if j < len(feature_names) else f'feature_{j}'
            feature_sequence = sequence_data[:, j] # Shape: (seq_len,)

            # Basic statistics
            sample_aggregated_features.append(np.mean(feature_sequence))
            if i == 0: current_sample_aggregated_feature_names.append(f'{feature_name}_mean')

            sample_aggregated_features.append(np.std(feature_sequence))
            if i == 0: current_sample_aggregated_feature_names.append(f'{feature_name}_std')

            sample_aggregated_features.append(np.max(feature_sequence))
            if i == 0: current_sample_aggregated_feature_names.append(f'{feature_name}_max')

            sample_aggregated_features.append(np.min(feature_sequence))
            if i == 0: current_sample_aggregated_feature_names.append(f'{feature_name}_min')

            # Range (Max - Min)
            sample_aggregated_features.append(np.max(feature_sequence) - np.min(feature_sequence))
            if i == 0: current_sample_aggregated_feature_names.append(f'{feature_name}_range')

            # First and Last values
            # Ensure sequence_data is not empty before accessing elements
            if seq_len > 0:
                sample_aggregated_features.append(feature_sequence[0])
                if i == 0: current_sample_aggregated_feature_names.append(f'{feature_name}_first')

                sample_aggregated_features.append(feature_sequence[-1])
                if i == 0: current_sample_aggregated_feature_names.append(f'{feature_name}_last')
            else:
                 sample_aggregated_features.extend([np.nan, np.nan]) # Append NaN if sequence is empty
                 if i == 0:
                      current_sample_aggregated_feature_names.append(f'{feature_name}_first')
                      current_sample_aggregated_feature_names.append(f'{feature_name}_last')


            # Change (Last - First)
            if seq_len > 0: # Needs at least one point
                 sample_aggregated_features.append(feature_sequence[-1] - feature_sequence[0])
                 if i == 0: current_sample_aggregated_feature_names.append(f'{feature_name}_change')
            else:
                 sample_aggregated_features.append(np.nan)
                 if i == 0: current_sample_aggregated_feature_names.append(f'{feature_name}_change')


            # Percentage Change ((Last - First) / First), handle division by zero
            if seq_len > 0 and feature_sequence[0] != 0 and not np.isnan(feature_sequence[0]):
                sample_aggregated_features.append((feature_sequence[-1] - feature_sequence[0]) / feature_sequence[0])
            else:
                sample_aggregated_features.append(np.nan) # Use np.nan for invalid percentage change
            if i == 0: current_sample_aggregated_feature_names.append(f'{feature_name}_pct_change')


            # Slope (simple linear regression slope)
            # Check for sufficient data points and handle NaNs
            if seq_len >= 2:
                 try:
                      # Ensure data is numeric before polyfit
                      numeric_feature_sequence = feature_sequence.astype(np.float64)
                      # Handle potential NaNs in polyfit
                      valid_indices = ~np.isnan(numeric_feature_sequence)
                      if np.sum(valid_indices) >= 2: # Need at least 2 non-NaN points for slope
                          slope, intercept = np.polyfit(np.arange(seq_len)[valid_indices], numeric_feature_sequence[valid_indices], 1)
                          sample_aggregated_features.append(slope)
                      else:
                           sample_aggregated_features.append(np.nan) # Not enough valid points for slope
                 except Exception: # Catch any potential errors in polyfit
                      sample_aggregated_features.append(np.nan)
            else:
                 sample_aggregated_features.append(np.nan) # Slope is undefined for single point or empty sequence
            if i == 0: current_sample_aggregated_feature_names.append(f'{feature_name}_slope')


            # Skewness and Kurtosis (measure of shape of the distribution)
            # Need at least 3 points for skew, 4 for kurtosis
            # Ensure data is numeric and handle NaNs for skew/kurtosis
            numeric_feature_sequence = feature_sequence.astype(np.float64) # Convert once
            if seq_len >= 3:
                 sample_aggregated_features.append(skew(numeric_feature_sequence, nan_policy='omit')) # Use nan_policy='omit'
                 if i == 0: current_sample_aggregated_feature_names.append(f'{feature_name}_skew')
            else:
                 sample_aggregated_features.append(np.nan)

            if seq_len >= 4:
                 sample_aggregated_features.append(kurtosis(numeric_feature_sequence, nan_policy='omit')) # Use nan_policy='omit'
                 if i == 0: current_sample_aggregated_feature_names.append(f'{feature_name}_kurtosis')
            else:
                 sample_aggregated_features.append(np.nan)


        # Append the aggregated features for the current sample
        aggregated_features_list.append(sample_aggregated_features)
        if i == 0: aggregated_feature_names = current_sample_aggregated_feature_names # Store names from the first sample


    # Convert the list of lists to a NumPy array
    # Use dtype=float64 directly, as np.nan will handle missing values
    try:
        aggregated_features_np = np.array(aggregated_features_list, dtype=np.float64)
    except ValueError:
        print("Error: Could not convert aggregated features to float64. Check data types in input sequences.")
        # Fallback to object or return empty if critical
        return np.array([])


    print(f"Finished creating aggregated features. Shape: {aggregated_features_np.shape}")
    print(f"Number of aggregated features: {len(aggregated_feature_names)}")
    # print(f"Aggregated feature names: {aggregated_feature_names}") # Optional: print feature names

    # Return the aggregated NumPy array
    return aggregated_features_np


# In[ ]:


import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# Assuming the following functions are defined in preceding cells:
# sliding_window
# transform_sequence_data
# transform_window_list_to_numpy_and_index
# scale_features # Although scaling of aggregated features happens later, the function is general.
# prepare_data_for_clustering # We are essentially extracting parts of this for the new function.


def prepare_3d_data(df, continuous_features, categorical_features, date_features, sequence_length=60, nan_threshold=200):
    """
    Performs data preparation steps up to creating and filtering the 3D NumPy array
    without performing clustering or saving results.

    Args:
        df (pd.DataFrame): The original dataframe.
        continuous_features (list): List of top 50 features to include.
        categorical_features (list): List of categorical feature names.
        date_features (list): List of date-related feature names.
        sequence_length (int): The size of the sliding window.
        nan_threshold (int): Threshold for dropping columns based on NaN count.

    Returns:
        tuple:
            - X_3d_numpy_filtered (np.ndarray): Filtered 3D NumPy array (n_samples, seq_len, n_features).
            - original_indices_filtered (pd.DatetimeIndex): DatetimeIndex corresponding to the start time of each sequence in X_3d_numpy_filtered.
            - original_feature_names_list (list): List of feature names present in the DataFrame *before* sliding window and dropping rows with NaNs.
    """
    print("--- Starting 3D Data Preparation ---")

    # Define the initial list of selected columns
    # Ensure these columns exist in the input df before selection
    initial_selected_columns = ["close"] + ['MA_t_6', 'MA_t_24', 'MA_t_72', 'MA_t_168'] + ['upper', 'lower'] + continuous_features + categorical_features + date_features

    # Filter columns that actually exist in the input DataFrame
    existing_selected_columns = [col for col in initial_selected_columns if col in df.columns]
    if len(existing_selected_columns) != len(initial_selected_columns):
         missing_cols = list(set(initial_selected_columns) - set(existing_selected_columns))
         print(f"Warning: The following selected columns are not found in the input DataFrame and will be skipped: {missing_cols}")

    if not existing_selected_columns:
         print("Error: No valid columns selected from the input DataFrame. Returning empty results.")
         return np.array([]), pd.DatetimeIndex([]), []

    df_processed = df[existing_selected_columns].copy()

    # Convert categorical features to category dtype if they exist and are selected
    existing_categorical_features_in_processed = [col for col in categorical_features if col in df_processed.columns]
    # Remove duplicates from existing_categorical_features_in_processed
    existing_categorical_features_in_processed = list(dict.fromkeys(existing_categorical_features_in_processed))
    if existing_categorical_features_in_processed:
        df_processed.loc[:, existing_categorical_features_in_processed] = df_processed.loc[:, existing_categorical_features_in_processed].astype('category')
        print(f"Converted columns to 'category' dtype: {existing_categorical_features_in_processed}")


    # Drop columns with excessive NaNs
    nan_counts = df_processed.isnull().sum()
    columns_to_drop = nan_counts[nan_counts >= nan_threshold].index.tolist()
    if columns_to_drop:
        print(f"Dropping columns with >= {nan_threshold} NaNs: {columns_to_drop}")
        df_processed = df_processed.drop(columns=columns_to_drop)

    # Drop rows with any remaining NaNs
    initial_rows = df_processed.shape[0]
    # Check if df_processed has a DatetimeIndex before dropping NaNs, or just use default index
    if isinstance(df_processed.index, pd.DatetimeIndex):
        original_index_before_dropna = df_processed.index
        df_processed = df_processed.dropna().copy()
        # Keep track of the index *after* dropping NaNs but *before* sliding window
        original_index_after_dropna = df_processed.index
    else:
        print("Warning: Input DataFrame does not have a DatetimeIndex. Using default index after dropna.")
        df_processed = df_processed.dropna().copy()
        original_index_after_dropna = df_processed.index # This will be a RangeIndex or similar


    print(f"Dropped {initial_rows - df_processed.shape[0]} rows with NaNs. Remaining rows: {df_processed.shape[0]}")

    # Check if any data remains after dropping NaNs
    if df_processed.shape[0] == 0:
        print("Error: No data remains after dropping NaNs. Returning empty results.")
        return np.array([]), pd.DatetimeIndex([]), []


    # Store the list of feature names after dropping columns but before sliding window
    # This list should correspond to the features in the last dimension of the 3D NumPy array
    original_feature_names_list = df_processed.columns.tolist()
    print(f"Features remaining after NaN processing: {len(original_feature_names_list)}")


    # Step 2: Create sliding windows
    print(f"Creating sliding windows with length {sequence_length}...")
    X_window_list = sliding_window(df_processed, sequence_length)
    print(f"Created {len(X_window_list)} sliding windows.")

    if not X_window_list:
         print("Warning: No sliding windows created. Returning empty results.")
         return np.array([]), pd.DatetimeIndex([]), original_feature_names_list


    # Step 3: Convert sliding window list to 3D NumPy array and get original indices
    # The original indices here correspond to the END time of each window by default in transform_window_list_to_numpy_and_index.
    # Let's adjust transform_window_list_to_numpy_and_index to return the START time index for consistency if needed later,
    # but for filtering it's usually the end time index that matters.
    # The clustering pipeline used the END time index to align with the aggregated features.
    # Let's stick to the END time index for filtering consistency with the original logic.
    print("Converting window list to 3D NumPy array and extracting original indices...")
    X_3d_numpy_all_sequences, original_end_indices_all_sequences = transform_window_list_to_numpy_and_index(X_window_list)

    print(f"Original X_3d_numpy shape: {X_3d_numpy_all_sequences.shape}")
    print(f"Original end indices length: {len(original_end_indices_all_sequences)}")

    # The rows in X_3d_numpy_all_sequences correspond to the windows created from df_processed.
    # The index of X_3d_numpy_all_sequences is implicitly 0 to n_samples-1.
    # original_end_indices_all_sequences gives the end timestamp of each of these windows.
    # The start index of each window is also important.
    # A window starting at index k in df_processed corresponds to the sequence X_3d_numpy_all_sequences[k].
    # Its start time is df_processed.index[k] and its end time is df_processed.index[k + sequence_length - 1].
    # The `transform_window_list_to_numpy_and_index` returns the END index.
    # Let's assume for downstream tasks we need the START index of each sequence.
    # We can reconstruct the start indices:
    original_start_indices_all_sequences = df_processed.index[:len(X_window_list)]


    # Filter the 3D NumPy array and its original indices to match any subsequent filtering (e.g., from aggregation)
    # In the original pipeline, filtering happened after aggregation due to NaN removal in aggregated features.
    # However, this new function only prepares the 3D data. It should return the 3D data and its indices *after*
    # initial NaN handling and column selection from the original df, but *before* any filtering based on aggregated features.
    # So, the filtering step based on `agg_X_scaled.index` is *not* needed here.

    # The output should be the 3D data and its corresponding start indices (or end indices, be consistent)
    # after dropping NaNs from the original DF and creating windows.
    # Let's return the start indices as they are often used as time_idx for the sequence.
    X_3d_numpy_filtered = X_3d_numpy_all_sequences # No filtering based on aggregation in this function
    original_indices_filtered = original_start_indices_all_sequences # Return the start indices


    print(f"Final X_3d_numpy shape: {X_3d_numpy_filtered.shape}")
    print(f"Final original_indices (start times) length: {len(original_indices_filtered)}")

    return X_3d_numpy_filtered, original_indices_filtered, original_feature_names_list


# # Task
# Modify the `perform_clustering_on_subset` function in the file "u_vzq_aaG3fq" to improve clustering by using a self-supervised contrastive learning approach (SimCLR/TS2Vec-like) with a 1D-CNN encoder and InfoNCE loss to generate latent vectors that emphasize individual differences within the same category, replacing the current Autoencoder-based method. The modified function should train the contrastive learning model, extract latent vectors, and then perform clustering (HDBSCAN or KMeans) on these vectors.

# ## 新しいエンコーダーモデル（1d-cnn）の定義
# 
# ### Subtask:
# Define a 1D-CNN based encoder model class for self-supervised contrastive learning.
# 

# **Reasoning**:
# The subtask is to define a 1D-CNN based encoder class. I will define the `CNNEncoder1D` class with convolutional and linear layers as described in the instructions.
# 
# 

# **Reasoning**:
# The previous command failed because `nn` from `torch.nn` was not defined. I need to import `torch.nn` to fix this error.
# 
# 

# In[ ]:


import torch.nn as nn

class CNNEncoder1D(nn.Module):
    def __init__(self, input_channels, seq_len, latent_dim):
        super().__init__()
        self.input_channels = input_channels
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        # Define convolutional layers
        # Input shape: (batch_size, input_channels, seq_len)
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3) # Output: (batch_size, 64, seq_len/2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1) # Output: (batch_size, 64, seq_len/4)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2) # Output: (batch_size, 128, seq_len/8)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1) # Output: (batch_size, 128, seq_len/16)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1) # Output: (batch_size, 256, seq_len/32)
        self.relu3 = nn.ReLU()
        # No pooling after last conv layer before flattening

        # Calculate the size of the flattened layer
        # Need a dummy forward pass or formula to calculate output shape after convolutions
        # Let's calculate based on the provided kernel_size, stride, padding
        def calculate_output_length(input_length, kernel_size, stride, padding):
             return (input_length + 2 * padding - kernel_size) // stride + 1

        conv1_out_len = calculate_output_length(seq_len, 7, 2, 3)
        pool1_out_len = calculate_output_length(conv1_out_len, 3, 2, 1)
        conv2_out_len = calculate_output_length(pool1_out_len, 5, 2, 2)
        pool2_out_len = calculate_output_length(conv2_out_len, 3, 2, 1)
        conv3_out_len = calculate_output_length(pool2_out_len, 3, 2, 1)


        # The flattened size is channels of last conv layer * its output length
        self.flattened_size = 256 * conv3_out_len

        # Define fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, latent_dim)

    def forward(self, x):
        # Input x shape: (batch_size, n_features, seq_len)
        # Ensure input features are treated as channels for Conv1D
        # If input is (batch_size, seq_len, n_features), transpose it
        if x.shape[1] == self.seq_len and x.shape[2] == self.input_channels:
             x = x.transpose(1, 2) # Transpose to (batch_size, n_features, seq_len)
        elif x.shape[1] != self.input_channels or x.shape[2] != self.seq_len:
             raise ValueError(f"Input shape {x.shape} does not match expected (batch_size, {self.input_channels}, {self.seq_len}) or (batch_size, {self.seq_len}, {self.input_channels})")


        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))

        # Flatten the output
        x = x.view(x.size(0), -1) # Flatten starting from the first dimension after batch size

        # Pass through fully connected layers
        x = self.relu4(self.fc1(x))
        latent_vector = self.fc2(x)

        return latent_vector


# ## コントラスト学習用データ拡張の実装
# 
# ### Subtask:
# 時系列データに対して、ランダムクロップ、ガウスノイズ追加などのデータ拡張を行う関数を実装します。
# 

# ## コントラスト学習モジュールの実装
# 
# ### Subtask:
# エンコーダー、データ拡張、InfoNCE損失を組み合わせたPyTorch Lightningモジュールを定義します。
# 

# **Reasoning**:
# Define the ContrastiveLearningModule class inheriting from LightningModule, including the init, forward, training_step, and configure_optimizers methods as specified in the instructions.
# 
# 

# In[ ]:


get_ipython().run_line_magic('pip', 'install pytorch-lightning')

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
# Assuming CNNEncoder1D is accessible from a previous cell

class ContrastiveLearningModule(pl.LightningModule):
    def __init__(self, encoder: nn.Module, learning_rate: float = 1e-3, temperature: float = 0.07):
        """
        Initializes the ContrastiveLearningModule.

        Args:
            encoder (nn.Module): The encoder model (e.g., CNNEncoder1D instance).
                                 It should output latent vectors.
            learning_rate (float): The learning rate for the optimizer.
            temperature (float): The temperature parameter for the InfoNCE loss.
        """
        super().__init__()
        self.save_hyperparameters(ignore=['encoder']) # Save other hyperparameters
        self.encoder = encoder
        self.learning_rate = learning_rate
        self.temperature = temperature

    def forward(self, x):
        """
        Passes the input through the encoder to get latent vectors.

        Args:
            x (torch.Tensor): Input tensor (batch_size, n_features, seq_len).

        Returns:
            torch.Tensor: Latent vectors (batch_size, latent_dim).
        """
        # Ensure input shape is compatible with the encoder if needed
        # Assuming encoder handles the expected input shape (batch_size, n_features, seq_len)
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        """
        Performs a training step calculating the InfoNCE loss.

        Args:
            batch (tuple): A tuple containing the original data and its augmentations,
                           e.g., (x, x_aug1, x_aug2).
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The calculated InfoNCE loss for the batch.
        """
        # Assuming batch contains at least two augmented views (x_aug1, x_aug2)
        # The first element x is the original data, which might not be used directly in training_step
        # if only augmented views are needed for contrastive loss.
        # Let's assume batch is (x, x_aug1, x_aug2) or (x_aug1, x_aug2)
        # We need at least two augmented views. Let's take the first two elements as views.
        if len(batch) < 2:
             raise ValueError("Batch must contain at least two augmented views for contrastive learning.")

        x_aug1 = batch[0]
        x_aug2 = batch[1]

        # Move data to the correct device
        x_aug1 = x_aug1.to(self.device)
        x_aug2 = x_aug2.to(self.device)

        # Get latent vectors from the encoder
        z1 = self.encoder(x_aug1) # Shape: (batch_size, latent_dim)
        z2 = self.encoder(x_aug2) # Shape: (batch_size, latent_dim)

        # Normalize latent vectors
        z1 = F.normalize(z1, dim=1) # Normalize along the latent dimension
        z2 = F.normalize(z2, dim=1)

        # Calculate cosine similarity matrix between z1 and z2
        # sim_matrix shape: (batch_size, batch_size)
        # Entry (i, j) is cosine similarity between z1[i] and z2[j]
        sim_matrix = torch.matmul(z1, z2.T) / self.temperature

        # InfoNCE Loss calculation
        # The diagonal elements are the positive pairs (z1[i] vs z2[i])
        positive_similarity = torch.diag(sim_matrix) # Shape: (batch_size,)

        # All other elements are negative pairs (z1[i] vs z2[j] where i != j)
        # For InfoNCE, the loss for z1 is based on its similarity to its positive pair (z2[i])
        # relative to its similarity to all other samples in z2 (negative pairs).
        # Similarly for z2 based on z1.

        # Loss for z1: log(exp(pos_sim) / sum(exp(all_sim_in_row)))
        # The sum includes the positive pair, so the numerator is one term in the sum.
        # We need log-softmax for numerical stability.
        # The target for log-softmax is the index of the positive sample in the row.
        # For z1, the positive pair is at index 'i' in the i-th row of sim_matrix (z1[i] vs z2[i]).
        # So, the target is the identity matrix (or a vector of batch indices).

        # Loss for z1
        # The target is that z1[i] should be most similar to z2[i].
        # In sim_matrix[i, :], the target is index i.
        loss_z1 = F.cross_entropy(sim_matrix, torch.arange(sim_matrix.size(0), device=self.device))

        # Loss for z2: log(exp(pos_sim) / sum(exp(all_sim_in_col)))
        # This is symmetric. The target for z2 is that z2[i] should be most similar to z1[i].
        # In sim_matrix[:, i], the target is index i.
        loss_z2 = F.cross_entropy(sim_matrix.T, torch.arange(sim_matrix.size(0), device=self.device))

        # Total loss is the average of the two symmetric losses
        loss = (loss_z1 + loss_z2) / 2.0

        # Log the loss
        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer for the model.

        Returns:
            torch.optim.Optimizer: The optimizer instance.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    # Optional: Add validation_step and test_step if needed for logging purposes,
    # but the core contrastive learning happens in training_step.
    # def validation_step(self, batch, batch_idx):
    #     # Similar logic as training_step but for validation data
    #     # Calculate loss and log validation_loss
    #     pass

    # def test_step(self, batch, batch_idx):
    #      # Similar logic as training_step but for test data
    #      # Calculate loss and log test_loss
    #      pass


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class CNNEncoder1D(pl.LightningModule):
    def __init__(self, input_channels, output_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.fc = nn.Linear(256, output_dim) # Assuming the output of conv layers will be pooled or flattened to 256 features before the linear layer

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # Global average pooling
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# **Reasoning**:
# The CNNEncoder1D module is defined. To train the model, I need to prepare the data using a PyTorch LightningDataModule. This involves creating a custom Dataset and DataLoader for the time series data. I will then define the DataModule to handle the train and validation splits.
# 
# 

# In[ ]:


from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class TimeSeriesDataModule(pl.LightningDataModule):
    def __init__(self, data, labels, batch_size=32):
        super().__init__()
        self.data = data
        self.labels = labels
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Split data into train and validation sets
        self.train_data, self.val_data, self.train_labels, self.val_labels = train_test_split(
            self.data, self.labels, test_size=0.2, random_state=42
        )

    def train_dataloader(self):
        train_dataset = TimeSeriesDataset(self.train_data, self.train_labels)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        val_dataset = TimeSeriesDataset(self.val_data, self.val_labels)
        return DataLoader(val_dataset, batch_size=self.batch_size)


# ## コントラスト学習用データ拡張の実装
# 
# ### Subtask:
# 時系列データに対して、ランダムクロップ、ガウスノイズ追加などのデータ拡張を行う関数を実装します。

# In[ ]:


import torch
import numpy as np

def random_crop(sequence: torch.Tensor, crop_length: int):
    """
    時系列データに対してランダムクロップを行います。

    Args:
        sequence (torch.Tensor): 入力時系列データ (seq_len, n_features) または (n_features, seq_len)。
                                  バッチ次元は想定していません。
        crop_length (int): クロップ後の長さ。

    Returns:
        torch.Tensor: クロップされた時系列データ。
    """
    seq_len = sequence.shape[-1] # Assume seq_len is the last dimension
    if crop_length > seq_len:
        raise ValueError(f"Crop length ({crop_length}) cannot be greater than sequence length ({seq_len}).")
    if crop_length == seq_len:
        return sequence # No cropping needed

    start_idx = torch.randint(0, seq_len - crop_length + 1, (1,)).item()
    # Determine which dimension is seq_len
    if sequence.shape[0] == seq_len: # Shape (seq_len, n_features)
         return sequence[start_idx : start_idx + crop_length, :]
    elif sequence.shape[-1] == seq_len: # Shape (n_features, seq_len) or (batch, n_features, seq_len)
         # If it's 2D (n_features, seq_len), crop last dim
         if sequence.ndim == 2:
             return sequence[:, start_idx : start_idx + crop_length]
         # If it's 3D (batch, n_features, seq_len), crop last dim for all in batch
         elif sequence.ndim == 3:
              return sequence[:, :, start_idx : start_idx + crop_length]
         else:
              raise ValueError(f"Unsupported input shape for random_crop: {sequence.shape}")
    else:
         raise ValueError(f"Could not identify sequence length dimension in shape {sequence.shape}")


def add_gaussian_noise(sequence: torch.Tensor, std: float = 0.01):
    """
    時系列データにガウスノイズを追加します。

    Args:
        sequence (torch.Tensor): 入力時系列データ。
        std (float): ノイズの標準偏差。

    Returns:
        torch.Tensor: ノイズが追加された時系列データ。
    """
    noise = torch.randn_like(sequence) * std
    return sequence + noise

def jitter(sequence: torch.Tensor, std: float = 0.01):
    """
    時系列データにジッター（各タイムステップに独立したノイズ）を追加します。
    Alias for add_gaussian_noise.
    """
    return add_gaussian_noise(sequence, std)


def random_scale(sequence: torch.Tensor, scale_range=(0.9, 1.1)):
    """
    時系列データ全体をランダムなスケールファクターでスケーリングします。

    Args:
        sequence (torch.Tensor): 入力時系列データ。
        scale_range (tuple): スケールファクターの範囲 (min_scale, max_scale)。

    Returns:
        torch.Tensor: スケーリングされた時系列データ。
    """
    scale_factor = torch.rand(1) * (scale_range[1] - scale_range[0]) + scale_range[0]
    return sequence * scale_factor

def random_permutation(sequence: torch.Tensor, max_segments: int = 5):
    """
    時系列データをランダムな数のセグメントに分割し、それらをランダムに並べ替えます。

    Args:
        sequence (torch.Tensor): 入力時系列データ (seq_len, n_features) または (n_features, seq_len)。
                                  バッチ次元は想定していません。
        max_segments (int): 分割する最大セグメント数。

    Returns:
        torch.Tensor: 並べ替えられた時系列データ。
    """
    seq_len = sequence.shape[-1] # Assume seq_len is the last dimension
    if seq_len < 2 or max_segments <= 1:
         return sequence # Cannot permute

    num_segments = torch.randint(2, max_segments + 1, (1,)).item()
    min_segment_length = seq_len // num_segments

    if min_segment_length == 0: # Not enough points for requested segments
         return sequence


    segment_boundaries = sorted(torch.randperm(seq_len - 1)[:num_segments - 1].tolist())
    segment_boundaries = [0] + segment_boundaries + [seq_len]

    segments = []
    for i in range(len(segment_boundaries) - 1):
        start_idx = segment_boundaries[i]
        end_idx = segment_boundaries[i+1]
        if start_idx < end_idx:
             # Determine which dimension is seq_len
             if sequence.shape[0] == seq_len: # Shape (seq_len, n_features)
                  segments.append(sequence[start_idx:end_idx, :])
             elif sequence.shape[-1] == seq_len: # Shape (n_features, seq_len)
                  segments.append(sequence[:, start_idx:end_idx])
             else:
                  raise ValueError(f"Could not identify sequence length dimension in shape {sequence.shape} for permutation.")
        # else: segment is empty, skip


    # Permute the segments
    if segments:
         permuted_segments = torch.randperm(len(segments)).tolist()
         reconstructed_sequence = torch.cat([segments[i] for i in permuted_segments], dim=-1 if sequence.shape[-1] == seq_len else 0) # Concatenate along seq_len dim
         # Ensure the reconstructed sequence has the original length (handle potential off-by-one from boundaries)
         # This simple permutation might not preserve exact length if segments are empty or boundaries are tricky.
         # A more robust method would ensure segments cover the whole sequence exactly.
         # For simplicity, let's assume ideal segmentation for now.

         # Simple check for length mismatch (might indicate issue with segmentation logic)
         if reconstructed_sequence.shape[-1 if sequence.shape[-1] == seq_len else 0] != seq_len:
              # print(f"Warning: Permuted sequence length mismatch. Original: {seq_len}, Reconstructed: {reconstructed_sequence.shape[-1 if sequence.shape[-1] == seq_len else 0]}. Returning original sequence.")
              return sequence # Return original if permutation failed to preserve length


         return reconstructed_sequence
    else:
         return sequence # Return original if no segments were created


# Example usage (assuming a dummy sequence):
# dummy_sequence = torch.randn(100, 10) # Shape (seq_len, n_features)
# cropped_seq = random_crop(dummy_sequence, 80)
# noise_seq = add_gaussian_noise(dummy_sequence, std=0.05)
# scaled_seq = random_scale(dummy_sequence, scale_range=(0.8, 1.2))
# permuted_seq = random_permutation(dummy_sequence, max_segments=3)

# print("Original shape:", dummy_sequence.shape)
# print("Cropped shape:", cropped_seq.shape)
# print("Noise shape:", noise_seq.shape)
# print("Scaled shape:", scaled_seq.shape)
# print("Permuted shape:", permuted_seq.shape)


# Note: These augmentations are designed for a single sequence (2D or 3D without batch).
# When used in a DataLoader, they should be applied to each sample individually.
# The ContrastiveLearningModule will receive batches and should apply augmentations
# to the batch, or the DataLoader should handle batching of augmented samples.
# A common approach is to apply augmentations within the Dataset's __getitem__ or
# have a custom collate_fn in the DataLoader.
# For SimCLR, we need two different augmented views of the *same* sample.
# So, the dataset/dataloader should yield (sample_aug1, sample_aug2).


# ## コントラスト学習モジュールの実装
# 
# ### Subtask:
# エンコーダー、データ拡張、InfoNCE 損失を組み合わせた PyTorch Lightning モジュールを定義します。

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
# Assuming CNNEncoder1D is accessible from a previous cell
# Assuming data augmentation functions (random_crop, add_gaussian_noise, etc.) are accessible

class ContrastiveLearningModule(pl.LightningModule):
    def __init__(self, encoder: nn.Module, learning_rate: float = 1e-3, temperature: float = 0.07, augmentation_strategies: list = None):
        """
        Initializes the ContrastiveLearningModule.

        Args:
            encoder (nn.Module): The encoder model (e.g., CNNEncoder1D instance).
                                 It should output latent vectors.
            learning_rate (float): The learning rate for the optimizer.
            temperature (float): The temperature parameter for the InfoNCE loss.
            augmentation_strategies (list, optional): A list of tuples, where each tuple
                                                     is (augmentation_function, kwargs).
                                                     If None, default augmentations will be used.
        """
        super().__init__()
        # Save hyperparameters, ignoring the complex encoder and augmentation_strategies objects
        self.save_hyperparameters(ignore=['encoder', 'augmentation_strategies'])
        self.encoder = encoder
        self.learning_rate = learning_rate
        self.temperature = temperature

        # Store augmentation strategies
        if augmentation_strategies is None:
            # Define some default augmentation strategies if none are provided
            # These functions should be implemented elsewhere and available in the scope
            self.augmentation_strategies = [
                (add_gaussian_noise, {'std': 0.01}),
                (random_scale, {'scale_range': (0.9, 1.1)}),
                # (random_crop, {'crop_length': int(self.hparams.seq_len * 0.8)}) # Requires seq_len hyperparameter
                # (random_permutation, {'max_segments': 5}) # Requires seq_len hyperparameter
            ]
            print("Using default augmentation strategies.")
        else:
            self.augmentation_strategies = augmentation_strategies
            print(f"Using {len(self.augmentation_strategies)} provided augmentation strategies.")

        # Check if required augmentation functions are available
        for aug_fn, _ in self.augmentation_strategies:
            if not callable(aug_fn):
                raise TypeError(f"Provided augmentation strategy {aug_fn} is not a callable function.")
            # Further checks like function signature could be added


    def forward(self, x):
        """
        Passes the input through the encoder to get latent vectors.

        Args:
            x (torch.Tensor): Input tensor (batch_size, n_features, seq_len).
                              Ensure this shape matches the encoder's expected input.

        Returns:
            torch.Tensor: Latent vectors (batch_size, latent_dim).
        """
        # Assuming encoder expects (batch_size, n_features, seq_len) based on CNNEncoder1D
        # If input comes as (batch_size, seq_len, n_features), transpose it here
        if x.ndim == 3 and x.shape[1] != self.encoder.input_channels: # Assuming input_channels is n_features
             # Try transposing if the shape looks like (batch, seq_len, n_features)
             if x.shape[2] == self.encoder.input_channels:
                  x = x.transpose(1, 2) # Transpose to (batch_size, n_features, seq_len)
             else:
                  print(f"Warning: Input shape {x.shape} does not match expected encoder input (batch_size, {self.encoder.input_channels}, seq_len).")
                  # Proceed, but expect potential errors in the encoder
                  pass


        return self.encoder(x)

    def apply_augmentation(self, sequence: torch.Tensor):
        """
        Applies a randomly chosen augmentation strategy to a single sequence.

        Args:
            sequence (torch.Tensor): A single time series sequence (n_features, seq_len).
                                     Batch dimension is NOT expected.

        Returns:
            torch.Tensor: The augmented sequence.
        """
        if not self.augmentation_strategies:
            return sequence # Return original if no augmentations are defined

        # Randomly select one augmentation strategy
        aug_fn, kwargs = self.augmentation_strategies[torch.randint(0, len(self.augmentation_strategies), (1,)).item()]

        # Apply the selected augmentation
        try:
            # Ensure the sequence is on the correct device for augmentation functions
            # Many augmentation functions assume CPU or handle device internally.
            # Pass to function and let it handle device or ensure it's on CPU before passing.
            # A safer approach is to move data to CPU for augmentation if functions are not device-aware.
            # Let's assume augmentation functions work on CPU and move data back after.
            device_orig = sequence.device
            augmented_sequence = aug_fn(sequence.cpu(), **kwargs).to(device_orig)
            return augmented_sequence

        except Exception as e:
            print(f"Warning: Failed to apply augmentation {aug_fn.__name__} with kwargs {kwargs}: {e}. Returning original sequence.")
            return sequence # Return original sequence if augmentation fails


    def training_step(self, batch, batch_idx):
        """
        Performs a training step calculating the InfoNCE loss.

        Args:
            batch (tuple or torch.Tensor): A batch of original time series data.
                                  If from DataLoader with TensorDataset(X_tensor), it's a tuple (X_tensor,).
                                  If from custom dataset yielding (x,), it's a tuple (x,).
                                  We expect a tuple containing the data tensor as the first element.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The calculated InfoNCE loss for the batch.
        """
        # The batch contains the original data (batch_size, n_features, seq_len)
        # Unpack the data tensor from the batch tuple
        if isinstance(batch, (list, tuple)):
             x_original = batch[0] # Assuming the data tensor is the first element
        else:
             x_original = batch # Assume batch is already the data tensor (e.g., if batch_size=1 and tuple unpacking was skipped)
             # print("Warning: Batch is not a tuple/list. Assuming it's the data tensor directly.")


        # Move original data to the device
        x_original = x_original.to(self.device)

        # Apply two different augmentations to the batch
        # Note: Applying augmentations batch-wise might require vectorized augmentation functions.
        # If augmentation functions work on single sequences, we need to iterate or use map.
        # Let's modify this to iterate for clarity, assuming augmentation functions work on (n_features, seq_len).
        # This is inefficient; vectorized augmentations or custom DataLoader collate_fn are better for performance.

        batch_size = x_original.size(0)
        x_aug1_list = []
        x_aug2_list = []

        for i in range(batch_size):
            # Apply two augmentations to the i-th sequence in the batch
            seq_i = x_original[i] # Shape (n_features, seq_len)
            x_aug1_list.append(self.apply_augmentation(seq_i))
            x_aug2_list.append(self.apply_augmentation(seq_i))

        # Stack the augmented sequences back into batches
        # Ensure all augmented sequences have the same shape before stacking
        # This is particularly important for `random_crop` if not handled carefully.
        # For robustness, pad or resize crops if necessary, or ensure augmentations preserve shape.
        # Assuming augmentations preserve shape (n_features, seq_len) for now, or random_crop is not used by default.

        # Check shapes before stacking (basic check)
        # Expected shape for stacking is (n_features, seq_len) for each sequence in the list
        expected_seq_shape = x_original.shape[1:] # (n_features, seq_len)

        if not all(aug_seq.shape == expected_seq_shape for aug_seq in x_aug1_list) or \
           not all(aug_seq.shape == expected_seq_shape for aug_seq in x_aug2_list):
             print(f"Warning: Augmented sequence shapes mismatch original shape {expected_seq_shape}. Skipping batch.")
             # Return 0 loss or raise error. Returning 0 loss allows training to continue but might mask issues.
             return torch.tensor(0.0, device=self.device) # Return zero loss

        x_aug1 = torch.stack(x_aug1_list, dim=0) # Shape (batch_size, n_features, seq_len)
        x_aug2 = torch.stack(x_aug2_list, dim=0) # Shape (batch_size, n_features, seq_len)

        # Move augmented data to the correct device (already handled in apply_augmentation if it moves to CPU)
        # Let's ensure they are on the module's device
        x_aug1 = x_aug1.to(self.device)
        x_aug2 = x_aug2.to(self.device)


        # Get latent vectors from the encoder
        # Ensure encoder input shape matches x_aug1, x_aug2 shape (batch_size, n_features, seq_len)
        try:
             z1 = self.encoder(x_aug1) # Shape: (batch_size, latent_dim)
             z2 = self.encoder(x_aug2) # Shape: (batch_size, latent_dim)
        except Exception as e:
             print(f"Error during encoder forward pass: {e}. Skipping batch.")
             return torch.tensor(0.0, device=self.device)


        # Normalize latent vectors
        # Handle potential NaN/Inf in latent vectors after encoding
        if torch.isnan(z1).any() or torch.isinf(z1).any() or torch.isnan(z2).any() or torch.isinf(z2).any():
             print("Warning: NaN or Inf detected in latent vectors after encoding. Skipping batch.")
             return torch.tensor(0.0, device=self.device)


        # Add a small epsilon before normalization if any vector is zero, to avoid NaNs
        epsilon = 1e-8
        z1 = F.normalize(z1 + epsilon, dim=1) # Normalize along the latent dimension
        z2 = F.normalize(z2 + epsilon, dim=1)

        # Calculate cosine similarity matrix between z1 and z2
        # sim_matrix shape: (batch_size, batch_size)
        # Entry (i, j) is cosine similarity between z1[i] and z2[j]
        # Ensure temperature is not zero or too close to zero
        effective_temperature = max(self.temperature, 1e-8) # Avoid division by zero

        # Calculate similarity matrix
        # Clamp values before matmul if they could be extremely large/small
        # Latent vectors are normalized, so this is less likely, but adding a safeguard.
        z1 = torch.clamp(z1, min=-1e6, max=1e6)
        z2 = torch.clamp(z2, min=-1e6, max=1e6)

        sim_matrix = torch.matmul(z1, z2.T) / effective_temperature

        # Clamp similarity matrix values before cross_entropy for numerical stability
        sim_matrix = torch.clamp(sim_matrix, min=-1e6, max=1e6)


        # InfoNCE Loss calculation
        # The diagonal elements are the positive pairs (z1[i] vs z2[i])
        # The target for F.cross_entropy(input, target) where input is logits (sim_matrix)
        # is the class index. For the i-th row of sim_matrix, the positive sample
        # (z2[i]) is at column index i. So the target is a vector of indices [0, 1, 2, ..., batch_size-1].

        # Create target indices [0, 1, ..., batch_size-1]
        targets = torch.arange(batch_size, device=self.device)

        # Calculate cross-entropy loss for z1 -> z2
        # loss_z1 = -torch.mean(F.log_softmax(sim_matrix, dim=1)[targets, targets]) # Manual InfoNCE
        loss_z1 = F.cross_entropy(sim_matrix, targets) # Using F.cross_entropy directly

        # Calculate cross-entropy loss for z2 -> z1 (symmetric loss)
        # loss_z2 = -torch.mean(F.log_softmax(sim_matrix.T, dim=1)[targets, targets]) # Manual InfoNCE
        loss_z2 = F.cross_entropy(sim_matrix.T, targets) # Using F.cross_entropy directly

        # Total loss is the average of the two symmetric losses
        loss = (loss_z1 + loss_z2) / 2.0

        # Log the loss
        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer for the model.

        Returns:
            torch.optim.Optimizer: The optimizer instance.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    # Optional: Add validation_step and test_step if needed for logging purposes,
    # but the core contrastive learning happens in training_step.
    # def validation_step(self, batch, batch_idx):
    #     # Similar logic as training_step but for validation data
    #     # Calculate loss and log validation_loss
    #     # Note: For validation, you might not need two augmentations if the goal
    #     # is just to monitor the loss on held-out *original* data pairs or
    #     # fixed augmented pairs. If you apply augmentations here, ensure consistency.
    #     # A common approach is to apply fixed augmentations for validation.
    #     pass

    # def test_step(self, batch, batch_idx):
    #      # Similar logic as validation_step
    #      pass


# ## コントラスト学習の訓練関数
# 
# ### Subtask:
# 定義したコントラスト学習モジュールを用いて、データローダー、オプティマイザー、トレーナーを設定し、コントラスト学習を行う関数を実装します。

# In[ ]:


import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import os

# Assuming ContrastiveLearningModule and CNNEncoder1D are defined in preceding cells
# Assuming data augmentation functions (add_gaussian_noise, random_scale, etc.) are defined

def train_contrastive_learning_model(
    X_data: np.ndarray, # Input data (n_samples, seq_len, n_features) - NumPy array
    input_channels: int, # Number of features (channels)
    seq_len: int,        # Sequence length
    latent_dim: int,     # Latent dimension for the encoder
    learning_rate: float = 1e-3,
    temperature: float = 0.07,
    augmentation_strategies: list = None,
    batch_size: int = 64,
    max_epochs: int = 100,
    encoder_save_path: str = None # Path to save the trained encoder state_dict
):
    """
    Trains a Contrastive Learning model (CNNEncoder1D + InfoNCE Loss).

    Args:
        X_data (np.ndarray): Training data as a NumPy array (n_samples, seq_len, n_features).
        input_channels (int): Number of features in the input data.
        seq_len (int): Length of each sequence in the input data.
        latent_dim (int): Dimension of the latent space.
        learning_rate (float): Learning rate for the optimizer.
        temperature (float): Temperature for the InfoNCE loss.
        augmentation_strategies (list, optional): List of augmentation functions and kwargs.
        batch_size (int): Batch size for training.
        max_epochs (int): Maximum number of training epochs.
        encoder_save_path (str, optional): Path to save the trained encoder's state_dict.
                                           If None, the encoder is not saved.

    Returns:
        torch.nn.Module: The trained encoder module.
        pl.Trainer: The trained PyTorch Lightning Trainer instance.
    """
    print("--- Starting Contrastive Learning Model Training ---")
    print(f"Input data shape: {X_data.shape}")
    print(f"Input channels: {input_channels}, Sequence length: {seq_len}, Latent dimension: {latent_dim}")
    print(f"Batch size: {batch_size}, Max epochs: {max_epochs}")

    # Ensure input data is a torch Tensor
    # Transpose from (n_samples, seq_len, n_features) to (n_samples, n_features, seq_len)
    # to match the expected input of CNNEncoder1D and ContrastiveLearningModule
    X_tensor = torch.tensor(X_data, dtype=torch.float32).transpose(1, 2)
    print(f"Debug: X_tensor shape after transpose for model input: {X_tensor.shape}")


    # Create a simple TensorDataset and DataLoader
    # For contrastive learning, the dataset yields the original data,
    # and the training_step applies augmentations within the module.
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count() // 2 or 1) # Use half CPU cores for workers


    # Initialize the encoder model
    # CNNEncoder1D expects (input_channels, output_dim) where output_dim is latent_dim
    # The seq_len argument is not needed for this version of CNNEncoder1D
    encoder = CNNEncoder1D(input_channels=input_channels, output_dim=latent_dim)
    print("CNNEncoder1D initialized.")

    # Initialize the Contrastive Learning module
    model = ContrastiveLearningModule(
        encoder=encoder,
        learning_rate=learning_rate,
        temperature=temperature,
        augmentation_strategies=augmentation_strategies # Pass augmentation strategies
    )
    print("ContrastiveLearningModule initialized.")


    # Initialize the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='auto', # Use GPU if available
        # Add logging, checkpointing etc. as needed
        # logger=pl.loggers.TensorBoardLogger("logs/", name="contrastive_learning"),
        # callbacks=[pl.callbacks.ModelCheckpoint(save_top_k=1, monitor="train_loss")]
    )
    print("PyTorch Lightning Trainer initialized.")


    # Train the model
    print("Starting model training...")
    try:
        trainer.fit(model, dataloader)
        print("Model training finished.")
    except Exception as e:
        print(f"Error during model training: {e}")
        # Optionally return None or raise the error if training is critical
        return None, None # Return None if training fails


    # Save the trained encoder if path is provided
    if encoder_save_path and encoder is not None: # Ensure encoder exists before saving
         try:
             # Ensure the directory exists
             save_dir = os.path.dirname(encoder_save_path)
             if save_dir and not os.path.exists(save_dir):
                 os.makedirs(save_dir)
                 print(f"Created directory for saving encoder: {save_dir}")

             # Save the state_dict of the encoder module
             torch.save(encoder.state_dict(), encoder_save_path)
             print(f"Saved trained encoder state_dict to {encoder_save_path}")
         except Exception as e:
             print(f"Warning: Failed to save trained encoder state_dict to {encoder_save_path}: {e}")


    # Return the trained encoder and trainer
    return encoder, trainer


# ## 潜在ベクトルの抽出関数
# 
# ### Subtask:
# 学習済みのコントラスト学習モジュール（エンコーダー部分）を用いて、入力データから潜在ベクトルを抽出する関数を実装します。

# In[ ]:


import torch
import numpy as np

# Assuming CNNEncoder1D is defined in a preceding cell
# Assuming ContrastiveLearningModule is defined in a preceding cell

def extract_latent_vectors(
    encoder: torch.nn.Module,
    X_data: np.ndarray, # Input data (n_samples, seq_len, n_features) - NumPy array
    batch_size: int = 64
) -> np.ndarray:
    """
    Extracts latent vectors from the trained encoder for the given input data.

    Args:
        encoder (torch.nn.Module): The trained encoder module.
        X_data (np.ndarray): Input data as a NumPy array (n_samples, seq_len, n_features).
        batch_size (int): Batch size for processing data.

    Returns:
        np.ndarray: Latent vectors as a NumPy array (n_samples, latent_dim).
    """
    print("--- Starting Latent Vector Extraction ---")
    print(f"Input data shape for extraction: {X_data.shape}")
    print(f"Batch size for extraction: {batch_size}")


    # Ensure input data is a torch Tensor and transpose to (n_samples, n_features, seq_len)
    X_tensor = torch.tensor(X_data, dtype=torch.float32).transpose(1, 2)
    print(f"Debug: X_tensor shape after transpose for extraction: {X_tensor.shape}")


    # Create DataLoader for efficient processing
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False) # No shuffling needed for extraction


    # Set encoder to evaluation mode and move to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.eval() # Set model to evaluation mode
    encoder.to(device) # Move model to device
    print(f"Encoder moved to device: {device}")


    latent_vectors_list = []

    # Extract latent vectors in batches
    print("Extracting latent vectors batch by batch...")
    with torch.no_grad(): # Disable gradient calculation for inference
        for i, batch in enumerate(dataloader):
            x_batch = batch[0].to(device) # Move batch to device
            try:
                 # Pass batch through the encoder
                 latent_batch = encoder(x_batch) # Shape: (batch_size, latent_dim)

                 # Move latent vectors back to CPU and convert to NumPy
                 latent_vectors_list.append(latent_batch.cpu().numpy())

                 # Optional: Print progress
                 if (i + 1) % 100 == 0:
                      print(f"Processed {i + 1} batches for latent extraction.")

            except Exception as e:
                 print(f"Error during latent extraction for batch {i}: {e}. Skipping batch.")
                 # Depending on error handling needs, you might want to stop or log more details.
                 pass # Continue processing other batches


    # Concatenate all batches of latent vectors
    if latent_vectors_list:
         latent_vectors_np = np.concatenate(latent_vectors_list, axis=0)
         print(f"Finished latent vector extraction. Concatenated shape: {latent_vectors_np.shape}")

         # Handle potential NaN/Inf in final latent vectors (should be less likely after normalization in CL module, but defensive)
         if np.isnan(latent_vectors_np).any() or np.isinf(latent_vectors_np).any():
             print("Warning: NaN or Inf detected in final extracted latent vectors. Replacing with 0 and clamping finite values.")
             latent_vectors_np = np.nan_to_num(latent_vectors_np, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)


    else:
         print("Warning: No latent vectors were extracted.")
         latent_vectors_np = np.array([]) # Return empty array if nothing was extracted

    return latent_vectors_np


# ## (任意) DECモジュールの実装と微調整
# 
# ### Subtask:
# DECのためのPyTorch Lightningモジュール（`DECModule`など）を新たに定義します。このモジュールは、ステップ2で得られたエンコーダーを使用し、クラスタリング層とKLダイバージェンス損失を組み込みます。エンコーダーのパラメータを固定するか、微調整可能にするかを選択できるようにします。

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.cluster import KMeans
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Assuming CNNEncoder1D is defined in a preceding cell
# Assuming extract_latent_vectors is defined in a preceding cell

class DECModule(pl.LightningModule):
    def __init__(self, encoder: nn.Module, n_clusters: int, alpha: float = 1.0,
                 learning_rate: float = 1e-3, finetune_encoder: bool = False):
        """
        Initializes the DECModule for fine-tuning latent space and clustering.

        Args:
            encoder (nn.Module): The trained encoder module from Contrastive Learning.
            n_clusters (int): The desired number of clusters.
            alpha (float): The alpha parameter for the Student-t distribution.
            learning_rate (float): The learning rate for the optimizer.
            finetune_encoder (bool): If True, the encoder parameters are fine-tuned.
                                     If False, only the clustering layer is trained.
        """
        super().__init__()
        self.save_hyperparameters(ignore=['encoder']) # Save other hyperparameters
        self.encoder = encoder
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.finetune_encoder = finetune_encoder

        # Freeze encoder parameters if not fine-tuning
        if not self.finetune_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Encoder parameters are frozen for DEC fine-tuning.")
        else:
             print("Encoder parameters are NOT frozen for DEC fine-tuning (fine-tuning enabled).")


        # Initialize cluster centroids (will be initialized later)
        self.cluster_centroids = nn.Parameter(torch.Tensor(n_clusters, self.encoder.fc2.out_features)) # Assuming latent_dim is the output of encoder's last layer (fc2)
        # Initialize centroids randomly for now; they will be updated with KMeans results
        nn.init.xavier_uniform_(self.cluster_centroids)


        # Target distribution P (will be computed dynamically)
        self.target_distribution = None

    def forward(self, x):
        """
        Passes input through the encoder and computes soft assignments to clusters.

        Args:
            x (torch.Tensor): Input tensor (batch_size, n_features, seq_len).

        Returns:
            torch.Tensor: Soft cluster assignments (batch_size, n_clusters).
        """
        # Ensure input shape is compatible with the encoder if needed
        # Assuming encoder expects (batch_size, n_features, seq_len)
        if x.ndim == 3 and x.shape[1] != self.encoder.input_channels:
             if x.shape[2] == self.encoder.input_channels:
                  x = x.transpose(1, 2) # Transpose to (batch_size, n_features, seq_len)
             else:
                  # This case should ideally not happen if data prep is consistent
                  print(f"Warning: DEC forward input shape {x.shape} does not match expected encoder input (batch_size, {self.encoder.input_channels}, seq_len).")
                  pass # Proceed, but expect potential errors

        # Get latent vectors from the encoder
        # Ensure gradients are tracked if finetune_encoder is True
        with torch.set_grad_enabled(self.finetune_encoder):
             z = self.encoder(x) # Shape: (batch_size, latent_dim)


        # Compute soft assignments using Student-t distribution
        # q_ij = (1 + ||z_i - mu_j||^2 / alpha)^(-(alpha+1)/2) / sum_k (1 + ||z_i - mu_k||^2 / alpha)^(-(alpha+1)/2)
        # where z_i is the latent vector for sample i, mu_j is the j-th cluster centroid.

        # Calculate squared Euclidean distance between latent vectors and centroids
        # z: (batch_size, latent_dim)
        # cluster_centroids: (n_clusters, latent_dim)
        # distances: (batch_size, n_clusters)
        distances = torch.cdist(z, self.cluster_centroids, p=2.0)**2 # Squared Euclidean distance

        # Compute numerator: (1 + distance / alpha)^(-(alpha+1)/2)
        numerator = torch.pow(1 + distances / self.alpha, -(self.alpha + 1) / 2)

        # Compute denominator: sum over k
        denominator = torch.sum(numerator, dim=1, keepdim=True)

        # Compute soft assignments q
        q = numerator / denominator # Shape: (batch_size, n_clusters)

        return q, z # Also return latent vectors (z) for potential logging or further use

    def compute_target_distribution(self, q):
        """
        Computes the target distribution P from soft assignments Q.
        p_ij = (q_ij^2 / sum_i q_ij) / sum_k (q_ik^2 / sum_i q_ik)

        Args:
            q (torch.Tensor): Soft cluster assignments (batch_size, n_clusters).

        Returns:
            torch.Tensor: Target distribution P (batch_size, n_clusters).
        """
        # Compute frequency of each cluster: sum_i q_ij
        # F_j = sum over batch_size of q_ij
        cluster_frequencies = torch.sum(q, dim=0) # Shape: (n_clusters,)

        # Compute p_ij numerator: q_ij^2 / sum_i q_ij
        # Handle potential division by zero if a cluster has zero frequency
        cluster_frequencies = torch.clamp(cluster_frequencies, min=1e-8) # Add epsilon for stability
        p_numerator = torch.pow(q, 2) / cluster_frequencies # Shape: (batch_size, n_clusters)

        # Compute p_ij denominator: sum over k (q_ik^2 / sum_i q_ik)
        # This is the sum of the numerators across clusters for each sample
        p_denominator = torch.sum(p_numerator, dim=1, keepdim=True)

        # Compute target distribution P
        # Handle potential division by zero if a sample has zero numerator sum
        p_denominator = torch.clamp(p_denominator, min=1e-8) # Add epsilon for stability
        p = p_numerator / p_denominator # Shape: (batch_size, n_clusters)

        # Ensure P sums to 1 across clusters for each sample (should be guaranteed by formula)
        # print(f"Debug: P sum check (should be close to 1): {torch.sum(p, dim=1)}") # Debug print


        return p

    def training_step(self, batch, batch_idx):
        """
        Performs a training step calculating the KL divergence loss.

        Args:
            batch (tuple): A tuple containing the input data (x).
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The calculated KL divergence loss for the batch.
        """
        x = batch[0] # Assuming batch is just the data tensor
        x = x.to(self.device) # Move data to device

        # Compute soft assignments q and latent vectors z
        q, z = self(x) # Calls the forward method

        # Compute target distribution P
        # P is computed dynamically based on the current batch's Q.
        # In standard DEC, P is computed over the *entire* dataset Q periodically.
        # For simplicity in a training step, we'll compute P using the current batch's Q.
        # A more robust implementation might require access to Q for the whole dataset
        # and update P periodically outside the training step.
        # Let's compute P based on the current batch's q for this step.
        p = self.compute_target_distribution(q)

        # Compute KL Divergence Loss: KL(P || Q) = sum_i sum_j p_ij * log(p_ij / q_ij)
        # Ensure numerical stability for log(p/q)
        # Loss = sum(P * log(P / Q))
        # KL divergence is non-negative. We want to minimize it.
        # Add epsilon for log stability
        q = torch.clamp(q, min=1e-8) # Avoid log(0)
        p = torch.clamp(p, min=1e-8) # Avoid log(0)

        kl_loss = torch.sum(p * torch.log(p / q), dim=1) # Sum over clusters for each sample
        kl_loss = torch.mean(kl_loss) # Mean over batch

        # Log the loss
        self.log("train_loss_dec", kl_loss)

        # Optional: Log other metrics like cluster assignment entropy or average silhouette score
        # This would require additional computation and potentially access to true labels (if available)
        # or external libraries (like silhouette).

        return kl_loss

    def configure_optimizers(self):
        """
        Configures the optimizer for the model.

        Returns:
            torch.optim.Optimizer: The optimizer instance.
        """
        # Optimize parameters that require gradients
        # If finetune_encoder is False, only self.cluster_centroids will be updated
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)
        return optimizer

    def on_train_epoch_start(self):
         """
         Hook called at the beginning of each training epoch.
         Here, we can update the target distribution P based on the current Q
         over the entire dataset if needed, or perform other epoch-level tasks.
         """
         # In standard DEC, P is updated periodically (e.g., every few epochs).
         # For simplicity in this module, P is computed per batch in training_step.
         # A more complex implementation would collect Q for the whole dataset here
         # or in a dedicated function and update self.target_distribution.
         pass # No epoch-start specific logic needed with per-batch P calculation


    def initialize_centroids(self, dataloader: DataLoader):
        """
        Initializes cluster centroids using KMeans on latent vectors of the dataset.
        Should be called BEFORE training starts.

        Args:
            dataloader (DataLoader): DataLoader for the dataset to extract latent vectors from.
        """
        print("Initializing DEC cluster centroids using KMeans...")
        # Extract latent vectors for the entire dataset
        # Need to ensure the encoder is on the correct device and in eval mode temporarily
        self.encoder.eval()
        device_orig = next(self.encoder.parameters()).device # Store original device
        self.encoder.to(self.device) # Move encoder to DEC module's device

        all_latent_vectors = []
        with torch.no_grad():
            for batch in dataloader:
                x_batch = batch[0].to(self.device) # Move batch to device
                z_batch = self.encoder(x_batch) # Get latent vectors
                all_latent_vectors.append(z_batch.cpu().numpy()) # Move to CPU and convert to NumPy

        all_latent_vectors_np = np.concatenate(all_latent_vectors, axis=0)
        print(f"Extracted {all_latent_vectors_np.shape[0]} latent vectors for centroid initialization.")

        # Perform KMeans clustering to get initial centroids
        # Ensure enough samples for KMeans
        if all_latent_vectors_np.shape[0] < self.n_clusters:
             print(f"Warning: Not enough samples ({all_latent_vectors_np.shape[0]}) for KMeans initialization with {self.n_clusters} clusters. Cannot initialize centroids.")
             # Centroids remain randomly initialized or handle error
             return

        try:
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10) # n_init=auto in newer sklearn
            kmeans.fit(all_latent_vectors_np)
            initial_centroids_np = kmeans.cluster_centers_
            print("KMeans initialization complete.")

            # Update the cluster_centroids parameter
            self.cluster_centroids.data.copy_(torch.tensor(initial_centroids_np, dtype=torch.float32).to(self.device))
            print("DEC cluster centroids initialized.")

        except Exception as e:
            print(f"Error during KMeans initialization: {e}. DEC centroids may not be properly initialized.")
            # Centroids remain randomly initialized


        # Restore encoder state if it was modified (e.g., eval mode)
        self.encoder.train(self.finetune_encoder) # Set back to train mode if finetuning is enabled
        self.encoder.to(device_orig) # Move encoder back to its original device (if different)


    # Optional: Add validation_step and test_step if needed
    # def validation_step(self, batch, batch_idx):
    #     x = batch[0].to(self.device)
    #     q, z = self(x)
    #     # Compute validation loss if target distribution P is available for validation set
    #     # Or compute validation metrics like silhouette score on latent vectors/assignments
    #     pass


# ## 潜在ベクトル上でのクラスタリング
# 
# ### Subtask:
# コントラスト学習（およびオプションでDEC微調整）によって得られた潜在ベクトルに対して、データの形状や特性に応じてHDBSCAN、GMM、KMeans++のいずれかの手法でクラスタリングを実行する関数を実装します。

# In[ ]:


import numpy as np
import pandas as pd
import hdbscan
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler # Might be needed for GMM/KMeans if not already scaled

# Optional: Functions to help evaluate clustering quality (e.g., silhouette score, Davies-Bouldin, Calinski-Harabasz)
# These functions are not implemented here but would be used to select the best parameters or method.

def perform_latent_clustering(
    latent_vectors: np.ndarray,
    clustering_method: str = 'hdbscan', # 'hdbscan', 'kmeans', 'gmm'
    hdbscan_params: dict = None,
    n_clusters_kmeans: int = 10, # For KMeans
    n_components_gmm: int = 10, # For GMM
    random_state: int = 42,
    evaluate_clustering: bool = False, # Flag to perform clustering evaluation
    X_for_evaluation: np.ndarray = None, # Original data (if needed for some metrics)
    metric_for_evaluation: str = 'silhouette' # 'silhouette', 'davies_bouldin', 'calinski_harabasz'
) -> tuple[np.ndarray | None, dict | None]:
    """
    Performs clustering on latent vectors using the specified method.

    Args:
        latent_vectors (np.ndarray): The latent vectors (n_samples, latent_dim).
        clustering_method (str): The clustering method to use ('hdbscan', 'kmeans', 'gmm').
        hdbscan_params (dict, optional): Parameters for HDBSCAN. Required if clustering_method is 'hdbscan'.
                                         If None, default HDBSCAN parameters will be used.
        n_clusters_kmeans (int): The number of clusters for KMeans. Used if clustering_method is 'kmeans'.
        n_components_gmm (int): The number of components for GMM. Used if clustering_method is 'gmm'.
        random_state (int): Random state for reproducible results (KMeans, GMM).
        evaluate_clustering (bool): If True, evaluate the clustering result using specified metric.
                                    Evaluation is only performed for methods that produce fixed clusters (KMeans, GMM).
                                    HDBSCAN evaluation (like persistence) is often done during parameter tuning externally.
        X_for_evaluation (np.ndarray, optional): Original data, needed for metrics like silhouette on original data.
                                                 Not typically used for evaluating latent space clustering itself.
        metric_for_evaluation (str): Metric to use for evaluation if evaluate_clustering is True.

    Returns:
        tuple: (cluster_labels, evaluation_metrics)
               - cluster_labels (np.ndarray or None): The cluster labels (n_samples,) or None if clustering fails/skipped.
               - evaluation_metrics (dict or None): Dictionary of evaluation metrics or None if evaluation is not requested or fails.
    """
    if latent_vectors is None or latent_vectors.shape[0] == 0:
        print("Warning: Latent vectors are empty. Skipping clustering.")
        return None, None

    n_samples = latent_vectors.shape[0]
    cluster_labels = None
    evaluation_metrics = None

    print(f"Starting clustering with method '{clustering_method}' on {n_samples} samples.")

    # Handle potential NaNs/Infs in latent vectors before clustering (defensive check)
    if np.isnan(latent_vectors).any() or np.isinf(latent_vectors).any():
        print("Warning: Latent vectors contain NaN or Inf values before clustering. Replacing with 0 and clamping finite values.")
        latent_vectors = np.nan_to_num(latent_vectors, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)


    try:
        if clustering_method == 'hdbscan':
            print("Using HDBSCAN clustering...")
            # Default HDBSCAN parameters if none provided
            if hdbscan_params is None:
                hdbscan_params = {
                    'min_cluster_size': max(10, int(np.sqrt(n_samples))), # Suggested starting point
                    'min_samples': None, # Often min_samples = min_cluster_size or slightly smaller
                    'cluster_selection_epsilon': 0.0, # Default 0.0
                    'gen_min_span_tree': False, # Can set to True for plotting, but False is faster
                    'random_state': random_state # HDBSCAN uses random state for some aspects
                }
                # If min_samples is None, HDBSCAN defaults it to min_cluster_size
                print(f"Using default HDBSCAN parameters: {hdbscan_params}")
            else:
                 print(f"Using provided HDBSCAN parameters: {hdbscan_params}")


            # Ensure enough samples for HDBSCAN
            min_samples_needed = hdbscan_params.get('min_cluster_size', 10)
            if n_samples < min_samples_needed:
                 print(f"Warning: Not enough samples ({n_samples}) for HDBSCAN with min_cluster_size={min_samples_needed}. Assigning all to -1.")
                 cluster_labels = np.full(n_samples, -1, dtype=int)
            else:
                clusterer = hdbscan.HDBSCAN(**hdbscan_params)
                # HDBSCAN fit_predict handles noise (-1 label)
                cluster_labels = clusterer.fit_predict(latent_vectors)
                print(f"HDBSCAN clustering complete. Found {len(np.unique(cluster_labels)) - (1 if -1 in cluster_labels else 0)} clusters (excluding noise).")
                print(f"Cluster label distribution:\n{pd.Series(cluster_labels).value_counts().sort_index()}")

            # HDBSCAN stability (persistence) is typically evaluated during parameter tuning,
            # not as a single metric after fitting. Evaluation flag is less applicable here.


        elif clustering_method == 'kmeans':
            print(f"Using KMeans clustering with n_clusters={n_clusters_kmeans}...")
            # Ensure enough samples for KMeans
            if n_samples < n_clusters_kmeans:
                 print(f"Warning: Not enough samples ({n_samples}) for KMeans with n_clusters={n_clusters_kmeans}. Assigning all to -1.")
                 cluster_labels = np.full(n_samples, -1, dtype=int) # Or raise error
            else:
                # Use KMeans++ initialization by default
                kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=random_state, n_init=10) # n_init='auto' in newer sklearn
                cluster_labels = kmeans.fit_predict(latent_vectors)
                print(f"KMeans clustering complete. Found {len(np.unique(cluster_labels))} clusters.")
                print(f"Cluster label distribution:\n{pd.Series(cluster_labels).value_counts().sort_index()}")

            # Evaluate KMeans if requested
            if evaluate_clustering and len(np.unique(cluster_labels)) > 1: # Need at least 2 clusters for evaluation metrics
                 print(f"Evaluating KMeans clustering using metric: {metric_for_evaluation}")
                 try:
                     if metric_for_evaluation == 'silhouette':
                         # Silhouette score requires distance metric (default Euclidean for KMeans is fine)
                         score = silhouette_score(latent_vectors, cluster_labels)
                         evaluation_metrics = {'silhouette_score': score}
                     elif metric_for_evaluation == 'davies_bouldin':
                         score = davies_bouldin_score(latent_vectors, cluster_labels)
                         evaluation_metrics = {'davies_bouldin_score': score} # Lower is better
                     elif metric_for_evaluation == 'calinski_harabasz':
                         score = calinski_harabasz_score(latent_vectors, cluster_labels)
                         evaluation_metrics = {'calinski_harabasz_score': score} # Higher is better
                     else:
                         print(f"Warning: Unknown evaluation metric '{metric_for_evaluation}'. Skipping evaluation.")

                 except Exception as e:
                     print(f"Error during KMeans evaluation: {e}. Skipping evaluation.")
                     evaluation_metrics = None


        elif clustering_method == 'gmm':
            print(f"Using Gaussian Mixture Model clustering with n_components={n_components_gmm}...")
            # Ensure enough samples for GMM (need at least n_components)
            if n_samples < n_components_gmm:
                 print(f"Warning: Not enough samples ({n_samples}) for GMM with n_components={n_components_gmm}. Assigning all to -1.")
                 cluster_labels = np.full(n_samples, -1, dtype=int) # Or raise error
            else:
                # GMM can be sensitive to scale, consider scaling latent_vectors if not already done
                # latent_vectors = StandardScaler().fit_transform(latent_vectors) # Optional scaling

                gmm = GaussianMixture(n_components=n_components_gmm, random_state=random_state)
                # GMM fit_predict assigns each sample to a cluster
                cluster_labels = gmm.fit_predict(latent_vectors)
                print(f"GMM clustering complete. Found {len(np.unique(cluster_labels))} components/clusters.")
                print(f"Cluster label distribution:\n{pd.Series(cluster_labels).value_counts().sort_index()}")

                # GMM also provides BIC/AIC for component selection, which is often done externally
                # bic = gmm.bic(latent_vectors)
                # aic = gmm.aic(latent_vectors)
                # print(f"GMM BIC: {bic}, AIC: {aic}")


            # Evaluate GMM if requested
            if evaluate_clustering and len(np.unique(cluster_labels)) > 1: # Need at least 2 clusters
                 print(f"Evaluating GMM clustering using metric: {metric_for_evaluation}")
                 try:
                     if metric_for_evaluation == 'silhouette':
                         score = silhouette_score(latent_vectors, cluster_labels)
                         evaluation_metrics = {'silhouette_score': score}
                     elif metric_for_evaluation == 'davies_bouldin':
                         score = davies_bouldin_score(latent_vectors, cluster_labels)
                         evaluation_metrics = {'davies_bouldin_score': score}
                     elif metric_for_evaluation == 'calinski_harabasz':
                         score = calinski_harabasz_score(latent_vectors, cluster_labels)
                         evaluation_metrics = {'calinski_harabasz_score': score}
                     else:
                         print(f"Warning: Unknown evaluation metric '{metric_for_evaluation}'. Skipping evaluation.")

                 except Exception as e:
                     print(f"Error during GMM evaluation: {e}. Skipping evaluation.")
                     evaluation_metrics = None


        else:
            print(f"Error: Unknown clustering method '{clustering_method}'. Supported methods are 'hdbscan', 'kmeans', 'gmm'. Skipping clustering.")
            # Assign all to -1 for unknown method
            cluster_labels = np.full(n_samples, -1, dtype=int)


    except Exception as e:
        print(f"An unexpected error occurred during clustering: {e}")
        # Assign all to -1 in case of unexpected errors
        cluster_labels = np.full(n_samples, -1, dtype=int)
        evaluation_metrics = None


    # Ensure cluster_labels is a NumPy array even if all samples are noise (-1)
    if cluster_labels is not None and not isinstance(cluster_labels, np.ndarray):
         cluster_labels = np.asarray(cluster_labels)


    return cluster_labels, evaluation_metrics


# # Task
# Implement a PyTorch function for supervised pre-training of a time series encoder for TFT, incorporating a classification head and using Cross-Entropy or Focal Loss, with options for SupCon, VICReg, or Barlow Twins regularization, and a mechanism to extract the encoder's latent vectors for downstream clustering and TFT static features.

# ## モデル構造の定義
# 
# ### Subtask:
# 潜在ベクトルを生成するエンコーダーと、それを受け取る小さな分類ヘッドを持つPyTorchモデルを定義します。既存のCNNEncoder1Dを再利用し、その上に分類層を追加することを検討します。
# 

# **Reasoning**:
# The subtask is to define a PyTorch model that combines an encoder and a classification head for supervised learning. I will define a new class `SupervisedEncoderClassifier` that uses the existing `CNNEncoder1D` and adds a linear classification layer on top.
# 
# 

# In[ ]:


import torch
import torch.nn as nn
# Assuming CNNEncoder1D is defined in a preceding cell and accessible

class SupervisedEncoderClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int):
        """
        Initializes the SupervisedEncoderClassifier model.

        Args:
            encoder (nn.Module): The encoder module (e.g., CNNEncoder1D instance).
                                 It should output latent vectors.
            num_classes (int): The number of output classes for the classification head.
        """
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes

        # Define the classification head
        # It takes the latent vector from the encoder as input
        # Assuming the encoder's last layer is a linear layer with an 'out_features' attribute
        if hasattr(self.encoder, 'fc') and hasattr(self.encoder.fc, 'out_features'): # Check for the specific CNNEncoder1D structure
             latent_dim = self.encoder.fc.out_features
        elif hasattr(self.encoder, 'fc2') and hasattr(self.encoder.fc2, 'out_features'): # Check for another potential encoder structure
             latent_dim = self.encoder.fc2.out_features
        else:
             # Fallback: try to infer latent_dim by passing a dummy tensor
             try:
                  print("Attempting to infer latent_dim from encoder output with dummy tensor...")
                  # Create a dummy tensor with shape (batch_size=1, input_channels, seq_len)
                  # Need input_channels and seq_len from the encoder or configuration
                  # Assuming encoder has input_channels attribute and a fixed seq_len is expected
                  dummy_seq_len = 60 # Example sequence length, adjust as needed
                  dummy_input_channels = getattr(self.encoder, 'input_channels', 1) # Default to 1 if attribute not found
                  dummy_input = torch.randn(1, dummy_input_channels, dummy_seq_len)
                  with torch.no_grad():
                       dummy_output = self.encoder(dummy_input)
                  latent_dim = dummy_output.shape[-1]
                  print(f"Inferred latent_dim: {latent_dim}")
             except Exception as e:
                  raise AttributeError(f"Could not determine latent_dim from encoder or dummy pass: {e}")


        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        """
        Passes the input through the encoder and classifier.

        Args:
            x (torch.Tensor): Input tensor (batch_size, n_features, seq_len).
                              Ensure this shape matches the encoder's expected input.

        Returns:
            tuple: (latent_vector, logits)
                   - latent_vector (torch.Tensor): Latent vectors from the encoder (batch_size, latent_dim).
                   - logits (torch.Tensor): Classification logits from the classifier (batch_size, num_classes).
        """
        # Pass input through the encoder
        # Ensure encoder expects (batch_size, n_features, seq_len) or handle transposition
        if x.ndim == 3 and x.shape[1] != self.encoder.input_channels: # Assuming input_channels is n_features
             if x.shape[2] == self.encoder.input_channels:
                  x = x.transpose(1, 2) # Transpose to (batch_size, n_features, seq_len)
             else:
                  print(f"Warning: Input shape {x.shape} does not match expected encoder input (batch_size, {self.encoder.input_channels}, seq_len).")
                  # Proceed, but expect potential errors in the encoder
                  pass


        latent_vector = self.encoder(x) # Shape: (batch_size, latent_dim)

        # Pass latent vector through the classifier
        logits = self.classifier(latent_vector) # Shape: (batch_size, num_classes)

        return latent_vector, logits


# ## 損失関数の選択と実装
# 
# ### Subtask:
# 多値分類のためのCross-Entropy損失、またはラベルの不均衡に対応するためのFocal Lossを選択し、必要に応じて実装またはPyTorchの既存機能を利用します。
# 

# **Reasoning**:
# Implement the Focal Loss function in PyTorch.
# 
# 

# In[ ]:


import torch.nn as nn
import torch.nn.functional as F
import torch

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Focal Loss for multi-class classification.

        Args:
            alpha (float or list): Alpha parameter. Can be a single value or a list of weights per class.
            gamma (float): Gamma parameter.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Calculates the Focal Loss.

        Args:
            inputs (torch.Tensor): Raw, unnormalized scores (logits) from the model (batch_size, num_classes).
            targets (torch.Tensor): Ground truth class labels (batch_size,) of type LongTensor.

        Returns:
            torch.Tensor: The calculated Focal Loss.
        """
        # inputs shape: (batch_size, num_classes)
        # targets shape: (batch_size,)

        # Ensure inputs and targets are on the same device
        inputs = inputs.to(targets.device)

        # Compute Cross-Entropy loss
        # This gives log(p_t) for each sample, where p_t is the probability of the ground truth class
        ce_loss = F.cross_entropy(inputs, targets, reduction='none') # Keep reduction='none' to apply focal scaling per sample

        # Get probabilities of the ground truth class
        # Use softmax to convert logits to probabilities
        pt = torch.exp(-ce_loss) # pt = exp(log(pt)) where log(pt) is -ce_loss

        # Compute the focal loss term: (1 - p_t)^gamma
        focal_term = torch.pow(1 - pt, self.gamma)

        # Apply alpha weighting
        if isinstance(self.alpha, (float, int)):
            # Single alpha value applied to all samples
            alpha_factor = self.alpha
        elif isinstance(self.alpha, (list, tuple, np.ndarray)):
            # Alpha weights per class. Need to select the alpha weight for the ground truth class of each sample.
            # Convert alpha list to a tensor and move to the same device as targets
            alpha_tensor = torch.tensor(self.alpha, dtype=inputs.dtype, device=targets.device)
            # Select the alpha weight for the target class of each sample
            alpha_factor = alpha_tensor.gather(0, targets) # Shape: (batch_size,)
        else:
            raise TypeError("alpha must be a float, int, list, tuple, or np.ndarray")


        # Compute the final Focal Loss per sample: alpha * (1 - p_t)^gamma * CE_loss
        focal_loss = alpha_factor * focal_term * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        elif self.reduction == 'none':
            return focal_loss
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")

# Example usage (assuming dummy data):
# inputs = torch.randn(10, 5) # Batch size 10, 5 classes (logits)
# targets = torch.randint(0, 5, (10,)) # Batch size 10 (labels)

# # Using Cross-Entropy
# ce_criterion = nn.CrossEntropyLoss()
# ce_loss = ce_criterion(inputs, targets)
# print(f"Cross-Entropy Loss: {ce_loss.item()}")

# # Using Focal Loss with default alpha and gamma
# focal_criterion_default = FocalLoss()
# focal_loss_default = focal_criterion_default(inputs, targets)
# print(f"Focal Loss (default): {focal_loss_default.item()}")

# # Using Focal Loss with class-specific alpha weights
# # Assuming class 0 is the minority class and needs higher weight
# num_classes = inputs.shape[1]
# alpha_weights = [0.5] + [0.25] * (num_classes - 1) # Example: weight 0.5 for class 0, 0.25 for others
# focal_criterion_weighted = FocalLoss(alpha=alpha_weights)
# focal_loss_weighted = focal_criterion_weighted(inputs, targets)
# print(f"Focal Loss (weighted alpha): {focal_loss_weighted.item()}")


# ## 正則化項の検討と実装
# 
# ### Subtask:
# SupCon、VICReg、またはBarlow Twinsのような正則化手法を組み込む方法を検討します。これらの手法は複雑なため、段階的に導入するか、最初はシンプルなCross-Entropyでベースラインを作成することを検討します。
# 

# ## 教師ありプレトレーニングの訓練関数実装
# 
# ### Subtask:
# 定義したモデルと損失関数、正則化項を用いて、教師ありデータでモデルを訓練するPyTorch Lightning Trainerを含む関数を実装します。訓練データとラベルを入力として受け取ります。
# 

# **Reasoning**:
# Implement a PyTorch Lightning module for the supervised encoder classifier. This module will encapsulate the model, loss function, optimizer, and training loop, including handling data batches and calculating the combined loss (classification + regularization).
# 
# 

# In[ ]:


get_ipython().run_line_magic('pip', 'install pytorch-lightning')

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader # Import DataLoader
import numpy as np # Import numpy for np.ndarray type hint and usage in extract_latent_vectors
import os # Import os for potential use in data loading/saving

# Assuming SupervisedEncoderClassifier, FocalLoss are defined
# Assuming data augmentation functions and SupConLoss are defined (or will be)

class SupervisedPretrainingModule(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        learning_rate: float = 1e-3,
        loss_fn_name: str = 'CrossEntropy', # 'CrossEntropy' or 'FocalLoss'
        focal_loss_params: dict = None, # Parameters for FocalLoss
        regularization_fn_name: str = None, # 'SupCon', 'VICReg', 'BarlowTwins'
        regularization_params: dict = None, # Parameters for regularization loss
        regularization_weight: float = 0.1, # Weight for the regularization loss
        augmentation_strategies: list = None # Augmentation strategies for SupCon
    ):
        """
        Initializes the SupervisedPretrainingModule.

        Args:
            encoder (nn.Module): The encoder module (e.g., CNNEncoder1D instance).
                                 It should output latent vectors.
            num_classes (int): The number of output classes for the classification head.
            learning_rate (float): The learning rate for the optimizer.
            loss_fn_name (str): Name of the classification loss function ('CrossEntropy' or 'FocalLoss').
            focal_loss_params (dict, optional): Parameters for FocalLoss if used.
            regularization_fn_name (str, optional): Name of the regularization loss function ('SupCon').
                                                   Other methods ('VICReg', 'BarlowTwins') are not fully implemented here yet.
            regularization_params (dict, optional): Parameters for the regularization loss.
            regularization_weight (float): Weight for the regularization loss term.
            augmentation_strategies (list, optional): A list of tuples for augmentation strategies,
                                                     (augmentation_function, kwargs), used for SupCon.
        """
        super().__init__()
        self.save_hyperparameters(ignore=['encoder', 'focal_loss_params', 'regularization_params', 'augmentation_strategies'])

        # Initialize the main supervised model (Encoder + Classifier)
        self.model = SupervisedEncoderClassifier(encoder=encoder, num_classes=num_classes)

        # Initialize the classification loss function
        if loss_fn_name == 'CrossEntropy':
            self.classification_criterion = nn.CrossEntropyLoss()
        elif loss_fn_name == 'FocalLoss':
            focal_params = focal_loss_params if focal_loss_params is not None else {}
            self.classification_criterion = FocalLoss(**focal_params)
        else:
            raise ValueError(f"Unsupported loss_fn_name: {loss_fn_name}")

        # Initialize the regularization loss function
        self.regularization_criterion = None
        if regularization_fn_name == 'SupCon':
            reg_params = regularization_params if regularization_params is not None else {}
            # Assuming SupConLoss class exists and takes temperature
            self.regularization_criterion = SupConLoss(**reg_params)
            # SupCon requires augmentations, store them if SupCon is used
            if augmentation_strategies is None:
                 # Define default augmentations if SupCon is used but no augmentations provided
                 # These functions should be implemented elsewhere and available in the scope
                 self.augmentation_strategies = [
                     ('add_gaussian_noise', {'std': 0.01}), # Use string names for default functions
                     ('random_scale', {'scale_range': (0.9, 1.1)}),
                 ]
                 print("Using default augmentation strategies for SupCon.")
            else:
                 self.augmentation_strategies = augmentation_strategies
                 print(f"Using {len(self.augmentation_strategies)} provided augmentation strategies for SupCon.")

            # Map string names to actual function objects
            self._map_augmentation_names_to_functions()


        elif regularization_fn_name in ['VICReg', 'BarlowTwins']:
             raise NotImplementedError(f"Regularization method '{regularization_fn_name}' is not yet implemented.")
        elif regularization_fn_name is not None:
             raise ValueError(f"Unsupported regularization_fn_name: {regularization_fn_name}")
        else:
             print("No regularization loss will be used.")
             self.augmentation_strategies = None # No augmentations needed if no SupCon


        self.regularization_weight = regularization_weight

    def _map_augmentation_names_to_functions(self):
        """Maps string names of augmentation functions to their actual function objects."""
        if self.augmentation_strategies:
            mapped_strategies = []
            available_fns = {
                'random_crop': random_crop,
                'add_gaussian_noise': add_gaussian_noise,
                'jitter': jitter,
                'random_scale': random_scale,
                'random_permutation': random_permutation,
                # Add other augmentation functions here
            }
            for aug_info in self.augmentation_strategies:
                if isinstance(aug_info, (list, tuple)) and len(aug_info) >= 1:
                    aug_fn_identifier = aug_info[0]
                    kwargs = aug_info[1] if len(aug_info) > 1 else {}

                    if isinstance(aug_fn_identifier, str):
                        # Look up by string name
                        if aug_fn_identifier in available_fns:
                            mapped_strategies.append((available_fns[aug_fn_identifier], kwargs))
                        else:
                            print(f"Warning: Augmentation function '{aug_fn_identifier}' not found in available functions. Skipping.")
                    elif callable(aug_fn_identifier):
                         # It's already a callable function
                         mapped_strategies.append((aug_fn_identifier, kwargs))
                    else:
                         print(f"Warning: Invalid augmentation strategy format: {aug_info}. Skipping.")
                else:
                    print(f"Warning: Invalid augmentation strategy format: {aug_info}. Skipping.")
            self.augmentation_strategies = mapped_strategies
            print(f"Mapped {len(mapped_strategies)} augmentation strategies.")


    def apply_augmentation(self, sequence: torch.Tensor):
        """
        Applies a randomly chosen augmentation strategy to a single sequence.

        Args:
            sequence (torch.Tensor): A single time series sequence (n_features, seq_len).
                                     Batch dimension is NOT expected.

        Returns:
            torch.Tensor: The augmented sequence.
        """
        if not self.augmentation_strategies:
            return sequence # Return original if no augmentations are defined

        # Randomly select one augmentation strategy
        aug_fn, kwargs = self.augmentation_strategies[torch.randint(0, len(self.augmentation_strategies), (1,)).item()]

        # Apply the selected augmentation
        try:
            # Move sequence to CPU for augmentation if needed, assuming aug functions work on CPU
            # A more robust approach is to ensure aug functions are device-agnostic or handle device transfer.
            # Let's assume aug functions work on tensors regardless of device for simplicity here.
            augmented_sequence = aug_fn(sequence, **kwargs)
            return augmented_sequence

        except Exception as e:
            print(f"Warning: Failed to apply augmentation {aug_fn.__name__} with kwargs {kwargs}: {e}. Returning original sequence.")
            return sequence # Return original sequence if augmentation fails


    def forward(self, x):
        """
        Standard forward pass through the supervised model.
        This is typically used for inference or validation, not for training step
        when regularization requires augmentations.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Performs a training step including classification and optional regularization loss.

        Args:
            batch (tuple): A tuple containing the input data (x) and labels (y).
                           (batch_size, n_features, seq_len), (batch_size,)
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The total calculated loss for the batch.
        """
        # Unpack data and labels
        x, y = batch
        x, y = x.to(self.device), y.to(self.device) # Move data to device

        total_loss = 0.0
        classification_loss = 0.0
        regularization_loss = 0.0

        # --- Classification Loss ---
        # For classification, we use the original data (or one view if using SupCon with multiple views)
        # Let's use the original data for the classification branch
        latent_vector, logits = self.model(x) # Pass original data through the supervised model

        # Calculate classification loss
        classification_loss = self.classification_criterion(logits, y)
        total_loss += classification_loss
        self.log("train_cls_loss", classification_loss)


        # --- Regularization Loss (Conditional) ---
        if self.regularization_criterion is not None and self.regularization_fn_name == 'SupCon':
            # SupCon requires multiple augmented views of the same sample.
            # The dataset/dataloader typically provides the original data (x, y).
            # We need to generate augmented views here in the training step.
            # This approach of augmenting within the training step is common but can be slow.
            # A custom DataLoader collate_fn is more efficient.
            # For simplicity here, let's apply augmentations to the batch.

            batch_size = x.size(0)
            # Check if augmentation strategies are available
            if not hasattr(self, 'augmentation_strategies') or not self.augmentation_strategies:
                 print("Warning: SupCon regularization enabled but no augmentation strategies available. Skipping regularization loss.")
                 regularization_loss = torch.tensor(0.0, device=self.device) # No regularization if no augs
            else:
                # Apply two different augmentations to each sample in the batch
                # This is done per sample and then stacked back
                x_aug1_list = []
                x_aug2_list = []
                # Use original data shape (n_features, seq_len) for augmentation functions
                original_seq_shape = x.shape[1:] # Shape (n_features, seq_len)

                for i in range(batch_size):
                    seq_i = x[i] # Shape (n_features, seq_len)
                    # Apply augmentation to the single sequence (n_features, seq_len)
                    x_aug1_list.append(self.apply_augmentation(seq_i))
                    x_aug2_list.append(self.apply_augmentation(seq_i))

                # Stack augmented sequences back into batches
                # Ensure shapes are consistent after augmentation
                if not all(aug_seq.shape == original_seq_shape for aug_seq in x_aug1_list) or \
                   not all(aug_seq.shape == original_seq_shape for aug_seq in x_aug2_list):
                     print(f"Warning: Augmented sequence shapes mismatch original shape {original_seq_shape}. Skipping SupCon batch.")
                     regularization_loss = torch.tensor(0.0, device=self.device) # No regularization if shapes mismatch
                else:
                    x_aug1 = torch.stack(x_aug1_list, dim=0).to(self.device) # (batch_size, n_features, seq_len)
                    x_aug2 = torch.stack(x_aug2_list, dim=0).to(self.device) # (batch_size, n_features, seq_len)

                    # Get latent vectors for augmented views using only the encoder part
                    # Ensure encoder is callable and expects (batch_size, n_features, seq_len)
                    # Use the encoder from the supervised model
                    try:
                         z_aug1 = self.model.encoder(x_aug1) # Shape: (batch_size, latent_dim)
                         z_aug2 = self.model.encoder(x_aug2) # Shape: (batch_size, latent_dim)

                         # Concatenate latent vectors from two views along the batch dimension
                         # Shape: (2 * batch_size, latent_dim)
                         features = torch.cat([z_aug1.unsqueeze(1), z_aug2.unsqueeze(1)], dim=1)
                         features = features.view(2 * batch_size, -1) # Flatten to (2 * batch_size, latent_dim)


                         # Calculate SupCon Loss
                         # Need labels for SupCon. Use the original labels 'y' replicated for the two views.
                         # Shape: (2 * batch_size,)
                         labels_supcon = torch.cat([y, y], dim=0)

                         regularization_loss = self.regularization_criterion(features, labels_supcon)
                         # print(f"Debug: SupCon Loss batch {batch_idx}: {regularization_loss.item()}")

                    except Exception as e:
                         print(f"Error during SupCon regularization calculation for batch {batch_idx}: {e}. Skipping regularization loss.")
                         regularization_loss = torch.tensor(0.0, device=self.device) # Skip regularization if error occurred


                total_loss += self.regularization_weight * regularization_loss
                self.log("train_reg_loss", regularization_loss)


        # --- Total Loss ---
        self.log("train_loss", total_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        Performs a validation step calculating the classification loss.
        Regularization is typically not applied during validation.
        """
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        # Pass data through the supervised model
        latent_vector, logits = self.model(x)

        # Calculate classification loss
        classification_loss = self.classification_criterion(logits, y)

        # Log validation loss
        self.log("val_cls_loss", classification_loss)

        # Optional: Log accuracy or other metrics
        # preds = torch.argmax(logits, dim=1)
        # accuracy = (preds == y).float().mean()
        # self.log("val_acc", accuracy)

        return classification_loss


    def configure_optimizers(self):
        """
        Configures the optimizer for the model.

        Returns:
            torch.optim.Optimizer: The optimizer instance.
        """
        # Optimize parameters of the entire supervised model
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    # Method to get the encoder part for downstream tasks
    def get_encoder(self):
        """Returns the trained encoder module."""
        return self.model.encoder

    # Method to extract latent vectors from the encoder
    def extract_latent_vectors(self, dataloader: DataLoader) -> np.ndarray:
        """
        Extracts latent vectors for data provided by a DataLoader using the trained encoder.
        Assumes the DataLoader yields batches of data tensors (e.g., (x,)).

        Args:
            dataloader (DataLoader): DataLoader for the data to extract latent vectors from.

        Returns:
            np.ndarray: Latent vectors as a NumPy array (n_samples, latent_dim).
        """
        print("--- Starting Latent Vector Extraction from Trained Pretraining Module ---")

        # Get the encoder and set to evaluation mode
        encoder = self.get_encoder()
        encoder.eval() # Set model to evaluation mode

        # Move encoder to appropriate device
        device = self.device # Use the module's device
        encoder.to(device)
        print(f"Encoder moved to device: {device}")


        latent_vectors_list = []

        # Extract latent vectors in batches
        print("Extracting latent vectors batch by batch...")
        with torch.no_grad(): # Disable gradient calculation for inference
            for i, batch in enumerate(dataloader):
                # Assuming batch contains the data tensor as the first element
                if isinstance(batch, (list, tuple)):
                     x_batch = batch[0].to(device) # Move batch to device
                else:
                     x_batch = batch.to(device) # Assume batch is already the data tensor

                try:
                    # Pass batch through the encoder
                    # Ensure encoder input shape matches x_batch (batch_size, n_features, seq_len)
                    # The SupervisedEncoderClassifier's forward handles transposition if needed.
                    # Here, we pass x_batch directly to the encoder. Ensure x_batch has the expected shape for the encoder.
                    # Assuming DataLoader provides batches in the expected shape (batch_size, n_features, seq_len).
                    # If DataLoader provides (batch_size, seq_len, n_features), transpose x_batch here.
                    # Let's check the encoder's expected input shape attribute if available.
                    encoder_input_channels = getattr(encoder, 'input_channels', None)
                    if x_batch.ndim == 3 and encoder_input_channels is not None and x_batch.shape[1] != encoder_input_channels:
                         if x_batch.shape[2] == encoder_input_channels:
                              x_batch = x_batch.transpose(1, 2) # Transpose
                         else:
                              print(f"Warning: Extraction batch shape {x_batch.shape} does not match expected encoder input (batch_size, {encoder_input_channels}, seq_len). Proceeding, but expect potential errors.")
                              # Proceed without transposition if shape doesn't match transpose logic


                    latent_batch = encoder(x_batch) # Shape: (batch_size, latent_dim)

                    # Move latent vectors back to CPU and convert to NumPy
                    latent_vectors_list.append(latent_batch.cpu().numpy())

                    # Optional: Print progress
                    if (i + 1) % 100 == 0:
                        print(f"Processed {i + 1} batches for latent extraction.")

                except Exception as e:
                    print(f"Error during latent extraction for batch {i}: {e}. Skipping batch.")
                    pass # Continue processing other batches


        # Concatenate all batches of latent vectors
        if latent_vectors_list:
            latent_vectors_np = np.concatenate(latent_vectors_list, axis=0)
            print(f"Finished latent vector extraction. Concatenated shape: {latent_vectors_np.shape}")

            # Handle potential NaN/Inf
            if np.isnan(latent_vectors_np).any() or np.isinf(latent_vectors_np).any():
                 print("Warning: NaN or Inf detected in extracted latent vectors. Replacing with 0 and clamping finite values.")
                 latent_vectors_np = np.nan_to_num(latent_vectors_np, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)


        else:
            print("Warning: No latent vectors were extracted.")
            latent_vectors_np = np.array([]) # Return empty array if nothing was extracted

        # Restore encoder to train mode if needed, but for downstream clustering it often stays in eval mode
        # encoder.train() # Restore train mode if needed

        return latent_vectors_np


# **Reasoning**:
# The previous command failed due to a missing package `pytorch_lightning`. I need to install it.
# 
# 

# In[ ]:


import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import numpy as np
import os

# Assuming SupervisedPretrainingModule, CNNEncoder1D, FocalLoss are defined
# Assuming data augmentation functions (if used with SupCon) are defined
# Assuming TimeSeriesDataset and TimeSeriesDataModule (if used for train/val split) are defined

def train_supervised_pretraining_model(
    X_data: np.ndarray, # Input data (n_samples, seq_len, n_features) - NumPy array
    y_labels: np.ndarray, # Labels (n_samples,) - NumPy array
    input_channels: int, # Number of features (channels)
    seq_len: int,        # Sequence length
    latent_dim: int,     # Latent dimension for the encoder
    num_classes: int,    # Number of output classes for classification
    learning_rate: float = 1e-3,
    loss_fn_name: str = 'CrossEntropy', # 'CrossEntropy' or 'FocalLoss'
    focal_loss_params: dict = None, # Parameters for FocalLoss
    regularization_fn_name: str = None, # 'SupCon'
    regularization_params: dict = None, # Parameters for regularization loss
    regularization_weight: float = 0.1, # Weight for regularization loss
    augmentation_strategies: list = None, # Augmentation strategies for SupCon
    batch_size: int = 64,
    max_epochs: int = 100,
    encoder_save_path: str = None, # Path to save the trained encoder state_dict
    supervised_model_save_path: str = None # Path to save the entire supervised model state_dict
):
    """
    Trains a supervised pretraining model (Encoder + Classifier) with optional regularization.

    Args:
        X_data (np.ndarray): Training data as a NumPy array (n_samples, seq_len, n_features).
        y_labels (np.ndarray): Training labels as a NumPy array (n_samples,).
        input_channels (int): Number of features in the input data.
        seq_len (int): Length of each sequence in the input data.
        latent_dim (int): Dimension of the latent space.
        num_classes (int): Number of output classes for classification.
        learning_rate (float): Learning rate for the optimizer.
        loss_fn_name (str): Name of the classification loss function ('CrossEntropy' or 'FocalLoss').
        focal_loss_params (dict, optional): Parameters for FocalLoss if used.
        regularization_fn_name (str, optional): Name of the regularization loss function ('SupCon').
        regularization_params (dict, optional): Parameters for the regularization loss.
        regularization_weight (float): Weight for the regularization loss term.
        augmentation_strategies (list, optional): List of augmentation functions and kwargs, used for SupCon.
        batch_size (int): Batch size for training.
        max_epochs (int): Maximum number of training epochs.
        encoder_save_path (str, optional): Path to save the trained encoder's state_dict.
        supervised_model_save_path (str, optional): Path to save the entire supervised model's state_dict.


    Returns:
        tuple: (trained_supervised_module, trainer)
               - trained_supervised_module (SupervisedPretrainingModule): The trained PyTorch Lightning module.
               - trainer (pl.Trainer): The trained PyTorch Lightning Trainer instance.
               Returns (None, None) if training fails.
    """
    print("--- Starting Supervised Pretraining Model Training ---")
    print(f"Input data shape: {X_data.shape}, Labels shape: {y_labels.shape}")
    print(f"Input channels: {input_channels}, Sequence length: {seq_len}, Latent dimension: {latent_dim}, Num classes: {num_classes}")
    print(f"Batch size: {batch_size}, Max epochs: {max_epochs}")
    print(f"Classification loss: {loss_fn_name}")
    if regularization_fn_name:
        print(f"Regularization: {regularization_fn_name} with weight {regularization_weight}")

    # Ensure input data and labels are torch Tensors
    # X_data needs to be transposed from (n_samples, seq_len, n_features) to (n_samples, n_features, seq_len)
    X_tensor = torch.tensor(X_data, dtype=torch.float32).transpose(1, 2)
    y_tensor = torch.tensor(y_labels, dtype=torch.long)
    print(f"Debug: X_tensor shape after transpose for model input: {X_tensor.shape}")
    print(f"Debug: y_tensor shape: {y_tensor.shape}")


    # Create a TensorDataset and DataLoader
    # For supervised learning, the dataset yields (data, label) pairs.
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count() // 2 or 1) # Use half CPU cores for workers


    # Initialize the encoder model
    # CNNEncoder1D expects (input_channels, output_dim) where output_dim is latent_dim
    # The seq_len argument is not needed for this version of CNNEncoder1D
    encoder = CNNEncoder1D(input_channels=input_channels, output_dim=latent_dim)
    print("CNNEncoder1D initialized.")

    # Initialize the Supervised Pretraining module
    model = SupervisedPretrainingModule(
        encoder=encoder,
        num_classes=num_classes,
        learning_rate=learning_rate,
        loss_fn_name=loss_fn_name,
        focal_loss_params=focal_loss_params,
        regularization_fn_name=regularization_fn_name,
        regularization_params=regularization_params,
        regularization_weight=regularization_weight,
        augmentation_strategies=augmentation_strategies # Pass augmentation strategies for SupCon
    )
    print("SupervisedPretrainingModule initialized.")


    # Initialize the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='auto', # Use GPU if available
        # Add logging, checkpointing etc. as needed
        # logger=pl.loggers.TensorBoardLogger("logs/", name="supervised_pretraining"),
        # callbacks=[pl.callbacks.ModelCheckpoint(save_top_k=1, monitor="train_loss")]
    )
    print("PyTorch Lightning Trainer initialized.")


    # Train the model
    print("Starting model training...")
    try:
        trainer.fit(model, dataloader)
        print("Model training finished.")
    except Exception as e:
        print(f"Error during model training: {e}")
        # Optionally return None or raise the error if training is critical
        return None, None # Return None if training fails


    # Save the trained encoder if path is provided
    if encoder_save_path and model.get_encoder() is not None: # Ensure encoder exists before saving
         try:
             # Ensure the directory exists
             save_dir = os.path.dirname(encoder_save_path)
             if save_dir and not os.path.exists(save_dir):
                 os.makedirs(save_dir)
                 print(f"Created directory for saving encoder: {save_dir}")

             # Save the state_dict of the encoder module
             torch.save(model.get_encoder().state_dict(), encoder_save_path)
             print(f"Saved trained encoder state_dict to {encoder_save_path}")
         except Exception as e:
             print(f"Warning: Failed to save trained encoder state_dict to {encoder_save_path}: {e}")

    # Save the entire supervised model if path is provided
    if supervised_model_save_path and model is not None:
         try:
             save_dir = os.path.dirname(supervised_model_save_path)
             if save_dir and not os.path.exists(save_dir):
                 os.makedirs(save_dir)
                 print(f"Created directory for saving supervised model: {save_dir}")

             # Save the state_dict of the entire supervised model
             torch.save(model.state_dict(), supervised_model_save_path)
             print(f"Saved trained supervised model state_dict to {supervised_model_save_path}")
         except Exception as e:
             print(f"Warning: Failed to save trained supervised model state_dict to {supervised_model_save_path}: {e}")


    # Return the trained module and trainer
    return model, trainer


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning Loss.

    It takes features and labels as input and computes the contrastive loss.
    The labels are used to determine positive and negative pairs.
    For features of a sample x_i, positive pairs are features of other
    augmented views of x_i and features of samples from the same class.
    Negative pairs are features of samples from different classes.
    """
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for either one view of features or more than one view.

        Args:
            features: shape [batch_size, num_views, feature_dim] or [batch_size, feature_dim].
            labels: ground truth of shape [batch_size].
            mask: adjacency matrix of shape [batch_size, batch_size], for positive pairs.
                  1 if sample i and sample j are from the same class.
                  Can be used instead of labels.
        """
        if len(features.shape) < 2:
            raise ValueError('`features` needs to be [BS * n_views, feature_dim] or [BS, n_views, feature_dim], at least 2 dimensions are required')
        if len(features.shape) == 2:
            # If input is [BS * n_views, feature_dim], reshape to [BS, n_views, feature_dim]
            # This assumes features are already flattened from [BS, n_views, feature_dim]
            # Need to infer num_views from batch_size and original batch_size
            # This requires knowing the original batch size or passing features as [BS, n_views, feature_dim]
            # Let's assume features are already shaped as [batch_size, num_views, feature_dim]
            raise ValueError('Input features should be of shape [batch_size, num_views, feature_dim]')

        # Get batch size and number of views
        batch_size = features.shape[0]
        num_views = features.shape[1] # Assuming shape [batch_size, num_views, feature_dim]
        feature_dim = features.shape[2]


        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            # If neither label nor mask is provided, assume each sample is its own class
            # and augmentations of the same sample are positive pairs.
            # This defaults to self-supervised contrastive loss (like SimCLR).
            mask = torch.eye(batch_size, dtype=torch.float32).to(features.device) # Identity mask (each sample is a positive pair with itself)
            # Replicate mask for multiple views: need mask of shape [BS * n_views, BS * n_views]
            # Assuming features are flattened to [BS * n_views, feature_dim] if labels is None.
            # Let's handle the case where features is [BS * n_views, feature_dim] and labels is None.
            # This function signature was initially based on [BS, n_views, feature_dim].
            # Let's modify the signature or add logic to handle [BS * n_views, feature_dim].

            # Let's assume the standard input is [BS, n_views, feature_dim] as per the comment.
            # If labels is None and mask is None, this means all pairs are negatives except for
            # same sample across views. This is SimCLR-like.

            # If labels is None, create pseudo-labels for SimCLR case
            # Assume original batch size before concatenating views was batch_size
            # The samples from index i to i + num_views - 1 are augmented views of the same sample.
            # This requires the input `features` to be [BS * n_views, feature_dim] where the first BS
            # belong to sample 1, the next BS to sample 2, etc. OR
            # The input is already structured as [BS, n_views, feature_dim].

            # Let's align with the training_step in SupervisedPretrainingModule which flattens features
            # to [BS * num_views, feature_dim] where num_views = 2.
            # Input features shape: [batch_size * 2, feature_dim]
            # Input labels shape: [batch_size * 2,] (original labels replicated)

            # Re-aligning with expected input shape from SupervisedPretrainingModule training_step
            # features: [BS * 2, feature_dim] where BS is the original batch size
            # labels: [BS * 2,] where labels are original labels repeated

            if len(features.shape) != 2:
                 raise ValueError('Input features must be of shape [BS * n_views, feature_dim]')
            if labels is None:
                 raise ValueError('Labels must be provided for Supervised Contrastive Loss')

            # Reshape features to [BS, num_views, feature_dim]
            # Infer num_views from features and labels shape
            # Batch size BS is labels.shape[0] // num_views
            # Let's assume num_views = 2 as per SupervisedPretrainingModule
            num_views = 2
            if labels.shape[0] % num_views != 0:
                 raise ValueError(f'Batch size ({labels.shape[0]}) must be divisible by number of views ({num_views})')
            batch_size = labels.shape[0] // num_views

            # Reshape features: [BS * num_views, feature_dim] -> [BS, num_views, feature_dim]
            features = features.view(batch_size, num_views, features.shape[1]) # Shape: [BS, 2, feature_dim]
            labels = labels.contiguous().view(batch_size, num_views) # Shape: [BS, 2] - labels should be the same for two views of a sample


            # Create mask from labels
            # mask[i, j] = 1 if sample i and sample j are from the same class (including same sample across views)
            # labels shape: [BS, num_views] (labels for view 1, labels for view 2)
            # We need to compare labels across the flattened batch [BS * num_views]
            labels_flattened = labels.view(-1, 1) # Shape [BS * num_views, 1]
            # mask shape: [BS * num_views, BS * num_views]
            # Entry (i, j) is 1 if labels_flattened[i] == labels_flattened[j]
            mask = torch.eq(labels_flattened, labels_flattened.T).float().to(features.device)


        # If mask is provided (instead of labels)
        # mask shape: [BS * n_views, BS * n_views]
        # Need to reshape features to [BS * n_views, feature_dim] if it came as [BS, n_views, feature_dim]
        if len(features.shape) == 3:
             features = features.view(batch_size * num_views, features.shape[2]) # Flatten features


        # Apply L2 normalization to features
        features = F.normalize(features, dim=1) # Shape: [BS * num_views, feature_dim]


        # Calculate cosine similarity between all pairs of samples in the flattened batch
        # Anchor: all samples (i)
        # Positive/Negative: all other samples (j)
        # Similarity matrix shape: [BS * num_views, BS * num_views]
        anchor_dot_contrast = torch.matmul(features, features.T) / self.temperature

        # For the numerator, we need similarity with positive pairs.
        # For the denominator, we need similarity with all other samples (positives and negatives).

        # Mask out self-contrastive pairs (similarity of a sample with itself)
        # The diagonal elements (i, i) correspond to self-similarity. We should exclude these.
        # Create a mask for the diagonal
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * num_views).view(-1, 1).to(features.device),
            0
        ) # Shape [BS * num_views, BS * num_views], 0 on diagonal, 1 elsewhere


        # Apply the self-contrastive mask to the similarity matrix
        anchor_dot_contrast = anchor_dot_contrast * logits_mask # Shape [BS * num_views, BS * num_views]


        # Apply the positive pair mask (from labels or provided mask)
        # We only want positive similarities for the numerator.
        # positive_mask shape: [BS * num_views, BS * num_views] (from labels or input mask)
        positive_mask = mask # Renaming for clarity

        # For a sample `i`, its positive pairs are `j` where `positive_mask[i, j] == 1` AND `i != j` (already handled by logits_mask).
        # Get similarities for positive pairs only
        # Shape [BS * num_views, BS * num_views], zero where not positive pair (or self)
        positive_sim = anchor_dot_contrast * positive_mask


        # Calculate log-likelihoods for InfoNCE
        # For each anchor `i`, the positive logit is the sum of similarities with its positive pairs.
        # The negative logits are similarities with all other samples `j` where `i != j` and `positive_mask[i, j] == 0`.
        # The denominator involves summing over all `j != i`.

        # InfoNCE log-likelihood for anchor `i`:
        # log( exp(sim(i, pos_j)/T) / sum_{k != i} exp(sim(i, k)/T) )
        # log( sum_{pos_j} exp(sim(i, pos_j)/T) / sum_{k != i} exp(sim(i, k)/T) )  <-- If multiple positive pairs
        # The standard SupCon loss takes the average over positive pairs for each anchor.

        # Reshape similarity matrix for log_softmax
        # Shape: [BS * num_views, BS * num_views]

        # Compute log-probabilities using log_softmax over all samples (excluding self)
        # This gives log(exp(sim(i, j)/T) / sum_{k != i} exp(sim(i, k)/T)) for each pair (i, j), j != i
        # Need to handle the self-similarity mask carefully before log_softmax.
        # A common way is to set self-similarity to a very small negative value before log_softmax.
        logits = anchor_dot_contrast # Still has zeros on diagonal and where not positive pair initially

        # Log-softmax over dimension 1 (over all potential contrastive samples for each anchor)
        # This includes positive and negative pairs, and the diagonal (which is zero)
        # Need to adjust logits before log_softmax to effectively exclude the diagonal.
        # Setting diagonal to -inf ensures exp(-inf) is 0 and they don't contribute to denominator sum.
        # For log_softmax, set diagonal to a very small number or use masked log_softmax.
        # Let's set diagonal to -1e9 before log_softmax.
        logits_adjusted = logits - torch.eye(batch_size * num_views).to(features.device) * 1e9 # Subtract large value on diagonal

        # Apply log-softmax
        log_prob = F.log_softmax(logits_adjusted, dim=1) # Shape [BS * num_views, BS * num_views]


        # Compute the positive log-likelihoods
        # For each anchor `i`, we need the log_prob values corresponding to its positive pairs `j`.
        # This is `log_prob[i, j]` where `positive_mask[i, j] == 1` AND `i != j`.
        # We can get these by element-wise multiplying log_prob with the positive mask (excluding diagonal).
        # positive_mask_no_self = positive_mask * logits_mask # Positive mask with diagonal zeroed
        # positive_log_prob = log_prob * positive_mask_no_self # Shape [BS * num_views, BS * num_views], non-zero only for positive pairs


        # Sum of positive log-probabilities for each anchor `i`
        # log_prob_pos_sum = torch.sum(positive_log_prob, dim=1) # Shape [BS * num_views,]

        # Compute the number of positive pairs for each anchor `i`
        # num_positive_pairs = torch.sum(positive_mask_no_self, dim=1) # Shape [BS * num_views,]
        # Handle potential division by zero if an anchor has no positive pairs (shouldn't happen in SupCon with at least two views)
        # num_positive_pairs = torch.clamp(num_positive_pairs, min=1.0) # Ensure at least 1 positive pair for robustness


        # The SupCon loss for anchor `i` is - (1 / num_positive_pairs) * sum_{pos_j} log(exp(sim(i, pos_j)/T) / sum_{k != i} exp(sim(i, k)/T))
        # This is equivalent to - (1 / num_positive_pairs) * sum_{pos_j} log_prob[i, j]

        # Loss per anchor = - (1 / num_positive_pairs) * sum_{pos_j} log_prob[i, j]
        # Calculate the negative sum of log probabilities over positive pairs for each anchor
        negative_log_prob_pos_sum = - torch.sum(log_prob * positive_mask, dim=1) # Sum over positive pairs, shape [BS * num_views,]
        # Note: this still includes the diagonal where positive_mask is 1 if labels are the same.
        # The diagonal in log_prob is effectively -inf, so sum will be -inf unless num_views > 1.
        # Need to correctly handle positive pairs including samples across views.

        # Revisit InfoNCE loss structure for SupCon:
        # For each anchor `i`, the loss is over pairs (i, j) where `j` is a positive sample.
        # L_i = - sum_{j in P(i)} log [ exp(sim(i, j)/T) / sum_{k in A(i)} exp(sim(i, k)/T) ] / |P(i)|
        # P(i) is the set of positive samples for anchor i (excluding i itself).
        # A(i) is the set of all samples in the batch (excluding i itself).
        # The denominator sum is over A(i). The log_softmax calculated `log_prob` already does sum over A(i) if `logits_adjusted` has diagonal -inf.

        # For each anchor `i`, we need `log_prob[i, j]` for `j` in P(i).
        # P(i) is the set of indices `j` where `labels_flattened[i] == labels_flattened[j]` and `i != j`.
        # This is exactly where `mask` (from labels) is 1 and `logits_mask` is 1.
        # So, `positive_mask_no_self` identifies the indices `j` in P(i) for anchor `i`.

        # Calculate `num_positive_pairs` for each anchor `i`.
        # This is the count of `j` where `positive_mask_no_self[i, j] == 1`.
        # Sum the `positive_mask_no_self` over dimension 1.
        # Note: Summing `mask` over dim 1 gives count of samples in the same class *in the flattened batch*.
        # Need to exclude self.
        # Count of positives for anchor i: sum(mask[i, :]) - 1 (excluding i itself)
        num_positive_pairs = torch.sum(mask, dim=1) - 1 # Shape [BS * num_views,]
        num_positive_pairs = torch.clamp(num_positive_pairs, min=1e-8) # Avoid division by zero


        # Calculate the sum of log_prob for positive pairs for each anchor
        # For each anchor `i`, sum `log_prob[i, j]` where `mask[i, j] == 1` AND `i != j`.
        # This is `torch.sum(log_prob[i, j] * mask[i, j] * logits_mask[i, j])` over `j`.
        # Simplified: `torch.sum(log_prob * mask * logits_mask, dim=1)`
        # Or even simpler: `torch.sum(log_prob * positive_mask, dim=1)` after setting diagonal of log_prob to -inf.
        # Using `logits_adjusted` which has diagonal -inf for log_softmax:
        # log_prob is correct.
        # Need to sum log_prob[i, j] for j in P(i).
        # The mask already has 1 for positive pairs (including self).
        # We need to sum over `j` where `mask[i, j] == 1`.
        # Sum of log_prob for all samples in the same class (including self)
        log_prob_sum_same_class = torch.sum(log_prob * mask, dim=1) # Shape [BS * num_views,]


        # SupCon loss per anchor i: - (1 / num_positive_pairs[i]) * log_prob_sum_same_class[i]
        loss_per_anchor = - log_prob_sum_same_class / num_positive_pairs # Shape [BS * num_views,]

        # Total loss is the mean over all anchors
        loss = torch.mean(loss_per_anchor)

        return loss


# In[ ]:


# ########################カテゴリごとにクラスタリングの実行###########################
# # Define parameters for clustering
# # Reverted latent_dim to 512 based on previous successful output
# seq_len = 60 # Sequence length
# latent_dim = 256 # AutoEncoder latent space dimension
# # Reverted hdbscan_params to original values based on previous successful output
# # hdbscan_params = {
# #     'min_cluster_size': 100,
# #     'min_samples': 10,
# #     'cluster_selection_epsilon': 0.2,
# #     'gen_min_span_tree': True
# # }
# # n_clusters = 10 # Default number of clusters for KMeans/GMM/DEC if not specified per strategy

# # Define default and strategy-specific K-Means parameters
# default_kmeans_params = {
#     'n_clusters': 10,
#     'random_state': 42,
#     'n_init': 10 # Number of initializations to run
# }

# # Updated strategy_kmeans_params to match the current strategy categories
# strategy_kmeans_params = {
#     'Uptrend': {'n_clusters': 15, 'random_state': 42, 'n_init': 10},
#     'Downtrend': {'n_clusters': 15, 'random_state': 42, 'n_init': 10},
#     'Reversal_Up': {'n_clusters': 10, 'random_state': 42, 'n_init': 10},
#     'Reversal_Down': {'n_clusters': 10, 'random_state': 42, 'n_init': 10},
#     'Range': {'n_clusters': 20, 'random_state': 42, 'n_init': 10},
#     'unknown': {'n_clusters': 5, 'random_state': 42, 'n_init': 10}, # Example for unknown, adjust as needed
# }


# print(f"Defined latent_dim: {latent_dim}")
# print(f"Defined default_kmeans_params: {default_kmeans_params}")
# print(f"Defined strategy_kmeans_params: {strategy_kmeans_params}")


# # Assume data_by_strategy, latent_dim, hdbscan_params are available from previous cells
# # Assume perform_clustering_on_subset function is defined in filtering.py and imported or available

# # Import UMAP and seaborn for visualization
# try:
#     import umap
#     import seaborn as sns
#     import matplotlib.pyplot as plt
#     from sklearn.cluster import KMeans # Import KMeans
#     print("Successfully imported umap, seaborn, matplotlib, and KMeans.")
# except ImportError:
#     print("Error: Could not import required libraries. Please ensure they are installed (`!pip install umap-learn seaborn matplotlib scikit-learn`).")
#     # Set flags or exit if essential libraries are missing
#     umap = None
#     sns = None
#     plt = None
#     KMeans = None


# # Check if required variables and function exist
# if 'data_by_strategy' not in locals() or 'latent_dim' not in locals() or KMeans is None:
#     print("Error: Required variables (data_by_strategy, latent_dim) or KMeans class are not defined.")
#     clustered_data_by_strategy = {} # Set empty dictionary to prevent errors
# elif 'filtering' not in locals() or not hasattr(filtering, 'perform_clustering_on_subset'):
#      print("Error: The function 'perform_clustering_on_subset' is not found in the 'filtering' module. Please ensure it has been defined and the module reloaded.")
#      clustered_data_by_strategy = {} # Set empty dictionary to prevent errors
# # Check if actual_sequence_length is defined
# elif 'actual_sequence_length' not in locals():
#      print("Error: Required variable 'actual_sequence_length' is not defined. Please ensure the cell calculating it (e.g., based on X_3d_numpy shape) has been executed.")
#      clustered_data_by_strategy = {} # Set empty dictionary to prevent errors
# # Check if feature_names_3d is defined (used by perform_clustering_on_subset)
# elif 'feature_names_3d' not in locals():
#      print("Error: Required variable 'feature_names_3d' is not defined. Please ensure feature names from the 3D data preparation are available.")
#      clustered_data_by_strategy = {} # Set empty dictionary to prevent errors
# else:
#     print("Starting clustering for each strategy category...")

#     # Dictionary to store clustering results for each strategy
#     clustered_data_by_strategy = {}

#     # --- Prepare AutoEncoder (Load or Initialize - REPLACE WITH YOUR ACTUAL AE) ---
#     # Example: Assuming AutoEncoder class is available globally or imported
#     # and you have a trained AutoEncoder model or path.
#     # trained_autoencoder = None # Replace with your trained AutoEncoder model instance if available
#     # autoencoder_model_path = None # Replace with the path to your trained AE state_dict if loading
#     # scaler_agg = None # Replace with your trained scaler for aggregated features if used


#     # Iterate through each strategy and perform clustering
#     for strategy_name, data_subset in data_by_strategy.items():
#         print(f"\n--- Clustering Strategy: {strategy_name} ---")
#         X_subset = data_subset.get('X')
#         original_indices_subset = data_subset.get('original_indices')

#         # Assuming feature_names is available from a previous cell
#         if 'feature_names' not in locals():
#              print("Error: 'feature_names' variable is not defined. Cannot pass it to clustering function. Skipping strategy.")
#              clustered_data_by_strategy[strategy_name] = {
#                  'latent': np.array([]), 'clusters': np.array([], dtype=int), 'original_indices': pd.Index([])
#              }
#              continue


#         if X_subset is None or len(X_subset) == 0:
#             print(f"No data available for strategy '{strategy_name}'. Skipping clustering.")
#             clustered_data_by_strategy[strategy_name] = {
#                 'latent': np.array([]),
#                 'clusters': np.array([], dtype=int),
#                 'original_indices': pd.Index([]) # Use empty Pandas Index
#             }
#             continue

#         # Get K-Means parameters for the current strategy, fallback to default if not specified
#         kmeans_params = strategy_kmeans_params.get(strategy_name, default_kmeans_params)
#         print(f"Using K-Means parameters for strategy '{strategy_name}': {kmeans_params}")


#         # Perform clustering on the subset using the function from filtering.py
#         # Pass the required arguments according to the function signature provided by the user
#         # def perform_clustering_on_subset(X_subset_3d, feature_names_3d, seq_len, original_indices_subset, latent_dim, hdbscan_params, ...):
#         try:
#              # Added debug prints before calling the function
#              print(f"Debug: Calling perform_clustering_on_subset with X_subset shape: {X_subset.shape}, feature_names_3d length: {len(feature_names_3d)}, seq_len: {actual_sequence_length}, original_indices_subset length: {len(original_indices_subset)}")

#              # Define clustering parameters for perform_clustering_on_subset
#              # Use spherical_kmeans, apply L2 normalization, use PCA
#              clustering_params_subset = {
#                  'clustering_method': 'spherical_kmeans',
#                  'n_clusters': kmeans_params.get('n_clusters', default_kmeans_params['n_clusters']), # Use strategy-specific or default n_clusters
#                  'kmeans_params': kmeans_params, # Pass strategy-specific or default KMeans parameters
#                  'apply_l2_normalization': True,
#                  'use_pca': True,
#                  'n_components_pca': 50, # Example: reduce to 50 dimensions after L2 norm
#                  'evaluate_clustering': True, # Optional: evaluate clustering quality
#                  'metric_for_evaluation': 'silhouette' # Example metric
#              }

#              # Define CL training parameters (only relevant if train_contrastive_learning_flag is True)
#              # If using a pre-trained encoder, train_contrastive_learning_flag should be False
#              # and trained_encoder or encoder_save_path should be provided.
#              cl_params_subset = {
#                  'train_contrastive_learning_flag': True, # Set to True to train CL model
#                  'max_epochs_cl_train': 50, # Example: train for 50 epochs (adjust as needed)
#                  # 'encoder_save_path': '/content/drive/MyDrive/Colab Notebooks/cryptoprice_prediction_tft/cnn_encoder_cl.pth', # Optional: save trained encoder
#                  # 'cl_augmentation_strategies': [...] # Optional: provide custom augmentations
#              }

#              # Define DEC fine-tuning parameters (optional)
#              dec_params_subset = {
#                  'use_dec_finetuning': False, # Set to True to enable DEC fine-tuning (adjust as needed)
#                  # 'dec_save_path': '/content/drive/MyDrive/Colab Notebooks/cryptoprice_prediction_tft/dec_model.pth', # Optional: save DEC model
#                  # 'max_epochs_dec_train': 50, # Example DEC epochs
#                  # 'dec_finetune_encoder': True # Set to True to finetune encoder during DEC
#              }


#              latent_strat, clusters_strat, indices_strat = filtering.perform_clustering_on_subset(
#               X_subset_3d=X_subset, # 3D 時系列生データ
#               feature_names_3d=feature_names_3d, # 特徴量名 (Use feature_names_3d which is expected by the function)
#               seq_len=actual_sequence_length, # シーケンス長
#               original_indices_subset=original_indices_subset, # 元のインデックス
#               latent_dim=latent_dim, # 潜在次元 (This is for the encoder output before L2/PCA)

#               # Pass the clustering parameters (includes method, n_clusters, etc.)
#               **clustering_params_subset,

#               # Pass the Contrastive Learning parameters
#               **cl_params_subset,

#               # Pass the DEC parameters
#               **dec_params_subset,

#               # Ensure other necessary arguments are passed if required by the function
#               # e.g., hdbscan_params=None since we are using kmeans
#               hdbscan_params=None # Not used for KMeans/Spherical KMeans
#           )

#              # 実行後の latent_strat には、DEC 微調整後の潜在ベクトル (use_dec_finetuning=True の場合)
#              # または CL 訓練後の潜在ベクトル (use_dec_finetuning=False の場合) が格納されます。
#              # clusters_strat には、その表現に対するクラスタリング結果が格納されます。
#              # indices_strat は入力の original_indices_subset と同じです。

#              # この後、得られた clusters_strat や latent_strat を使って
#              # クラスタ特徴量抽出、戦略ラベル付け、バイナリラベル生成などの後続処理に進みます。
#              print("\n--- perform_clustering_on_subset 実行結果 ---")
#              print(f"最終表現の形状 (latent_strat): {latent_strat.shape if latent_strat is not None else 'None'}")
#              print(f"クラスタラベルの形状 (clusters_strat): {clusters_strat.shape if clusters_strat is not None else 'None'}")
#              print(f"処理された元のインデックス数 (indices_strat): {len(indices_strat) if indices_strat is not None else 'None'}")
#              if clusters_strat is not None:
#                  print(f"生成されたクラスタ数 (ノイズ含む): {len(np.unique(clusters_strat))}")
#                  print("クラスタ分布:")
#                  print(pd.Series(clusters_strat).value_counts().sort_index())

#              # エラーが発生した場合 (返り値が None を含む場合) のハンドリングを追加してください
#              if latent_strat is None or clusters_strat is None:
#                  print("\n エラーが発生したため、クラスタリング結果は得られませんでした。")
#                  # エラーに応じた処理（例: プログラム終了、代替処理など）
#              else:
#                  print("\n 成功: 表現学習とクラスタリングが完了しました。後続のラベル生成に進めます。")

#                  # ここから後続処理（クラスタ特徴量抽出、戦略ラベル付けなど）を続けます。
#              # --- Visualization using UMAP and Matplotlib/Seaborn ---
#              # Only visualize if latent vectors were successfully generated and UMAP/plotting libraries are available
#              # Note: latent_strat here is the final representation *after* L2/PCA if applied
#              if latent_strat is not None and latent_strat.shape[0] > 0 and umap is not None and sns is not None and plt is not None:
#                  print(f"\n--- Visualizing Latent Space for Strategy: {strategy_name} ---")
#                  try:
#                      # Check if clusters_strat is available and use it for coloring if possible
#                      if clusters_strat is not None and len(clusters_strat) == latent_strat.shape[0]:
#                           print(f"Using cluster labels for coloring. Found {len(np.unique(clusters_strat))} unique labels.")
#                           # Ensure cluster labels are integers
#                           cluster_labels_vis = clusters_strat.astype(str) # Convert to string for categorical coloring
#                           # Replace -1 with 'Noise' for plotting clarity (KMeans doesn't produce -1, but keep for consistency if methods change)
#                           cluster_labels_vis[cluster_labels_vis == '-1'] = 'Noise'
#                           # If KMeans was used and found < n_clusters, some labels might be missing or unexpected.
#                           # Ensure all labels are strings for consistent handling by seaborn.
#                           cluster_labels_vis = np.array([str(label) for label in clusters_strat])
#                           cluster_labels_vis[cluster_labels_vis == '-1'] = 'Noise'


#                      else:
#                          print("Cluster labels not available or length mismatch. Coloring points uniformly.")
#                          cluster_labels_vis = None # Don't use labels for coloring


#                      # Perform UMAP dimensionality reduction on the final representation (latent_strat)
#                      print(f"Applying UMAP to final representation with shape: {latent_strat.shape}")
#                      reducer = umap.UMAP(n_components=2, random_state=42) # Use 2 components for 2D plot
#                      # Handle potential NaN/Inf values in latent_strat before UMAP
#                      if np.isnan(latent_strat).any() or np.isinf(latent_strat).any():
#                           print("Warning: Final representation contains NaN or Inf values before UMAP. Replacing with 0 and clamping finite values.")
#                           # Create a copy to avoid modifying the original latent_strat array
#                           latent_strat_cleaned = latent_strat.copy()
#                           latent_strat_cleaned[np.isnan(latent_strat_cleaned)] = 0
#                           latent_strat_cleaned[np.isinf(latent_strat_cleaned)] = np.finfo(latent_strat_cleaned.dtype).max # Clamp Inf to max float value
#                           # Clamp negative Inf to min float value
#                           latent_strat_cleaned[np.isneginf(latent_strat_cleaned)] = np.finfo(latent_strat_cleaned.dtype).min
#                           print("Replaced NaN/Inf values for UMAP.")
#                           latent_2d = reducer.fit_transform(latent_strat_cleaned)
#                      else:
#                           latent_2d = reducer.fit_transform(latent_strat)

#                      print(f"UMAP reduced latent vectors to shape: {latent_2d.shape}")


#                      # Create DataFrame for seaborn plotting
#                      plot_df = pd.DataFrame(latent_2d, columns=['UMAP_Dim1', 'UMAP_Dim2'])

#                      # Add cluster labels if available
#                      if cluster_labels_vis is not None:
#                          plot_df['Cluster'] = cluster_labels_vis
#                          # Determine number of unique non-noise clusters for palette size
#                          unique_clusters = np.unique(cluster_labels_vis)
#                          num_non_noise_clusters = len([c for c in unique_clusters if c != 'Noise'])
#                          # Use a palette that accommodates all unique labels, including 'Noise'
#                          # Add a distinct color for 'Noise' if needed
#                          palette = 'viridis' # Default palette
#                          hue_order = sorted(unique_clusters) # Plot Noise last


# In[ ]:


# ########################カテゴリごとにクラスタリングの実行###########################
# # Define parameters for clustering
# # Reverted latent_dim to 512 based on previous successful output
# seq_len = 60 # Sequence length
# latent_dim = 256 # AutoEncoder latent space dimension
# # Reverted hdbscan_params to original values based on previous successful output
# # hdbscan_params = {
# #     'min_cluster_size': 100,
# #     'min_samples': 10,
# #     'cluster_selection_epsilon': 0.2,
# #     'gen_min_span_tree': True
# # }
# # n_clusters = 10 # Default number of clusters for KMeans/GMM/DEC if not specified per strategy

# # Define default and strategy-specific K-Means parameters
# default_kmeans_params = {
#     'n_clusters': 10,
#     'random_state': 42,
#     'n_init': 10 # Number of initializations to run
# }

# # Updated strategy_kmeans_params to match the current strategy categories
# strategy_kmeans_params = {
#     'Uptrend': {'n_clusters': 15, 'random_state': 42, 'n_init': 10},
#     'Downtrend': {'n_clusters': 15, 'random_state': 42, 'n_init': 10},
#     'Reversal_Up': {'n_clusters': 10, 'random_state': 42, 'n_init': 10},
#     'Reversal_Down': {'n_clusters': 10, 'random_state': 42, 'n_init': 10},
#     'Range': {'n_clusters': 20, 'random_state': 42, 'n_init': 10},
#     'unknown': {'n_clusters': 5, 'random_state': 42, 'n_init': 10}, # Example for unknown, adjust as needed
# }


# print(f"Defined latent_dim: {latent_dim}")
# print(f"Defined default_kmeans_params: {default_kmeans_params}")
# print(f"Defined strategy_kmeans_params: {strategy_kmeans_params}")


# # Assume data_by_strategy, latent_dim, hdbscan_params are available from previous cells
# # Assume perform_clustering_on_subset function is defined in filtering.py and imported or available

# # Import UMAP and seaborn for visualization
# try:
#     import umap
#     import seaborn as sns
#     import matplotlib.pyplot as plt
#     from sklearn.cluster import KMeans # Import KMeans
#     print("Successfully imported umap, seaborn, matplotlib, and KMeans.")
# except ImportError:
#     print("Error: Could not import required libraries. Please ensure they are installed (`!pip install umap-learn seaborn matplotlib scikit-learn`).")
#     # Set flags or exit if essential libraries are missing
#     umap = None
#     sns = None
#     plt = None
#     KMeans = None


# # Check if required variables and function exist
# if 'data_by_strategy' not in locals() or 'latent_dim' not in locals() or KMeans is None:
#     print("Error: Required variables (data_by_strategy, latent_dim) or KMeans class are not defined.")
#     clustered_data_by_strategy = {} # Set empty dictionary to prevent errors
# elif 'filtering' not in locals() or not hasattr(filtering, 'perform_clustering_on_subset'):
#      print("Error: The function 'perform_clustering_on_subset' is not found in the 'filtering' module. Please ensure it has been defined and the module reloaded.")
#      clustered_data_by_strategy = {} # Set empty dictionary to prevent errors
# # Check if actual_sequence_length is defined
# elif 'actual_sequence_length' not in locals():
#      print("Error: Required variable 'actual_sequence_length' is not defined. Please ensure the cell calculating it (e.g., based on X_3d_numpy shape) has been executed.")
#      clustered_data_by_strategy = {} # Set empty dictionary to prevent errors
# # Check if feature_names_3d is defined (used by perform_clustering_on_subset)
# elif 'feature_names_3d' not in locals():
#      print("Error: Required variable 'feature_names_3d' is not defined. Please ensure feature names from the 3D data preparation are available.")
#      clustered_data_by_strategy = {} # Set empty dictionary to prevent errors
# else:
#     print("Starting clustering for each strategy category...")

#     # Dictionary to store clustering results for each strategy
#     clustered_data_by_strategy = {}

#     # --- Prepare AutoEncoder (Load or Initialize - REPLACE WITH YOUR ACTUAL AE) ---
#     # Example: Assuming AutoEncoder class is available globally or imported
#     # and you have a trained AutoEncoder model or path.
#     # trained_autoencoder = None # Replace with your trained AutoEncoder model instance if available
#     # autoencoder_model_path = None # Replace with the path to your trained AE state_dict if loading
#     # scaler_agg = None # Replace with your trained scaler for aggregated features if used


#     # Iterate through each strategy and perform clustering
#     for strategy_name, data_subset in data_by_strategy.items():
#         print(f"\n--- Clustering Strategy: {strategy_name} ---")
#         X_subset = data_subset.get('X')
#         original_indices_subset = data_subset.get('original_indices')

#         # Assuming feature_names is available from a previous cell
#         if 'feature_names' not in locals():
#              print("Error: 'feature_names' variable is not defined. Cannot pass it to clustering function. Skipping strategy.")
#              clustered_data_by_strategy[strategy_name] = {
#                  'latent': np.array([]), 'clusters': np.array([], dtype=int), 'original_indices': pd.Index([])
#              }
#              continue


#         if X_subset is None or len(X_subset) == 0:
#             print(f"No data available for strategy '{strategy_name}'. Skipping clustering.")
#             clustered_data_by_strategy[strategy_name] = {
#                 'latent': np.array([]),
#                 'clusters': np.array([], dtype=int),
#                 'original_indices': pd.Index([]) # Use empty Pandas Index
#             }
#             continue

#         # Get K-Means parameters for the current strategy, fallback to default if not specified
#         kmeans_params = strategy_kmeans_params.get(strategy_name, default_kmeans_params)
#         print(f"Using K-Means parameters for strategy '{strategy_name}': {kmeans_params}")


#         # Perform clustering on the subset using the function from filtering.py
#         # Pass the required arguments according to the function signature provided by the user
#         # def perform_clustering_on_subset(X_subset_3d, feature_names_3d, seq_len, original_indices_subset, latent_dim, hdbscan_params, ...):
#         try:
#              # Added debug prints before calling the function
#              print(f"Debug: Calling perform_clustering_on_subset with X_subset shape: {X_subset.shape}, feature_names_3d length: {len(feature_names_3d)}, seq_len: {actual_sequence_length}, original_indices_subset length: {len(original_indices_subset)}")

#              # Define clustering parameters for perform_clustering_on_subset
#              # Use spherical_kmeans, apply L2 normalization, use PCA
#              clustering_params_subset = {
#                  'clustering_method': 'spherical_kmeans',
#                  'n_clusters': kmeans_params.get('n_clusters', default_kmeans_params['n_clusters']), # Use strategy-specific or default n_clusters
#                  'kmeans_params': kmeans_params, # Pass strategy-specific or default KMeans parameters
#                  'apply_l2_normalization': True,
#                  'use_pca': True,
#                  'n_components_pca': 50, # Example: reduce to 50 dimensions after L2 norm
#                  'evaluate_clustering': True, # Optional: evaluate clustering quality
#                  'metric_for_evaluation': 'silhouette' # Example metric
#              }

#              # Define CL training parameters (only relevant if train_contrastive_learning_flag is True)
#              # If using a pre-trained encoder, train_contrastive_learning_flag should be False
#              # and trained_encoder or encoder_save_path should be provided.
#              cl_params_subset = {
#                  'train_contrastive_learning_flag': True, # Set to True to train CL model
#                  'max_epochs_cl_train': 50, # Example: train for 50 epochs (adjust as needed)
#                  # 'encoder_save_path': '/content/drive/MyDrive/Colab Notebooks/cryptoprice_prediction_tft/cnn_encoder_cl.pth', # Optional: save trained encoder
#                  # 'cl_augmentation_strategies': [...] # Optional: provide custom augmentations
#              }

#              # Define DEC fine-tuning parameters (optional)
#              dec_params_subset = {
#                  'use_dec_finetuning': False, # Set to True to enable DEC fine-tuning (adjust as needed)
#                  # 'dec_save_path': '/content/drive/MyDrive/Colab Notebooks/cryptoprice_prediction_tft/dec_model.pth', # Optional: save DEC model
#                  # 'max_epochs_dec_train': 50, # Example DEC epochs
#                  # 'dec_finetune_encoder': True # Set to True to finetune encoder during DEC
#              }


#              latent_strat, clusters_strat, indices_strat = filtering.perform_clustering_on_subset(
#               X_subset_3d=X_subset, # 3D 時系列生データ
#               feature_names_3d=feature_names_3d, # 特徴量名 (Use feature_names_3d which is expected by the function)
#               seq_len=actual_sequence_length, # シーケンス長
#               original_indices_subset=original_indices_subset, # 元のインデックス
#               latent_dim=latent_dim, # 潜在次元 (This is for the encoder output before L2/PCA)

#               # Pass the clustering parameters (includes method, n_clusters, etc.)
#               **clustering_params_subset,

#               # Pass the Contrastive Learning parameters
#               **cl_params_subset,

#               # Pass the DEC parameters
#               **dec_params_subset,

#               # Ensure other necessary arguments are passed if required by the function
#               # e.g., hdbscan_params=None since we are using kmeans
#               hdbscan_params=None # Not used for KMeans/Spherical KMeans
#           )

#              # 実行後の latent_strat には、DEC 微調整後の潜在ベクトル (use_dec_finetuning=True の場合)
#              # または CL 訓練後の潜在ベクトル (use_dec_finetuning=False の場合) が格納されます。
#              # clusters_strat には、その表現に対するクラスタリング結果が格納されます。
#              # indices_strat は入力の original_indices_subset と同じです。

#              # この後、得られた clusters_strat や latent_strat を使って
#              # クラスタ特徴量抽出、戦略ラベル付け、バイナリラベル生成などの後続処理に進みます。
#              print("\n--- perform_clustering_on_subset 実行結果 ---")
#              print(f"最終表現の形状 (latent_strat): {latent_strat.shape if latent_strat is not None else 'None'}")
#              print(f"クラスタラベルの形状 (clusters_strat): {clusters_strat.shape if clusters_strat is not None else 'None'}")
#              print(f"処理された元のインデックス数 (indices_strat): {len(indices_strat) if indices_strat is not None else 'None'}")
#              if clusters_strat is not None:
#                  print(f"生成されたクラスタ数 (ノイズ含む): {len(np.unique(clusters_strat))}")
#                  print("クラスタ分布:")
#                  print(pd.Series(clusters_strat).value_counts().sort_index())

#              # エラーが発生した場合 (返り値が None を含む場合) のハンドリングを追加してください
#              if latent_strat is None or clusters_strat is None:
#                  print("\n エラーが発生したため、クラスタリング結果は得られませんでした。")
#                  # エラーに応じた処理（例: プログラム終了、代替処理など）
#              else:
#                  print("\n 成功: 表現学習とクラスタリングが完了しました。後続のラベル生成に進めます。")

#                  # ここから後続処理（クラスタ特徴量抽出、戦略ラベル付けなど）を続けます。
#              # --- Visualization using UMAP and Matplotlib/Seaborn ---
#              # Only visualize if latent vectors were successfully generated and UMAP/plotting libraries are available
#              # Note: latent_strat here is the final representation *after* L2/PCA if applied
#              if latent_strat is not None and latent_strat.shape[0] > 0 and umap is not None and sns is not None and plt is not None:
#                  print(f"\n--- Visualizing Latent Space for Strategy: {strategy_name} ---")
#                  try:
#                      # Check if clusters_strat is available and use it for coloring if possible
#                      if clusters_strat is not None and len(clusters_strat) == latent_strat.shape[0]:
#                           print(f"Using cluster labels for coloring. Found {len(np.unique(clusters_strat))} unique labels.")
#                           # Ensure cluster labels are integers
#                           cluster_labels_vis = clusters_strat.astype(str) # Convert to string for categorical coloring
#                           # Replace -1 with 'Noise' for plotting clarity (KMeans doesn't produce -1, but keep for consistency if methods change)
#                           cluster_labels_vis[cluster_labels_vis == '-1'] = 'Noise'
#                           # If KMeans was used and found < n_clusters, some labels might be missing or unexpected.
#                           # Ensure all labels are strings for consistent handling by seaborn.
#                           cluster_labels_vis = np.array([str(label) for label in clusters_strat])
#                           cluster_labels_vis[cluster_labels_vis == '-1'] = 'Noise'


#                      else:
#                          print("Cluster labels not available or length mismatch. Coloring points uniformly.")
#                          cluster_labels_vis = None # Don't use labels for coloring


#                      # Perform UMAP dimensionality reduction on the final representation (latent_strat)
#                      print(f"Applying UMAP to final representation with shape: {latent_strat.shape}")
#                      reducer = umap.UMAP(n_components=2, random_state=42) # Use 2 components for 2D plot
#                      # Handle potential NaN/Inf values in latent_strat before UMAP
#                      if np.isnan(latent_strat).any() or np.isinf(latent_strat).any():
#                           print("Warning: Final representation contains NaN or Inf values before UMAP. Replacing with 0 and clamping finite values.")
#                           # Create a copy to avoid modifying the original latent_strat array
#                           latent_strat_cleaned = latent_strat.copy()
#                           latent_strat_cleaned[np.isnan(latent_strat_cleaned)] = 0
#                           latent_strat_cleaned[np.isinf(latent_strat_cleaned)] = np.finfo(latent_strat_cleaned.dtype).max # Clamp Inf to max float value
#                           # Clamp negative Inf to min float value
#                           latent_strat_cleaned[np.isneginf(latent_strat_cleaned)] = np.finfo(latent_strat_cleaned.dtype).min
#                           print("Replaced NaN/Inf values for UMAP.")
#                           latent_2d = reducer.fit_transform(latent_strat_cleaned)
#                      else:
#                           latent_2d = reducer.fit_transform(latent_strat)

#                      print(f"UMAP reduced latent vectors to shape: {latent_2d.shape}")


#                      # Create DataFrame for seaborn plotting
#                      plot_df = pd.DataFrame(latent_2d, columns=['UMAP_Dim1', 'UMAP_Dim2'])

#                      # Add cluster labels if available
#                      if cluster_labels_vis is not None:
#                          plot_df['Cluster'] = cluster_labels_vis
#                          # Determine number of unique non-noise clusters for palette size
#                          unique_clusters = np.unique(cluster_labels_vis)
#                          num_non_noise_clusters = len([c for c in unique_clusters if c != 'Noise'])
#                          # Use a palette that accommodates all unique labels, including 'Noise'
#                          # Add a distinct color for 'Noise' if needed
#                          palette = 'viridis' # Default palette
#                          hue_order = sorted(unique_clusters) # Plot Noise last


# # Task
# Generate a Python function `generate_future_based_binary_labels` that takes a pandas DataFrame `df` with a 'close' column, and parameters `H`, `L`, `W`, `p_pos`, `p_neg`, `alpha`, `M` to generate binary labels based on future price movements, incorporating dynamic thresholds and hysteresis. The function should calculate a future-based score for each time step, determine dynamic thresholds based on a rolling window of scores, generate candidate labels, and apply hysteresis to produce the final labels.

# ## Define a new function `generate future based binary labels`
# 
# ### Subtask:
# Define the main function `generate_future_based_binary_labels` that orchestrates the labeling process based on future price movements, dynamic thresholds, and hysteresis.
# 

# **Reasoning**:
# The subtask is to define the main function `generate_future_based_binary_labels` and implement the initial steps of score calculation. I will define the function signature, initialize the scores array, and iterate through the DataFrame to calculate scores for each time step based on the future window.
# 
# 

# **Reasoning**:
# The previous command defined the main function `generate_future_based_binary_labels` and calculated the scores. The next step is to calculate the dynamic thresholds based on these scores using the `calculate_dynamic_threshold` function.
# 
# 

# **Reasoning**:
# The previous command failed because the `calculate_dynamic_threshold` function was not defined. I need to include the definition of this function.
# 
# 

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import percentileofscore # Import for percentile calculation

# --- compute_slope Function (assuming it's available from another cell) ---
# Include the definition of compute_slope as it's used by calculate_future_score internally
# However, the provided calculate_future_score does NOT use compute_slope, it uses percentage change.
# So, keeping this commented out unless the scoring method is changed to use slope.
# def compute_slope(series):
#     """単純な線形回帰で傾きを返す"""
#     x = np.arange(len(series))
#     if np.isnan(series).any() or len(series) < 2:
#         return np.nan
#     if np.all(series == series[0]):
#         return 0.0
#     return linregress(x, series).slope


# --- calculate_future_score Function ---
# Modified signature to accept original_df and original_indices_strat
def calculate_future_score_from_processed(
    df_processed: pd.DataFrame,
    sequence_length: int,
    horizon: int,
    col_close: str = "close",
):
    """
    df_processed（dropna後 & sliding window の元になったDF）を基準に
    各シーケンスの future return を計算する。

    戻り値:
        future_scores: shape = (num_sequences,)
        各 i について:
           シーケンス i は df_processed[i : i+sequence_length]
           ラベルは end_pos = i+sequence_length-1 の行を現在とみなし、
           future_pos = end_pos + horizon の close との %変化（×100）
    """
    n_rows = len(df_processed)
    if col_close not in df_processed.columns:
        raise ValueError(f"{col_close} column not found in df_processed.")

    close_vals = df_processed[col_close].to_numpy().astype(float)

    # 何個のシーケンスが作られているか（prepare_3d_data と一致）
    num_sequences = n_rows - sequence_length + 1
    if num_sequences <= 0:
        return np.array([], dtype=np.float32)

    # 各シーケンスの終端位置と未来位置
    end_pos = np.arange(sequence_length - 1, sequence_length - 1 + num_sequences)
    future_pos = end_pos + horizon

    future_scores = np.full(num_sequences, np.nan, dtype=np.float32)

    valid_mask = future_pos < n_rows
    valid_end_pos = end_pos[valid_mask]
    valid_future_pos = future_pos[valid_mask]

    current_close = close_vals[valid_end_pos]
    future_close = close_vals[valid_future_pos]

    non_zero_mask = (current_close != 0) & ~np.isnan(current_close) & ~np.isnan(future_close)
    idx = np.where(valid_mask)[0][non_zero_mask]

    # %変化 ×100
    future_scores[idx] = (
        (future_close[non_zero_mask] - current_close[non_zero_mask])
        / current_close[non_zero_mask] * 100.0
    ).astype(np.float32)

    return future_scores

def calculate_future_score(
    sequences_3d,
    feature_names,
    horizon,
    df_processed,               # ★ original_df ではなく df_processed を渡す
    original_indices_filtered,  # ★ prepare_3d_data が返す start index（df_processed.index）
    seq_length,
    col_close="close",
):
    """
    df_processed（NaN drop 後）の時間軸に完全準拠した future return (%)
    を返す calculate_future_score。

    入力:
        sequences_3d: (num_seq, seq_length, n_feat)
        feature_names: X_3d_numpy の feature 名（使わないがインタフェース維持）
        horizon: 未来ステップ数
        df_processed: prepare_3d_data が返す dropna 後の DF
        original_indices_filtered: 各シーケンスの start index（df_processed.index）
        seq_length: シーケンス長
        col_close: "close"

    出力:
        future_scores: shape = (num_seq,)
    """
    num_seq = sequences_3d.shape[0]

    # df_processed の close を numpy 化
    close_arr = df_processed[col_close].to_numpy()
    n_rows = len(df_processed)

    # future_scores (初期 NaN)
    future_scores = np.full(num_seq, np.nan, dtype=np.float32)

    # START index → df_processed の位置に統一
    # original_indices_filtered は df_processed.index と整合している
    try:
        start_positions = df_processed.index.get_indexer(original_indices_filtered)
    except Exception as e:
        print("Error mapping original_indices_filtered to df_processed.index:", e)
        return future_scores

    # end index = start_pos + (seq_length - 1)
    end_positions = start_positions + (seq_length - 1)
    future_positions = end_positions + horizon

    # valid future positions only
    valid_mask = (end_positions >= 0) & (end_positions < n_rows) & (future_positions < n_rows)
    valid_end = end_positions[valid_mask]
    valid_fut = future_positions[valid_mask]

    current_close = close_arr[valid_end]
    future_close = close_arr[valid_fut]

    # 計算可能な部分だけ更新（ゼロ除算/NaN除外）
    nz_mask = (current_close != 0) & ~np.isnan(current_close) & ~np.isnan(future_close)
    valid_idx = np.where(valid_mask)[0][nz_mask]

    future_scores[valid_idx] = (
        (future_close[nz_mask] - current_close[nz_mask]) / current_close[nz_mask] * 100.0
    ).astype(np.float32)

    return future_scores

# --- calculate_rolling_threshold Function ---
def calculate_rolling_threshold(future_scores, window_size, target_positive_rate, ewma_alpha):
    """
    Calculates a dynamic threshold for each timestep using a rolling window of past scores
    and a target positive rate, applying EWMA smoothing.

    Args:
        future_scores (np.ndarray): 1D numpy array of future scores for each sequence.
        window_size (int): The size of the rolling window for threshold calculation.
        target_positive_rate (float): The desired positive rate (between 0 and 1).
        ewma_alpha (float): The smoothing factor for the EWMA (between 0 and 1).

    Returns:
        np.ndarray: A 1D numpy array containing the calculated and smoothed dynamic thresholds.
                    The first (window_size - 1) thresholds will be NaN due to the rolling window burn-in.
    """
    num_scores = len(future_scores)
    rolling_thresholds = np.full(num_scores, np.nan, dtype=np.float32)
    smoothed_thresholds = np.full(num_scores, np.nan, dtype=np.float32)

    if num_scores == 0 or window_size <= 0:
         print("Warning: Input scores are empty or window_size is not positive. Cannot calculate rolling threshold.")
         return smoothed_thresholds # Return NaNs

    # Ensure scores are in a pandas Series for easy rolling window operations
    scores_series = pd.Series(future_scores)

    # Calculate the percentile corresponding to the target positive rate
    # If target_positive_rate is p*, we want the value such that p* proportion of scores are >= that value.
    # This is the (1 - p*)th percentile.
    # Ensure percentile_q is within [0, 100]
    percentile_q = max(0.0, min(100.0, (1 - target_positive_rate) * 100))


    # Calculate the rolling threshold (raw)
    # The rolling window operation includes the current point and looks back.
    # .quantile(q / 100) calculates the q-th percentile.
    # min_periods=1 allows calculating quantile even with fewer than window_size points initially,
    # but we only want results after the full window, so we'll rely on the NaNs before window_size.
    # Use `interpolation='lower'` or 'higher' if needed for specific percentile definition
    raw_rolling_thresholds = scores_series.rolling(window=window_size).quantile(percentile_q / 100.0).values

    # Store the raw rolling thresholds
    # Rolling window results in NaNs for the first (window_size - 1) elements
    rolling_thresholds[:] = raw_rolling_thresholds


    # Apply EWMA smoothing
    # Find the index where the first non-NaN raw threshold appears
    first_valid_idx = np.argmax(~np.isnan(rolling_thresholds)) if np.any(~np.isnan(rolling_thresholds)) else -1

    if first_valid_idx != -1:
        # Initialize EWMA with the first valid raw threshold
        smoothed_thresholds[first_valid_idx] = rolling_thresholds[first_valid_idx]

        # Apply EWMA smoothing from the first valid index + 1
        for t in range(first_valid_idx + 1, num_scores):
            # Apply smoothing if the current raw threshold is not NaN
            if not np.isnan(rolling_thresholds[t]):
                smoothed_thresholds[t] = ewma_alpha * rolling_thresholds[t] + (1 - ewma_alpha) * smoothed_thresholds[t-1]
            else:
                 # If a raw threshold is NaN after the first valid one, carry forward the previous smoothed value.
                 # This might happen if there are consecutive NaNs in the input scores within a window.
                 # In such cases, it's reasonable to hold the last known good smoothed threshold.
                 smoothed_thresholds[t] = smoothed_thresholds[t-1]

    else:
         # If no valid raw thresholds were calculated (e.g., all scores are NaN or not enough scores)
         print(f"Warning: No valid raw thresholds calculated. Cannot apply EWMA smoothing.")
         # smoothed_thresholds remains all NaNs as initialized.

    return smoothed_thresholds


# --- generate_candidate_labels Function ---
def generate_candidate_labels(future_scores, smoothed_thresholds, lower_threshold_multiplier=1.0):
    """
    Generates initial candidate binary labels (1, 0, or -1 for the central band)
    based on future scores and smoothed dynamic thresholds.

    Args:
        future_scores (np.ndarray): 1D numpy array of future scores for each sequence.
        smoothed_thresholds (np.ndarray): 1D numpy array of smoothed dynamic thresholds (upper boundary).
        lower_threshold_multiplier (float, optional): Multiplier for the lower threshold.
                                                      Defaults to 1.0 (lower threshold equals upper threshold).

    Returns:
        np.ndarray: A 1D numpy array containing the candidate labels (1, 0, or -1).
                    -1 indicates the score is within the central band or NaN.
    """
    if len(future_scores) != len(smoothed_thresholds):
        print("Error: future_scores and smoothed_thresholds must have the same length.")
        return np.full(len(future_scores), -1, dtype=int) # Return array of -1 if lengths mismatch

    num_samples = len(future_scores)
    # Initialize candidate labels with -1 (representing the central band or undecided/NaN)
    candidate_labels = np.full(num_samples, -1, dtype=int)

    if num_samples == 0:
         print("Warning: Input arrays are empty. Returning empty candidate labels.")
         return candidate_labels


    # Calculate the lower threshold
    lower_thresholds = smoothed_thresholds * lower_threshold_multiplier

    # Using numpy's vectorized operations for efficiency
    # Condition for label 1: score >= upper threshold
    # Need to handle NaNs explicitly in comparisons if needed, but standard numpy comparisons
    # involving NaN result in False, which is suitable here as initial state is -1.
    condition_label_1 = (future_scores >= smoothed_thresholds)
    candidate_labels[condition_label_1] = 1

    # Condition for label 0: score <= lower threshold
    condition_label_0 = (future_scores <= lower_thresholds)
    candidate_labels[condition_label_0] = 0

    # The default value of -1 handles the case where score is between
    # lower_threshold and smoothed_threshold (upper threshold), and also
    # where future_scores or smoothed_thresholds are NaN, because comparisons
    # involving NaN result in False, leaving the initial -1 value.

    return candidate_labels

# --- apply_hysteresis Function ---
def apply_hysteresis(candidate_labels, M, on_value=1, off_value=0, ignore_value=-1):
    """
    Applies hysteresis smoothing to a sequence of labels.

    Args:
        candidate_labels (np.ndarray): 1D numpy array of candidate labels (e.g., 1, 0, ignore_value).
        M (int): The required number of continuous timesteps to confirm a label or transition.
                 A value of M=1 means no hysteresis (transitions immediately).
        on_value: The value considered as the "on" state (e.g., 1 for positive).
        off_value: The value considered as the "off" state (e.g., 0 for negative).
        ignore_value: The value in candidate_labels to ignore in counting consecutive states (e.g., -1 or NaN).

    Returns:
        np.ndarray: A 1D numpy array containing the final smoothed labels (on_value, off_value, or ignore_value).
                    ignore_value indicates an unconfirmed state.
    """
    num_samples = len(candidate_labels)
    # Initialize final_labels with the ignore_value
    final_labels = np.full(num_samples, ignore_value, dtype=candidate_labels.dtype)

    if num_samples == 0 or M <= 0:
        print("Warning: Input is empty or M is not positive. Returning initial labels (ignore_value).")
        return final_labels

    current_state = ignore_value # The last *confirmed* label (on_value, off_value, or ignore_value)
    consecutive_count = 0 # Counts consecutive non-ignore candidate labels of the same value

    for i in range(num_samples):
        candidate = candidate_labels[i]

        # If the candidate is the ignore value, reset count and candidate value
        if candidate == ignore_value or np.isnan(candidate): # Handle both -1 and NaN
            consecutive_count = 0
            # Keep the current_state (last confirmed label) as the label for the current step
            final_labels[i] = current_state

        # If the candidate is a non-ignore value
        else:
            # If the candidate is the same as the current state (last confirmed label)
            if candidate == current_state:
                consecutive_count += 1
                # If the count reaches M, the state is confirmed again (already was, but maintains count)
                if consecutive_count >= M:
                     # Final label is the current state
                     final_labels[i] = current_state
                else:
                     # Count is increasing but not yet M, label is still the current state
                     final_labels[i] = current_state

            # If the candidate is DIFFERENT from the current state
            elif candidate != current_state:
                 # If the candidate is the 'on' value and the current state is 'off' or 'ignore'
                 if candidate == on_value and (current_state == off_value or current_state == ignore_value):
                      # Start counting consecutive 'on' candidates
                      if consecutive_count < M or final_labels[i-1] != on_value if i > 0 else True: # Start new count only if not already counting this value recently below threshold
                           consecutive_count = 1
                      else:
                           consecutive_count += 1 # Continue counting if previous was also 'on' below threshold

                      if consecutive_count >= M:
                           # Transition confirmed to 'on'
                           current_state = on_value
                           final_labels[i] = on_value
                           consecutive_count = M # Keep count at M after confirmation

                      else:
                           # Not enough consecutive 'on' yet, label is still the current (old) state
                           final_labels[i] = current_state # Carry forward the last confirmed state ('off' or 'ignore')


                 # If the candidate is the 'off' value and the current state is 'on' or 'ignore'
                 elif candidate == off_value and (current_state == on_value or current_state == ignore_value):
                      # Start counting consecutive 'off' candidates
                      if consecutive_count < M or final_labels[i-1] != off_value if i > 0 else True: # Start new count only if not already counting this value recently below threshold
                           consecutive_count = 1
                      else:
                           consecutive_count += 1 # Continue counting if previous was also 'off' below threshold

                      if consecutive_count >= M:
                           # Transition confirmed to 'off'
                           current_state = off_value
                           final_labels[i] = off_value
                           consecutive_count = M # Keep count at M after confirmation
                      else:
                           # Not enough consecutive 'off' yet, label is still the current (old) state
                           final_labels[i] = current_state # Carry forward the last confirmed state ('on' or 'ignore')

                 # If candidate is neither 'on' nor 'off' (shouldn't happen if input is only 0, 1, ignore_value)
                 # Or if candidate is same as current state but candidate was ignore_value (handled above)
                 else:
                      # unexpected candidate value or logic branch
                      consecutive_count = 0 # Reset count
                      # Keep the current_state as the label for the current step
                      final_labels[i] = current_state


    return final_labels



# --- create_strategy_specific_binary_labels Function ---
# Modified signature to accept original_df
def create_strategy_specific_binary_labels(
    original_df, # Added original_df here
    X_3d_numpy,
    original_indices_filtered, # This is the full filtered original indices (aligned with X_3d_numpy)
    feature_names_3d,
    strategy_labels, # This is expected to be the integrated_strategy_names array
    horizon,
    rolling_window_size,
    target_positive_rate, # Target positive rate for the *final* re-quantilization
    dynamic_threshold_percentile_q=95, # Percentile for the *initial* dynamic threshold (e.g., 95 for top 5%)
    ewma_alpha=0.1,
    lower_threshold_multiplier=1.0, # Multiplier for the lower threshold relative to the upper threshold (for candidate labels)
    hysteresis_M=3 # The required number of continuous timesteps for label confirmation/transition (for final labels)
):
    """
    Creates strategy-specific smoothed binary labels (1/0) for each sequence,
    incorporating dynamic thresholding, percentile-based re-quantilization,
    and hysteresis smoothing.

    Steps:
    1. Calculate future scores for each sequence using the original_df for accurate future lookups.
    2. Estimate dynamic upper/lower thresholds based on a rolling window of *past* scores using percentiles (e.g., 95th percentile for upper).
    3. Generate initial candidate labels (1/0/-1) by comparing future scores to these dynamic thresholds.
    4. Re-quantilize the *valid* scores (excluding NaNs and initial rolling window burn-in) using a single percentile threshold derived from the target_positive_rate, assigning final 1/0 labels. Sequences with invalid scores remain -1.
    5. Apply hysteresis smoothing to these 1/0/-1 labels to introduce temporal continuity with a light setting (e.g., M=3).

    Args:
        original_df (pd.DataFrame): The original, unfiltered time series DataFrame.
                                    Must contain a 'close' column and have a time-based index.
        X_3d_numpy (np.ndarray): The full 3D feature data (num_total_sequences, seq_length, num_features).
                                  Expected to be the data *after* initial NaN handling and column selection,
                                  but *before* any strategy-specific filtering or aggregation.
        original_indices_filtered (pd.Index or np.ndarray): The full original indices
                                                             (e.g., timestamps) after filtering the original_df,
                                                             aligned with the first dimension of X_3d_numpy.
                                                             These are the START indices of the sequences.
        feature_names_3d (list): List of feature names corresponding to the last dimension of X_3d_numpy.
        strategy_labels (np.ndarray): A 1D numpy array of strategy names (strings) for each
                                      sequence in X_3d_numpy, aligned with its first dimension.
        horizon (int): The prediction horizon (number of future steps) for scoring.
        rolling_window_size (int): The size of the rolling window for dynamic threshold calculation based on *past* scores.
        target_positive_rate (float): The desired positive rate (between 0 and 1) for the *final* percentile-based re-quantilization step.
        dynamic_threshold_percentile_q (float): The percentile (0-100) to use for the *initial* dynamic upper threshold based on the rolling window of *past* scores. E.g., 95 means the 95th percentile.
        ewma_alpha (float): The smoothing factor for the EWMA of the dynamic threshold (between 0 and 1).
        lower_threshold_multiplier (float): Multiplier for the lower threshold relative to the upper threshold, used in candidate label generation (Step 3).
        hysteresis_M (int): The required number of continuous timesteps for final label confirmation/transition (Step 5).


    Returns:
        dict: A dictionary where keys are strategy names and values are 1D numpy arrays
              containing the final smoothed binary labels (1, 0, or -1) for all sequences
              in X_3d_numpy, with labels for sequences not belonging to a strategy or
              where processing failed set to -1 (or NaN if preferred).
              Returns an empty dict if inputs are invalid.
    """
    if original_df is None or not isinstance(original_df, pd.DataFrame) or 'close' not in original_df.columns:
         print("Error: original_df is invalid or missing 'close' column. Cannot proceed.")
         return {}
    if not isinstance(X_3d_numpy, np.ndarray) or X_3d_numpy.ndim != 3:
        print("Error: X_3d_numpy must be a 3D numpy array.")
        return {}
    if not isinstance(strategy_labels, np.ndarray) or strategy_labels.ndim != 1:
         print("Error: strategy_labels must be a 1D numpy array.")
         return {}
    if X_3d_numpy.shape[0] != len(strategy_labels):
        print("Error: Mismatch in number of sequences between X_3d_numpy and strategy_labels.")
        return {}
    if X_3d_numpy.shape[2] != len(feature_names_3d):
        print("Error: Mismatch in number of features between X_3d_numpy and feature_names_3d.")
        return {}
    # Check original_indices_filtered length
    if len(original_indices_filtered) != X_3d_numpy.shape[0]:
         print("Error: Mismatch in number of sequences between X_3d_numpy and original_indices_filtered.")
         return {}
    if not 0 <= target_positive_rate <= 1:
         print("Error: target_positive_rate must be between 0 and 1.")
         return {}
    if not 0 <= dynamic_threshold_percentile_q <= 100:
         print("Error: dynamic_threshold_percentile_q must be between 0 and 100.")
         return {}
    if not 0 <= ewma_alpha <= 1:
         print("Error: ewma_alpha must be between 0 and 1.")
         return {}
    if hysteresis_M < 0:
         print("Error: hysteresis_M must be non-negative.")
         return {}


    n_total_sequences = X_3d_numpy.shape[0]
    seq_length = X_3d_numpy.shape[1] # Get seq_length from the data

    # Initialize a full array to store labels for all sequences, default to -1 (unprocessed/noise)
    # Use dtype=float to allow np.nan if needed, although -1 is used for "unconfirmed/noise"
    full_binary_labels = np.full(n_total_sequences, -1, dtype=float)

    print(f"Total sequences to process: {n_total_sequences}")

    # Ensure original_df index is sorted for efficient lookup in calculate_future_score
    if not original_df.index.is_monotonic_increasing:
        print("Warning: original_df index is not monotonic increasing. Sorting for lookup.")
        original_df = original_df.sort_index()

    # --- Step 1: Calculate future scores for ALL sequences in X_3d_numpy ---
    # Call calculate_future_score with the full X_3d_numpy and original_indices_filtered
    print("\n--- Step 1: Calculating future scores for all sequences ---")
    future_scores_all = calculate_future_score(
         X_3d_numpy,
         feature_names_3d,
         horizon,
         original_df,
         original_indices_filtered, # Pass the full original_indices_filtered (start indices)
         seq_length
    )
    print(f"Calculated {len(future_scores_all)} future scores.")

    # Check for sufficient valid scores for thresholding and labeling
    valid_scores_mask_all = ~np.isnan(future_scores_all)
    num_valid_scores_all = np.sum(valid_scores_mask_all)

    if num_valid_scores_all == 0:
        print("No valid future scores calculated. Cannot proceed with labeling.")
        # full_binary_labels remains all -1 (or NaN)
        return {'all': full_binary_labels} # Return dictionary with a single key 'all'

    if num_valid_scores_all < rolling_window_size:
         print(f"Warning: Not enough valid scores ({num_valid_scores_all}) to calculate rolling threshold with window size {rolling_window_size}. Skipping dynamic thresholding and labeling for most sequences.")
         # Fallback: Re-quantilize based on all valid scores and apply hysteresis if possible
         print("Attempting to re-quantilize and apply hysteresis to all valid scores...")
         # Proceed to Step 4 logic using all valid scores
         valid_scores_subset_for_relabeling = future_scores_all[valid_scores_mask_all]
         if len(valid_scores_subset_for_relabeling) > 0:
              print(f"Re-quantilizing based on {len(valid_scores_subset_for_relabeling)} valid scores.")
              # Calculate the threshold based on the target positive rate over the *valid* scores
              percentile_for_relabeling = (1 - target_positive_rate) * 100
              relabeling_threshold = np.percentile(valid_scores_subset_for_relabeling, percentile_for_relabeling)
              print(f"Re-quantilization threshold ({percentile_for_relabeling}th percentile of valid scores): {relabeling_threshold}")

              # Create new labels based on this threshold and the original valid scores
              relabeling_labels_valid_scores = np.full(len(valid_scores_subset_for_relabeling), 0, dtype=int) # Default to 0
              relabeling_labels_valid_scores[valid_scores_subset_for_relabeling >= relabeling_threshold] = 1 # Assign 1 if score >= threshold

              # Create temporary labels array for hysteresis, initialized with -1
              temp_labels_for_hysteresis = np.full(n_total_sequences, -1, dtype=float)
              temp_labels_for_hysteresis[valid_scores_mask_all] = relabeling_labels_valid_scores

              # Apply hysteresis to these temp labels
              final_labels_all = apply_hysteresis(
                  temp_labels_for_hysteresis,
                  hysteresis_M,
                  on_value=1,
                  off_value=0,
                  ignore_value=-1 # Use -1 as the ignore value
              )
              print(f"Applied hysteresis to get {len(final_labels_all)} final labels.")
              print(f"Final label distribution (fallback): {np.unique(final_labels_all, return_counts=True)}")

              # Assign these labels to the full_binary_labels array
              full_binary_labels[:] = final_labels_all

              # Store labels for all sequences under a single key 'all' or iterate through strategies
              # Let's store under 'all' for this fallback case
              return {'all': full_binary_labels}

         else:
              print("No valid scores available even for fallback re-quantilization. Cannot label.")
              return {'all': full_binary_labels} # Return all -1 (or NaN)


    # --- Step 2: Estimate dynamic thresholds based on rolling window of PAST scores ---
    # Calculate rolling threshold using ALL future scores (NaNs will propagate initially)
    print("\n--- Step 2: Calculating dynamic thresholds based on rolling window of past scores ---")

    # Use the percentile specific for the dynamic threshold calculation
    smoothed_thresholds_all = calculate_rolling_threshold(
        future_scores_all, # Use all scores, rolling function handles NaNs
        rolling_window_size,
        # Use dynamic_threshold_percentile_q converted to a rate for quantile calculation
        target_positive_rate = 1.0 - (dynamic_threshold_percentile_q / 100.0), # Convert percentile to target rate
        ewma_alpha = ewma_alpha # Use EWMA alpha for smoothing
    )
    print(f"Calculated {len(smoothed_thresholds_all)} smoothed dynamic thresholds.")

    # Check if any valid smoothed thresholds were calculated
    if np.sum(~np.isnan(smoothed_thresholds_all)) == 0:
         print("No valid smoothed dynamic thresholds calculated. Cannot proceed with candidate labeling based on dynamic threshold.")
         # Fallback: Re-quantilize based on all valid scores as done in the insufficient samples case
         print("Falling back to re-quantilization and hysteresis on all valid scores...")
         valid_scores_subset_for_relabeling = future_scores_all[valid_scores_mask_all]
         if len(valid_scores_subset_for_relabeling) > 0:
              print(f"Re-quantilizing based on {len(valid_scores_subset_for_relabeling)} valid scores.")
              percentile_for_relabeling = (1 - target_positive_rate) * 100
              relabeling_threshold = np.percentile(valid_scores_subset_for_relabeling, percentile_for_relabeling)
              print(f"Re-quantilization threshold ({percentile_for_relabeling}th percentile of valid scores): {relabeling_threshold}")
              relabeling_labels_valid_scores = np.full(len(valid_scores_subset_for_relabeling), 0, dtype=int)
              relabeling_labels_valid_scores[valid_scores_subset_for_relabeling >= relabeling_threshold] = 1
              temp_labels_for_hysteresis = np.full(n_total_sequences, -1, dtype=float)
              temp_labels_for_hysteresis[valid_scores_mask_all] = relabeling_labels_valid_scores
              final_labels_all = apply_hysteresis(temp_labels_for_hysteresis, hysteresis_M, on_value=1, off_value=0, ignore_value=-1)
              print(f"Applied hysteresis to get {len(final_labels_all)} final labels.")
              print(f"Final label distribution (fallback): {np.unique(final_labels_all, return_counts=True)}")
              full_binary_labels[:] = final_labels_all
              return {'all': full_binary_labels}
         else:
              print("No valid scores available even for fallback re-quantilization. Cannot label.")
              return {'all': full_binary_labels} # Return all -1 (or NaN)


    # --- Step 3: Generate initial candidate labels (1/0/-1) ---
    # Compare future scores to the smoothed dynamic thresholds
    print("\n--- Step 3: Generating initial candidate labels ---")
    candidate_labels_all = generate_candidate_labels(
        future_scores_all, # Use all scores
        smoothed_thresholds_all, # Use all aligned smoothed thresholds (includes initial NaNs)
        lower_threshold_multiplier # Use the specified lower threshold multiplier
    )
    print(f"Generated {len(candidate_labels_all)} candidate labels (1/0/-1).")
    print(f"Candidate label distribution: {np.unique(candidate_labels_all, return_counts=True)}")

    # --- Step 4: Re-quantilize valid scores based on target positive rate ---
    # Use the original future_scores_all for this step, specifically the valid ones.
    # This step overwrites the candidate labels for valid sequences.
    print("\n--- Step 4: Re-quantilizing valid scores based on target positive rate ---")

    # Get the subset of future scores that are valid (not NaN)
    valid_scores_subset = future_scores_all[valid_scores_mask_all] # Re-use mask from Step 1

    if len(valid_scores_subset) > 0:
         # Calculate the threshold based on the target positive rate over the *valid* scores
         # If target_positive_rate is q, we want the score threshold such that q% of valid scores are >= threshold.
         # This is the (1 - target_positive_rate)th percentile of the valid scores.
         percentile_for_relabeling = (1 - target_positive_rate) * 100

         # Handle edge cases for percentile (0 and 100) - use `np.percentile` which handles this gracefully
         # Ensure percentile_for_relabeling is within [0, 100]
         percentile_for_relabeling = max(0.0, min(100.0, percentile_for_relabeling))

         # Calculate the relabeling threshold
         relabeling_threshold = np.percentile(valid_scores_subset, percentile_for_relabeling)

         print(f"Re-quantilization threshold ({percentile_for_relabeling}th percentile of valid scores): {relabeling_threshold}")

         # Create new labels based on this threshold and the original valid scores
         relabeling_labels_valid_scores = np.full(len(valid_scores_subset), 0, dtype=int) # Default to 0
         relabeling_labels_valid_scores[valid_scores_subset >= relabeling_threshold] = 1 # Assign 1 if score >= threshold
         print(f"Generated {len(relabeling_labels_valid_scores)} re-quantilized labels for valid scores.")
         print(f"Re-quantilized label distribution for valid scores: {np.unique(relabeling_labels_valid_scores, return_counts=True)}")

         # Create a temporary array for hysteresis, initialized with -1 (ignore)
         # This array will have shape (n_total_sequences,)
         temp_labels_for_hysteresis = np.full(n_total_sequences, -1, dtype=float)

         # Place the re-labeled values back into the temporary array using the valid_scores_mask
         # Sequences with NaN scores will remain -1.
         temp_labels_for_hysteresis[valid_scores_mask_all] = relabeling_labels_valid_scores

         print(f"Prepared {len(temp_labels_for_hysteresis)} labels for hysteresis (1/0/-1).")
         print(f"Distribution before hysteresis: {np.unique(temp_labels_for_hysteresis, return_counts=True)}")


    else:
        print("No valid scores available for re-quantilization. Cannot perform Step 4. Labels will be based on Step 3 candidates passed to hysteresis.")
        # If no valid scores, pass the candidate labels from Step 3 to hysteresis.
        # The candidate labels already have -1 where scores/thresholds were NaN.
        temp_labels_for_hysteresis = candidate_labels_all.astype(float) # Ensure float dtype for consistency with ignore_value=-1.0


    # --- Step 5: Apply hysteresis smoothing ---
    # Apply hysteresis to the labels from Step 4 (1/0/-1 array)
    print("\n--- Step 5: Applying hysteresis smoothing ---")

    # Ensure hysteresis_M is an integer
    hysteresis_M_int = int(hysteresis_M)
    if hysteresis_M_int < 0: hysteresis_M_int = 0 # Ensure non-negative

    final_labels_all = apply_hysteresis(
        temp_labels_for_hysteresis, # Use the labels from Step 4 (re-quantilized or candidates)
        hysteresis_M_int, # Use the specified hysteresis window size
        on_value=1, # Define the 'on' state
        off_value=0, # Define the 'off' state
        ignore_value=-1 # Define the ignore value
    )
    print(f"Applied hysteresis to get {len(final_labels_all)} final labels.")
    print(f"Final label distribution (after hysteresis): {np.unique(final_labels_all, return_counts=True)}")


    # --- Assign final labels back based on original strategy labels ---
    # The `final_labels_all` array is aligned with the original X_3d_numpy and strategy_labels.
    # We need to return a dictionary where keys are strategy names and values are
    # the subset of `final_labels_all` corresponding to that strategy.

    strategy_specific_final_labels = {}
    unique_strategies = np.unique(strategy_labels)

    for strategy_name in unique_strategies:
        # Filter final labels for the current strategy
        strategy_indices_in_full = np.where(strategy_labels == strategy_name)[0]

        if len(strategy_indices_in_full) > 0:
            strategy_specific_final_labels[strategy_name] = final_labels_all[strategy_indices_in_full]
            # print(f"Strategy '{strategy_name}': Extracted {len(strategy_indices_in_full)} final labels.")
            # print(f"Strategy '{strategy_name}' final label distribution: {np.unique(strategy_specific_final_labels[strategy_name], return_counts=True)}")
        else:
             # This case should theoretically not happen if unique_strategies are from strategy_labels
             strategy_specific_final_labels[strategy_name] = np.array([], dtype=float) # Empty array


    # Optional: Add labels for sequences not belonging to any identified strategy (-1 in strategy_labels)
    noise_indices_in_full = np.where(strategy_labels == -1)[0]
    if len(noise_indices_in_full) > 0:
         strategy_specific_final_labels[-1] = final_labels_all[noise_indices_in_full]
         # print(f"Strategy '-1' (Noise): Extracted {len(noise_indices_in_full)} final labels.")
         # print(f"Strategy '-1' final label distribution: {np.unique(strategy_specific_final_labels[-1], return_counts=True)}")


    print("\n--- Finished generating strategy-specific binary labels ---")


    # Return the dictionary containing final labels per strategy
    return strategy_specific_final_labels


# In[ ]:


import numpy as np
import pandas as pd

def compute_ATR_simple(df, window=24):
    """
    シンプルなATR計算。original_df に high, low, close がない場合は NaN を返す。
    """
    required_cols = {"high", "low", "close"}
    if not required_cols.issubset(df.columns):
        print("Warning: 'high', 'low', 'close' が不足しているため ATR フィルタは無効化します。")
        return pd.Series(np.nan, index=df.index)

    high_low  = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close  = (df["low"]  - df["close"].shift(1)).abs()
    tr  = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr


def create_strategy_specific_binary_labels_simple(
    original_df,
    X_3d_numpy,
    original_indices_filtered,
    feature_names_3d,
    strategy_labels,
    horizon,
    rolling_window_size,
    target_positive_rate,
    dynamic_threshold_percentile_q=95,
    ewma_alpha=0.1,
    lower_threshold_multiplier=1.0,
    hysteresis_M=3,
):
    """
    【簡略版ラベル生成】
    - 未来リターン（%） future_scores_all を計算
    - high vol フィルタを original_df ベースで作成
    - 全シーケンス共通の分布から quantile を計算
    - strategy_labels ∈ {uptrend, downtrend, range, reversal_up, reversal_down} ごとに
      異なるルールで 1/0/-1 ラベルを付与
    - -1 は「学習対象外」

    既存の create_strategy_specific_binary_labels と同じ引数シグネチャを持ちますが、
    動的閾値・ヒステリシス等は使わず、単純なルールベースです。
    """
    # --- 基本チェック ---
    if original_df is None or not isinstance(original_df, pd.DataFrame) or "close" not in original_df.columns:
        print("Error: original_df is invalid or missing 'close' column. Cannot proceed.")
        return {}

    if not isinstance(X_3d_numpy, np.ndarray) or X_3d_numpy.ndim != 3:
        print("Error: X_3d_numpy must be a 3D numpy array.")
        return {}

    if not isinstance(strategy_labels, np.ndarray) or strategy_labels.ndim != 1:
        print("Error: strategy_labels must be a 1D numpy array.")
        return {}

    if X_3d_numpy.shape[0] != len(strategy_labels):
        print("Error: Mismatch in number of sequences between X_3d_numpy and strategy_labels.")
        return {}

    if X_3d_numpy.shape[2] != len(feature_names_3d):
        print("Error: Mismatch in number of features between X_3d_numpy and feature_names_3d.")
        return {}

    if len(original_indices_filtered) != X_3d_numpy.shape[0]:
        print("Error: Mismatch in number of sequences between X_3d_numpy and original_indices_filtered.")
        return {}

    n_total_sequences, seq_length, _ = X_3d_numpy.shape

    # インデックス整列
    if not original_df.index.is_monotonic_increasing:
        original_df = original_df.sort_index()

    # --- Step 1: 未来リターン（%）を既存関数で計算 ---
    future_scores_all = calculate_future_score(
        X_3d_numpy,
        feature_names_3d,
        horizon,
        original_df,
        original_indices_filtered,
        seq_length,
    )

    valid_scores_mask = ~np.isnan(future_scores_all)
    if valid_scores_mask.sum() == 0:
        print("No valid future scores. All labels will be -1.")
        full_labels = np.full(n_total_sequences, -1, dtype=float)
        # 形式だけ従来どおり strategy 名ごとに返す
        result = {}
        for s in np.unique(strategy_labels):
            idx = np.where(strategy_labels == s)[0]
            result[s] = full_labels[idx]
        return result

    # --- Step 2: 高ボラフィルタ作成（ATR24 / close > 3%） ---
    atr24 = compute_ATR_simple(original_df, window=24)
    vol_ratio = atr24 / original_df["close"]


    # --- 出来高条件の準備（例：全体の60%タイル以上） ---
    if "ETH_volume" in original_df.columns:
        vol_series = original_df["ETH_volume"]
        vol_threshold = vol_series.quantile(0.60)  # 好きな値に調整可
    else:
        vol_series = None
        vol_threshold = None

    vol_filter_seq = np.zeros(n_total_sequences, dtype=bool)

    for i in range(n_total_sequences):
        start_idx = original_indices_filtered[i]
        try:
            start_pos = original_df.index.get_loc(start_idx)
        except KeyError:
            vol_filter_seq[i] = False
            continue

        end_pos = start_pos + (seq_length - 1)
        if end_pos >= len(original_df.index):
            vol_filter_seq[i] = False
            continue

        # ATR条件
        if pd.isna(vol_ratio.iloc[end_pos]):
            atr_cond = False
        else:
            atr_cond = (vol_ratio.iloc[end_pos] > 0.03)  # 3% 閾値

        # 出来高条件（列があれば）
        if vol_series is not None and not pd.isna(vol_series.iloc[end_pos]):
            vol_cond = (vol_series.iloc[end_pos] >= vol_threshold)
        else:
            vol_cond = True  # volume が無いなら条件なし扱いでもOK

        # 両方を満たしたときのみ True
        vol_filter_seq[i] = (atr_cond and vol_cond)

    # ATR が計算できなかった場合（全部 NaN）の救済
    if (~vol_filter_seq).all():
        print("Warning: ATR-based vol_filter could not be applied. Disabling vol filter (all True).")
        vol_filter_seq[:] = True

    # --- Step 3: 過去リターン（%）を簡易計算（reversal用） ---
    past_scores_all = np.full(n_total_sequences, np.nan, dtype=np.float32)
    closes = original_df["close"].values
    idx_array = original_df.index.to_numpy()

    for i in range(n_total_sequences):
        start_idx = original_indices_filtered[i]
        try:
            start_pos = original_df.index.get_loc(start_idx)
        except KeyError:
            continue
        past_end = start_pos - 1
        past_start = start_pos - horizon  # 「過去 horizon 本」の変化を見る（簡易）
        if past_start < 0 or past_end < 0:
            continue
        c0 = closes[past_start]
        c1 = closes[past_end]
        if c0 != 0 and not (np.isnan(c0) or np.isnan(c1)):
            past_scores_all[i] = (c1 - c0) / c0 * 100.0

    # --- Step 4: quantile を計算（未来・過去両方） ---
    valid_future = future_scores_all[valid_scores_mask]
    q20 = np.percentile(valid_future, 20)
    q40 = np.percentile(valid_future, 40)
    q60 = np.percentile(valid_future, 60)
    q80 = np.percentile(valid_future, 80)

    abs_future = np.abs(valid_future)
    q_abs30 = np.percentile(abs_future, 30)
    q_abs70 = np.percentile(abs_future, 70)

    valid_past = past_scores_all[~np.isnan(past_scores_all)]
    if len(valid_past) > 0:
        rp_q30 = np.percentile(valid_past, 30)
        rp_q70 = np.percentile(valid_past, 70)
    else:
        rp_q30 = rp_q70 = 0.0  # フォールバック（reversal_label はほぼ -1 になる）

    # --- Step 5: 戦略ごとのラベル付け関数 ---
    def label_downtrend(r_f):
        if np.isnan(r_f):
            return -1
        if r_f <= q20:
            return 1
        elif r_f >= q40:
            return 0
        else:
            return -1

    def label_uptrend(r_f):
        if np.isnan(r_f):
            return -1
        if r_f >= q80:
            return 1
        elif r_f <= q60:
            return 0
        else:
            return -1

    def label_range(r_f):
        if np.isnan(r_f):
            return -1
        a = abs(r_f)
        if a <= q_abs30:
            return 1   # レンジ継続（あまり動かない）
        elif a >= q_abs70:
            return 0   # ブレイク（大きく動く）
        else:
            return -1

    def label_reversal_up(r_p, r_f):
        if np.isnan(r_p) or np.isnan(r_f):
            return -1
        cond_past_downish = (r_p <= rp_q30)
        cond_future_up_strong = (r_f >= q80)
        cond_no_reversal = (r_f <= q60)
        if cond_past_downish and cond_future_up_strong:
            return 1
        elif cond_no_reversal:
            return 0
        else:
            return -1

    def label_reversal_down(r_p, r_f):
        if np.isnan(r_p) or np.isnan(r_f):
            return -1
        cond_past_upish = (r_p >= rp_q70)
        cond_future_down_strong = (r_f <= q20)
        cond_no_reversal = (r_f >= q40)
        if cond_past_upish and cond_future_down_strong:
            return 1
        elif cond_no_reversal:
            return 0
        else:
            return -1

    # --- Step 6: 全シーケンスに対して一括ラベリング ---
    full_labels = np.full(n_total_sequences, -1, dtype=float)

    for i in range(n_total_sequences):
        # 未来スコアが NaN → -1
        r_f = future_scores_all[i]
        r_p = past_scores_all[i]
        if np.isnan(r_f):
            full_labels[i] = -1
            continue

        strat = strategy_labels[i]
        # 文字列に統一（万が一 int などが混じっていても大丈夫なように）
        if isinstance(strat, bytes):
            strat_str = strat.decode("utf-8")
        else:
            strat_str = str(strat)
        strat_l = strat_str.lower()

        if strat_l == "downtrend":
            lab = label_downtrend(r_f)
        elif strat_l == "uptrend":
            lab = label_uptrend(r_f)
        elif strat_l == "range":
            lab = label_range(r_f)
        elif strat_l in ("reversal_up", "reversal-up", "rev_up"):
            lab = label_reversal_up(r_p, r_f)
        elif strat_l in ("reversal_down", "reversal-down", "rev_down"):
            lab = label_reversal_down(r_p, r_f)
        else:
            lab = -1

        # 高ボラ条件を満たさない場合は学習対象外
        if not vol_filter_seq[i]:
            lab = -1

        full_labels[i] = lab

    # --- Step 7: 戦略ごとの配列に分割して返す（従来関数と同じインターフェイス） ---
    result = {}
    unique_strats = np.unique(strategy_labels)
    for s in unique_strats:
        idx = np.where(strategy_labels == s)[0]
        result[s] = full_labels[idx]

    return result


# In[ ]:


import numpy as np
import pandas as pd

def calculate_future_score(
    X_3d_numpy: np.ndarray,
    feature_names_3d: list,
    horizon: int,
    original_df: pd.DataFrame,
    original_indices_filtered: pd.DatetimeIndex,
    seq_length: int,
    col_close: str = "close"
) -> np.ndarray:
    """
    各シーケンスの未来リターン（%変化）を計算する。

    Args:
        X_3d_numpy (np.ndarray): 3D NumPy配列 (n_samples, seq_len, n_features)。
        feature_names_3d (list): X_3d_numpyの3次元目の特徴量名のリスト。
        horizon (int): 未来のステップ数。
        original_df (pd.DataFrame): 基準となる元のDataFrame（closeカラムを含む）。
        original_indices_filtered (pd.DatetimeIndex): 各シーケンスの開始時刻のインデックス。
        seq_length (int): 各シーケンスの長さ。
        col_close (str): closeカラムの名前。

    Returns:
        np.ndarray: 各シーケンスに対応する未来リターン（%変化）のNumPy配列 (n_samples,).
                    計算できない場合はNaN。
    """
    n_sequences = X_3d_numpy.shape[0]
    future_scores = np.full(n_sequences, np.nan, dtype=np.float32)

    if col_close not in original_df.columns:
        print(f"Error: '{col_close}' column not found in original_df.")
        return future_scores

    close_vals = original_df[col_close].to_numpy().astype(float)
    df_index = original_df.index

    # original_indices_filtered は各シーケンスの開始時刻（start_time）を表す
    # 各シーケンスの終端時刻（end_time）は start_time + (seq_length-1) に対応するoriginal_dfの行になる

    for i in range(n_sequences):
        start_time_of_sequence = original_indices_filtered[i]

        # original_dfにおけるシーケンスの開始位置（行番号）
        try:
            start_pos_in_df = df_index.get_loc(start_time_of_sequence)
        except KeyError:
            # 開始時刻が original_df に存在しない場合、スキップ
            continue

        # シーケンスの終端時刻に対応するoriginal_dfの行番号
        end_pos_in_df = start_pos_in_df + (seq_length - 1)

        # 未来の時刻に対応するoriginal_dfの行番号
        future_pos_in_df = end_pos_in_df + horizon

        # 終端位置と未来位置がoriginal_dfの範囲内であることを確認
        if end_pos_in_df < 0 or end_pos_in_df >= len(df_index) or \
           future_pos_in_df < 0 or future_pos_in_df >= len(df_index):
            continue

        current_close = close_vals[end_pos_in_df]
        future_close = close_vals[future_pos_in_df]

        if current_close != 0 and not np.isnan(current_close) and not np.isnan(future_close):
            future_scores[i] = (future_close - current_close) / current_close * 100.0

    return future_scores


# In[ ]:


import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

# 既存：compute_ATR_simple / calculate_future_score がある前提


def create_strategy_specific_binary_labels_simple_crypto_3cat(
    original_df: pd.DataFrame,
    X_3d_numpy: np.ndarray,
    original_indices_filtered,
    feature_names_3d,
    strategy_labels: np.ndarray,
    horizon: int,
    rolling_window_size: int,           # 未使用（互換のため残す）
    target_positive_rate: float,        # ★ 3戦略共通で pos 比率を固定（train-fit向け）
    dynamic_threshold_percentile_q=95,  # 未使用（互換のため残す）
    ewma_alpha=0.1,                     # 未使用
    lower_threshold_multiplier=1.0,     # 未使用
    hysteresis_M=3,                     # 未使用
    *,
    # --- 追加オプション（必要なら調整） ---
    strategy_id_map: Optional[Dict[int, str]] = None,
    # vol filter
    atr_window: int = 24,
    atr_ratio_th: float = 0.03,         # ATR/close の閾値（例: 3%）
    volume_col: str = "ETH_volume",
    volume_q: float = 0.60,             # 出来高フィルタの分位
    use_vol_filter: bool = True,
    # -1 を使わない運用に寄せる（株式版と整合）
    # True なら「曖昧帯」も 0 に落とす（学習対象外 -1 を極力作らない）
    force_binary_no_minus1: bool = True,
):
    """
    Crypto 3戦略(Downtrend/Range/Uptrend)用の簡略ラベル生成（fold内train-fitに使う前提）

    ✅ 変更点（クリプト3戦略版）
    - reversal_up/down を廃止し、3戦略のみ
    - target_positive_rate を再導入：
        Uptrend   : future_ret >= q(1 - p) を 1
        Downtrend : future_ret <= q(p)      を 1
        Range     : |future_ret| <= q(p)    を 1  （「小さい変動=レンジ継続」を正例）
      ※ p = target_positive_rate
    - 既存の高ボラ(ATR/close) + 出来高フィルタはオプションで維持
    - デフォルトは -1 を使わず 0/1 に寄せる（force_binary_no_minus1=True）

    返り値：
      result: dict[strategy_name] -> np.ndarray (その戦略に属するシーケンス分の 0/1（必要なら-1）)
      互換目的なら「従来通り dict」を返すのが安全。
    """

    # ----------------
    # 基本チェック
    # ----------------
    if original_df is None or not isinstance(original_df, pd.DataFrame) or "close" not in original_df.columns:
        print("Error: original_df is invalid or missing 'close' column. Cannot proceed.")
        return {}

    if not isinstance(X_3d_numpy, np.ndarray) or X_3d_numpy.ndim != 3:
        print("Error: X_3d_numpy must be a 3D numpy array.")
        return {}

    if not isinstance(strategy_labels, np.ndarray) or strategy_labels.ndim != 1:
        print("Error: strategy_labels must be a 1D numpy array.")
        return {}

    if X_3d_numpy.shape[0] != len(strategy_labels):
        print("Error: Mismatch in number of sequences between X_3d_numpy and strategy_labels.")
        return {}

    if X_3d_numpy.shape[2] != len(feature_names_3d):
        print("Error: Mismatch in number of features between X_3d_numpy and feature_names_3d.")
        return {}

    if len(original_indices_filtered) != X_3d_numpy.shape[0]:
        print("Error: Mismatch in number of sequences between X_3d_numpy and original_indices_filtered.")
        return {}

    if target_positive_rate <= 0 or target_positive_rate >= 0.5:
        # Rangeを |r| <= q(p) で作る都合、p>=0.5 はレンジ正例が過大になりやすい
        print("Warning: target_positive_rate should be in (0, 0.5). Using clipped value.")
        target_positive_rate = float(np.clip(target_positive_rate, 1e-3, 0.49))

    n_total_sequences, seq_length, _ = X_3d_numpy.shape

    # index整列
    if not original_df.index.is_monotonic_increasing:
        original_df = original_df.sort_index()

    # ----------------
    # Step 1: future score（%）計算
    # ----------------
    future_scores_all = calculate_future_score(
        X_3d_numpy,
        feature_names_3d,
        horizon,
        original_df,
        original_indices_filtered,
        seq_length,
    ).astype(np.float32)

    valid_scores_mask = np.isfinite(future_scores_all)
    if valid_scores_mask.sum() == 0:
        print("No valid future scores. All labels will be -1 (or 0 if force_binary_no_minus1).")
        fill = 0.0 if force_binary_no_minus1 else -1.0
        full_labels = np.full(n_total_sequences, fill, dtype=float)
        result = {}
        for s in np.unique(strategy_labels):
            idx = np.where(strategy_labels == s)[0]
            result[str(s)] = full_labels[idx]
        return result

    # ----------------
    # Step 2: vol filter（任意）
    # ----------------
    vol_filter_seq = np.ones(n_total_sequences, dtype=bool)

    if use_vol_filter:
        atr24 = compute_ATR_simple(original_df, window=atr_window)
        vol_ratio = atr24 / original_df["close"]

        if volume_col in original_df.columns:
            vol_series = original_df[volume_col]
            vol_threshold = vol_series.quantile(volume_q)
        else:
            vol_series = None
            vol_threshold = None

        vol_filter_seq = np.zeros(n_total_sequences, dtype=bool)

        for i in range(n_total_sequences):
            start_idx = original_indices_filtered[i]
            try:
                start_pos = original_df.index.get_loc(start_idx)
            except KeyError:
                vol_filter_seq[i] = False
                continue

            end_pos = start_pos + (seq_length - 1)
            if end_pos >= len(original_df.index):
                vol_filter_seq[i] = False
                continue

            # ATR条件
            vr = vol_ratio.iloc[end_pos]
            atr_cond = bool(np.isfinite(vr) and (vr > atr_ratio_th))

            # 出来高条件（列があれば）
            if vol_series is not None:
                vv = vol_series.iloc[end_pos]
                vol_cond = bool(np.isfinite(vv) and (vv >= vol_threshold))
            else:
                vol_cond = True

            vol_filter_seq[i] = (atr_cond and vol_cond)

        # ATR が計算できなかった等で全Falseなら無効化
        if (~vol_filter_seq).all():
            print("Warning: vol_filter became all False. Disabling vol filter (all True).")
            vol_filter_seq[:] = True

    # ----------------
    # Step 3: 3戦略ごとの閾値（全体分布から作る：この関数自体は fold-fit 想定）
    # ----------------
    valid_future = future_scores_all[valid_scores_mask]
    p = target_positive_rate

    # Uptrend: 上位pを正例 → thr_up = q(1-p)
    thr_up = float(np.quantile(valid_future, 1.0 - p))

    # Downtrend: 下位pを正例 → thr_dn = q(p)
    thr_dn = float(np.quantile(valid_future, p))

    # Range: |r| の下位pを正例 → thr_rg = q(|r|, p)
    thr_rg = float(np.quantile(np.abs(valid_future), p))

    # ----------------
    # Step 4: 戦略名の正規化
    # ----------------
    if strategy_id_map is None:
        # int id で来てもOK、文字列で来てもOK
        strategy_id_map = {0: "Downtrend", 1: "Range", 2: "Uptrend"}

    def _to_strat_name(x) -> str:
        # bytes/np.str_/int 全部受ける
        if isinstance(x, (bytes, bytearray)):
            s = x.decode("utf-8", errors="ignore")
        else:
            s = str(x)

        # "0"/"1"/"2" のような文字列idも考慮
        if s.isdigit():
            sid = int(s)
            return strategy_id_map.get(sid, s)

        # すでに名前ならそのまま（大小のみ正規化）
        s_low = s.lower()
        if s_low in ("up", "uptrend"):
            return "Uptrend"
        if s_low in ("down", "downtrend"):
            return "Downtrend"
        if s_low in ("range", "sideways"):
            return "Range"

        return s

    # ----------------
    # Step 5: ラベリング（0/1（必要なら-1））
    # ----------------
    if force_binary_no_minus1:
        default_invalid = 0.0
    else:
        default_invalid = -1.0

    full_labels = np.full(n_total_sequences, default_invalid, dtype=float)

    for i in range(n_total_sequences):
        r_f = float(future_scores_all[i])
        if not np.isfinite(r_f):
            full_labels[i] = default_invalid
            continue

        strat_name = _to_strat_name(strategy_labels[i])

        if strat_name == "Uptrend":
            lab = 1.0 if (r_f >= thr_up) else 0.0
        elif strat_name == "Downtrend":
            lab = 1.0 if (r_f <= thr_dn) else 0.0
        elif strat_name == "Range":
            lab = 1.0 if (abs(r_f) <= thr_rg) else 0.0
        else:
            # 不明戦略は学習対象外（または0）
            lab = default_invalid

        # vol filter を満たさないなら学習対象外（または0）
        if use_vol_filter and (not bool(vol_filter_seq[i])):
            lab = default_invalid

        full_labels[i] = lab

    # ----------------
    # Step 6: 戦略ごとに分割して返す（従来互換）
    # ----------------
    result = {}
    unique_strats = np.unique(strategy_labels)
    for s in unique_strats:
        idx = np.where(strategy_labels == s)[0]
        # dictキーは「正規化後の戦略名」に寄せる
        result[_to_strat_name(s)] = full_labels[idx].astype(float)

    # （診断用）閾値を表示したい場合はここでprintしてもOK
    # print({"thr_dn": thr_dn, "thr_rg": thr_rg, "thr_up": thr_up})

    return result


# ----------------------------
# 便利：dict → 全シーケンス配列へ戻す（必要なら）
# ----------------------------
def merge_strategy_label_dict_to_full(
    strategy_labels: np.ndarray,
    label_dict: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    create_strategy_specific_binary_labels_simple_crypto_3cat の返り値(dict)を、
    元の順番の full_labels (N_seq,) に戻すヘルパー。
    """
    n = len(strategy_labels)
    out = np.full(n, np.nan, dtype=float)

    # label_dict の key は正規化後の "Uptrend"/"Downtrend"/"Range"
    def norm(s):
        if isinstance(s, (bytes, bytearray)):
            s = s.decode("utf-8", errors="ignore")
        s = str(s).lower()
        if s in ("up", "uptrend"):
            return "Uptrend"
        if s in ("down", "downtrend"):
            return "Downtrend"
        if s in ("range", "sideways"):
            return "Range"
        # "0"/"1"/"2" を持ってる場合は呼び出し側で変換してください
        return str(s)

    # 各戦略の位置に埋め戻す
    for s in np.unique(strategy_labels):
        key = norm(s)
        idx = np.where(strategy_labels == s)[0]
        arr = label_dict.get(key, None)
        if arr is None or len(arr) != len(idx):
            continue
        out[idx] = arr

    return out


# **Reasoning**:
# The previous command failed because the variables `scores`, `W`, `p_pos`, `alpha`, `n_steps`, and `ignore_value` were not defined in that code block. These variables were defined in the initial part of the `generate_future_based_binary_labels` function. I need to combine the definition of the main function and the subsequent steps of calculating thresholds and generating candidate labels into a single code block to ensure all necessary variables are in scope. I also need to include the definition of `compute_slope` as it was used in the first code block of the main function.
# 
# 

# ## Implement future-based scoring
# 
# ### Subtask:
# Modify or create a function to calculate a score for each sequence based on its future time steps (H to H+L), incorporating metrics like Future Return / ATR, Future MA order/strength, Future Slope. This function will be called within the main labeling function for each sequence.
# 

# ## Calculate dynamic thresholds
# 
# ### Subtask:
# Within the main labeling function, iterate through the sequences and maintain a history of calculated scores. For each sequence, calculate the dynamic threshold based on a rolling window of this score history using percentile and EWMA smoothing, leveraging the `calculate_dynamic_threshold` function.
# 

# ## Summary:
# 
# ### Data Analysis Key Findings
# 
# *   The labeling process involves calculating a future-based score for each time step by analyzing the price movement (specifically, the slope) within a future window defined by a prediction horizon ($H$) and evaluation length ($L$).
# *   Dynamic thresholds for positive and negative labels are determined based on a rolling window ($W$) of these calculated scores, using specified percentiles ($p\_pos$ and $p\_neg$) and Exponentially Weighted Moving Average (EWMA) smoothing ($\alpha$).
# *   Candidate binary labels (1 for positive, 0 for negative, or ignored) are generated by comparing the score at each time step to the corresponding dynamic positive and negative thresholds.
# *   Hysteresis is applied to the candidate labels using a window size ($M$) to smooth transitions and reduce noise, resulting in the final binary labels.
# *   The process includes checks to ensure the DataFrame is long enough to extract the future windows and that required columns are present.
# *   Labels are assigned an `ignore_value` (defaulting to NaN) if scores or thresholds are invalid, or if the hysteresis logic cannot determine a clear state within the specified window.
# 
# ### Insights or Next Steps
# 
# *   The choice of parameters ($H, L, W, p\_pos, p\_neg, \alpha, M$) significantly impacts the resulting labels and requires careful tuning based on the specific dataset and trading strategy.
# *   Consider implementing alternative scoring methods (e.g., percentage change, volatility-adjusted return) within the `generate_future_based_binary_labels` function to explore different ways of quantifying future price movement.
# 
