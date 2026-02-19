#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import cohen_kappa_score
import numpy as np
import torch
import torch.nn as nn
import timm
from timm.data.transforms_factory import create_transform
from pytorch_forecasting import Baseline
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import CrossEntropy
from pytorch_forecasting.data import TimeSeriesDataSet, GroupNormalizer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.core.module import LightningModule
from torchmetrics import Accuracy, Precision, Recall, F1Score
from pytorch_forecasting.metrics import QuantileLoss
import pandas as pd
import model as crypto_model


class TFTModel(LightningModule):  # Define a LightningModule subclass
    def __init__(self,  dataset,
                 hidden_size,
                 lstm_layers,
                 attention_head_size,
                 dropout,
                 hidden_continuous_size,
                 loss,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters for logging
        self.dataset = dataset
        # Create your TemporalFusionTransformer model here
        self.model = TemporalFusionTransformer.from_dataset(
            self.dataset, # datasetはグローバル変数として定義されていると仮定します。
            hidden_size=self.hparams.hidden_size,
            lstm_layers=self.hparams.lstm_layers,
            attention_head_size=self.hparams.attention_head_size,
            dropout=self.hparams.dropout,
            hidden_continuous_size=self.hparams.hidden_continuous_size,
            loss=self.hparams.loss,
            **kwargs
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = {k: v.to(self.device) for k, v in x.items()}
        output = self.model(x)
        loss = output["loss"]
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = {k: v.to(self.device) for k, v in x.items()}  # 入力データをGPUへ
        output = self.model(x)
        loss = output["loss"]
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Define your optimizer here
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


def hyperparameter_tuning_forecasting(df_train, train_labels, n_trials):
    # デバイスの設定 (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def objective(trial):
        # ハイパーパラメータの定義
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
        num_epochs = trial.suggest_int("num_epochs", 10, 20)
        hidden_size = trial.suggest_int("hidden_size", 16, 256)
        lstm_layers = trial.suggest_int("lstm_layers", 1, 3)  # LSTM層数を調整
        dropout = trial.suggest_float("dropout", 0.1, 0.3)   # ドロップアウト率を調整
        hidden_continuous_size = trial.suggest_int("hidden_continuous_size", 8, 64)  # hidden_continuous_sizeを調整
        max_encoder_length = trial.suggest_int("max_encoder_length", 10, 30)
        max_prediction_length = 12


        # TimeSeriesDataSetの作成
        # 必要なカラムを指定し、TimeSeriesDataSetを作成します
        # 'time_idx', 'target', 'group_ids', 'static_categoricals', 'time_varying_known_categoricals',
        # 'time_varying_known_reals', 'time_varying_unknown_reals' などのカラムが必要です
        # 既存のdf_train、train_labels、weightsから適切なカラムを選択して使用してください

        # 例：
        data = df_train.copy()  # df_trainをコピー

        if isinstance(data.index, pd.DatetimeIndex):
            data["time_idx"] = pd.to_datetime(data.index).to_period('H').astype(int)
        else:
            data["time_idx"] = range(len(data))
        data["target"] = train_labels  # 'target'カラムを作成
        data["series_id"] = 0 # 時系列IDを付与（仮想通貨なので常に1つのseries扱い）
        time_varying_unknown_reals = data.columns.tolist()
        time_varying_unknown_reals.remove('time_idx')
        time_varying_unknown_reals.remove('series_id')


        # data = data.replace([np.inf, -np.inf], np.nan)
        # data = data.dropna()



        # 他の必要なカラムを追加
        # ...

        dataset = TimeSeriesDataSet(
            data,
            time_idx="time_idx",
            target="target",
            group_ids=["series_id"],
            min_encoder_length=max_encoder_length,
            max_encoder_length=max_encoder_length,  # 最大エンコーダー長を調整
            min_prediction_length=max_prediction_length,
            max_prediction_length=max_prediction_length,
            static_categoricals=[],
            static_reals=[],
            time_varying_known_categoricals=[],
            time_varying_known_reals=["time_idx"],  # 時刻情報
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals= time_varying_unknown_reals,
            target_normalizer=GroupNormalizer(groups=['series_id']) ,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=False,
        )


        # TimeSeriesSplitで交差検証の準備
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        cumulative_returns = []

        for fold, (train_index, val_index) in enumerate(tscv.split(dataset)):

            # データ分割
            train_data = torch.utils.data.Subset(dataset, train_index)
            val_data = torch.utils.data.Subset(dataset, val_index)
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=False, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False, pin_memory=True)

            # ... (トレーニングループ)
            for batch_idx, (x, y) in enumerate(train_loader):
                # xとyはDataLoaderの作成時にpin_memory=Trueを設定したため、既にページロックメモリに格納されている
                x = x.to(device, non_blocking=True)  # non_blocking=Trueで非同期転送
                y = y.to(device, non_blocking=True)
                # ...

            # ... (バリデーションループ)
            for batch_idx, (x, y) in enumerate(val_loader):
                # xとyはDataLoaderの作成時にpin_memory=Trueを設定したため、既にページロックメモリに格納されている
                x = x.to(device, non_blocking=True)  # non_blocking=Trueで非同期転送
                y = y.to(device, non_blocking=True)

            # モデルの初期化
            # TFTモデルの初期化
            model = TFTModel(
                dataset,
                hidden_size=hidden_size,
                lstm_layers=lstm_layers,
                attention_head_size=hidden_size,
                dropout=dropout,
                hidden_continuous_size=hidden_continuous_size,
                loss=CrossEntropy(),
                output_size=2,
                log_interval=10,
                reduce_on_plateau_patience=4,
                learning_rate=learning_rate,

            )
            # model = TemporalFusionTransformer.from_dataset(
            #     dataset,
            #     learning_rate=learning_rate,
            #     hidden_size=hidden_size,
            #     lstm_layers=lstm_layers,
            #     dropout=dropout,
            #     hidden_continuous_size=hidden_continuous_size,
            #     loss=CrossEntropy(),
            #     output_size=2,
            #     log_interval=10,
            #     reduce_on_plateau_patience=4,
            # )
            model.to(device)
            #　回帰モデル
            #model = StockPricePredictionModel(num_features, hidden_dim, num_static_features, embedding_dim, sequence_length)
            # カスタムトレーナーのインスタンス化
            #model = CustomTrainer(dataset, decay_lambda=0.02)  # 減衰率を0.02に設定

            # 凍結（オプション）
            # for param in model.swin_transformer.parameters():
            #     param.requires_grad = False

            #　分類モデル
            # criterion = nn.CrossEntropyLoss()
            #criterion = stock_model.WeightedKappaLoss(num_classes)
            # criterion = nn.CrossEntropyLoss()

            #　回帰モデル
            # criterion = ListNetLoss()
            # criterion = nn.MSELoss()
            # criterion = nn.HuberLoss()


            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            # # Early Stoppingのパラメータ
            # patience = 3  # 検証データセットでの性能が向上しなくなってから、学習を継続するエポック数
            # best_loss = float('inf')  # 最良の検証データセットでの損失
            # epochs_without_improvement = 0  # 性能が向上していないエポック数

            # Early Stoppingの設定
            early_stopping_callback = EarlyStopping(
                monitor="val_loss", # 監視する指標
                patience=3, # 性能が向上しないエポック数
                mode="min", # 指標が最小値を目指す場合
            )

            # 学習率スケジューラの種類を選択
            scheduler_type = trial.suggest_categorical("scheduler", ["StepLR", "CosineAnnealingLR"])

            # StepLRの場合
            if scheduler_type == "StepLR":
                step_size = trial.suggest_int("step_size", 10, 30)
                gamma = trial.suggest_float("gamma", 0.1, 0.9)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        # CosineAnnealingLRの場合
            else:
                T_max = trial.suggest_int("T_max", 5, 20)
                eta_min = trial.suggest_float("eta_min", 1e-5, 1e-3, log=True)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

            # Trainerの作成
            trainer = Trainer(
                max_epochs=num_epochs,
                accelerator="gpu" if torch.cuda.is_available() else "cpu", # acceleratorとdevicesで指定
                devices=1 if torch.cuda.is_available() else None,
                logger=TensorBoardLogger(save_dir="lightning_logs", name="tft_optuna"), # ログ出力先
                callbacks=[early_stopping_callback], # コールバックの設定
            )

            # 学習の実行
            trainer.fit(
                model,
                train_dataloaders=dataset.to_dataloader(train=True, batch_size=32, num_workers=8),
                val_dataloaders=dataset.to_dataloader(train=False, batch_size=32, num_workers=8),
            )

            # 各エポックの学習損失をOptunaに報告
            for epoch, logs in enumerate(trainer.logged_metrics.items()):
                trial.report(logs["train_loss"], epoch)


            # 各エポックでの学習
            # for epoch in range(num_epochs):

            #     # train_loss = crypto_model.train(model, train_loader, criterion, optimizer, device)
            #     # print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")
            #     # Trainerの作成

            #     # PyTorch LightningのTrainer
            #     #trainer = pl.Trainer(max_epochs=30, gradient_clip_val=0.1)

            #     # 学習の実行


            #     # 各エポックのtrain_lossをOptunaに報告
            #     trial.report(train_loss, epoch)

            #     # 学習率スケジューラの更新
            #     scheduler.step()

            #     # 検証データセットでの評価
            #     val_loss = crypto_model.evaluate_loss(model, val_loader, device, criterion)

            #     # Early Stoppingのチェック
            #     if val_loss < best_loss:
            #         best_loss = val_loss
            #         epochs_without_improvement = 0

            #     else:
            #         epochs_without_improvement += 1
            #         if epochs_without_improvement >= patience:
            #             print(f'Early stopping at epoch {epoch}')
            #             break
    ###############分類モデル#####################################################################
            # 検証フェーズ
            predictions, actuals = crypto_model.evaluate(model, val_loader, device)
            # 検証データのリターンを取得
            # qwk = cohen_kappa_score(actuals, predictions, weights='quadratic')
            # print(f"Fold {fold + 1} QWK: {qwk:.4f}")
            # scores.append(qwk)
            # precision = precision_score(actuals, predictions, average='macro')
            accuracy = accuracy_score(actuals, predictions)
            print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")
            scores.append(accuracy)

        # 平均スコア
        # average_qwk = np.mean(scores)
        # print(f"Mean Cross-Validation Score: {average_qwk:.4f}")
        # return average_qwk
        average_accuracy = np.mean(scores)
        print(f"Mean Cross-Validation Score: {average_accuracy:.4f}")
        return average_accuracy


    # Optunaによる最適化
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials)  # 試行回数を指定

    # 最適なハイパーパラメータの取得
    best_params = study.best_params
    print(f"Best hyperparameters: {best_params}")
    return best_params


# In[ ]:


def objective(trial):
    # ... (ハイパーパラメータの定義)

    # ... (TimeSeriesDataSetの作成)

    # TFTモデルの初期化
    model = TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        loss=CrossEntropy(),
        output_size=2,
        log_interval=10, # ログ出力間隔
        reduce_on_plateau_patience=4,
    )

    # Early Stoppingの設定
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", # 監視する指標
        patience=3, # 性能が向上しないエポック数
        mode="min", # 指標が最小値を目指す場合
    )

    # 学習率スケジューラの設定
    # ...

    # Trainerの作成
    trainer = Trainer(
        max_epochs=num_epochs,
        logger=TensorBoardLogger(save_dir="lightning_logs", name="tft_optuna"), # ログ出力先
        callbacks=[early_stopping_callback], # コールバックの設定
    )

    # 学習の実行
    trainer.fit(
        model,
        train_dataloaders=dataset.to_dataloader(train=True, batch_size=32, num_workers=8),
        val_dataloaders=dataset.to_dataloader(train=False, batch_size=32, num_workers=8),
    )

    # 各エポックの学習損失をOptunaに報告
    for epoch, logs in enumerate(trainer.logged_metrics.items()):
        trial.report(logs["train_loss"], epoch)

    # ... (評価)


# In[ ]:


import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import cohen_kappa_score
import numpy as np
import torch
import torch.nn as nn
import timm
from timm.data.transforms_factory import create_transform
import dataset as stock_dataset
import model as stock_model


def hyperparameter_tuning(df_train, train_labels, weights, n_trials):
    # デバイスの設定 (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def objective(trial, df_train=df_train, train_labels=train_labels, weights=weights):

      # ハイパーパラメータの定義
      learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
      hidden_dim = trial.suggest_int("hidden_dim", 32, 256, step=32)
      sequence_length = trial.suggest_int("sequence_length", 15, 30)
      num_epochs = trial.suggest_int("num_epochs", 10, 20)
      # future_period = trial.suggest_int("future_period", 1, 7)
      # volatility_multiplier = trial.suggest_float("volatility_multiplier", 1.0, 2.0, step=0.1)
      # decay_rate = trial.suggest_float("decay_rate", 0.01, 0.1, step=0.01)
      # quantile = trial.suggest_float("quantile", 0.0, 1.0, step=0.1)

      num_static_features = 0
      embedding_dim = 16
      num_classes = 2



      dataset = stock_dataset.StockDataset(df_train, train_labels, weights, sequence_length)
  #     dataset = TimeSeriesDataSet(
  #     data,
  #     time_idx="time_idx",
  #     target="target",
  #     group_ids=["group_id"],
  #     static_categoricals=["static_cat"],
  #     time_varying_known_categoricals=["known_cat"],
  #     time_varying_known_reals=["known_real"],
  #     time_varying_unknown_reals=["unknown_real"],
  #     max_encoder_length=max_encoder_length,
  #     max_prediction_length=max_prediction_length,
  #     target_normalizer="auto",  # 正規化
  #     add_relative_time_idx=True,
  #     add_target_scales=True,
  #     add_encoder_length=True
  # )

      num_features = df_train.shape[1]


      # TimeSeriesSplitで交差検証の準備
      tscv = TimeSeriesSplit(n_splits=5)
      scores = []
      cumulative_returns = []

      for fold, (train_index, val_index) in enumerate(tscv.split(dataset)):

          # データ分割
          train_data = torch.utils.data.Subset(dataset, train_index)
          val_data = torch.utils.data.Subset(dataset, val_index)
          train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=False)
          val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False)

          # モデルの初期化
          #　分類モデル
          model = stock_model.StockPricePredictionModel(num_features, hidden_dim, sequence_length, num_classes)
          #　回帰モデル
          #model = StockPricePredictionModel(num_features, hidden_dim, num_static_features, embedding_dim, sequence_length)
          # カスタムトレーナーのインスタンス化
          #model = CustomTrainer(dataset, decay_lambda=0.02)  # 減衰率を0.02に設定
          model.to(device)

          # 凍結（オプション）
          # for param in model.swin_transformer.parameters():
          #     param.requires_grad = False

          #　分類モデル
          criterion = nn.CrossEntropyLoss()
          #criterion = stock_model.WeightedKappaLoss(num_classes)
          # criterion = nn.CrossEntropyLoss()

          #　回帰モデル
          # criterion = ListNetLoss()
          # criterion = nn.MSELoss()
          # criterion = nn.HuberLoss()


          optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

          # Early Stoppingのパラメータ
          patience = 3  # 検証データセットでの性能が向上しなくなってから、学習を継続するエポック数
          best_loss = float('inf')  # 最良の検証データセットでの損失
          epochs_without_improvement = 0  # 性能が向上していないエポック数

          # 学習率スケジューラの種類を選択
          scheduler_type = trial.suggest_categorical("scheduler", ["StepLR", "CosineAnnealingLR"])

          # StepLRの場合
          if scheduler_type == "StepLR":
              step_size = trial.suggest_int("step_size", 10, 30)
              gamma = trial.suggest_float("gamma", 0.1, 0.9)
              scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

      # CosineAnnealingLRの場合
          else:
              T_max = trial.suggest_int("T_max", 5, 20)
              eta_min = trial.suggest_float("eta_min", 1e-5, 1e-3, log=True)
              scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)


          # 各エポックでの学習
          for epoch in range(num_epochs):

              train_loss = stock_model.train(model, train_loader, criterion, optimizer, device)
              print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")
              # PyTorch LightningのTrainer
              #trainer = pl.Trainer(max_epochs=30, gradient_clip_val=0.1)

              # 学習の実行
              #trainer.fit(model, train_dataloader)

              # 各エポックのtrain_lossをOptunaに報告
              trial.report(train_loss, epoch)

              # 学習率スケジューラの更新
              scheduler.step()

              # 検証データセットでの評価
              val_loss = stock_model.evaluate_loss(model, val_loader, device, criterion)

              # Early Stoppingのチェック
              if val_loss < best_loss:
                  best_loss = val_loss
                  epochs_without_improvement = 0

              else:
                  epochs_without_improvement += 1
                  if epochs_without_improvement >= patience:
                      print(f'Early stopping at epoch {epoch}')
                      break
  ###############分類モデル#####################################################################
          # 検証フェーズ
          predictions, actuals = stock_model.evaluate(model, val_loader, device)
          # 検証データのリターンを取得
          # qwk = cohen_kappa_score(actuals, predictions, weights='quadratic')
          # print(f"Fold {fold + 1} QWK: {qwk:.4f}")
          # scores.append(qwk)
          # precision = precision_score(actuals, predictions, average='macro')
          accuracy = accuracy_score(actuals, predictions)
          print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")
          scores.append(accuracy)

      # 平均スコア
      # average_qwk = np.mean(scores)
      # print(f"Mean Cross-Validation Score: {average_qwk:.4f}")
      # return average_qwk
      average_accuracy = np.mean(scores)
      print(f"Mean Cross-Validation Score: {average_accuracy:.4f}")
      return average_accuracy


    # Optunaによる最適化
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials)  # 試行回数を指定

    # 最適なハイパーパラメータの取得
    best_params = study.best_params
    print(f"Best hyperparameters: {best_params}")
    return best_params


# In[1]:


def focal_loss_lgb(y_pred, dtrain, alpha=0.25, gamma=2.0):
    y_true = dtrain.get_label()
    p = 1 / (1 + np.exp(-y_pred))

    grad = alpha * y_true * (1 - p)**gamma * (gamma * p * np.log(p + 1e-12) + p - 1) + \
           (1 - alpha) * (1 - y_true) * p**gamma * (-gamma * (1 - p) * np.log(1 - p + 1e-12) + p)

    hess = alpha * y_true * (1 - p)**gamma * (
        (gamma * (1 - p - p * np.log(p + 1e-12))) + 1
    ) * p * (1 - p) + \
           (1 - alpha) * (1 - y_true) * p**gamma * (
               (gamma * (p - (1 - p) * np.log(1 - p + 1e-12))) + 1
           ) * p * (1 - p)

    return grad, hess


# In[ ]:


# F1スコアを返す feval 関数
def focal_loss_eval(y_pred, data):
    y_true = data.get_label()
    y_prob = 1.0 / (1.0 + np.exp(-y_pred))
    y_pred_label = (y_prob > 0.5).astype(int)
    return 'f1', f1_score(y_true, y_pred_label), True  # True = 高い方が良い


# In[ ]:


import lightgbm as lgb
import optuna
from imblearn.combine import SMOTEENN
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score,f1_score  # 分類問題なのでaccuracy_scoreに変更

def hyperparameter_tuning_lgbm_classification(df_train, train_labels, n_trials):
    """
    LightGBMで分類モデルのハイパーパラメータチューニングを行う関数

    Args:
        df_train (pd.DataFrame): 訓練データ
        train_labels (pd.Series): 訓練データのラベル
        weights (pd.Series): 訓練データの重み
        n_trials (int): 試行回数

    Returns:
        dict: 最適なハイパーパラメータ
    """

    def objective(trial):

        # # 正例と負例の件数を取得
        # num_pos = len(train_labels[train_labels == 1])
        # num_neg = len(train_labels[train_labels == 0])

        # # scale_pos_weightを計算
        # scale_pos_weight = num_neg / num_pos
        # ハイパーパラメータの定義

        params = {
        "objective":focal_loss_lgb,
        "metric": "None",
        "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart", "goss"]) ,
        "verbosity": -1,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 20.0),
         }
        for col in df_train.select_dtypes(include=['object']).columns:
            try:
                df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
            except ValueError:
                df_train[col] = df_train[col].astype('category').cat.codes

        # GOSSが使用される場合にbaggingを無効にする条件
        if params['boosting_type'] == 'goss':
            params['bagging_fraction'] = 1.0
            params['bagging_freq'] = 0

        # TimeSeriesSplitで交差検証の準備
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []

        for fold, (train_index, val_index) in enumerate(tscv.split(df_train)):
            # データ分割
            X_train, X_val = df_train.iloc[train_index], df_train.iloc[val_index]
            y_train, y_val = train_labels.iloc[train_index], train_labels.iloc[val_index]

            # X_train.fillna(0, inplace=True)
            # X_val.fillna(0, inplace=True)

            # # SMOTEENNを用いて訓練データを前処理
            # smote_enn = SMOTEENN(sampling_strategy='minority')
            # X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)

            #LightGBMデータセットの作成
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val)

            model = lgb.train(
                    params,  # ハイパーパラメータ
                    train_data,  # 学習データ
                    valid_sets=[train_data, val_data],  # 検証データ
                    feval=focal_loss_eval,  # カスタム評価関数
                    num_boost_round=1000,  # 最大イテレーション回数
                    callbacks=[lgb.early_stopping(stopping_rounds=100),lgb.log_evaluation(period=100)],  # ログ出力の設定
                )

            # 予測
            y_pred_prob = model.predict(X_val)

            # しきい値の調整も含めて最適化
            threshold = trial.suggest_float("threshold", 0.1, 0.9)
            y_pred = (y_pred_prob > threshold).astype(int)


            # 評価指標の計算 (f1スコア)
            score = f1_score(y_val, y_pred)
            scores.append(score)

        # 平均スコア
        average_score = np.mean(scores)
        return average_score

    # Optunaによる最適化 (Accuracyを最大化するように設定)
    study = optuna.create_study(direction='maximize')  # directionをmaximizeに変更
    study.optimize(objective, n_trials=n_trials)

    # 最適なハイパーパラメータの取得
    best_params = study.best_params
    print(f"Best hyperparameters: {best_params}")
    return best_params


# In[ ]:


import lightgbm as lgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_log_error

def hyperparameter_tuning_lgbm_regression(df_train, train_labels, weights, n_trials):
    """
    LightGBMで回帰モデルのハイパーパラメータチューニングを行う関数

    Args:
        df_train (pd.DataFrame): 訓練データ
        train_labels (pd.Series): 訓練データのラベル
        weights (pd.Series): 訓練データの重み
        n_trials (int): 試行回数

    Returns:
        dict: 最適なハイパーパラメータ
    """

    def objective(trial):
        # ハイパーパラメータの定義
        params = {
            'objective': 'regression',  # 回帰に変更
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
            'num_leaves': trial.suggest_int('num_leaves', 31, 127),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
        }

        # GOSSが使用される場合にbaggingを無効にする条件
        if params['boosting_type'] == 'goss':
            params['bagging_fraction'] = 1.0
            params['bagging_freq'] = 0

        # TimeSeriesSplitで交差検証の準備
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []

        for fold, (train_index, val_index) in enumerate(tscv.split(df_train)):
            # データ分割
            X_train, X_val = df_train.iloc[train_index], df_train.iloc[val_index]
            y_train, y_val = train_labels.iloc[train_index], train_labels.iloc[val_index]

            # LightGBMデータセットの作成
            train_data = lgb.Dataset(X_train, label=y_train, weight=weights.iloc[train_index])
            val_data = lgb.Dataset(X_val, label=y_val, weight=weights.iloc[val_index])

            # モデルの学習
            model = lgb.train(params, train_data, valid_sets=[train_data, val_data], callbacks=[lgb.log_evaluation(period=100)], )

            # 検証フェーズ
            predictions = model.predict(X_val, num_iteration=model.best_iteration)

            # 評価指標の計算 (MSLE)
            score = mean_squared_log_error(y_val, predictions) # KLダイバージェンスでもよい
            scores.append(score)

        # 平均スコア (RMSE)
        average_score = np.mean(scores)
        return average_score  # MSLEを返す

    # Optunaによる最適化 (RMSEを最小化するように設定)
    study = optuna.create_study(direction='minimize')  # directionをminimizeに変更
    study.optimize(objective, n_trials=n_trials)

    # 最適なハイパーパラメータの取得
    best_params = study.best_params
    print(f"Best hyperparameters: {best_params}")
    return best_params


# In[ ]:


import numpy as np
from scipy.stats import entropy

# 真値と予測値
y_true = np.random.randn(1000)  # 例として、正規分布に従う乱数を生成
y_pred = np.random.randn(1000) + 0.1  # 予測値は真値に少しノイズを加えたもの

# ヒストグラムを用いて分布を推定
hist_true, bins_true = np.histogram(y_true, bins=50, density=True)
hist_pred, bins_pred = np.histogram(y_pred, bins=bins_true, density=True)  # binsを揃える

# KLダイバージェンスを計算
kl_divergence = entropy(hist_true, hist_pred)

print(f"KL Divergence: {kl_divergence}")

