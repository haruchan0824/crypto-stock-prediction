#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import talib
from scipy.stats import entropy

# 曜日や月（例：day_of_week, month）
def create_date_features(df):
  """
  曜日や月を特徴量としてデータフレームに追加する関数

  Args:
    df: Pandas DataFrame.

  Returns:
    特徴量が追加されたPandas DataFrame.
  """

  df["hour"] = df.index.hour
  df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
  df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

  df["day"] = df.index.dayofyear # 1年のうちの経過日数
  df["day_sin"] = np.sin(2 * np.pi * df["day"] / 365.25) # うるう年考慮
  df["day_cos"] = np.cos(2 * np.pi * df["day"] / 365.25) # うるう年考慮

  df['day_of_week'] = df.index.dayofweek  # 曜日 (0:月曜日, 1:火曜日, ..., 6:日曜日)
  df['month'] = df.index.month  # 月 (1:1月, 2:2月, ..., 12:12月)
  df['hour_of_day'] = df.index.hour  # 時間 (0:00:00, 1:01:00, ..., 23:23:00
  return df

def compute_sign_entropy(series: pd.Series, window: int = 24) -> pd.Series:
    """
    ローリングウィンドウ内のリターン符号列（+/-）のエントロピーを計算
    """
    # リターンの符号 (+1 or -1)
    sign_series = np.sign(series.diff().fillna(0))

    def rolling_entropy(x):
        counts = np.bincount((x + 1).astype(int))  # -1→0, 0→1, +1→2
        probs = counts / counts.sum()
        return entropy(probs, base=2)

    return sign_series.rolling(window=window, min_periods=window).apply(rolling_entropy, raw=True)

def detect_ma_cross(price: pd.Series, short_window: int, long_window: int) -> pd.Series:
    short_ma = price.rolling(window=short_window).mean()
    long_ma = price.rolling(window=long_window).mean()

    # 差分の符号の変化で交差を検出
    signal = short_ma - long_ma

    # ゴールデンクロス (signalが0以上になり、前が0未満) または デッドクロス (signalが0以下になり、前が0超)
    cross_flag = np.where(
        (signal.shift(1) < 0) & (signal >= 0), 1,  # ゴールデンクロス
        np.where(
            (signal.shift(1) > 0) & (signal <= 0), -1,  # デッドクロス
            0 # それ以外
        )
    )
    return pd.Series(cross_flag, index=price.index).fillna(0)

def generate_features(df: pd.DataFrame, train_mask: pd.Series | None = None) -> tuple[pd.DataFrame, list[str], dict]:
    """
    指定された特徴量カテゴリに基づいてtalibを用いて特徴量を作成し、標準化する関数。
    新しいフラグ特徴量（トレンド強度×ボラ正規化、モメンタム持続、押し目/ブレイク、出来高×価格方向一致）を追加。
    train_maskを使用して、訓練期間で統計量をフィットさせ、全期間に適用する。

    Args:
        df (pd.DataFrame): 入力データフレーム
        train_mask (pd.Series | None): 訓練期間を示すブールマスク。Noneの場合は最初の70%を訓練期間とする。

    Returns:
        tuple[pd.DataFrame, list[str], dict]: 特徴量を追加・標準化したデータフレーム、連続特徴量カラム名一覧、fitした統計量params
    """
    if train_mask is None:
        train_mask = pd.Series(False, index=df.index)
        train_mask.iloc[: int(len(df) * 0.7)] = True
    train_mask = train_mask.fillna(False)

    # def _winsorize_by_train(s: pd.Series, q_low: float, q_high: float, mask: pd.Series):
    #     # trainで分位点fit → 全期間クリップ
    #     tri = s.loc[mask].dropna()
    #     if len(tri) == 0:
    #         return s, (np.nan, np.nan)
    #     try: # 少数データなどでエラーになる可能性考慮
    #         lo, hi = np.quantile(tri, [q_low, q_high])
    #     except:
    #          return s, (np.nan, np.nan)
    #     return s.clip(lower=lo, upper=hi), (float(lo), float(hi))

    # def _standardize_by_train(s: pd.Series, mask: pd.Series):
    #     tri = s.loc[mask].dropna()
    #     if len(tri) == 0:
    #         return s, (np.nan, np.nan)
    #     mu, sd = tri.mean(), tri.std()
    #     if sd == 0 or not np.isfinite(sd):
    #         sd = 1.0
    #     return (s - mu) / sd, (float(mu), float(sd))

    features = []
    cont_cols: list[str] = [] # 標準化する連続特徴量のリスト
    params: dict = {"winsor": {}, "standardize": {}} # 標準化に使用したパラメータ

    df_processed = df.copy() # 元のDataFrameを変更しないようにコピー

    # 元のデータフレームに存在し、標準化したい連続値カラムを追加
    original_cont_cols = ['open', 'high', 'low', 'close', 'volume_USD', 'volume_ETH'] # 例としてこれらのカラムを追加
    cont_cols.extend([col for col in original_cont_cols if col in df_processed.columns])


    # 1) リターン（変化率）：方向性の一次情報
    df_processed['ret_1h'] = df_processed['close'].pct_change(1)
    df_processed['ret_6h'] = df_processed['close'].pct_change(6)
    df_processed['ret_12h'] = df_processed['close'].pct_change(12)
    cont_cols.extend(['ret_1h', 'ret_6h', 'ret_12h'])

    # リターン系列 (log return)
    df_processed['log_return_1h'] = np.log(df_processed['close'] / df_processed['close'].shift(1))
    df_processed['log_return_3h'] = np.log(df_processed['close'] / df_processed['close'].shift(3))
    df_processed['log_return_6h'] = np.log(df_processed['close'] / df_processed['close'].shift(6))
    df_processed['log_return_24h'] = np.log(df_processed['close'] / df_processed['close'].shift(24))
    cont_cols.extend(['log_return_1h', 'log_return_3h', 'log_return_6h', 'log_return_24h'])


    # 2) モメンタム＆加速度：継続/反転の兆候
    df_processed['momentum_6h'] = df_processed['close'] - df_processed['close'].shift(6)
    df_processed['momentum_12h'] = df_processed['close'] - df_processed['close'].shift(12)
    # 加速度はモメンタムの変化率や差分で定義することが多い
    df_processed['acceleration_6h'] = df_processed['momentum_6h'].diff() # 6時間モメンタムの差分を加速度として定義
    cont_cols.extend(['momentum_6h', 'momentum_12h', 'acceleration_6h'])

    # 3) ボラティリティとその変化（拡張/収縮）
    df_processed['volatility_6h'] = df_processed['close'].rolling(window=6).std()
    df_processed['volatility_12h'] = df_processed['close'].rolling(window=12).std()
    df_processed['volatility_change_6h'] = df_processed['volatility_6h'].pct_change(6) # 6時間ボラティリティの変化率
    cont_cols.extend(['volatility_6h', 'volatility_12h', 'volatility_change_6h'])

    # 4) トレンド強度＆傾き（短期MAと長期MAの関係）
    df_processed['ma_6h'] = df_processed['close'].rolling(window=6).mean()
    df_processed['ma_24h'] = df_processed['close'].rolling(window=24).mean()
    df_processed['trend_strength_6_24'] = df_processed['ma_6h'] - df_processed['ma_24h'] # 短期MAと長期MAの差
    df_processed['trend_slope_6h'] = df_processed['ma_6h'].diff() # 短期MAの傾き
    cont_cols.extend(['ma_6h', 'ma_24h', 'trend_strength_6_24', 'trend_slope_6h'])


    # テクニカル指標 (talibを用いて書き換え)
    # Talibの結果は標準化対象とする
    df_processed['RSI'] = talib.RSI(df_processed['close'], timeperiod=14)
    df_processed['MACD'], df_processed['MACD_signal'], df_processed['MACD_hist'] = talib.MACD(df_processed['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df_processed['upper'], df_processed['middle'], df_processed['lower'] = talib.BBANDS(df_processed['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df_processed['BB_width'] = df_processed['upper'] - df_processed['lower']
    df_processed['EMA_12'] = talib.EMA(df_processed['close'], timeperiod=12)
    df_processed['EMA_26'] = talib.EMA(df_processed['close'], timeperiod=26)
    df_processed['EMA_ratio'] = df_processed['EMA_12'] / (df_processed['EMA_26'] + 1e-9) # ゼロ除算防止
    # ストキャスティクス
    df_processed['slowk'], df_processed['slowd'] = talib.STOCH(df_processed['high'], df_processed['low'], df_processed['close'], fastk_period=14, slowk_period=3)
    # ATR
    df_processed['ATR24'] = talib.ATR(df_processed['high'], df_processed['low'], df_processed['close'], timeperiod=24)
    cont_cols.extend(['RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'upper', 'middle', 'lower', 'BB_width', 'EMA_12', 'EMA_26', 'EMA_ratio', 'slowk', 'slowd', 'ATR', 'ATR24'])


    # エントロピー特徴量（ローリング・リターン符号列）
    df_processed['sign_entropy_24'] = compute_sign_entropy(df_processed['close'], window=24)
    df_processed['sign_entropy_12'] = compute_sign_entropy(df_processed['close'], window=12)
    df_processed['sign_entropy_12_ma3'] = df_processed['sign_entropy_12'].rolling(window=3).mean()
    df_processed['sign_entropy_12_ma3_slope'] = df_processed['sign_entropy_12_ma3'].diff()
    df_processed['sign_entropy_24_ma3'] = df_processed['sign_entropy_24'].rolling(window=3).mean()
    df_processed['sign_entropy_24_ma3_slope'] = df_processed['sign_entropy_24_ma3'].diff()
    cont_cols.extend(['sign_entropy_24', 'sign_entropy_12', 'sign_entropy_12_ma3', 'sign_entropy_12_ma3_slope', 'sign_entropy_24_ma3', 'sign_entropy_24_ma3_slope'])


    # 移動平均 (MA) - 上で定義した ma_6h, ma_24h と重複するが、期間違いを追加
    df_processed['MA_t_6'] = df_processed['close'].rolling(window=6).mean()
    df_processed['MA_t_12'] = df_processed['close'].rolling(window=12).mean()
    df_processed['MA_t_24'] = df_processed['close'].rolling(window=24).mean()
    df_processed['MA_t_48'] = df_processed['close'].rolling(window=48).mean() # 24時間移動平均を例として追加
    df_processed['MA_t_72'] = df_processed['close'].rolling(window=72).mean() # 24時間移動平均を例として追加
    df_processed['MA_t_96'] = df_processed['close'].rolling(window=96).mean() # 24時間移動平均を例として追加
    df_processed['MA_t_168'] = df_processed['close'].rolling(window=168).mean() # 24時間移動平均を例として追加
    df_processed['MA_t_336'] = df_processed['close'].rolling(window=336).mean() # 24時間移動平均を例として追加
    cont_cols.extend(['MA_t_6', 'MA_t_12', 'MA_t_24', 'MA_t_48', 'MA_t_72', 'MA_t_96', 'MA_t_168', 'MA_t_336'])



    # ラグ特徴量 (Lag)
    df_processed['Lag_t_1'] = df_processed['close'].shift(1)  # 1時間前の値を例として追加
    df_processed['Lag_t_3'] = df_processed['close'].shift(3)  # 3時間前の値を例として追加
    cont_cols.extend(['Lag_t_1', 'Lag_t_3'])

    # トレンド差分 (Diff)
    df_processed['Diff_t_6'] = df_processed['close'] - df_processed['MA_t_6']  # 6時間移動平均との差分を例として追加
    df_processed['Diff_t_24'] = df_processed['close'] - df_processed['MA_t_24'] # 24時間移動平均との差分を例として追加
    cont_cols.extend(['Diff_t_6', 'Diff_t_24'])


    # 移動平均の傾き - 上で定義した trend_slope_6h と重複するが、期間違いを追加
    df_processed['MA_t_6_slope'] = df_processed['MA_t_6'].diff()
    df_processed['MA_t_12_slope'] = df_processed['MA_t_12'].diff()
    df_processed['MA_t_24_slope'] = df_processed['MA_t_24'].diff()
    df_processed['MA_t_48_slope'] = df_processed['MA_t_48'].diff()
    df_processed['MA_t_72_slope'] = df_processed['MA_t_72'].diff()
    df_processed['MA_t_96_slope'] = df_processed['MA_t_96'].diff()
    df_processed['MA_t_168_slope'] = df_processed['MA_t_168'].diff()
    df_processed['MA_t_336_slope'] = df_processed['MA_t_336'].diff()
    # 既に cont_cols に追加済み: ['MA_t_6_slope', 'MA_t_12_slope', 'MA_t_24_slope', 'MA_t_48_slope', 'MA_t_72_slope', 'MA_t_96_slope', 'MA_t_168_slope', 'MA_t_336_slope']


    # 移動平均の傾きの変化率
    df_processed['MA_t_6_slope_pct_change'] = df_processed['MA_t_6_slope'].pct_change()
    df_processed['MA_t_12_slope_pct_change'] = df_processed['MA_t_12_slope'].pct_change()
    df_processed['MA_t_24_slope_pct_change'] = df_processed['MA_t_24_slope'].pct_change()
    df_processed['MA_t_48_slope_pct_change'] = df_processed['MA_t_48_slope'].pct_change()
    df_processed['MA_t_72_slope_pct_change'] = df_processed['MA_t_72_slope'].pct_change()
    df_processed['MA_t_96_slope_pct_change'] = df_processed['MA_t_96_slope'].pct_change()
    df_processed['MA_t_168_slope_pct_change'] = df_processed['MA_t_168_slope'].pct_change()
    df_processed['MA_t_336_slope_pct_change'] = df_processed['MA_t_336_slope'].pct_change()
    cont_cols.extend(['MA_t_6_slope_pct_change', 'MA_t_12_slope_pct_change', 'MA_t_24_slope_pct_change', 'MA_t_48_slope_pct_change', 'MA_t_72_slope_pct_change', 'MA_t_96_slope_pct_change', 'MA_t_168_slope_pct_change', 'MA_t_336_slope_pct_change'])

    # 収束と発散
    df_processed['MA_convergence_divergence'] = df_processed['MA_t_6'] - df_processed['MA_t_24']  # 収束なら値は小さく、発散なら値は大きくなる
    cont_cols.append('MA_convergence_divergence')

    # 移動平均のクロスフラグ (カテゴリ型なので標準化対象外)
    df_processed['MA_6_24_cross_flag'] = pd.Series(np.where(df_processed['MA_t_6'] > df_processed['MA_t_24'], 1, 0), index=df_processed.index).diff().fillna(0).astype('category')
    df_processed['MA_12_48_cross_flag'] = pd.Series(np.where(df_processed['MA_t_12'] > df_processed['MA_t_48'], 1, 0), index=df_processed.index).diff().fillna(0).astype('category')
    df_processed['MA_24_72_cross_flag'] = pd.Series(np.where(df_processed['MA_t_24'] > df_processed['MA_t_72'], 1, 0), index=df_processed.index).diff().fillna(0).astype('category')


    # 移動平均の傾きの符号変化フラグ (カテゴリ型なので標準化対象外)
    df_processed['MA_slope_6_24_change_flag'] = pd.Series(np.where(df_processed['MA_t_6_slope'] * df_processed['MA_t_24_slope'] < 0, 1, 0), index=df_processed.index).astype('category')
    df_processed['MA_slope_12_48_change_flag'] = pd.Series(np.where(df_processed['MA_t_12_slope'] * df_processed['MA_t_48_slope'] < 0, 1, 0), index=df_processed.index).astype('category')
    df_processed['MA_slope_24_72_change_flag'] = pd.Series(np.where(df_processed['MA_t_24_slope'] * df_processed['MA_t_72_slope'] < 0, 1, 0), index=df_processed.index).astype('category')

    # 移動平均の傾きの変化率の符号変化フラグ (カテゴリ型なので標準化対象外)
    df_processed['MA_slope_pct_change_6_24_change_flag'] = pd.Series(np.where(df_processed['MA_t_6_slope_pct_change'] * df_processed['MA_t_24_slope_pct_change'] < 0, 1, 0), index=df_processed.index).astype('category')
    df_processed['MA_slope_pct_change_12_48_change_flag'] = pd.Series(np.where(df_processed['MA_t_12_slope_pct_change'] * df_processed['MA_t_48_slope_pct_change'] < 0, 1, 0), index=df_processed.index).astype('category')
    df_processed['MA_slope_pct_change_24_72_change_flag'] = pd.Series(np.where(df_processed['MA_t_24_slope_pct_change'] * df_processed['MA_t_72_slope_pct_change'] < 0, 1, 0), index=df_processed.index).astype('category')

    # 短期(6), 中期(24), 長期(72)移動平均の順序と傾きに基づいたフラグ (カテゴリ型なので標準化対象外)
    # (MA_6 > MA_24 > MA_72) かつ (MA_6_slope > 0, MA_24_slope > 0, MA_72_slope > 0)
    df_processed['MA_6_24_72_trend_flag'] = pd.Series(np.where(
        (df_processed['MA_t_6'] > df_processed['MA_t_24']) &
        (df_processed['MA_t_24'] > df_processed['MA_t_72']) &
        (df_processed['MA_t_6_slope'] > 0) &
        (df_processed['MA_t_24_slope'] > 0) &
        (df_processed['MA_t_72_slope'] > 0),
        1,
        0
    ), index=df_processed.index).astype('category') # Convert to Series first


    # ボラティリティの変化フラグ
    # apply_cusum_test関数が別のセルで定義されている前提
    # df['volatility_change_flag'] = apply_cusum_test(df['rolling_std_6h']/df['rolling_std_12h'], quantile=0.90, drift=0.0).astype('category')
    # 一時的にコメントアウト、必要に応じて apply_cusum_test 関数をこのセルに含めるか、使用前に定義されていることを確認してください。
    # または、CUSUM検定なしの簡易的なボラティリティ変化フラグを実装することも可能です。
    # 例：直近のボラティリティが過去N期間の平均よりX%以上高い場合など

    # --- 新しいフラグ特徴量の追加（元データの列があれば連続値として標準化対象に） ---

    # 1. トレンド強度×ボラ正規化フラグ
    # ATR_W が定義されていないため、既存の ATR を使用します。
    # q_hi は分位数計算のために必要ですが、ここでは例として上位90%と下位10%を閾値とします。
    if 'MA_t_6' in df_processed.columns and 'MA_t_24' in df_processed.columns and 'ATR' in df_processed.columns:
        # MA差分をATRで正規化
        ma_diff_normalized = (df_processed['MA_t_6'] - df_processed['MA_t_24']) / (df_processed['ATR'] + 1e-9) # ゼロ除算防止
        df_processed['trend_strength_vol_flag_cont'] = ma_diff_normalized # 連続値として追加
        cont_cols.append('trend_strength_vol_flag_cont')

        # 分位数を計算 (データ全体ではなく、計算可能な期間で) - これはフラグ生成にのみ使用
        # q_hi = ma_diff_normalized.quantile(0.90) # 例: 上位90%
        # q_lo = ma_diff_normalized.quantile(0.10) # 例: 下位10%

        # フラグ設定 (カテゴリ型なので標準化対象外)
        # df_processed['trend_strength_vol_flag'] = pd.Series(np.where(
        #     ma_diff_normalized > q_hi, 1, # 強上昇
        #     np.where(
        #         ma_diff_normalized < q_lo, -1, # 強下降
        #         0 # その他
        #     )
        # ), index=df_processed.index).astype('category') # Convert to Series first
    else:
        print("Warning: Required columns for 'trend_strength_vol_flag_cont' not found.")


    # 2. モメンタム持続（K-of-L）
    # L=5, K=4 を例とします。直近5本のうち4本で log_return_1h > 0 が成立
    L = 5
    K = 4
    if 'log_return_1h' in df_processed.columns:
        # log_return_1h > 0 の場合に1、そうでなければ0
        positive_return_flag = (df_processed['log_return_1h'] > 0).astype(int)
        # 直近L期間の1の数をカウント
        rolling_positive_sum = positive_return_flag.rolling(window=L, min_periods=L).sum()
        # カウント値を連続値として追加
        df_processed['momentum_persistence_flag_cont'] = rolling_positive_sum
        cont_cols.append('momentum_persistence_flag_cont')
        # カウントがK以上であればモメンタム持続フラグを1とする (カテゴリ型なので標準化対象外)
        # df_processed['momentum_persistence_flag'] = (rolling_positive_sum >= K).astype(int).astype('category')
    else:
        print("Warning: 'log_return_1h' column not found. Skipping 'momentum_persistence_flag_cont'.")


    # 3. 押し目/ブレイクの判定
    # M=3 を例とします。close > upper_band の連続本数 ≥ 3 で「ブレイク」フラグ
    # RSI∈[45,65] の条件も追加
    M = 3
    if 'close' in df_processed.columns and 'upper' in df_processed.columns and 'MA_t_24' in df_processed.columns and 'RSI' in df_processed.columns:
        # close > upper_band の連続本数をカウント
        close_above_upper = (df_processed['close'] > df_processed['upper']).astype(int)
        rolling_above_upper_sum = close_above_upper.rolling(window=M, min_periods=M).sum()
        df_processed['breakout_flag_cont'] = rolling_above_upper_sum # 連続値として追加
        cont_cols.append('breakout_flag_cont')

        # close > MA_24 かつ RSI∈[45,65] の条件
        pullback_condition = (df_processed['close'] > df_processed['MA_t_24']) & (df_processed['RSI'] >= 45) & (df_processed['RSI'] <= 65)
        df_processed['pullback_condition_cont'] = pullback_condition.astype(float) # 連続値として追加
        cont_cols.append('pullback_condition_cont')

        # フラグ統合: ブレイクアウトの場合は1、押し目維持の場合は-1、その他は0 (カテゴリ型なので標準化対象外)
        # df_processed['pullback_breakout_flag'] = pd.Series(np.where(
        #     rolling_above_upper_sum == M, 1, # ブレイクアウト (簡易判定)
        #     np.where(
        #         pullback_condition, -1, # 押し目維持
        #         0 # その他
        #     )
        # ), index=df_processed.index).astype('category')
    else:
        print("Warning: Required columns for 'breakout_flag_cont' or 'pullback_condition_cont' not found.")


    # 4. 出来高×価格方向の一致
    # volume_zscore が定義されていないため、volume_USD のZ-scoreを使用します。
    # z0 = 1.0 を例とします。
    z0 = 1.0
    if 'volume_USD' in df_processed.columns and 'log_return_1h' in df_processed.columns:
        # 出来高のZ-scoreを計算 (ローリングZ-score window=24)
        volume_rolling_mean = df_processed['volume_USD'].rolling(window=24).mean()
        volume_rolling_std = df_processed['volume_USD'].rolling(window=24).std()
        volume_zscore = (df_processed['volume_USD'] - volume_rolling_mean) / (volume_rolling_std + 1e-9) # ゼロ除算防止
        df_processed['volume_zscore_24h'] = volume_zscore # 連続値として追加
        cont_cols.append('volume_zscore_24h')

        # 条件: 出来高zscore > z0 かつ log_return_1h > 0
        bullish_volume_condition = (volume_zscore > z0) & (df_processed['log_return_1h'] > 0)
        df_processed['bullish_volume_condition_cont'] = bullish_volume_condition.astype(float) # 連続値として追加
        cont_cols.append('bullish_volume_condition_cont')

        # 条件: 出来高zscore > z0 かつ log_return_1h < 0
        bearish_volume_condition = (volume_zscore > z0) & (df_processed['log_return_1h'] < 0)
        df_processed['bearish_volume_condition_cont'] = bearish_volume_condition.astype(float) # 連続値として追加
        cont_cols.append('bearish_volume_condition_cont')


        # フラグ設定: 強気一致なら1、弱気一致なら-1、その他0 (カテゴリ型なので標準化対象外)
        # df_processed['volume_price_direction_flag'] = pd.Series(np.where(
        #     bullish_volume_condition, 1, # 強気一致 (順張り優位)
        #     np.where(
        #         bearish_volume_condition, -1, # 弱気一致
        #         0 # その他
        #     )
        # ), index=df_processed.index).astype('category') # Convert to Series first
    else:
        print("Warning: Required columns for volume/price direction features not found.")

    # # --- 標準化処理 ---
    # winsor_q = (0.005, 0.995) # winsorizeの分位点

    # # 連続特徴量をWinsorize & Standardize
    # # cont_cols リストに含まれる全ての特徴量を処理
    # for col in cont_cols:
    #     if col in df_processed.columns:
    #         # Winsorize
    #         df_processed[col], wq = _winsorize_by_train(df_processed[col], winsor_q[0], winsor_q[1], train_mask)
    #         params["winsor"][col] = {"low": wq[0], "high": wq[1]}

    #         # Standardize
    #         df_processed[col], st = _standardize_by_train(df_processed[col], train_mask)
    #         params["standardize"][col] = {"mean": st[0], "std": st[1]}
    #     else:
    #          print(f"Warning: Continuous feature column '{col}' not found after processing. Skipping standardization.")


    def _roll_mean(s, w):  # NaN安全な rolling mean
        return s.rolling(int(w), min_periods=1).mean()

    def _roll_std(s, w):
        return s.rolling(int(w), min_periods=1).std()

    def _eps():
        return 1e-8

    # 以下の特徴量は既に上の連続特徴量生成部分で考慮されている可能性があるが、
    # 構造を維持するため、ここでは標準化せずに追加する。
    # もし標準化が必要であれば、cont_cols に追加する必要がある。

    # 1) リターン（変化率）：方向性の一次情報
    # すでにcont_colsに追加・標準化済み: "ret_1h", "ret_6h", "ret_12h"

    # 2) モメンタム＆加速度：継続/反転の兆候
    # すでにcont_colsに追加・標準化済み: "momentum_6h", "acceleration_6h"

    # 3) ボラティリティとその変化（拡張/収縮）
    # すでにcont_colsに追加・標準化済み: "volatility_6h", "volatility_change_6h"

    # 4) トレンド強度＆傾き（短期MAと長期MAの関係）
    # すでにcont_colsに追加・標準化済み: "ma_6h", "ma_24h", "trend_strength_6_24", "trend_slope_6h"


    # 5) ボリンジャーバンド幅とその変化（ブレイクの前兆）
    _ma20  = _roll_mean(df_processed["close"], 20)
    _std20 = _roll_std(df_processed["close"], 20)
    df_processed["bb_width_20"] = (2.0 * _std20) / (_ma20.abs() + _eps()) # raw値を追加し標準化
    df_processed["bb_width_change_6"] = df_processed["bb_width_20"].pct_change(6) # raw値を追加し標準化
    cont_cols.extend(["bb_width_20", "bb_width_change_6"])


    # 6) 出来高×価格の整合（フロー強度）
    #    ・短期の価格リターンと出来高の関係
    # volume_ETH が元データに存在しない場合はスキップ
    if 'volume_ETH' in df_processed.columns:
        df_processed["volume_6h_mean"] = _roll_mean(df_processed["volume_ETH"], 6)
        df_processed["vol_norm_6h"]    = df_processed["volume_ETH"] / (df_processed["volume_6h_mean"] + _eps()) # raw値を追加し標準化
        # 価格6h変化で重みづけしたフロー
        # ret_6h は log_return_6h と同じなので、そちらを標準化済みとして利用
        df_processed["flow_strength_6h"] = df_processed["vol_norm_6h"] * np.sign(df_processed["log_return_6h"].fillna(0.0)) # raw値を追加し標準化
        # 出来高と価格のローリング相関（6h）
        df_processed["vol_price_corr_6h"] = (
            df_processed["volume_ETH"].rolling(6, min_periods=3).corr(df_processed["log_return_1h"].rolling(6, min_periods=3).mean())
        ) # raw値を追加し標準化
        cont_cols.extend(["volume_6h_mean", "vol_norm_6h", "flow_strength_6h", "vol_price_corr_6h"])
    else:
        print("Warning: 'volume_ETH' column not found. Skipping related features.")


    # 7) Zスコア移動（直近ボラで正規化した価格の一歩分の動き）
    # ret_1h は log_return_1h と同じなので、そちらを標準化済みとして利用
    if 'log_return_1h' in df_processed.columns:
        ret1 = df_processed["log_return_1h"].fillna(0.0)
        ret1_std6 = _roll_std(ret1, 6).replace(0.0, _eps())
        df_processed["z_move_6h"] = (ret1 / ret1_std6).clip(-10, 10)  # 外れ値抑制は生値にも適用
        cont_cols.append("z_move_6h")


    # 8) 価格×出来高の比（価格変化に対する出来高の“燃料”）
    if 'volume_ETH' in df_processed.columns and 'log_return_6h' in df_processed.columns:
        df_processed["volume_pchg_6h"] = df_processed["volume_ETH"].pct_change(6) # raw値を追加し標準化
        df_processed["price_volume_ratio_6h"] = df_processed["log_return_6h"] / (df_processed["volume_pchg_6h"].replace(0.0, _eps()) + _eps()) # raw値を追加し標準化
        cont_cols.extend(["volume_pchg_6h", "price_volume_ratio_6h"])

    # 9) （任意）簡易“圧縮→拡張”フラグの連続値版（連続値で学習 → 閾値は後段で）
    #    BB幅の縮小度（最近対過去中央値比）で圧縮度を測る
    if 'bb_width_20' in df_processed.columns:
        bb_med_168 = df_processed["bb_width_20"].rolling(168, min_periods=24).median()
        df_processed["bb_compression_ratio"] = (df_processed["bb_width_20"] / (bb_med_168 + _eps())).clip(0, 10) # raw値を追加し標準化
        df_processed["bb_reexpansion_speed_6h"] = df_processed["bb_width_20"].pct_change(6).clip(-5, 5) # raw値を追加し標準化
        cont_cols.extend(["bb_compression_ratio", "bb_reexpansion_speed_6h"])


    # 10) 安全な数値範囲にクリップ（極端値による学習不安定化を防ぐ）
    # Standardize処理の中でwinsorizeを行うため、ここではクリップ処理は不要だが、
    # 元のロジックを維持するため、raw値に対してクリップを適用する。
    _clip_specs = {
        "log_return_1h": (-0.2, 0.2), "log_return_6h": (-0.5, 0.5), "log_return_12h": (-0.8, 0.8),
        "momentum_6h": (-1e5, 1e5), "acceleration_6h": (-1e5, 1e5),
        "volatility_6h": (0.0, 1.0), "volatility_change_6h": (-1.0, 1.0),
        "trend_strength_6_24": (-2.0, 2.0), "trend_slope_6h": (-1.0, 1.0),
        "bb_width_20": (0.0, 1.0), "bb_width_change_6": (-1.0, 1.0),
        "vol_norm_6h": (0.0, 10.0), "flow_strength_6h": (-10.0, 10.0),
        "vol_price_corr_6h": (-1.0, 1.0),
        "z_move_6h": (-10.0, 10.0),
        "volume_pchg_6h": (-1.0, 5.0), "price_volume_ratio_6h": (-50.0, 50.0),
        "bb_compression_ratio": (0.0, 10.0), "bb_reexpansion_speed_6h": (-5.0, 5.0),
    }
    for k, (lo, hi) in _clip_specs.items():
        if k in df_processed.columns:
            df_processed[k] = df_processed[k].astype(float).clip(lo, hi)

    # --- 標準化処理（再度、cont_colsに含まれる全ての連続特徴量に対してWinsorize & Standardizeを適用） ---
    # raw値の特徴量名に '_zn' をつけて標準化済みとして追加する
    cont_cols_to_standardize = []
    # params ディクショナリをクリアして再構築
    params: dict = {"winsor": {}, "standardize": {}}


    # for col in cont_cols:
    #      if col in df_processed.columns:
    #         standardized_col = f"{col}_zn"
    #         # Winsorize
    #         df_processed[col], wq = _winsorize_by_train(df_processed[col], winsor_q[0], winsor_q[1], train_mask)
    #         params["winsor"][col] = {"low": wq[0], "high": wq[1]}

    #         # Standardize
    #         df_processed[standardized_col], st = _standardize_by_train(df_processed[col], train_mask)
    #         params["standardize"][col] = {"mean": st[0], "std": st[1]}
    #         cont_cols_to_standardize.append(standardized_col)
    #      else:
    #          print(f"Warning: Continuous feature column '{col}' not found before final standardization. Skipping.")


    def hysteresis_flag(series, enter_thr, exit_thr):
        on = False
        out = []
        for v in series:
            if not on and v >= enter_thr: on = True
            elif on and v <= exit_thr:    on = False
            out.append(1 if on else 0)
        return pd.Series(out, index=series.index)

    # トレンド順序 (カテゴリ型なので標準化対象外)
    df_processed["ma_6"]  = df_processed["close"].rolling(6, 1).mean()
    df_processed["ma_24"] = df_processed["close"].rolling(24, 1).mean()
    df_processed["ma_72"] = df_processed["close"].rolling(72, 1).mean()
    df_processed["ma_order_flag"] = ((df_processed["ma_6"] > df_processed["ma_24"]) & (df_processed["ma_24"] > df_processed["ma_72"])).astype(int).astype('category')

    # ボラレジーム (カテゴリ型なので標準化対象外)
    std24 = df_processed["close"].pct_change().rolling(24, 12).std()
    med  = std24.expanding(24).median()
    df_processed["high_vol_flag"] = (std24 > med * 1.5).astype(int)

    # スクイーズ/再拡張 (カテゴリ型なので標準化対象外)
    bb = df_processed["close"].rolling(20, 10).std() * 2 / (df_processed["close"].rolling(20, 10).mean().abs() + 1e-8)
    q20 = bb.expanding(100).quantile(0.2)
    df_processed["squeeze_flag"] = (bb < q20).astype(int)
    df_processed["squeeze_reexp_flag"] = (bb.pct_change(6) > 0.3).astype(int)

    # RSI過熱（ヒステリシス） (カテゴリ型なので標準化対象外)
    chg = df_processed["close"].diff()
    gain = chg.clip(lower=0).rolling(14, 1).mean()
    loss = (-chg.clip(upper=0)).rolling(14, 1).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - 100/(1+rs)
    df_processed["rsi_overbought_flag"] = hysteresis_flag(rsi, enter_thr=70, exit_thr=60)
    df_processed["rsi_oversold_flag"]   = hysteresis_flag(100-rsi, enter_thr=70, exit_thr=60)

    # ブレイク (カテゴリ型なので標準化対象外)
    roll_max24 = df_processed["close"].rolling(24, 1).max()
    roll_min24 = df_processed["close"].rolling(24, 1).min()
    df_processed["break_up_flag"]   = (df_processed["close"] > roll_max24.shift(1)).astype(int)
    df_processed["break_down_flag"] = (df_processed["close"] < roll_min24.shift(1)).astype(int)

    # 元のraw値を削除するか、保持するかは利用目的による。
    # ここでは標準化済みの特徴量のみを返すリストに含める。
    # raw値が必要であれば別途取得する必要がある。

    # 欠損値が5割以上の特徴量列をドロップ (標準化後に行う)
    threshold = 0.5 * len(df_processed)
    cols_to_drop = df_processed.columns[df_processed.isnull().sum() > threshold]
    df_processed = df_processed.drop(columns=cols_to_drop)
    print(f"Dropped columns with more than 50% NaN: {list(cols_to_drop)}")

    # 欠損値処理 (残った行をドロップ)
    df_processed = df_processed.dropna()

    # 標準化された連続特徴量のリストを更新 (ドロップされた列を除く)
    final_cont_cols = [col for col in cont_cols_to_standardize if col in df_processed.columns]

    return df_processed


# In[ ]:


def calculate_volume_profile(df, num_bins=20):
    """
    価格帯別出来高（Volume Profile）の特徴量を計算し、DataFrameに追加する。

    Parameters:
        df (pd.DataFrame): OHLCVデータ（"Open", "High", "Low", "Close", "Volume"が必要）
        num_bins (int): 価格帯を分割するビンの数

    Returns:
        pd.DataFrame: Volume Profileの特徴量を追加したDataFrame
    """

    # 価格帯の範囲設定
    min_price = df["low"].min()
    max_price = df["high"].max()

    # 価格帯をビンに分割
    bins = np.linspace(min_price, max_price, num_bins + 1)

    # 各価格帯の出来高を計算
    volume_profile = np.zeros(num_bins)
    volume_by_price = []

    for i in range(num_bins):
        # 価格帯の範囲
        lower_bound = bins[i]
        upper_bound = bins[i + 1]

        # この範囲に含まれる出来高を集計
        mask = (df["low"] < upper_bound) & (df["high"] > lower_bound)
        volume_profile[i] = df.loc[mask, "volume_USD"].sum()
        volume_by_price.append(volume_profile[i])

    volume_by_price = np.array(volume_by_price)
    # 統計量を計算（平均値、最大値、標準偏差）
    mean_volume = np.mean(volume_by_price)
    max_volume = np.max(volume_by_price)
    std_volume = np.std(volume_by_price)


    # POC（出来高最大の価格帯）
    poc_index = np.argmax(volume_profile)
    poc_price = (bins[poc_index] + bins[poc_index + 1]) / 2

    # Value Area (出来高の70%範囲)
    sorted_indices = np.argsort(volume_profile)[::-1]  # 出来高の多い順にソート
    cumulative_volume = np.cumsum(volume_profile[sorted_indices])
    total_volume = cumulative_volume[-1]

    # 出来高合計の70%をカバーする価格範囲を決定
    value_area_indices = sorted_indices[cumulative_volume <= total_volume * 0.7]
    value_area_prices = [(bins[i] + bins[i + 1]) / 2 for i in value_area_indices]

    # VAH（Value Area High）とVAL（Value Area Low）
    vah_price = max(value_area_prices)
    val_price = min(value_area_prices)

    # VWAP 計算
    vwap = (df["close"] * df["volume_USD"]).sum() / df["volume_USD"].sum()

    # VWAPの標準偏差
    vwap_std = np.sqrt(((df["close"] - vwap) ** 2 * df["volume_USD"]).sum() / df["volume_USD"].sum())


    # 結果をDataFrameに追加
    df["vbp_mean"] = mean_volume
    df["vbp_max"] = max_volume
    df["vbp_std"] = std_volume
    df["poc_price"] = poc_price
    df["vah_price"] = vah_price
    df["val_price"] = val_price
    df["vwap"] = vwap
    df["vwap_std"] = vwap_std

    # 移動平均
    df["vbp_mean_MA_t_6"] = df["vbp_mean"].rolling(window=6).mean()
    df["vbp_mean_MA_t_24"] = df["vbp_mean"].rolling(window=24).mean()
    df["vbp_max_MA_t_6"] = df["vbp_max"].rolling(window=6).mean()
    df["vbp_max_MA_t_24"] = df["vbp_max"].rolling(window=24).mean()
    df["vbp_std_MA_t_6"] = df["vbp_std"].rolling(window=6).mean()
    df["vbp_std_MA_t_24"] = df["vbp_std"].rolling(window=24).mean()
    df["poc_price_MA_t_6"] = df["poc_price"].rolling(window=6).mean()
    df["poc_price_MA_t_24"] = df["poc_price"].rolling(window=24).mean()
    df["vah_price_MA_t_6"] = df["vah_price"].rolling(window=6).mean()
    df["vah_price_MA_t_24"] = df["vah_price"].rolling(window=24).mean()
    df["val_price_MA_t_6"] = df["val_price"].rolling(window=6).mean()
    df["val_price_MA_t_24"] = df["val_price"].rolling(window=24).mean()
    df["vwap_MA_t_6"] = df["vwap"].rolling(window=6).mean()
    df["vwap_MA_t_24"] = df["vwap"].rolling(window=24).mean()
    df["vwap_std_MA_t_6"] = df["vwap_std"].rolling(window=6).mean()
    df["vwap_std_MA_t_24"] = df["vwap_std"].rolling

    # ラグ特徴量
    df["vbp_mean_Lag_t_1"] = df["vbp_mean"].shift(1)
    df["vbp_mean_Lag_t_3"] = df["vbp_mean"].shift(3)
    df["vbp_max_Lag_t_1"] = df["vbp_max"].shift(1)
    df["vbp_max_Lag_t_3"] = df["vbp_max"].shift(3)
    df["vbp_std_Lag_t_1"] = df["vbp_std"].shift(1)
    df["vbp_std_Lag_t_3"] = df["vbp_std"].shift(3)
    df["poc_price_Lag_t_1"] = df["poc_price"].shift(1)
    df["poc_price_Lag_t_3"] = df["poc_price"].shift(3)
    df["vah_price_Lag_t_1"] = df["vah_price"].shift(1)
    df["vah_price_Lag_t_3"] = df["vah_price"].shift(3)
    df["val_price_Lag_t_1"] = df["val_price"].shift(1)
    df["val_price_Lag_t_3"] = df["val_price"].shift(3)
    df["vwap_Lag_t_1"] = df["vwap"].shift(1)
    df["vwap_Lag_t_3"] = df["vwap"].shift(3)


    return df


# In[ ]:


def cusum_filter(series: pd.Series, threshold: float = 2.0):
    """CUSUM変化検知: 上昇変化 (1) と下降変化 (-1) をフラグ化"""
    pos, neg = 0.0, 0.0
    up_flags = np.zeros(len(series))
    down_flags = np.zeros(len(series))
    mean = series.mean() # 基準となる平均値

    for i, x in enumerate(series):
        # 上方向への累積和
        pos = max(0.0, pos + (x - mean))
        # 下方向への累積和
        neg = min(0.0, neg + (x - mean))

        # 上方向の変化検出
        if pos > threshold:
            up_flags[i] = 1
            pos = 0.0 # リセット
        # 下方向の変化検出
        elif neg < -threshold:
            down_flags[i] = 1 # 下降変化を示すため、ここでは1を立てて区別する
            neg = 0.0 # リセット

    return pd.Series(up_flags, index=series.index), pd.Series(down_flags, index=series.index)


def add_cusum_flags(df: pd.DataFrame, threshold_dict=None):
    """
    指定されたカラムにCUSUM変化検知フラグ (上昇/下降) を追加する関数。

    Parameters:
        df (pd.DataFrame): 入力DataFrame。
        threshold_dict (dict): {カラム名: しきい値} の辞書。

    Returns:
        pd.DataFrame: CUSUMフラグが追加されたDataFrame。
    """
    df = df.copy() # 元のDataFrameを変更しないようにコピー

    if threshold_dict is None:
        print("Warning: threshold_dict is None. No CUSUM flags will be added.")
        return df

    for col, thr in threshold_dict.items():
        if col in df.columns:
            # cusum_filter から上昇/下降それぞれのフラグ系列を取得
            up_flag_series, down_flag_series = cusum_filter(df[col], threshold=thr)

            # 新しいカラムとしてDataFrameに追加
            df[f'{col}_cusum_up_flag'] = up_flag_series
            df[f'{col}_cusum_down_flag'] = down_flag_series
        else:
            print(f"Warning: Column '{col}' not found in DataFrame. Skipping CUSUM flags for this column.")

    return df


# In[ ]:


import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    オンチェーンデータの特徴量を作成する関数

    Args:
        df (pd.DataFrame): オンチェーンデータのDataFrame

    Returns:
        pd.DataFrame: 特徴量を追加したDataFrame
    """


    # 'date' 列を datetime 型に変換
    # df['date'] = pd.to_datetime(df['date']) # すでにインデックスになっている可能性があるのでコメントアウト
    # 'date' 列をインデックスに設定
    # df.set_index('date', inplace=True) # すでにインデックスになっている可能性があるのでコメントアウト

    # 複製して元のデータフレームを変更しないようにする
    df = df.copy()

    # 必要なカラムを選択
    # 入力データフレームのカラムに合わせて修正
    diff_cols = ['total_flow_sum', 'address_count_sum',
                 'active_receivers', 'active_senders',
                 'dex_volume', 'contract_calls', 'whale_tx_count']

    # 差分
    for col in diff_cols:
        df[f'{col}_diff'] = df[col].diff()

    # 変化率（%変化）
    for col in diff_cols:
        df[f'{col}_pct_change'] = df[col].pct_change()

        # 移動平均 (MA)
        df[f'{col}_MA_t_6'] = df[col].rolling(window=7).mean()  # 7時間移動平均を例として追加
        df[f'{col}_MA_t_12'] = df[col].rolling(window=12).mean() # 12時間移動平均を例として追加
        df[f'{col}_MA_t_24'] = df[col].rolling(window=24).mean() # 24時間移動平均を例として追加
        df[f'{col}_MA_t_48'] = df[col].rolling(window=48).mean() # 48時間移動平均を例として追加
        df[f'{col}_MA_t_72'] = df[col].rolling(window=72).mean() # 72時間移動平均を例として追加
        df[f'{col}_MA_t_96'] = df[col].rolling(window=96).mean() # 96時間移動平均を例として追加
        df[f'{col}_MA_t_168'] = df[col].rolling(window=168).mean() # 168時間移動平均を例として追加

        # 移動平均の傾き
        # 重複している可能性のある行を修正（最初のMA_t_6_slopeのみ残すなど）
        # df[f'{col}_MA_slope'] = df[f'{col}_MA_t_6'].diff() # この行を残すか検討
        # df[f'{col}_MA_slope'] = df[f'{col}_MA_t_12'].diff() # 他の行は削除または別の特徴量名に変更
        # df[f'{col}_MA_slope'] = df[f'{col}_MA_t_24'].diff()
        # df[f'{col}_MA_slope'] = df[f'{col}_MA_t_48'].diff()
        # df[f'{col}_MA_slope'] = df[f'{col}_MA_t_72'].diff()
        # df[f'{col}_MA_slope'] = df[f'{col}_MA_t_96'].diff()
        # df[f'{col}_MA_slope'] = df[f'{col}_MA_t_168'].diff()
        # ここでは MA_t_6 の傾きのみを生成するように修正します
        df[f'{col}_MA_t_6_slope'] = df[f'{col}_MA_t_6'].diff()


        # ラグ特徴量 (Lag)
        df[f'{col}_Lag_t_1'] = df[col].shift(1)  # 1時間前の値を例として追加
        df[f'{col}_Lag_t_3'] = df[col].shift(3)  # 3時間前の値を例として追加

        # トレンド差分 (Diff)
        df[f'{col}_Diff_t_6'] = df[col] - df[f'{col}_MA_t_6']  # 7時間移動平均との差分を例として追加
        df[f'{col}_Diff_t_24'] = df[col] - df[f'{col}_MA_t_24'] # 30時間移動平均との差分を例として追加

        # 標準偏差
        df[f'{col}_rolling_std_7'] = df[col].rolling(window=7, center=True).std()
        df[f'{col}_rolling_std_30'] = df[col].rolling(window=30, center=True).std()




    # active_receiversとactive_sendersの比率
    df['active_ratio'] = df['active_receivers'] / (df['active_senders'] + 1e-8)

    # --- 新しいオンチェーンデータ特徴量の追加 ---

    # 1. 大口アクティビティ急増（whale_tx_count のラグ差分）
    # (whale_t - whale_(t-3)) / rolling_std > z1 → 1。
    z1 = 2.0 # 例として閾値z1を2.0に設定
    if 'whale_tx_count' in df.columns:
        # ラグ差分を計算
        whale_lag_diff = df['whale_tx_count'] - df['whale_tx_count'].shift(3)
        # ラグ差分のローリング標準偏差を計算 (ウィンドウサイズは例として12)
        whale_lag_diff_rolling_std = whale_lag_diff.rolling(window=12).std()
        # 正規化されたラグ差分
        whale_lag_diff_normalized = whale_lag_diff / (whale_lag_diff_rolling_std + 1e-9) # ゼロ除算防止

        # フラグ設定 - numpy array -> pandas Series -> category の順に変換
        df['whale_activity_surge_flag'] = pd.Series((whale_lag_diff_normalized > z1).astype(int), index=df.index).astype('category')
    else:
        print("Warning: 'whale_tx_count' column not found. Skipping 'whale_activity_surge_flag'.")
        df['whale_activity_surge_flag'] = np.nan


    # 2. DEXフローの偏り (K-of-L)
    # buy_sell_imbalance > q_hi をK-of-Lで成立 → 1。
    # buy_sell_imbalance は通常オーダーブックデータから計算されます。
    # engineer_features 関数がオンチェーンデータのみを扱う場合、この計算はここではできません。
    # もしbuy_sell_imbalanceが既存の列として存在しない場合、以下のコードはエラーになるか、NaNを生成します。
    # 仮に、buy_sell_imbalance が df に含まれていると仮定してロジックを記述します。
    # 実際のデータに合わせて、この部分を make_orderbook_features 関数に移動するか検討してください。
    L_dex = 5 # 例としてウィンドウサイズL=5
    K_dex = 3 # 例として閾値K=3
    q_hi_dex = 0.75 # 例として上位75%分位数を閾値q_hiに設定

    if 'buy_sell_imbalance' in df.columns:
         # buy_sell_imbalance の上位分位数を超えているか判定
        imbalance_exceeds_threshold = (df['buy_sell_imbalance'] > df['buy_sell_imbalance'].quantile(q_hi_dex)).astype(int)
        # 直近L期間の条件一致数をカウント
        rolling_imbalance_sum = imbalance_exceeds_threshold.rolling(window=L_dex, min_periods=K_dex).sum()
        # カウントがK以上であればフラグを1とする - numpy array -> pandas Series -> category の順に変換
        df['dex_flow_imbalance_persistence_flag'] = pd.Series((rolling_imbalance_sum >= K_dex).astype(int), index=df.index).astype('category')
    else:
        print("Warning: 'buy_sell_imbalance' column not found. Skipping 'dex_flow_imbalance_persistence_flag'.")
        df['dex_flow_imbalance_persistence_flag'] = np.nan


    # 3. オンチェーン利用の増勢
    # active_receivers_growth = (MA_t_6 - MA_t_24)/MA_t_24
    # growth が上位分位 → 1、下位 → -1。
    if 'active_receivers_MA_t_6' in df.columns and 'active_receivers_MA_t_24' in df.columns:
        # 成長率を計算
        # MA_t_24 が0の場合のゼロ除算を防ぐ
        active_receivers_growth = (df['active_receivers_MA_t_6'] - df['active_receivers_MA_t_24']) / (df['active_receivers_MA_t_24'].replace(0, np.nan) + 1e-9)

        # 成長率の分位数を計算 (計算可能な期間で)
        q_hi_growth = active_receivers_growth.quantile(0.90) # 例: 上位90%
        q_lo_growth = active_receivers_growth.quantile(0.10) # 例: 下位10%

        # フラグ設定 - numpy array -> pandas Series -> category の順に変換
        df['active_receivers_growth_flag'] = pd.Series(np.where(
            active_receivers_growth > q_hi_growth, 1, # 強増勢
            np.where(
                active_receivers_growth < q_lo_growth, -1, # 強減勢
                0 # その他
            )
        ), index=df.index).astype('category')
    else:
        print("Warning: Required MA columns for 'active_receivers_growth_flag' not found.")
        df['active_receivers_growth_flag'] = np.nan


    # 欠損値が2割以上の特徴量列をドロップ
    threshold = 0.5 * len(df)
    df = df.dropna(axis=1, thresh=len(df) - threshold)

    # 欠損値処理
    df = df.dropna()

    def _eps(): return 1e-8
    def _roll_mean(s, w):  return s.rolling(int(w), min_periods=1).mean()
    def _roll_std(s, w):   return s.rolling(int(w), min_periods=1).std()
    def _safe_ratio(a, b): return a / (b + _eps())

    def _exists(col): return (col in df.columns)

    onchain_cols = []  # 追加カラムをここで管理

    # --- 0) 存在確認のヘルパ ---
    def _add_if_exists(name, series):
        df[name] = series
        onchain_cols.append(name)

    # --- 1) ネットワーク活性度（アクティブアドレス/送受信者） ---
    # 成長率（6h）とZスコア
    for base in ["active_addresses", "active_senders", "active_receivers"]:
        if _exists(base):
            g6 = df[base].pct_change(6)
            _add_if_exists(f"{base}_growth_6h", g6)
            roll = _roll_std(df[base].pct_change().fillna(0.0), 24).replace(0.0, _eps())
            z   = (df[base].pct_change().fillna(0.0) / roll).clip(-10, 10)
            _add_if_exists(f"{base}_z_24h", z)

    # --- 2) 取引量・DEX動向 ---
    if _exists("transfer_volume"):
        _add_if_exists("transfer_vol_growth_6h", df["transfer_volume"].pct_change(6))

    if _exists("dex_volume") and _exists("transfer_volume"):
        _add_if_exists("dex_vol_ratio", _safe_ratio(df["dex_volume"], df["transfer_volume"]))

    # --- 3) 取引所フロー（売り圧/買い圧 proxy） ---
    if _exists("exchange_inflow") and _exists("exchange_outflow"):
        infl, out = df["exchange_inflow"], df["exchange_outflow"]
        _add_if_exists("ex_in_out_ratio", _safe_ratio(infl, out))
        # ネットフローとその6h変化
        net = infl - out
        _add_if_exists("ex_netflow", net)
        _add_if_exists("ex_netflow_change_6h", net.diff(6))

    # --- 4) 供給動態（取引所保有/長期保有） ---
    if _exists("supply_on_exchange"):
        _add_if_exists("ex_supply_change_24h", df["supply_on_exchange"].diff(24))
        if _exists("total_supply"):
            _add_if_exists("ex_supply_ratio", _safe_ratio(df["supply_on_exchange"], df["total_supply"]))

    if _exists("top10_holdings") and _exists("total_supply"):
        _add_if_exists("top10_hold_ratio", _safe_ratio(df["top10_holdings"], df["total_supply"]))

    # --- 5) クジラ行動・集中度 ---
    if _exists("whale_tx_count") and _exists("tx_count"):
        _add_if_exists("whale_tx_ratio", _safe_ratio(df["whale_tx_count"], df["tx_count"]))
        # クジラ比の6h移動平均と変化
        wma6 = _roll_mean(df["whale_tx_count"], 6)
        tma6 = _roll_mean(df["tx_count"], 6).replace(0.0, _eps())
        _add_if_exists("whale_ratio_ma6", _safe_ratio(wma6, tma6))
        _add_if_exists("whale_ratio_change_6h", df["whale_tx_count"].pct_change(6))

    # --- 6) ガス/契約アクティビティ（DeFi/NFT/レイヤ2活況のproxy） ---
    if _exists("gas_used"):
        _add_if_exists("gas_used_change_6h", df["gas_used"].pct_change(6))
        # 正規化Z（ボラ変化吸収）
        z = df["gas_used"].pct_change().fillna(0.0) / (_roll_std(df["gas_used"].pct_change().fillna(0.0), 24).replace(0.0, _eps()))
        _add_if_exists("gas_used_z_24h", z.clip(-10, 10))

    if _exists("unique_contracts"):
        _add_if_exists("unique_contracts_growth_24h", df["unique_contracts"].pct_change(24))

    # --- 7) オンチェーン“センチメント”合成（軽量な派生） ---
    # 例: 活性度 × 需要 / 売り圧 を対数でまとめる（欠損に強いようにクリップ）
    _comp = []
    if _exists("active_addresses"): _comp.append(np.log(np.clip(df["active_addresses"].astype(float), 1.0, None)))
    if _exists("transfer_volume"):  _comp.append(np.log(np.clip(df["transfer_volume"].astype(float),  1.0, None)))
    if _exists("exchange_inflow"):  _comp.append(-np.log(np.clip(df["exchange_inflow"].astype(float), 1.0, None)))  # 売り圧はマイナス寄与
    if len(_comp) >= 2:
        onchain_sent = sum(_comp) / len(_comp)
        _add_if_exists("onchain_sentiment", onchain_sent)

    # --- 8) ボラ正規化の“動き”指標（全体フレームに応じた相対化） ---
    #    オンチェーン系列のpct_change()を短期stdで割ってZ化 → regime差を吸収
    def _z_move(series, std_win=24):
        pc = series.pct_change().fillna(0.0)
        den = _roll_std(pc, std_win).replace(0.0, _eps())
        return (pc / den).clip(-10, 10)

    for base in ["active_addresses","transfer_volume","exchange_inflow","exchange_outflow","dex_volume"]:
        if _exists(base):
            _add_if_exists(f"{base}_zmove_24h", _z_move(df[base], 24))

    # --- 9) クリップで安定化（外れ値による学習不安定を抑制） ---
    _clip_specs = {
        "active_addresses_growth_6h": (-1.0, 3.0),
        "active_senders_growth_6h":   (-1.0, 3.0),
        "active_receivers_growth_6h": (-1.0, 3.0),
        "active_addresses_z_24h":     (-10.0, 10.0),
        "active_senders_z_24h":       (-10.0, 10.0),
        "active_receivers_z_24h":     (-10.0, 10.0),

        "transfer_vol_growth_6h":     (-1.0, 3.0),
        "dex_vol_ratio":              (0.0, 10.0),

        "ex_in_out_ratio":            (0.0, 10.0),
        "ex_netflow":                 (-1e12, 1e12),
        "ex_netflow_change_6h":       (-1e12, 1e12),

        "ex_supply_change_24h":       (-1e12, 1e12),
        "ex_supply_ratio":            (0.0, 1.0),
        "top10_hold_ratio":           (0.0, 1.0),

        "whale_tx_ratio":             (0.0, 1.0),
        "whale_ratio_ma6":            (0.0, 1.0),
        "whale_ratio_change_6h":      (-1.0, 3.0),

        "gas_used_change_6h":         (-1.0, 3.0),
        "gas_used_z_24h":             (-10.0, 10.0),

        "unique_contracts_growth_24h":(-1.0, 3.0),

        "onchain_sentiment":          (-20.0, 20.0),

        "active_addresses_zmove_24h": (-10.0, 10.0),
        "transfer_volume_zmove_24h":  (-10.0, 10.0),
        "exchange_inflow_zmove_24h":  (-10.0, 10.0),
        "exchange_outflow_zmove_24h": (-10.0, 10.0),
        "dex_volume_zmove_24h":       (-10.0, 10.0),
    }
    for k, (lo, hi) in _clip_specs.items():
        if k in df.columns:
            df[k] = df[k].astype(float).clip(lo, hi)

    return df


# In[ ]:


def add_velocity_features(df, col="velocity_usd_24h"):
    # Z-score
    roll_mean = df[col].rolling(200).mean()
    roll_std = df[col].rolling(200).std()
    df[f"{col}_zn"] = (df[col] - roll_mean) / (roll_std + 1e-8)

    # Lag, delta はそのまま（OK）
    df[f"{col}_lag1"] = df[col].shift(1)
    df[f"{col}_lag2"] = df[col].shift(2)
    df[f"{col}_lag3"] = df[col].shift(3)

    df[f"delta_{col}"] = df[col].diff()
    df[f"delta2_{col}"] = df[f"delta_{col}"].diff()
    df[f"delta_{col}_zn"] = df[f"{col}_zn"].diff()

    # spike/crash, regime は今のままでもOK（未来見てない）
    mu = roll_mean
    sigma = roll_std
    df[f"{col}_spike_flag"] = (df[col] > mu + 2 * sigma).astype(int)
    df[f"{col}_crash_flag"] = (df[col] < mu - 2 * sigma).astype(int)

    ema_s = df[col].ewm(span=24).mean()
    ema_l = df[col].ewm(span=72).mean()
    df[f"{col}_regime_up_flag"] = (ema_s > ema_l).astype(int)
    df[f"{col}_regime_down_flag"] = (ema_s < ema_l).astype(int)

    # --- CUSUM（各時点のローカル平均と閾値を使う） ---
    threshold = roll_std * 0.5

    cusum_pos = 0.0
    cusum_neg = 0.0
    up_flag = []
    down_flag = []

    for v, m, th in zip(df[col], mu, threshold):
        if pd.isna(m) or pd.isna(th):
            cusum_pos = cusum_neg = 0.0
            up_flag.append(0)
            down_flag.append(0)
            continue

        if v - m > th:
            cusum_pos += v - m
            cusum_neg = 0.0
        elif m - v > th:
            cusum_neg += m - v
            cusum_pos = 0.0
        else:
            cusum_pos = cusum_neg = 0.0

        up_flag.append(int(cusum_pos > th))
        down_flag.append(int(cusum_neg > th))

    df[f"{col}_cusum_up_flag"] = up_flag
    df[f"{col}_cusum_down_flag"] = down_flag

    return df


# In[ ]:


# ==============================================================
# 連続オンチェーン特徴量（リーク防止つき）
#  - 入力: df, diff_cols, train_mask（Trueがtrain期間）
#  - 出力: df, 追加した連続特徴量名リスト, 学習時にfitした統計量params
# ==============================================================

def create_onchain_cont_features(
    df: pd.DataFrame,
    diff_cols: list[str],
    train_mask: pd.Series | None = None,
    short_win: int = 6,    # ≈ 6h
    mid_win:   int = 24,   # ≈ 24h
    long_win:  int = 72,   # ≈ 72h
    ema_alpha_short: float = 2/(6+1),
    ema_alpha_mid:   float = 2/(24+1),
    winsor_q: tuple[float, float] = (0.005, 0.995),  # 標準化前の外れ値クリップ（trainでfit）
    eps: float = 1e-8,
):
    """
    7系列（例: total_flow_sum, address_count_sum, ...）から
    動的/比率/標準化/スコア系の連続特徴量をまとめて生成。
    ・勝手にval/test情報を使わないよう、標準化などは train で fit→全期間に適用。
    返り値:
      df: 追加列込み
      cont_cols: 追加した連続特徴量カラム名一覧
      params: trainでfitした分位点・平均・分散など
    """
    if train_mask is None:
        train_mask = pd.Series(False, index=df.index)
        train_mask.iloc[: int(len(df) * 0.7)] = True
    train_mask = train_mask.fillna(False)

    def _ema(x: pd.Series, alpha: float):
        return x.ewm(alpha=alpha, adjust=False, min_periods=1).mean()

    def _winsorize_by_train(s: pd.Series, q_low: float, q_high: float, mask: pd.Series):
        # trainで分位点fit → 全期間クリップ
        tri = s.loc[mask].dropna()
        if len(tri) == 0:
            return s, (np.nan, np.nan)
        lo, hi = np.quantile(tri, [q_low, q_high])
        return s.clip(lower=lo, upper=hi), (float(lo), float(hi))

    def _standardize_by_train(s: pd.Series, mask: pd.Series):
        tri = s.loc[mask].dropna()
        if len(tri) == 0:
            return s, (np.nan, np.nan)
        mu, sd = tri.mean(), tri.std()
        if sd == 0 or not np.isfinite(sd):
            sd = 1.0
        return (s - mu) / sd, (float(mu), float(sd))

    cont_cols: list[str] = []
    params: dict = {"winsor": {}, "standardize": {}}

    for c in diff_cols:
        if c not in df.columns:
            continue

        # ---- 基本変換（log1pはスケール安定化に効く） ----
        logc = f"{c}_log1p"
        df[logc] = np.log1p(df[c])

        # ---- 短・中・長の移動平均（MA）, 移動STD, Zスコア ----
        ma_s = f"{c}_ma_s{short_win}"
        ma_m = f"{c}_ma_m{mid_win}"
        ma_l = f"{c}_ma_l{long_win}"
        std_s = f"{c}_std_s{short_win}"
        std_m = f"{c}_std_m{mid_win}"

        df[ma_s]  = df[c].rolling(short_win, min_periods=1).mean()
        df[ma_m]  = df[c].rolling(mid_win, min_periods=1).mean()
        df[ma_l]  = df[c].rolling(long_win, min_periods=1).mean()
        df[std_s] = df[c].rolling(short_win, min_periods=1).std().replace(0, np.nan)
        df[std_m] = df[c].rolling(mid_win,  min_periods=1).std().replace(0, np.nan)

        z_s = f"{c}_z_short"
        z_m = f"{c}_z_mid"
        df[z_s] = (df[c] - df[ma_s]) / (df[std_s])
        df[z_m] = (df[c] - df[ma_m]) / (df[std_m])

        # ---- 比率系（短/中、短/長）と乖離量 ----
        r_sm = f"{c}_ratio_s_m"
        r_sl = f"{c}_ratio_s_l"
        d_sm = f"{c}_dev_s_m"     # |短/中 - 1|
        d_sl = f"{c}_dev_s_l"     # |短/長 - 1|
        df[r_sm] = df[ma_s] / (df[ma_m] + eps)
        df[r_sl] = df[ma_s] / (df[ma_l] + eps)
        df[d_sm] = (df[r_sm] - 1.0).abs()
        df[d_sl] = (df[r_sl] - 1.0).abs()

        # ---- 変化率（パーセントチェンジ） ----
        pct1 = f"{c}_pct1"
        pct_s = f"{c}_pct{short_win}"
        pct_m = f"{c}_pct{mid_win}"
        df[pct1] = df[c].pct_change(1)
        df[pct_s] = df[c].pct_change(short_win)
        df[pct_m] = df[c].pct_change(mid_win)

        # ---- EMAとクロス（短・中） ----
        ema_s = f"{c}_ema_s"
        ema_m = f"{c}_ema_m"
        ema_x = f"{c}_ema_cross"   # 短-中
        df[ema_s] = _ema(df[c], ema_alpha_short)
        df[ema_m] = _ema(df[c], ema_alpha_mid)
        df[ema_x] = df[ema_s] - df[ema_m]

        # ---- 簡易スロープ（短窓での平均傾き） ----
        # polyfitは重いので差分平均で近似（実務的には充分に効く）
        slope_s = f"{c}_slope_s{short_win}"
        df[slope_s] = (df[c] - df[c].shift(short_win)) / (short_win + eps)

        # ---- 連続スコア（イベント/レジームの“強さ”） ----
        # event_score: 短期Z と 短期%変化の「強い方」
        # regime_score: 中期Z と |短/中 - 1| の「強い方」
        e_score = f"{c}_event_score"
        r_score = f"{c}_regime_score"
        # 正規化のため、まずは%変化をtrainでwinsorize→標準化しておく
        pct_s_w, wq = _winsorize_by_train(df[pct_s], winsor_q[0], winsor_q[1], train_mask)
        params["winsor"][pct_s] = {"low": wq[0], "high": wq[1]}
        pct_s_n, st = _standardize_by_train(pct_s_w, train_mask)
        params["standardize"][pct_s] = {"mean": st[0], "std": st[1]}

        z_s_n, st_zs = _standardize_by_train(df[z_s], train_mask)
        params["standardize"][z_s] = {"mean": st_zs[0], "std": st_zs[1]}

        # 強さを同一スケールで比較（train標準化後にmax）
        df[e_score] = pd.concat([z_s_n, pct_s_n], axis=1).max(axis=1)

        # レジームスコア: z_mid と 乖離量（dev_s_m）をtrain標準化 → max
        dev_sm_n, st_dev = _standardize_by_train(df[d_sm], train_mask)
        params["standardize"][d_sm] = {"mean": st_dev[0], "std": st_dev[1]}

        z_m_n, st_zm = _standardize_by_train(df[z_m], train_mask)
        params["standardize"][z_m] = {"mean": st_zm[0], "std": st_zm[1]}

        df[r_score] = pd.concat([z_m_n, dev_sm_n], axis=1).max(axis=1)

        # ---- 仕上げ：選抜して「標準化版」を用意（学習の安定化に効く）
        # 代表的に学習に効く連続特徴を standardized で追加（接尾辞 _zn）
        def add_std(name: str):
            s = df[name]
            s_w, wq2 = _winsorize_by_train(s, winsor_q[0], winsor_q[1], train_mask)
            params["winsor"][name] = {"low": wq2[0], "high": wq2[1]}
            s_n, st2 = _standardize_by_train(s_w, train_mask)
            params["standardize"][name] = {"mean": st2[0], "std": st2[1]}
            zn = f"{name}_zn"
            df[zn] = s_n
            cont_cols.append(zn)

        # 基本推奨：z_s, z_m, pct_s, pct_m, ema_cross, slope, ratio, event/regime score
        add_std(z_s)
        add_std(z_m)
        add_std(pct_s)
        add_std(pct_m)
        add_std(ema_x)
        add_std(slope_s)
        add_std(r_sm)
        add_std(r_sl)
        add_std(e_score)
        add_std(r_score)

        # 生のlog1pやMAも（必要なら）追加（標準化版）
        add_std(logc)
        add_std(ma_s)
        add_std(ma_m)
        add_std(ma_l)

    # 返り値：df（追加済み）、連続特徴列（*_zn）、fit済みパラメータ
    return df, cont_cols, params


# In[ ]:


def create_onchain_flags(
    df: pd.DataFrame,
    diff_cols: list[str],
    train_mask: pd.Series | None = None,
    # 近短・中期窓（時系列は1h想定。別解像度なら適宜変更）
    short_win: int = 6,     # ≈ 6h
    mid_win:   int = 24,    # ≈ 24h
    # 分位点しきい（イベントは上位15%を既定、状態は上位40%）
    event_q_enter: float = 0.85,
    event_q_exit:  float = 0.70,
    state_q_enter: float = 0.60,
    state_q_exit:  float = 0.45,
    # Zスコアの固定しきい（分位が壊れた時のフォールバック）
    z_enter_default: float = 1.5,
    z_exit_default:  float = 0.5,
) -> tuple[pd.DataFrame, list[str], dict]:
    """
    7系列のみから作れる“最小だけど効く”フラグ群を生成。
      - スパイク系（event）: 短期Zスコア / 変化率の上位分位をON
      - 状態系（state）   : 中期Zスコア / 短長比の上位分位をON
    しきいは train 窓でfit→全期間適用（リーク防止）。enter/exitのヒステリシス付き。
    """
    def _exists(c): return (c in df.columns)

    # --- trainマスク（未指定なら先頭70%をtrain） ---
    if train_mask is None:
        train_mask = pd.Series(False, index=df.index)
        train_mask.iloc[: int(len(df) * 0.7)] = True
    train_mask = train_mask.fillna(False)

    # --- 小道具：分位点fit / ヒステリシス ---
    def fit_quantiles(series: pd.Series, qs: list[float]) -> dict[float, float]:
        s = series.loc[train_mask].dropna().values
        if s.size == 0:
            return {q: np.nan for q in qs}
        vals = np.quantile(s, qs)
        return {q: float(v) for q, v in zip(qs, vals)}

    def hysteresis_from_series(x: pd.Series, thr_enter: float, thr_exit: float, mode: str = "high") -> pd.Series:
        """
        mode="high": 値が高いほどON（enter: >=thr_enter, exit: <=thr_exit）
        mode="low" : 値が低いほどON（enter: <=thr_enter, exit: >=thr_exit）
        """
        on = False
        out = []
        for v in x.fillna(np.nan):
            if np.isnan(v):
                out.append(1 if on else 0)
                continue
            if mode == "high":
                if (not on) and (v >= thr_enter): on = True
                elif on and (v <= thr_exit):      on = False
            else:
                if (not on) and (v <= thr_enter): on = True
                elif on and (v >= thr_exit):      on = False
            out.append(1 if on else 0)
        return pd.Series(out, index=x.index, dtype=int)

    flag_cols: list[str] = []
    thresholds_used: dict[str, dict] = {}

    # 事前に短期・中期の統計を作る
    for c in diff_cols:
        if not _exists(c):
            continue
        # 中期Zスコア（状態系向け）
        mu_mid = df[c].rolling(mid_win, min_periods=1).mean()
        sd_mid = df[c].rolling(mid_win, min_periods=1).std().replace(0, np.nan)
        df[f"{c}_z_mid"] = (df[c] - mu_mid) / sd_mid

        # 短期Z（イベント向け）、短長比、短期変化率
        mu_s = df[c].rolling(short_win, min_periods=1).mean()
        sd_s = df[c].rolling(short_win, min_periods=1).std().replace(0, np.nan)
        df[f"{c}_z_short"] = (df[c] - mu_s) / sd_s
        df[f"{c}_ratio_s_over_m"] = (mu_s / (mu_mid.replace(0, np.nan))).replace([np.inf, -np.inf], np.nan)
        df[f"{c}_pctchg_{short_win}"] = df[c].pct_change(short_win)

    # ---- 各系列ごとに フラグ2種（event/state）を作る ----
    for c in diff_cols:
        if not _exists(c):
            continue

        # 1) スパイク（event）: 短期Z と 短期変化率の“強い方”を使う
        #    ・enter/exitはtrain分位でfit
        s_event_src1 = df[f"{c}_z_short"]
        s_event_src2 = df[f"{c}_pctchg_{short_win}"]
        # 2系列を標準化してmaxをとる（どちらかが強スパイクなら拾う）
        se1 = (s_event_src1 - s_event_src1.loc[train_mask].mean()) / (s_event_src1.loc[train_mask].std() or 1.0)
        se2 = (s_event_src2 - s_event_src2.loc[train_mask].mean()) / (s_event_src2.loc[train_mask].std() or 1.0)
        s_event = pd.concat([se1, se2], axis=1).max(axis=1)

        qs_event = fit_quantiles(s_event, [event_q_enter, event_q_exit])
        enter_e = qs_event[event_q_enter]
        exit_e  = qs_event[event_q_exit]
        # フォールバック（分位がNaNのときは固定Zのしきい）
        if not np.isfinite(enter_e): enter_e = z_enter_default
        if not np.isfinite(exit_e):  exit_e  = z_exit_default

        flag_e = hysteresis_from_series(s_event, enter_e, exit_e, mode="high")
        name_e = f"{c}_spike_flag"
        df[name_e] = flag_e
        flag_cols.append(name_e)
        thresholds_used[name_e] = {"enter": float(enter_e), "exit": float(exit_e), "mode": "high", "base": "max(z_short, pctchg_norm)"}

        # 2) 状態（state）: 中期Z と 短長比の“強い方”を使う
        s_state_src1 = df[f"{c}_z_mid"]
        s_state_src2 = df[f"{c}_ratio_s_over_m"]
        # 比率は1中心→(x-1)の絶対値で“偏り”も候補に
        s_state = pd.concat([
            s_state_src1,
            (s_state_src2 - 1.0).abs()
        ], axis=1).max(axis=1)

        qs_state = fit_quantiles(s_state, [state_q_enter, state_q_exit])
        enter_s = qs_state[state_q_enter]
        exit_s  = qs_state[state_q_exit]
        # フォールバック
        if not np.isfinite(enter_s): enter_s = 0.5   # 状態系はやや緩く
        if not np.isfinite(exit_s):  exit_s  = 0.2

        flag_s = hysteresis_from_series(s_state, enter_s, exit_s, mode="high")
        name_s = f"{c}_state_flag"
        df[name_s] = flag_s
        flag_cols.append(name_s)
        thresholds_used[name_s] = {"enter": float(enter_s), "exit": float(exit_s), "mode": "high", "base": "max(z_mid, |ratio-1|)"}

    # 参考: 合成フラグ（例：売り圧＝流入↑ & アドレス増）などは最小構成では作らず、後段でAND条件を自由設計
    return df, flag_cols, thresholds_used


# ### `engineer_features` 関数の修正と解説
# 
# `generate_features` 関数で発生した `TypeError: data type 'category' not understood` と同様の問題が、`engineer_features` 関数でも発生する可能性があります。これは、NumPyの `np.where` などによって生成されたNumPy配列を、直接Pandasのカテゴリ型 (`astype('category')`) に変換しようとした際に起こります。
# 
# この問題を解決するため、修正後のコードでは、NumPy配列として生成されたフラグや判定結果を、一度 **`pd.Series()`** を用いてPandas Seriesに変換し、その後に **`.astype('category')`** を適用するように変更しました。これにより、Pandasがカテゴリ型として正しくデータを解釈できるようになります。
# 
# 以下の3つの新しいフラグ特徴量の生成箇所で、この修正を適用しました。
# 
# 1.  `whale_activity_surge_flag`
# 2.  `dex_flow_imbalance_persistence_flag`
# 3.  `active_receivers_growth_flag`
# 
# #### engineer_features 関数の概要
# 
# この `engineer_features` 関数は、主にオンチェーンデータ（ブロックチェーン上の取引やアドレスの活動に関するデータ）を基にした特徴量を作成します。オンチェーンデータは、市場参加者の活動やネットワークの状態を反映するため、価格予測において重要な情報源となり得ます。
# 
# 関数内で生成される主な特徴量は以下の通りです。
# 
# *   **差分 (`_diff`)**: 特定の指標の直近の変化量を示します。
# *   **変化率 (`_pct_change`)**: 特定の指標の直近の相対的な変化率を示します。
# *   **移動平均 (`_MA_t_X`)**: 指定された期間（X時間）における指標の平均値で、短期的なノイズを平滑化しトレンドを捉えます。
# *   **移動平均の傾き (`_MA_t_X_slope`)**: 移動平均の増減の勢いを示します。
# *   **ラグ特徴量 (`_Lag_t_X`)**: 現在の時点からX時間前の指標の値を示し、過去の値との関係性を捉えます。
# *   **トレンド差分 (`_Diff_t_X`)**: 現在の指標の値と移動平均との差分で、現在の状態がトレンドからどれだけ乖離しているかを示します。
# *   **標準偏差 (`_rolling_std_X`)**: 指定された期間における指標のばらつき（ボラティリティ）を示します。
# *   **アクティブ比率 (`active_ratio`)**: アクティブな受信者数と送信者数の比率で、ネットワーク上での資金の流れの偏りを示唆します。
# *   **大口アクティビティ急増フラグ (`whale_activity_surge_flag`)**: 大口取引数がある閾値を超えて急増したかを示すフラグです。
# *   **DEXフローの偏りフラグ (`dex_flow_imbalance_persistence_flag`)**: Decentralized Exchange (DEX) における売買の偏りがある期間継続しているかを示すフラグです（`buy_sell_imbalance` 列が存在する場合）。
# *   **オンチェーン利用の増勢フラグ (`active_receivers_growth_flag`)**: アクティブな受信者数の移動平均の成長率が一定の閾値を超えたかを示すフラグです。
# 
# これらの特徴量を組み合わせることで、オンチェーンデータから市場の潜在的な動きや参加者の行動に関する有用なシグナルを抽出することが期待できます。

# In[ ]:


def make_orderbook_features(agg):

    # 基本的な派生特徴量
    agg['buy_sell_ratio'] = agg['buy_qty_sum'] / (agg['sell_qty_sum'] + 1e-8)
    agg['volatility'] = (agg['price_max'] - agg['price_min']) / agg['price_mean']
    agg['price_change'] = agg['price_last'] - agg['price_first']
    agg['pseudo_spread'] = (agg['price_max'] - agg['price_min']) / agg['price_mean']

    # 平均取引サイズ（全体・買い・売り）
    agg['avg_trade_size'] = agg['qty_sum'] / agg['qty_count']
    agg['avg_buy_size'] = agg['buy_qty_sum'] / (agg['buy_trade_sum'] + 1e-8)
    agg['avg_sell_size'] = agg['sell_qty_sum'] / (agg['sell_trade_sum'] + 1e-8)

    # 成行買い取引の割合（件数ベース）
    agg['buy_trade_ratio'] = agg['buy_trade_sum'] / (agg['buy_trade_sum'] + agg['sell_trade_sum'] + 1e-8)

    # 売買圧力の変化（1時間前との差分）
    agg['delta_buy_qty'] = agg['buy_qty_sum'].diff()
    agg['delta_sell_qty'] = agg['sell_qty_sum'].diff()
    agg['delta_buy_sell_ratio'] = agg['buy_sell_ratio'].diff()

    # 買い注文と売り注文の数量差（Buy-Sell Order Imbalance）
    agg['buy_sell_imbalance'] = agg['buy_qty_sum'] - agg['sell_qty_sum']

    agg["order_imbalance_norm"] = (
    (agg["buy_qty_sum"] - agg["sell_qty_sum"]) /
    (agg["buy_qty_sum"] + agg["sell_qty_sum"] + 1e-8)
    )

    agg["buy_pressure_momentum"] = agg["buy_sell_ratio"].diff(2)

    agg["depth_imbalance"] = (
    agg["avg_buy_size"] - agg["avg_sell_size"]
    )

    agg["spread_change_rate"] = agg["pseudo_spread"].pct_change()

    agg["buy_qty_z"] = (
    (agg["buy_qty_sum"] - agg["buy_qty_sum"].rolling(24).mean()) /
    agg["buy_qty_sum"].rolling(24).std()
    )
    agg["sell_qty_z"] = (
        (agg["sell_qty_sum"] - agg["sell_qty_sum"].rolling(24).mean()) /
        agg["sell_qty_sum"].rolling(24).std()
    )

    agg["buy_sell_diff_smooth"] = agg["buy_sell_imbalance"].rolling(6).mean()
    agg["buy_sell_diff_smooth_rate"] = agg["buy_sell_diff_smooth"].diff()

    agg["ratio_short_long"] = (
    agg["buy_sell_ratio"].rolling(6).mean() /
    (agg["buy_sell_ratio"].rolling(72).mean() + 1e-8)
    )

    agg["buy_dominant_flag"] = (agg["order_imbalance_norm"] > 0).astype(int)

    agg["buy_overheat_flag"] = (agg["buy_qty_z"] > 2.0).astype(int)
    agg["sell_overheat_flag"] = (agg["sell_qty_z"] < -2.0).astype(int)

    agg["spread_spike_flag"] = (agg["spread_change_rate"].abs() > 0.5).astype(int)

    agg["imbalance_flip_flag"] = (
    np.sign(agg["buy_sell_imbalance"]).diff().abs() > 0
    ).astype(int)

    agg["low_liquidity_flag"] = (agg["volatility"] > agg["volatility"].quantile(0.8)).astype(int)

    # # 最良気配値付近の注文数量（Order Depth Near Best Bid/Ask）最良買い気配値から0.1%以内の買い注文数量を計算
    # agg['best_bid_depth'] = agg[(agg['price'] >= agg['best_bid'] * 0.999) & (agg['price'] <= agg['best_bid'] * 1.001)]['buy_qty_sum'].sum()

    # # 大口注文の発生頻度（Large Order Frequency）
    # agg['large_order_freq'] = agg[agg['qty'] >= 100]['qty'].count() / len(agg)

    # # 大口注文の平均数量
    # agg['avg_large_order_size'] = agg[agg['qty'] >= 100]['qty'].mean()

    # 移動平均 (MA)
    agg['price_mean_MA_t_6'] = agg['price_mean'].rolling(window=6).mean()  # 6時間移動平均を例として追加
    agg['price_mean_MA_t_12'] = agg['price_mean'].rolling(window=12).mean() # 12時間移動平均を例として追加
    agg['price_mean_MA_t_24'] = agg['price_mean'].rolling(window=24).mean() # 24時間移動平均を例として追加
    agg['price_mean_MA_t_48'] = agg['price_mean'].rolling(window=48).mean() # 48時間移動平均を例として追加
    agg['price_mean_MA_t_72'] = agg['price_mean'].rolling(window=72).mean() # 72時間移動平均を例として追加
    agg['price_mean_MA_t_96'] = agg['price_mean'].rolling(window=96).mean() # 96時間移動平均を例として追加
    agg['price_mean_MA_t_168'] = agg['price_mean'].rolling(window=168).mean() # 168時間移動平均を例として追加
    agg['buy_qty_sum_MA_t_6'] = agg['buy_qty_sum'].rolling(window=6).mean()  # 6時間移動平均を例として追加
    agg['buy_qty_sum_MA_t_12'] = agg['buy_qty_sum'].rolling(window=12).mean() # 12時間移動平均を例として追加
    agg['buy_qty_sum_MA_t_24'] = agg['buy_qty_sum'].rolling(window=24).mean() # 24時間移動平均を例として追加
    agg['sell_qty_sum_MA_t_6'] = agg['sell_qty_sum'].rolling(window=6).mean()  # 6時間移動平均を例として追加
    agg['sell_qty_sum_MA_t_24'] = agg['sell_qty_sum'].rolling(window=24).mean() # 24時間移動平均を例として追加
    agg['buy_sell_ratio_MA_t_6'] = agg['buy_sell_ratio'].rolling(window=6).mean()  # 6時間移動平均を例として追加
    agg['buy_sell_ratio_MA_t_24'] = agg['buy_sell_ratio'].rolling(window=24).mean() # 24時間移動平均を例として追加

    # 移動平均の変化率
    agg['price_mean_MA_slope'] = agg['price_mean_MA_t_6'].diff()
    agg['price_mean_MA_slope'] = agg['price_mean_MA_t_24'].diff()
    agg['buy_qty_sum_MA_slope'] = agg['buy_qty_sum_MA_t_6'].diff()
    agg['buy_qty_sum_MA_slope'] = agg['buy_qty_sum_MA_t_24'].diff()
    agg['sell_qty_sum_MA_slope'] = agg['sell_qty_sum_MA_t_6'].diff()
    agg['sell_qty_sum_MA_slope'] = agg['sell_qty_sum_MA_t_24'].diff()
    agg['buy_sell_ratio_MA_slope'] = agg['buy_sell_ratio_MA_t_6'].diff()
    agg['buy_sell_ratio_MA_slope'] = agg['buy_sell_ratio_MA_t_24'].diff()

    # ラグ特徴量 (Lag)
    agg['price_mean_Lag_t_1'] = agg['price_mean'].shift(1)  # 1時間前の値を例として追加
    agg['price_mean_Lag_t_3'] = agg['price_mean'].shift(3)  # 3時間前の値を例として追加
    agg['buy_qty_sum_Lag_t_1'] = agg['buy_qty_sum'].shift(1)  # 1時間前の値を例として追加
    agg['buy_qty_sum_Lag_t_3'] = agg['buy_qty_sum'].shift(3)  # 3時間前の値を例として追加
    agg['sell_qty_sum_Lag_t_1'] = agg['sell_qty_sum'].shift(1)  # 1時間前の値を例として追加
    agg['sell_qty_sum_Lag_t_3'] = agg['sell_qty_sum'].shift(3)  # 3時間前の値を例として追加

    # トレンド差分 (Diff)
    agg['price_mean_Diff_t_6'] = agg['price_mean'] - agg['price_mean_MA_t_6']  # 6時間移動平均との差分を例として追加
    agg['price_mean_Diff_t_24'] = agg['price_mean'] - agg['price_mean_MA_t_24'] # 24時間移動平均との差分を例として追加
    agg['buy_qty_sum_Diff_t_6'] = agg['buy_qty_sum'] - agg['buy_qty_sum_MA_t_6']  # 6時間移動平均との差分を例として追加
    agg['buy_qty_sum_Diff_t_24'] = agg['buy_qty_sum'] - agg['buy_qty_sum_MA_t_24'] # 24時間移動平均との差分を例として追加
    agg['sell_qty_sum_Diff_t_6'] = agg['sell_qty_sum'] - agg['sell_qty_sum_MA_t_6']  # 6時間移動平均との差分を例として追加
    agg['sell_qty_sum_Diff_t_24'] = agg['sell_qty_sum'] - agg['sell_qty_sum_MA_t_24'] # 24時間移動平均との差分を例として追加



    return agg


# In[ ]:


def fractional_diff(series, d, threshold=1e-2):
    """
    分数次差分を計算する関数
    series: 時系列データ (1D numpy array or list)
    d: 分数次階数 (例えば 0.5)
    threshold: 小さすぎる係数をカットするしきい値 (計算コスト削減)
    """
    series = np.asarray(series, dtype=np.float64)
    n = len(series)
    weights = [1.0]
    k = 1
    while True:
        weight = -weights[-1] * (d - k + 1) / k
        if abs(weight) < threshold:
            break
        weights.append(weight)
        k += 1
    weights = np.array(weights[::-1])  # 逆順にする

    diff_series = np.zeros(n)
    for i in range(len(weights)-1, n):
        diff_series[i] = np.dot(weights, series[i-len(weights)+1:i+1])

    return diff_series


# In[ ]:


# 分数次差分の適用
def apply_fractional_differencing(df, column, d=0.4):


    fd = fractional_diff(df[column].values, d=d)
    df[f"{column}_fracdiff"] = fd
    return df.dropna()


# In[ ]:


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = calculate_volume_profile(df, features, num_bins=20)
    df = engineer_features(df)
    df = generate_features(df)
    df = make_orderbook_features(df)
    return df


# # Task
# Create flag features based on moving averages with periods 6, 24, and 72, where the flag is 1 if the short-term moving average is greater than the medium-term moving average, which is greater than the long-term moving average, and all slopes are positive, and 0 otherwise. Additionally, discuss whether hysteresis should be incorporated into feature engineering given that the target variable is labeled with hysteresis (3 out of 4 conditions met within a window of 4 with a future horizon of 6), and create other potentially useful features for predicting this target variable, such as moving average deviation.

# ## ヒステリシスを特徴量に組み込むことについての検討と説明
# 
# ### Subtask:
# 目的変数のヒステリシスを考慮したラベリングとの関連性を踏まえ、特徴量にヒステリシスを組み込むことの意義や方法について説明します。
# 

# **Reasoning**:
# Explain the rationale behind incorporating hysteresis in the target variable and its implications for model training, discuss the benefits of considering hysteresis in feature engineering and potential approaches, and consider how hysteresis can be applied to specific features for this task.
# 
# 

# In[ ]:


# 目的変数にヒステリシスが導入されている理由と、それがモデル学習に与える影響について説明します。
print("目的変数にヒステリシスが導入されている理由:")
print("予測対象のイベント（例: 価格の大幅な上昇/下降）をより頑健に捉えるためです。特に金融時系列データでは、価格の短期的なノイズや一時的な変動が多く含まれます。ヒステリシスを導入することで、単一時点での条件一致だけでなく、ある程度の期間にわたる条件の持続や複数条件の同時発生を考慮することができ、誤ったシグナルを減らし、より信頼性の高いトレンドやイベントを特定することが期待できます。これにより、モデルはノイズに惑わされにくくなり、より本質的な市場の動きに基づいた学習が可能になります。")
print("\nモデル学習への影響:")
print("ヒステリシスを導入した目的変数は、より明確で識別しやすいパターンを持つ傾向があります。これにより、モデルは予測すべきイベントの開始と終了をより正確に学習できるようになります。ただし、ヒステリシスの設定（ウィンドウサイズ、条件数など）によっては、イベントの発生タイミングが実際の市場の動きから遅れる可能性があり、これがモデルの予測精度や取引戦略のパフォーマンスに影響を与えることも考慮する必要があります。また、ヒステリシスによってイベントの発生頻度が減少する場合、学習データにおける目的変数のクラスバランスが偏る可能性があり、適切なクラス不均衡対策が必要になる場合があります。")

# 特徴量エンジニアリングにおいてヒステリシスを考慮することの潜在的なメリットについて説明します。
print("\n特徴量エンジニアリングにおいてヒステリシスを考慮することの潜在的なメリット:")
print("1. ノイズの低減: 短期的な価格ノイズや一時的な指標の変動に起因する偽のシグナルを特徴量から取り除くことができます。例えば、移動平均のクロス判定に短い遅延を設けることで、瞬間的なクロスターンによる誤判断を防ぐことができます。")
print("2. トレンドの識別向上: ある特徴量が示すトレンドやパターンが一時的なものではなく、ある程度の期間持続しているかを評価するのに役立ちます。これにより、より安定したトレンドフォロー戦略や、トレンド転換の早期かつ信頼性の高い検出が可能になります。")
print("3. モデルの解釈性向上: ヒステリシスを組み込んだ特徴量は、より市場の「勢い」や「持続性」を反映するため、モデルが学習するパターンがより直感的で経済合理性に基づいたものになる可能性があります。")
print("4. 過学習の抑制: 短期的なノイズに過度に反応する特徴量を減らすことで、モデルが訓練データ固有のノイズパターンを学習してしまうリスク（過学習）を低減する効果が期待できます。")

# ヒステリシスを特徴量に組み込むための一般的なアプローチやアイデアについていくつか言及します。
print("\nヒステリシスを特徴量に組み込むための一般的なアプローチやアイデア:")
print("1. 条件の連続性チェック: ある条件（例: MA5 > MA20）がN期間連続して満たされているかを判定するフラグ特徴量を作成します。")
print("2. 条件一致カウンター: あるウィンドウ内で、特定の条件（例: RSI > 70）が満たされた回数をカウントする特徴量を作成します。")
print("3. 時間遅延を持つ判定: 移動平均のクロスなどの判定において、クロスが発生してから実際にフラグを立てるまでに数期間の遅延を設けます。")
print("4. 移動平均乖離の持続性: 価格と移動平均の乖離率が、ある閾値を超えてN期間持続しているかを判定する特徴量を作成します。")
print("5. CUSUM検定の応用: CUSUM検定のように、累積的な変化が閾値を超えた場合にシグナルを出すアプローチを、特定の特徴量の変化に適用します。")

# 今回のタスクで生成する可能性のある具体的な特徴量に対して、どのようにヒステリシスを適用できるかについて考察を述べます。
print("\n今回のタスクで生成する可能性のある具体的な特徴量へのヒステリシス適用に関する考察:")
print("- 移動平均のクロスフラグ (MA_6_24_cross_flagなど): 現在のコードでは単一時点でのクロスを検出していますが、これを「MA6がMA24を上回った状態が3時間以上継続した場合に1とする」といったように、連続期間の条件一致を組み込むことが考えられます。")
print("- 価格と移動平均の乖離 (Diff_t_6, Diff_t_24など): 「価格がMA24から±X%以上乖離した状態が2時間以上継続しているか」を示すフラグや、乖離率が閾値を超えている連続時間の長さなどを特徴量として追加できます。")
print("- モメンタム系指標 (momentum_6hなど): モメンタムがプラス（またはマイナス）の状態が一定期間続いているか、あるいはモメンタムの増加（減少）がN期間連続しているかなどを判定する特徴量が考えられます。")
print("- ボラティリティ関連特徴量 (rolling_std_6h, BB_widthなど): ボラティリティがある閾値を超えて高止まり（または低迷）している期間の長さや、ボラティリティの増加傾向が持続しているかなどを特徴量化できます。")
print("- オンチェーンデータ特徴量 (total_flow_sum_diff, active_ratioなど): 特定のオンチェーン指標の急増（急減）が一時的なものか、あるいは数時間にわたって継続しているかを判断するヒステリシスを組み込んだ特徴量が有効かもしれません。例えば、「active_receivers_diffが過去4時間のうち3時間以上プラスであった場合に1とする」といった特徴量が考えられます。")
print("\nこれらのヒステリシスを考慮した特徴量を導入することで、目的変数のラベリングロジックとの整合性を高め、モデルがより効果的にイベントを学習できる可能性があります。")


# ## ヒステリシスを組み込んだ特徴量の実装方法の検討
# 
# ### Subtask:
# 具体的にどのような特徴量にヒステリシスを適用するか、そのロジックや実装方法を検討します。必要であればコードを生成します。
# 

# **Reasoning**:
# Implement hysteresis for moving average cross flags and add it to the generate_features function.
# 
# 

# **Reasoning**:
# The previous command failed because the pandas library was not imported in the current code block. I will regenerate the code block with the necessary imports.
# 
# 

# In[ ]:


import pandas as pd
import numpy as np

def generate_features_with_hysteresis(df: pd.DataFrame) -> pd.DataFrame:
    """
    指定された特徴量カテゴリに基づいてtalibを用いて特徴量を作成する関数。
    ヒステリシスを考慮した特徴量を追加する。

    Args:
        df (pd.DataFrame): 入力データフレーム

    Returns:
        pd.DataFrame: 特徴量を追加したデータフレーム
    """
    df = df.copy()

    # 既存の特徴量生成ロジックをここにコピー＆ペーストするか、generate_features関数を呼び出す
    # 今回は既存のgenerate_featuresを呼び出す形式で記述します。
    # ただし、 generate_features 内でdropna() が呼ばれているとヒステリシス特徴量の計算に影響するため、
    # generate_features 関数を修正するか、dropna() を最後に移動する必要があります。
    # ここでは、generate_features を呼び出した後、不要な列をドロップし、ヒステリシス特徴量を計算します。

    # 既存の特徴量を生成
    # generate_features 関数内で dropna() が実行されるため、ヒステリシス特徴量計算前にデータが欠損する可能性がある。
    # 理想的には generate_features 関数を修正して最後に dropna() するか、
    # generate_features が返す DataFrame に対して改めてヒステリシス特徴量を計算し、最後に dropna() する。
    # ここでは、generate_features が返す DataFrame に特徴量を追加する形で実装します。
    # generate_features 関数がどのような DataFrame を返すか不明なため、エラーが発生する可能性はあります。
    # 実際の利用時には generate_features 関数の実装を確認し、必要に応じて修正してください。

    try:
        df = generate_features(df)
        # generate_features が返す DataFrame に基づいて further processing
    except NameError:
        print("Warning: 'generate_features' function not found. Skipping base feature generation.")
        # generate_features が見つからない場合は、最低限必要な列があることを想定して処理を続行
        # 実際のデータに合わせてこの部分を調整する必要があります。
        # 例: df = create_date_features(df) # 日付特徴量だけ生成する場合
        # 例: df = pd.DataFrame(...) # ダミーデータを作成する場合
        pass # 何もしない場合は、既存のdfに対して特徴量追加を試みる

    # ヒステリシスを導入する特徴量のリスト
    # 既存の generate_features 関数によって生成される特徴量名を正確に反映させる必要がある
    features_for_hysteresis = [
        'MA_6_24_cross_flag',
        'MA_12_48_cross_flag',
        'MA_24_72_cross_flag',
        'MA_6_24_72_trend_flag',
        'MA_slope_6_24_change_flag',
        'MA_slope_12_48_change_flag',
        # 'MA_slope_pct_change_6_24_72_change_flag' # generate_features に存在しないためコメントアウト
    ]

    # ヒステリシスを考慮したフラグ特徴量を生成するヘルパー関数
    def create_hysteresis_flag(series: pd.Series, window: int = 3, threshold: int = 2) -> pd.Series:
        """
        指定されたウィンドウ内で、値が1である期間が閾値回数以上ある場合に1となるヒステリシス付きフラグを作成
        Args:
            series (pd.Series): 元のフラグ系列 (0 or 1)
            window (int): 判定を行うローリングウィンドウサイズ
            threshold (int): ウィンドウ内で1であると判定する最小回数
        Returns:
            pd.Series: ヒステリシス付きフラグ系列 (0 or 1)
        """
        # rolling sum でウィンドウ内の1の数をカウント
        # nanが含まれている場合を考慮
        rolling_sum = series.rolling(window=window, min_periods=threshold).sum() # min_periods を threshold に設定
        # rolling sum が閾値以上であれば1、そうでなければ0
        hysteresis_flag = (rolling_sum >= threshold).astype(float) # float にして NaN を保持
        return pd.Series(hysteresis_flag, index=series.index).fillna(0).astype('category') # NaN を 0 で埋めてカテゴリ型に変換


    # 各フラグ特徴量にヒステリシスを適用
    for feature in features_for_hysteresis:
        # 既存のgenerate_featuresによって生成された特徴量にヒステリシスを適用
        # カテゴリ型を一時的にintに変換して計算
        if feature in df.columns:
            # dropna() によって一部データが失われている可能性があるため、元のフラグ列のインデックスを使用
            original_flag_series = df[feature].astype(int) # カテゴリ型をintに変換
            df[f'{feature}_hysteresis'] = create_hysteresis_flag(original_flag_series, window=4, threshold=3) # 例としてwindow=4, threshold=3を設定
        else:
            print(f"Warning: Feature '{feature}' not found in DataFrame. Skipping hysteresis for this feature.")


    # その他のヒステリシス適用例の検討と実装
    # 例：価格とMAの乖離が一定期間継続
    def create_deviation_persistence_flag(price_series: pd.Series, ma_series: pd.Series, threshold_pct: float, window: int, min_periods: int) -> pd.Series:
        """
        価格が移動平均から一定割合以上乖離した状態が指定期間継続した場合のフラグを作成
        Args:
            price_series (pd.Series): 価格系列
            ma_series (pd.Series): 移動平均系列
            threshold_pct (float): 乖離率の閾値 (例: 0.01 for 1%)
            window (int): 継続期間のウィンドウサイズ
            min_periods (int): 計算に必要な最小期間
        Returns:
            pd.Series: 乖離継続フラグ系列 (0 or 1)
        """
        # 乖離率を計算
        # MAが0の場合はNaNを返すことでゼロ除算を防ぐ
        deviation_pct = (price_series - ma_series) / ma_series.replace(0, np.nan)
        # 閾値以上の乖離があるか判定 (絶対値で)
        deviation_exceeds_threshold = (deviation_pct.abs() >= threshold_pct).astype(float) # float にして NaN を保持
        # rolling sum でウィンドウ内の閾値超えの数をカウント
        rolling_deviation_sum = deviation_exceeds_threshold.rolling(window=window, min_periods=min_periods).sum()
        # ウィンドウ内の閾値超えの数が min_periods 以上で、かつ全ての期間で条件が満たされている (rolling_deviation_sum == window) 場合に1
        # あるいは、rolling_deviation_sum >= min_periods を条件とするなど、継続の定義を調整可能
        persistence_flag = (rolling_deviation_sum == window).astype(float) # float にして NaN を保持
        return pd.Series(persistence_flag, index=price_series.index).fillna(0).astype('category') # NaN を 0 で埋めてカテゴリ型に変換

    # 例として、closeとMA_t_24の乖離が1%以上で3期間継続するフラグを追加
    if 'close' in df.columns and 'MA_t_24' in df.columns:
         df['close_MA_24_deviation_1pct_persist_3h'] = create_deviation_persistence_flag(df['close'], df['MA_t_24'], threshold_pct=0.01, window=3, min_periods=3)
    else:
        print("Warning: 'close' or 'MA_t_24' column not found. Skipping deviation persistence flag.")


    # 例：モメンタムが一定期間プラス（またはマイナス）を継続
    def create_momentum_persistence_flag(momentum_series: pd.Series, window: int, min_periods: int, sign: str = 'positive') -> pd.Series:
        """
        モメンタムが一定期間、指定された符号を継続した場合のフラグを作成
        Args:
            momentum_series (pd.Series): モメンタム系列
            window (int): 継続期間のウィンドウサイズ
            min_periods (int): 計算に必要な最小期間
            sign (str): 'positive' または 'negative'
        Returns:
            pd.Series: モメンタム継続フラグ系列 (0 or 1)
        """
        if sign == 'positive':
            condition_met = (momentum_series > 1e-9).astype(float) # ゼロ付近のノイズを避けるため微小な値と比較
        elif sign == 'negative':
            condition_met = (momentum_series < -1e-9).astype(float) # ゼロ付近のノイズを避けるため微小な値と比較
        else:
            raise ValueError("sign must be 'positive' or 'negative'")

        # rolling sum でウィンドウ内の条件一致数をカウント
        rolling_condition_sum = condition_met.rolling(window=window, min_periods=min_periods).sum()
        # ウィンドウ内の条件一致の数が window と等しい場合に1 (厳密な継続)
        # あるいは、rolling_condition_sum >= min_periods を条件とするなど、継続の定義を調整可能
        persistence_flag = (rolling_condition_sum == window).astype(float) # float にして NaN を保持
        return pd.Series(persistence_flag, index=momentum_series.index).fillna(0).astype('category') # NaN を 0 で埋めてカテゴリ型に変換

    # 例として、momentum_6hがプラスで3期間継続するフラグを追加
    if 'momentum_6h' in df.columns:
        df['momentum_6h_positive_persist_3h'] = create_momentum_persistence_flag(df['momentum_6h'], window=3, min_periods=3, sign='positive')
    else:
         print("Warning: 'momentum_6h' column not found. Skipping momentum persistence flag.")


    # オンチェーンデータの特徴量へのヒステリシス適用例
    # 例：active_receivers_diff が一定期間プラスを継続
    if 'active_receivers_diff' in df.columns:
        df['active_receivers_diff_positive_persist_3h'] = create_momentum_persistence_flag(df['active_receivers_diff'], window=3, min_periods=3, sign='positive')
    else:
        print("Warning: 'active_receivers_diff' column not found. Skipping active_receivers_diff persistence flag.")

    # 例：active_ratio が一定期間閾値を超過
    def create_ratio_threshold_persistence_flag(ratio_series: pd.Series, threshold: float, window: int, min_periods: int, direction: str = 'above') -> pd.Series:
        """
        比率がある閾値を一定期間超過（または下回る）した場合のフラグを作成
        Args:
            ratio_series (pd.Series): 比率系列
            threshold (float): 閾値
            window (int): 継続期間のウィンドウサイズ
            min_periods (int): 計算に必要な最小期間
            direction (str): 'above' または 'below'
        Returns:
            pd.Series: 閾値継続フラグ系列 (0 or 1)
        """
        if direction == 'above':
            condition_met = (ratio_series > threshold).astype(float)
        elif direction == 'below':
            condition_met = (ratio_series < threshold).astype(float)
        else:
            raise ValueError("direction must be 'above' or 'below'")

        rolling_condition_sum = condition_met.rolling(window=window, min_periods=min_periods).sum()
        persistence_flag = (rolling_condition_sum == window).astype(float) # float にして NaN を保持
        return pd.Series(persistence_flag, index=ratio_series.index).fillna(0).astype('category') # NaN を 0 で埋めてカテゴリ型に変換

    # 例として、active_ratioが1.5を超過して3期間継続するフラグを追加
    if 'active_ratio' in df.columns:
        df['active_ratio_above_1.5_persist_3h'] = create_ratio_threshold_persistence_flag(df['active_ratio'], threshold=1.5, window=3, min_periods=3, direction='above')
    else:
        print("Warning: 'active_ratio' column not found. Skipping active_ratio persistence flag.")


    # 欠損値処理は最後にまとめて行うことを推奨
    # generate_features 関数内で既に dropna() が実行されている可能性が高いですが、
    # ヒステリシス特徴量を計算したことで新たな NaN が発生している可能性もあるため、
    # 必要に応じてここで再度 dropna() または適切な欠損値補完を行います。
    # 現状の実装では、create_hysteresis_flag 等の中で fillna(0) しているため、明示的な dropna は不要かもしれません。

    return df

# 注意: この関数は generate_features 関数を呼び出しており、generate_features 内で dropna() が実行されます。
# ヒステリシス特徴量は rolling 処理を含むため、dropna() の位置によっては計算結果に影響が出ます。
# 理想的には、全ての特徴量計算を終えた後に一度だけ dropna() を実行するのが良いです。
# 既存の generate_features 関数を編集できない場合は、generate_features_with_hysteresis の中で
# generate_features が返す DataFrame に対して改めて全てのヒステリシス特徴量を計算し、
# 最後にまとめて dropna() を行う必要があります。
# 上記コードは generate_features が返す DataFrame にヒステリシス特徴量を「追加」する想定で記述しています。
# generate_features 関数を編集可能であれば、generate_features 内で全ての特徴量を生成し、最後に dropna() するのが最も効率的です。


# ## その他の予測に役立つ特徴量の検討と実装
# 
# ### Subtask:
# 移動平均乖離率など、ヒステリシスを考慮した目的変数の予測に役立つ可能性のある他の特徴量を検討し、実装コードを生成します。
# 

# **Reasoning**:
# Calculate the deviation percentage of the close price from various moving averages and add them as new features.
# 
# 

# In[ ]:


def generate_additional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    移動平均乖離率など、目的変数の予測に役立つ可能性のある追加の特徴量を作成する関数。

    Args:
        df (pd.DataFrame): 入力データフレーム（価格データを含む）

    Returns:
        pd.DataFrame: 追加特徴量が追加されたデータフレーム
    """
    df = df.copy()

    # 1. 価格と様々な期間の移動平均との乖離率
    ma_periods = [6, 12, 24, 48, 72, 96, 168, 336]
    for period in ma_periods:
        ma_col = f'MA_t_{period}'
        # MAが0の場合はNaNを返すことでゼロ除算を防ぐ
        df[f'deviation_pct_MA_{period}'] = (df['close'] - df[ma_col]) / df[ma_col].replace(0, np.nan)

    # 2. 乖離率の移動平均や標準偏差
    deviation_features = [f'deviation_pct_MA_{period}' for period in ma_periods]
    rolling_windows = [6, 24] # 例として6時間、24時間のウィンドウ

    for dev_feature in deviation_features:
        if dev_feature in df.columns:
            for window in rolling_windows:
                df[f'{dev_feature}_MA_{window}'] = df[dev_feature].rolling(window=window).mean()
                df[f'{dev_feature}_std_{window}'] = df[dev_feature].rolling(window=window).std()

    # 3. 乖離率の符号フラグ
    for dev_feature in deviation_features:
         if dev_feature in df.columns:
            # 乖離率がプラスなら1、マイナスなら-1、ゼロなら0のフラグ
            df[f'{dev_feature}_sign'] = np.sign(df[dev_feature]).astype('category')

    # 4. 移動平均乖離率に関連するその他の有用な特徴量
    for dev_feature in deviation_features:
         if dev_feature in df.columns:
            # 乖離率の傾き
            df[f'{dev_feature}_slope'] = df[dev_feature].diff()
            # 乖離率の傾きの符号変化フラグ
            # 傾きの符号が変わった場合に1、それ以外は0
            df[f'{dev_feature}_slope_change_flag'] = (np.sign(df[f'{dev_feature}_slope'].shift(1)) != np.sign(df[f'{dev_feature}_slope'])).astype(int).astype('category')

    # 5. オンチェーンデータやオーダーブックデータからの追加特徴量
    # ここでは例としてダミーの特徴量名を記載します。
    # 実際のデータと、ycHL_mGUqgF4、ru9UUANT_1eq のセルで定義された関数（engineer_features, make_orderbook_features）
    # によって生成される特徴量名に合わせて調整してください。
    # 既に engineer_features や make_orderbook_features 関数で多くの特徴量が生成されていますが、
    # さらにヒステリシスを考慮した目的変数に特化する可能性のある特徴量を追加検討・実装します。

    # オンチェーンデータ例：特定の閾値を超えた大口取引の頻度 (仮)
    # 'whale_tx_count' があると仮定し、その移動平均からの乖離や、一定期間の合計などを特徴量とする
    if 'whale_tx_count' in df.columns:
        df['whale_tx_count_MA_24'] = df['whale_tx_count'].rolling(window=24).mean()
        df['whale_tx_count_deviation_MA_24'] = df['whale_tx_count'] - df['whale_tx_count_MA_24']
        # 過去N期間における大口取引の合計や最大値なども有用かもしれない
        df['whale_tx_count_sum_72h'] = df['whale_tx_count'].rolling(window=72).sum()
        df['whale_tx_count_max_72h'] = df['whale_tx_count'].rolling(window=72).max()

    # オーダーブックデータ例：オーダーブックの厚み（例えば、現在価格から±X%以内のbid/ask数量合計）の変化率 (仮)
    # 'buy_qty_sum', 'sell_qty_sum' があると仮定し、それらの合計を簡易的な厚み指標とする
    if 'buy_qty_sum' in df.columns and 'sell_qty_sum' in df.columns:
        df['total_orderbook_depth_proxy'] = df['buy_qty_sum'] + df['sell_qty_sum']
        df['total_orderbook_depth_proxy_pct_change'] = df['total_orderbook_depth_proxy'].pct_change()
        df['total_orderbook_depth_proxy_MA_6'] = df['total_orderbook_depth_proxy'].rolling(window=6).mean()
        df['total_orderbook_depth_proxy_deviation_MA_6'] = df['total_orderbook_depth_proxy'] - df['total_orderbook_depth_proxy_MA_6']

    # 目的変数のラベリングロジック（ウィンドウ4、条件3/4、将来6時間）を考慮すると、
    # 現在から将来6時間にかけての変化や、過去の類似パターンとの比較なども特徴量として有効かもしれないが、
    # 将来の情報（目的変数計算に使用される期間）を特徴量に直接含めることはリークとなるため避ける。
    # 代わりに、過去のデータに基づいて計算される、将来の変化を示唆するような特徴量を検討する。
    # 例：過去のボラティリティや乖離の拡大/縮小のパターン、特定のイベント（大口取引集中など）発生後の価格変動

    # 6. 実装した特徴量生成コードを関数にまとめる (generate_additional_features 関数として定義済み)

    # 7. 欠損値処理
    # ここまでで新たなNaNが多く発生しているため、dropna() を適用
    df = df.dropna()

    return df


# In[ ]:


# =========================================================
# 共通ヘルパー
# =========================================================

def _zscore(series: pd.Series, window: int | None = None) -> pd.Series:
    """
    Z-score を計算するヘルパー。
    window が None の場合は全体平均・分散、
    window が指定されている場合は rolling ベースで計算。
    """
    s = series.astype(float)
    if window is None:
        mu = s.mean()
        sigma = s.std()
        return (s - mu) / (sigma + 1e-8)
    else:
        mu = s.rolling(window).mean()
        sigma = s.rolling(window).std()
        return (s - mu) / (sigma + 1e-8)


def _ema(series: pd.Series, span: int) -> pd.Series:
    """単純な EMA 計算ヘルパー"""
    return series.ewm(span=span, adjust=False).mean()


def _ema_regime_flags(series: pd.Series, short_span: int, long_span: int, prefix: str):
    """
    EMA 短期・長期のクロスからレジームフラグを作成。
    返り値:
        (regime_up_flag, regime_down_flag)
    """
    ema_s = _ema(series, span=short_span)
    ema_l = _ema(series, span=long_span)
    up_flag = (ema_s > ema_l).astype(int)
    down_flag = (ema_s < ema_l).astype(int)
    up_flag.name = f"{prefix}_regime_up_flag"
    down_flag.name = f"{prefix}_regime_down_flag"
    return up_flag, down_flag


def _cusum_flags(series: pd.Series, threshold_sigma: float = 0.5, prefix: str = ""):
    """
    CUSUM によるスパイク検知フラグ（上方向・下方向）を作成。

    threshold_sigma:
        rolling std の何倍を閾値とするか（0.5〜1.0 あたりがおすすめ）
    """
    s = series.astype(float)
    # ここでは「比較的ゆっくり変化する基準」を想定して rolling mean & std
    rolling_mean = s.rolling(200, min_periods=50).mean()
    rolling_std = s.rolling(200, min_periods=50).std()
    threshold = rolling_std * threshold_sigma

    cusum_pos = 0.0
    cusum_neg = 0.0
    up_flags = []
    down_flags = []

    for v, mu, th in zip(s, rolling_mean, threshold):
        if np.isnan(v) or np.isnan(mu) or np.isnan(th):
            cusum_pos = 0.0
            cusum_neg = 0.0
            up_flags.append(0)
            down_flags.append(0)
            continue

        # 上方向
        if v - mu > th:
            cusum_pos += v - mu
            cusum_neg = 0.0
        # 下方向
        elif mu - v > th:
            cusum_neg += mu - v
            cusum_pos = 0.0
        else:
            cusum_pos = 0.0
            cusum_neg = 0.0

        up_flags.append(int(cusum_pos > th))
        down_flags.append(int(cusum_neg > th))

    up_flags = pd.Series(up_flags, index=s.index, name=f"{prefix}_cusum_up_flag")
    down_flags = pd.Series(down_flags, index=s.index, name=f"{prefix}_cusum_down_flag")
    return up_flags, down_flags


# =========================================================
# 1. VIX 用特徴量
# =========================================================

def add_vix_features(
    df: pd.DataFrame,
    col: str = "VIX_close",
    rolling_z_window: int = 252,
    ema_short: int = 10,
    ema_long: int = 30,
    cusum_sigma: float = 0.5,
) -> pd.DataFrame:
    """
    df に VIX 関連の特徴量を追加する。

    前提:
        df[col] に VIX の終値などが入っていること。
        （日足を1hに forward-fill しておくなど）
    追加される列:
        - VIX_zn              : rolling Z-score
        - delta_VIX           : 1ステップ前との差分
        - VIX_regime_up_flag  : EMA短期>長期
        - VIX_regime_down_flag
        - VIX_cusum_up_flag   : CUSUM上方向スパイク
        - VIX_cusum_down_flag : CUSUM下方向スパイク
    """
    df = df.copy()
    if col not in df.columns:
        raise ValueError(f"{col} not found in df.columns")

    s = df[col].astype(float)

    # Z-score
    df["VIX_zn"] = _zscore(s, window=rolling_z_window)

    # 差分
    df["delta_VIX"] = s.diff()

    # EMA レジーム
    up_flag, down_flag = _ema_regime_flags(s, short_span=ema_short, long_span=ema_long, prefix="VIX")
    df[up_flag.name] = up_flag
    df[down_flag.name] = down_flag

    # CUSUM スパイク
    up_cusum, down_cusum = _cusum_flags(s, threshold_sigma=cusum_sigma, prefix="VIX")
    df[up_cusum.name] = up_cusum
    df[down_cusum.name] = down_cusum

    return df


# =========================================================
# 2. Funding rate 用特徴量
# =========================================================

def add_funding_features(
    df: pd.DataFrame,
    col: str = "fundingRate",
    rolling_z_window: int = 200,
    ema_short: int = 10,
    ema_long: int = 30,
    cusum_sigma: float = 0.5,
) -> pd.DataFrame:
    """
    df に Funding rate 関連の特徴量を追加する。

    前提:
        df[col] に funding rate (例: Binance ETHUSDT の 8h funding) が入っていること。
        必要に応じて 1h 足に forward-fill 済であること。
    追加される列:
        - funding_zn
        - delta_funding
        - funding_regime_up_flag / funding_regime_down_flag
        - funding_cusum_up_flag / funding_cusum_down_flag
    """
    df = df.copy()
    if col not in df.columns:
        raise ValueError(f"{col} not found in df.columns")

    s = df[col].astype(float)

    df["funding_zn"] = _zscore(s, window=rolling_z_window)
    df["delta_funding"] = s.diff()

    up_flag, down_flag = _ema_regime_flags(s, short_span=ema_short, long_span=ema_long, prefix="funding")
    df[up_flag.name] = up_flag
    df[down_flag.name] = down_flag

    up_cusum, down_cusum = _cusum_flags(s, threshold_sigma=cusum_sigma, prefix="funding")
    df[up_cusum.name] = up_cusum
    df[down_cusum.name] = down_cusum

    return df


# =========================================================
# 3. Global Liquidity Index 用特徴量
# =========================================================

def add_gli_features(
    df: pd.DataFrame,
    col: str = "GLI_norm",
    rolling_z_window: int = 252,
    ema_short: int = 30,
    ema_long: int = 90,
    cusum_sigma: float = 0.5,
) -> pd.DataFrame:
    """
    df に Global Liquidity Index(GLI) 関連の特徴量を追加する。

    前提:
        build_global_liquidity_index などで作成した GLI_norm (Z-score済み) を
        マージして df[col] として持っていること。
        （日足 → 1h に forward-fill 済など）
    追加される列:
        - gli_zn             : 追加の rolling z-score（中長期の偏差をとる用）
        - delta_gli          : 1ステップ差分
        - gli_regime_up_flag / gli_regime_down_flag
        - gli_cusum_up_flag / gli_cusum_down_flag
    """
    df = df.copy()
    if col not in df.columns:
        raise ValueError(f"{col} not found in df.columns")

    s = df[col].astype(float)

    df["gli_zn"] = _zscore(s, window=rolling_z_window)
    df["delta_gli"] = s.diff()

    up_flag, down_flag = _ema_regime_flags(s, short_span=ema_short, long_span=ema_long, prefix="gli")
    df[up_flag.name] = up_flag
    df[down_flag.name] = down_flag

    up_cusum, down_cusum = _cusum_flags(s, threshold_sigma=cusum_sigma, prefix="gli")
    df[up_cusum.name] = up_cusum
    df[down_cusum.name] = down_cusum

    return df


# =========================================================
# 4. BTC dominance 用特徴量
# =========================================================

def add_btc_dominance_features(
    df: pd.DataFrame,
    col: str = "btc_dominance",
    rolling_z_window: int = 252,
    ema_short: int = 20,
    ema_long: int = 60,
    cusum_sigma: float = 0.5,
) -> pd.DataFrame:
    """
    df に BTC dominance 関連の特徴量を追加する。

    前提:
        df[col] に BTC dominance (0〜1 または 0〜100%) が入っていること。
        日足を 1h に forward-fill してもよい。
    追加される列:
        - btc_dom_zn
        - delta_btc_dom
        - btc_dom_regime_up_flag / btc_dom_regime_down_flag
        - btc_dom_cusum_up_flag / btc_dom_cusum_down_flag
    """
    df = df.copy()
    if col not in df.columns:
        raise ValueError(f"{col} not found in df.columns")

    s = df[col].astype(float)

    df["btc_dom_zn"] = _zscore(s, window=rolling_z_window)
    df["delta_btc_dom"] = s.diff()

    up_flag, down_flag = _ema_regime_flags(s, short_span=ema_short, long_span=ema_long, prefix="btc_dom")
    df[up_flag.name] = up_flag
    df[down_flag.name] = down_flag

    up_cusum, down_cusum = _cusum_flags(s, threshold_sigma=cusum_sigma, prefix="btc_dom")
    df[up_cusum.name] = up_cusum
    df[down_cusum.name] = down_cusum

    return df


# In[ ]:


# =========================================================
# 共通ヘルパー
# =========================================================

def _zscore(series: pd.Series, window: int | None = None) -> pd.Series:
    """
    Z-score を計算するヘルパー。
    window=None の場合は全期間で標準化、
    window>0 の場合は rolling ベースで標準化。
    """
    s = series.astype(float)
    if window is None:
        mu = s.mean()
        sigma = s.std()
        return (s - mu) / (sigma + 1e-8)
    else:
        mu = s.rolling(window).mean()
        sigma = s.rolling(window).std()
        return (s - mu) / (sigma + 1e-8)


def _ema(series: pd.Series, span: int) -> pd.Series:
    """EMA"""
    return series.ewm(span=span, adjust=False).mean()


def _ema_regime_flags(series: pd.Series, short_span: int, long_span: int, prefix: str):
    """
    EMA短期/長期クロスからレジームフラグを作成。
    返り値:
        up_flag  : prefix + "_regime_up_flag"
        down_flag: prefix + "_regime_down_flag"
    """
    s = series.astype(float)
    ema_s = _ema(s, span=short_span)
    ema_l = _ema(s, span=long_span)
    up_flag = (ema_s > ema_l).astype(int)
    down_flag = (ema_s < ema_l).astype(int)
    up_flag.name = f"{prefix}_regime_up_flag"
    down_flag.name = f"{prefix}_regime_down_flag"
    return up_flag, down_flag


def _cusum_flags(series: pd.Series, threshold_sigma: float = 0.5, prefix: str = ""):
    """
    CUSUM によるスパイク検知フラグ（上方向・下方向）を作成。
    threshold_sigma:
        rolling std の何倍を閾値とするか（0.5〜1.0 程度）
    """
    s = series.astype(float)
    # ゆっくり変化する基準線
    rolling_mean = s.rolling(200, min_periods=50).mean()
    rolling_std = s.rolling(200, min_periods=50).std()
    threshold = rolling_std * threshold_sigma

    cusum_pos = 0.0
    cusum_neg = 0.0
    up_flags = []
    down_flags = []

    for v, mu, th in zip(s, rolling_mean, threshold):
        if np.isnan(v) or np.isnan(mu) or np.isnan(th):
            cusum_pos = 0.0
            cusum_neg = 0.0
            up_flags.append(0)
            down_flags.append(0)
            continue

        if v - mu > th:
            cusum_pos += v - mu
            cusum_neg = 0.0
        elif mu - v > th:
            cusum_neg += mu - v
            cusum_pos = 0.0
        else:
            cusum_pos = 0.0
            cusum_neg = 0.0

        up_flags.append(int(cusum_pos > th))
        down_flags.append(int(cusum_neg > th))

    up_flags = pd.Series(up_flags, index=s.index, name=f"{prefix}_cusum_up_flag")
    down_flags = pd.Series(down_flags, index=s.index, name=f"{prefix}_cusum_down_flag")
    return up_flags, down_flags


# =========================================================
# 1. Stablecoin Velocity 関連特徴量
# =========================================================

def add_stablecoin_features(
    df: pd.DataFrame,
    vol_col: str = "stablecoin_volume_usd",
    vel_col: str = "velocity_stablecoin_24h",
    rolling_z_window: int = 24 * 7,
    ema_short: int = 24,
    ema_long: int = 24 * 7,
    cusum_sigma: float = 0.5,
) -> pd.DataFrame:
    """
    ステーブルコイン転送量 & Velocity から特徴量を作成。

    追加される列の例:
        - stablecoin_volume_usd_zn
        - delta_stablecoin_volume_usd
        - velocity_stablecoin_24h_zn
        - delta_velocity_stablecoin_24h
        - velocity_stablecoin_regime_up_flag / _down_flag
        - velocity_stablecoin_cusum_up_flag / _down_flag
    """
    df = df.copy()
    for c in [vol_col, vel_col]:
        if c not in df.columns:
            raise ValueError(f"{c} not found in df.columns")

    vol = df[vol_col].astype(float)
    vel = df[vel_col].astype(float)

    # Volume の水準と変化
    df[f"{vol_col}_zn"] = _zscore(vol, window=rolling_z_window)
    df[f"delta_{vol_col}"] = vol.diff()

    # Velocity の水準と変化
    df[f"{vel_col}_zn"] = _zscore(vel, window=rolling_z_window)
    df[f"delta_{vel_col}"] = vel.diff()

    # Velocity ベースのレジーム
    up_flag, down_flag = _ema_regime_flags(
        vel.fillna(method="ffill"),
        short_span=ema_short,
        long_span=ema_long,
        prefix="velocity_stablecoin_24h"
    )
    df[up_flag.name] = up_flag
    df[down_flag.name] = down_flag

    # Velocity のスパイク検知
    up_cusum, down_cusum = _cusum_flags(
        vel.fillna(method="ffill"),
        threshold_sigma=cusum_sigma,
        prefix="velocity_stablecoin_24h"
    )
    df[up_cusum.name] = up_cusum
    df[down_cusum.name] = down_cusum

    return df


# =========================================================
# 2. DEX 流動性（Uniswap など）
# =========================================================

def add_dex_liquidity_features(
    df: pd.DataFrame,
    dex_vol_col: str = "dex_volume_usd",
    rolling_z_window: int = 24 * 7,
    ema_short: int = 24,
    ema_long: int = 24 * 7,
    cusum_sigma: float = 0.5,
) -> pd.DataFrame:
    """
    DEX 取引量から流動性・トレンド強度系の特徴量を作成。

    追加される列:
        - dex_volume_usd_zn
        - delta_dex_volume_usd
        - dex_liquidity_regime_up_flag / _down_flag
        - dex_liquidity_cusum_up_flag / _down_flag
    """
    df = df.copy()
    if dex_vol_col not in df.columns:
        raise ValueError(f"{dex_vol_col} not found in df.columns")

    s = df[dex_vol_col].astype(float)

    df[f"{dex_vol_col}_zn"] = _zscore(s, window=rolling_z_window)
    df[f"delta_{dex_vol_col}"] = s.diff()

    up_flag, down_flag = _ema_regime_flags(
        s.fillna(method="ffill"),
        short_span=ema_short,
        long_span=ema_long,
        prefix="dex_volume"
    )
    df[up_flag.name] = up_flag
    df[down_flag.name] = down_flag

    up_cusum, down_cusum = _cusum_flags(
        s.fillna(method="ffill"),
        threshold_sigma=cusum_sigma,
        prefix="dex_volume"
    )
    df[up_cusum.name] = up_cusum
    df[down_cusum.name] = down_cusum

    return df


# =========================================================
# 3. Activity 指標（gas fee / tx count）
# =========================================================

def add_activity_features(
    df: pd.DataFrame,
    gas_col: str = "median_gas_price",
    tx_col: str = "tx_count",
    rolling_z_window: int = 24 * 7,
    cusum_sigma: float = 0.5,
) -> pd.DataFrame:
    """
    ガス代 & Tx 数からネットワーク Activity 特徴量を作成。

    追加される列:
        - median_gas_price_zn, delta_median_gas_price,
          gas_spike_up_flag, gas_spike_down_flag
        - tx_count_zn, delta_tx_count,
          tx_spike_up_flag, tx_spike_down_flag
    """
    df = df.copy()
    for c in [gas_col, tx_col]:
        if c not in df.columns:
            raise ValueError(f"{c} not found in df.columns")

    gas = df[gas_col].astype(float)
    tx = df[tx_col].astype(float)

    # Gas
    df[f"{gas_col}_zn"] = _zscore(gas, window=rolling_z_window)
    df[f"delta_{gas_col}"] = gas.diff()
    up, down = _ema_regime_flags(
        gas.fillna(method="ffill"), short_span=12, long_span=12*7, prefix="gas_fee"
    )
    df[up.name] = up
    df[down.name] = down
    up_c, down_c = _cusum_flags(
        gas.fillna(method="ffill"),
        threshold_sigma=cusum_sigma,
        prefix="gas_fee"
    )
    df[up_c.name] = up_c
    df[down_c.name] = down_c

    # Tx Count
    df[f"{tx_col}_zn"] = _zscore(tx, window=rolling_z_window)
    df[f"delta_{tx_col}"] = tx.diff()
    up, down = _ema_regime_flags(
        tx.fillna(method="ffill"), short_span=24, long_span=24*7, prefix="tx"
    )
    df[up.name] = up
    df[down.name] = down

    up_c, down_c = _cusum_flags(
        tx.fillna(method="ffill"), threshold_sigma=2.0, prefix="tx"
    )
    df[up_c.name] = up_c
    df[down_c.name] = down_c

    return df


# =========================================================
# 4. Realized Volatility 特徴量
# =========================================================

def add_realized_vol_features(
    df: pd.DataFrame,
    vol_col: str = "realized_vol_24h",
    rolling_z_window: int = 24 * 14,
    high_vol_quantile: float = 0.7,
    low_vol_quantile: float = 0.3,
) -> pd.DataFrame:
    """
    realized_vol_24h を元にボラティリティレジームの特徴量を作成。

    追加される列:
        - realized_vol_24h_zn
        - delta_realized_vol_24h
        - high_vol_flag / low_vol_flag
    """
    df = df.copy()
    if vol_col not in df.columns:
        raise ValueError(f"{vol_col} not found in df.columns")

    s = df[vol_col].astype(float)

    df[f"{vol_col}_zn"] = _zscore(s, window=rolling_z_window)
    df[f"delta_{vol_col}"] = s.diff()

    # 分位に基づく High / Low vol flag
    q_high = s.quantile(high_vol_quantile)
    q_low = s.quantile(low_vol_quantile)

    df["high_vol_flag"] = (s >= q_high).astype(int)
    df["low_vol_flag"] = (s <= q_low).astype(int)

    return df


# =========================================================
# 5. ETH Volume Dominance 特徴量
# =========================================================

def add_eth_volume_dominance_features(
    df: pd.DataFrame,
    dom_col: str = "eth_volume_dominance",
    rolling_z_window: int = 24 * 14,
    cusum_sigma: float = 0.5,
) -> pd.DataFrame:
    """
    ETH / BTC volume dominance proxy から特徴量を作成。

    追加される列:
        - eth_volume_dominance_zn
        - delta_eth_volume_dominance
        - eth_dom_cusum_up_flag / _down_flag
        - alt_season_flag / btc_season_flag（閾値は0.5付近で判定）
    """
    df = df.copy()
    if dom_col not in df.columns:
        raise ValueError(f"{dom_col} not found in df.columns")

    s = df[dom_col].astype(float)

    df[f"{dom_col}_zn"] = _zscore(s, window=rolling_z_window)
    df[f"delta_{dom_col}"] = s.diff()

    up_cusum, down_cusum = _cusum_flags(
        s.fillna(method="ffill"),
        threshold_sigma=cusum_sigma,
        prefix="eth_dom"
    )
    df[up_cusum.name] = up_cusum      # eth_dom_cusum_up_flag
    df[down_cusum.name] = down_cusum  # eth_dom_cusum_down_flag

    # シンプルな regime flag：ETH優位 or BTC優位
    df["alt_season_flag"] = (s > 0.5).astype(int)   # ETH volume が優位
    df["btc_season_flag"] = (s < 0.5).astype(int)   # BTC volume が優位

    return df


# =========================================================
# 6. Dune の funding_rate 列用特徴量
# =========================================================

def add_funding_features_from_dune(
    df: pd.DataFrame,
    col: str = "long_short_imbalance",
    rolling_z_window: int = 24 * 7,
    ema_short: int = 24,
    ema_long: int = 24 * 7,
    cusum_sigma: float = 0.5,
) -> pd.DataFrame:
    """
    Dune から取得した funding proxy (long_short_imbalance) を使った特徴量を生成。

    追加される特徴量:
        - long_short_imbalance_zn
        - delta_long_short_imbalance
        - funding_regime_up_flag
        - funding_regime_down_flag
        - funding_cusum_up_flag
        - funding_cusum_down_flag

    備考:
        funding の「本物の funding rate 」ではなく、
        Dune の perp 取引量ベースの long/short volume imbalance を使用する。
    """
    df = df.copy()

    if col not in df.columns:
        raise ValueError(f"{col} not found in df.columns")

    s = df[col].astype(float)

    # ---- Z-score ----
    df[f"{col}_zn"] = _zscore(s, window=rolling_z_window)

    # ---- Difference ----
    df[f"delta_{col}"] = s.diff()

    # ---- EMA-based regime (trend direction of funding bias) ----
    up_flag, down_flag = _ema_regime_flags(
        s.fillna(method="ffill"),
        short_span=ema_short,
        long_span=ema_long,
        prefix="funding"
    )

    df[up_flag.name] = up_flag
    df[down_flag.name] = down_flag

    # ---- CUSUM (detect funding regime shifts) ----
    up_cusum, down_cusum = _cusum_flags(
        s.fillna(method="ffill"),
        threshold_sigma=cusum_sigma,
        prefix="funding"
    )

    df[up_cusum.name] = up_cusum
    df[down_cusum.name] = down_cusum

    return df


# ## 特徴量エンジニアリングの全体的なアドバイス
# 
# ### Subtask:
# 特徴量エンジニアリング全般に関して、今回のケースに特化したアドバイスや注意点などを提供します。
# 

# **Reasoning**:
# Provide advice and considerations for feature engineering based on the previous steps and the characteristics of the target variable, covering feature selection, scaling, encoding, time series specifics, and future ideas.
# 
# 

# In[ ]:


# 1. これまでの特徴量エンジニアリングのプロセス（ヒステリシス特徴量、追加特徴量）を振り返ります。
print("### 特徴量エンジニアリングのプロセス振り返り ###")
print("これまでのプロセスでは、以下の特徴量群を生成しました。")
print("- **日時特徴量**: 曜日、月、時間、年内の経過日数などを周期成分（sin/cos）やカテゴリ変数として追加しました。")
print("- **価格・出来高ベースの特徴量**: リターン、ボラティリティ、モメンタム、移動平均（MA）、ボリンジャーバンド、RSI、MACD、ストキャスティクス、ATRなどをtalibを用いて計算しました。")
print("- **エントロピー特徴量**: リターン符号列のエントロピーを計算しました。")
print("- **Volume Profile特徴量**: 価格帯別出来高の統計量（平均、最大、標準偏差）、POC、VAH、VAL、VWAPとその統計量などを計算しました。")
print("- **オンチェーンデータ特徴量**: 総フロー量、アドレス数、アクティブ送受信者数、Dex出来高、コントラクトコール数、大口取引数などの差分、変化率、移動平均、ラグ、トレンド差分、標準偏差、アクティブ比率などを計算しました。")
print("- **オーダーブックデータ特徴量**: 売買数量合計、価格統計量、売買比率、ボラティリティ、価格変化、スプレッド、平均取引サイズ、売買件数比率、売買数量差分、売買圧力変化などを計算しました。")
print("- **移動平均乖離率特徴量**: 価格と多様な期間のMAとの乖離率、その移動平均、標準偏差、符号、傾き、傾き符号変化フラグなどを計算しました。")
print("- **ヒステリシス特徴量**: MAクロスフラグ、価格-MA乖離、モメンタム、オンチェーン/オーダーブック関連指標など、特定の条件が一定期間継続しているかを示すフラグを導入しました。")
print("\nこれらの特徴量は、価格のトレンド、モメンタム、ボラティリティ、市場の需給バランス、ネットワーク活動など、多角的な側面を捉えようとしています。")

# 2. 目的変数の特性（ヒステリシスを含むラベリング）を踏まえ、どのような特徴量がモデルにとって特に重要になるかを考察し、アドバイスを提供します。
print("\n### 目的変数の特性を踏まえた重要特徴量とアドバイス ###")
print("目的変数がヒステリシス（ウィンドウ4、条件3/4、将来6時間）を含むラベリングであることから、モデルは単なる短期的な価格変動ではなく、ある程度の期間にわたる持続的なトレンドや市場の状態変化を捉える特徴量を重視すると考えられます。")
print("特に重要になる可能性のある特徴量:")
print("- **中期〜長期の移動平均とその傾き・乖離**: 目的変数が将来6時間の価格変化に基づいているため、短期的なノイズに強く、より長期的なトレンドを示すMA（例: MA_t_24, MA_t_72, MA_t_168）とその傾き（トレンドの方向と勢い）、および価格や短期MAとの乖離（買われすぎ/売られすぎ、トレンドからの乖離度）が重要です。乖離率特徴量（deviation_pct_MA_period）やその統計量、傾き特徴量（MA_t_period_slope）は特に有効でしょう。")
print("- **ヒステリシス特徴量**: 目的変数と同様に「持続性」を組み込んだ特徴量は、ラベリングロジックとの整合性が高く、モデルが学習しやすいパターンを提供する可能性があります。MAクロスの持続、乖離の持続、モメンタムの持続などを示すフラグ特徴量（例: MA_6_24_cross_flag_hysteresis, close_MA_24_deviation_1pct_persist_3h）は重要です。")
print("- **ボラティリティ関連特徴量**: 将来の価格変動の大きさに関連するため、ローリング標準偏差（rolling_std_6h, rolling_std_12h）、ボリンジャーバンド幅（BB_width）、ATRなどが重要です。ボラティリティの変化やその持続性を示す特徴量（volatility_change_flag）も有用です。")
print("- **市場の需給バランスを示す特徴量**: オーダーブックデータやオンチェーンデータから得られる売買圧力、出来高の偏り、大口取引の動向などは、将来の価格動向に影響を与える可能性があります。buy_sell_ratio, buy_sell_imbalance, total_orderbook_depth_proxy, whale_tx_count関連の特徴量などが考えられます。これらの特徴量の変化率や移動平均、そしてヒステリシスを考慮した特徴量も有効です。")
print("- **モメンタム系指標**: RSI、MACD、ストキャスティクスなどは、買われすぎ/売られすぎやトレンドの勢いを示唆するため、目的変数の発生を先行して示唆する可能性があります。これらの指標自体の値だけでなく、その変化やクロス、特定の閾値の超過なども特徴量として検討すべきです。")

print("\nアドバイス:")
print("1. **重要度評価**: モデル学習後、特徴量の重要度を評価し、予測に寄与しない特徴量を削減することを検討してください（過学習抑制、計算コスト削減）。線形モデルの係数、ツリー系モデルの特徴量重要度などが参考になります。")
print("2. **ドメイン知識の活用**: 生成した特徴量が、市場の動きやオンチェーン活動においてどのような意味を持つかを理解し、仮説に基づいて特徴量エンジニアリングを進めることが、より効果的な特徴量発見につながります。")
print("3. **目的変数との相関分析**: 各特徴量と目的変数（またはその先行/遅行バージョン）との相関を分析することで、有用な特徴量の候補を絞り込むことができます。ただし、非線形な関係もあるため、相関だけでは判断できません。")

# 3. 特徴量の選択や削減（例: 相関の高い特徴量の扱い、重要度の低い特徴量の除外）に関する一般的なアドバイスと、今回のデータに特化した注意点を述べます。
print("\n### 特徴量選択・削減に関するアドバイス ###")
print("一般的なアドバイス:")
print("- **高相関特徴量の扱い**: 複数の特徴量が互いに高い相関を持つ場合（多重共線性）、モデルの安定性や解釈性に悪影響を与える可能性があります。片方を削除するか、PCAなどで次元削減するか、あるいは両方を維持しつつ正則化手法（Lassoなど）を検討します。ただし、ツリー系モデル（Random Forest, LightGBMなど）は線形モデルほど多重共線性の影響を受けにくい傾向があります。")
print("- **重要度の低い特徴量の除外**: モデル学習後の特徴量重要度に基づいて、寄与の小さい特徴量を除外することで、モデルの複雑さを減らし、過学習を抑制し、学習時間を短縮できます。")
print("- **特徴量選択手法の利用**: ラッパー法（Recursive Feature Elimination: RFEなど）、フィルター法（相関係数、分散分析など）、埋め込み法（Lassoなどモデル内部で重要度を計算）など、体系的な特徴量選択手法を試すことも有効です。")

print("\n今回のデータに特化した注意点:")
print("- **時系列相関**: 金融・仮想通貨データでは、特徴量間に強い自己相関や相互相関が存在するのが一般的です。単純な相関係数だけでなく、ラグを考慮した相互相関関数なども分析すると良いでしょう。")
print("- **特徴量間の非線形関係**: 価格、出来高、オンチェーンデータなど、異なる種類のデータから生成された特徴量間には、複雑な非線形関係が存在する可能性があります。高相関でも安易に削除せず、モデルがその非線形性を捉えられるか検討が必要です。ツリー系モデルやニューラルネットワークは非線形関係の学習が得意です。")
print("- **特徴量の時間的安定性**: ある時期には重要だった特徴量が、別の時期にはそうでないという、特徴量重要度の時間的な変動があり得ます。これは市場構造の変化を反映している可能性があります。モデルの頑健性を高めるためには、時間的な変化を考慮した特徴量選択や、アンサンブル学習、モデルの定期的な再学習などが有効です。")
print("- **オンチェーン/オーダーブック特徴量のスパース性・ノイズ**: これらのデータは取引活動やネットワークの状態を直接反映するため有用ですが、突発的な大口取引やbot活動などによるノイズやスパース性を持つ場合があります。ローリング統計量やヒステリシスを適用することで、ノイズを抑制し、より安定したシグナルを抽出することが重要です。")

# 4. 特徴量のスケーリングやエンコーディング（カテゴリ特徴量の扱い）に関するアドバイスを提供します。
print("\n### スケーリング・エンコーディングに関するアドバイス ###")
print("スケーリング（標準化/正規化）:")
print("- **必要性**: 多くの機械学習アルゴリズム（特に線形モデル、SVM、ニューラルネットワーク、距離ベースの手法など）は、特徴量のスケールに敏感です。スケールが大きく異なる特徴量があると、学習が不安定になったり、特定のスケールの大きい特徴量に過度に影響されたりする可能性があります。勾配降下法を用いるモデルでは収束速度にも影響します。")
print("- **推奨手法**: 標準化 (StandardScaler) がよく用いられます。平均を0、標準偏差を1に変換します。外れ値に強いMinMaxScalerやRobustScalerなども検討できます。")
print("- **今回のデータへの適用**: 価格関連、出来高関連、オンチェーンデータ関連など、スケールが大きく異なる特徴量が多く含まれるため、スケーリングは必須と考えられます。時間軸に沿ってリークしないように注意が必要です（訓練データで学習したスケーラーをテストデータに適用）。")

print("エンコーディング（カテゴリ特徴量）:")
print("- **対象**: 曜日（day_of_week）、月（month）、時間（hour_of_day）、移動平均クロスフラグ（MA_..._cross_flag）、ヒステリシスフラグ（..._hysteresis）など、カテゴリ型の特徴量が該当します。")
print("- **推奨手法**:")
print("  - **One-Hot Encoding**: カテゴリ数が多い場合は次元が増大しますが、最も一般的で安全な手法です。各カテゴリをバイナリのダミー変数に変換します。")
print("  - **Label Encoding**: 順序性のあるカテゴリ（例: 低, 中, 高）には有効ですが、順序性のないカテゴリに使うとモデルが誤った順序関係を学習する可能性があります。ツリー系モデルは内部的にカテゴリ変数を扱えるものもあります（LightGBMなど）。")
print("  - **Target Encoding**: 目的変数の情報を使ってエンコードしますが、リークしやすいリスクがあります。時系列データでは特に注意が必要です（過去のデータのみを使用）。")
print("- **今回のデータへの適用**: 曜日、月、時間などの周期的なカテゴリは、sin/cos変換で表現する方が滑らかな関係を捉えられる場合がありますが、One-Hot Encodingも有効です。フラグ特徴量（0/1またはカテゴリ型）は、モデルによってはそのまま扱える場合もありますが、One-Hot Encodingしておくと汎用性が高まります。")

# 5. 時系列データにおける特徴量エンジニアリング特有の注意点（例: リークの回避、時間軸に沿った分割）について再度強調し、注意喚起を行います。
print("\n### 時系列データにおける特有の注意点 ###")
print("- **未来からの情報リーク (Look-ahead Bias) の回避**: 特徴量を計算する際に、予測対象の時点よりも未来のデータを使用しては絶対にいけません。これはモデルが未来を知っているかのような状態になり、訓練データでの性能が過剰に高くなる（現実では再現できない）「リーク」を引き起こします。")
print("  - **ローリング統計量**: `rolling(...).mean()`, `rolling(...).std()` などは、デフォルトでは現在の時点を含めて計算されますが、`shift(1)` などで未来の情報を参照しないように注意が必要です。今回のコードでは適切に過去のデータのみを使用するように実装されています。")
print("  - **ラグ特徴量**: `shift(n)` で過去のデータを参照するのは正しい使い方です。")
print("  - **目的変数に基づいた特徴量**: 目的変数自体や、目的変数を計算するのに使われる未来の情報（例: 将来の価格変化）を特徴量に直接使用してはいけません。ターゲットエンコーディングなど、目的変数を利用する手法を使う場合は、訓練データ内の過去のデータのみを使用するように厳密に制御する必要があります。")
print("- **時間軸に沿ったデータ分割**: 訓練データ、検証データ、テストデータは、必ず時間的に連続するように分割する必要があります。ランダムシャッフルして分割すると、訓練データに未来の情報が混入し、リークの原因となります。訓練データでモデルを学習し、その直後の期間を検証データ、さらにその後の期間をテストデータとするように分割します（時系列クロスバリデーションも同様）。")
print("- **ウィンドウサイズの考慮**: 移動平均やローリング統計量、ヒステリシス特徴量などで使用するウィンドウサイズは、予測したい期間や市場の特性に合わせて適切に設定する必要があります。ウィンドウが短すぎるとノイズを拾いやすく、長すぎると変化への追従が遅れます。")
print("- **初期期間のNaN**: ローリング計算や差分などで発生する初期期間のNaNは、モデル学習前に適切に処理（削除または補完）する必要があります。")

# 6. 今後の特徴量エンジニアリングの改善や探索に関するアイデア（例: 交互作用特徴量、非線形特徴量）を提案します。
print("\n### 今後の特徴量エンジニアリング改善・探索アイデア ###")
print("- **交互作用特徴量 (Interaction Features)**: 複数の特徴量を組み合わせることで、単独では捉えられないパターンを表現できます。例: 'RSI' * 'volume_USD', 'MA_t_6_slope' * 'volatility' など。ドメイン知識に基づいて有効そうな組み合わせを検討します。")
print("- **非線形特徴量**: 特徴量の多項式変換（例: price**2, volume_USD**0.5）や対数変換（例: np.log(volume_USD)）などで非線形性を導入します。特に線形モデルを用いる場合に有効です。")
print("- **時系列モデルからの特徴量**: ARIMAモデルやProphetモデルなどで時系列成分（トレンド、季節性、残差）を抽出し、それを特徴量として追加するアプローチもあります。")
print("- **異なる時間スケールの特徴量**: 現在の1時間足だけでなく、4時間足や日足などのデータも集計し、そこから特徴量（例: 日足のMA、日足のRSI）を抽出して組み合わせることで、より広範な市場状況を捉えることができます。")
print("- **外部データ**: 仮想通貨市場に関連する外部データ（ニュースセンチメント、SNS感情、主要インフルエンサーの投稿数、規制動向、技術開発ニュースなど）を収集し、特徴量として組み込むことで、予測精度が向上する可能性があります。ただし、データの取得や構造化が課題となります。")
print("- **異常検知**: 価格、出来高、オンチェーンデータなどの異常値を検知し、その発生をフラグ特徴量として追加することも、市場の特異な動きを捉える上で有効かもしれません。")
print("- **特徴量の自動生成 (Feature Toolsなど)**: 特徴量エンジニアリングを自動化するライブラリ（Feature Toolsなど）を利用して、網羅的に特徴量を生成し、その中から有用なものを選択するアプローチも考えられます。")
print("\nこれらのアイデアを参考に、さらに予測力の高い特徴量セットの構築を目指してください。")


# **Reasoning**:
# Continue printing the advice on encoding and then move on to the specific considerations for time series data and future feature engineering ideas, as outlined in the instructions.
# 
# 

# ## Summary:
# 
# ### Data Analysis Key Findings
# 
# *   Hysteresis was successfully incorporated into feature engineering by creating flag features that assess the persistence of conditions (e.g., moving average crosses, deviations from moving averages, momentum direction) over a defined window and threshold.
# *   Helper functions were developed to apply hysteresis logic to different types of features, handling rolling calculations and potential NaN values.
# *   Additional features considered useful for predicting the target variable were implemented, including percentage deviation of price from various moving averages, rolling statistics (mean, standard deviation) of these deviations, sign flags of deviations, and the slope and slope change flag of deviations.
# *   Placeholder logic was included to demonstrate the integration of potential features from on-chain and order book data, such as rolling statistics and differences of assumed relevant columns (e.g., whale transaction count, buy/sell quantity sum).
# *   Comprehensive advice on feature engineering for this specific time series prediction task was provided, covering the importance of mid/long-term moving averages and hysteresis features, strategies for feature selection/reduction (handling correlation, importance), methods for scaling and encoding, crucial time series considerations (avoiding leakage, time-based splitting), and ideas for further feature exploration (interactions, non-linear transformations, external data).
# 
# ### Insights or Next Steps
# 
# *   Evaluate the predictive power and importance of the newly created hysteresis and deviation features using appropriate model training and evaluation techniques, paying close attention to time series cross-validation to avoid leakage.
# *   Refine the integration of on-chain and order book features by obtaining actual data and implementing the placeholder logic with specific, relevant metrics that reflect market sentiment, demand/supply dynamics, and large-scale movements.
# 
