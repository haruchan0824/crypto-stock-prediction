#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install ccxt')
import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta
import torch
import numpy as np
import glob
import os
import features
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from google.colab import drive

# データ取得関数
def fetch_ohlcv_all(exchange, symbol, timeframe, since, limit=1000):
    all_data = []
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not ohlcv:
            break
        all_data += ohlcv
        since = ohlcv[-1][0] + 1  # 次の開始時間（+1ms）
        time.sleep(exchange.rateLimit / 1000)  # API制限対策
        if len(ohlcv) < limit:
            break
    return all_data


# In[ ]:


import requests
import pandas as pd
from datetime import datetime, timedelta
import time

def get_eth_hourly_data(api_key, start_date, end_date):
    """
    ETHの1時間足データをstart_dateからend_dateの範囲で取得する関数。

    Parameters:
        api_key (str): CryptocompareのAPIキー。
        start_date (str): 開始日 (例: '2023-01-01')
        end_date (str): 終了日 (例: '2023-01-31')

    Returns:
        pd.DataFrame: 指定期間の1時間足データ（時間、OHLCV）。
    """
    all_data = []
    limit = 2000  # APIのリクエスト制限

    # start_dateとend_dateをタイムスタンプに変換
    start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
    end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())

    toTs = end_timestamp  # 初期値はend_dateのタイムスタンプ

    while toTs > start_timestamp:
        url = 'https://min-api.cryptocompare.com/data/v2/histohour'
        params = {
            'fsym': 'ETH',
            'tsym': 'USD',
            'limit': limit,
            'toTs': toTs,
            'api_key': api_key
        }

        response = requests.get(url, params=params)
        data = response.json()

        if data['Response'] != 'Success':
            raise Exception(f"API Error: {data.get('Message', 'No message')}")

        batch = data['Data']['Data']
        all_data.extend(batch)

        # 次回のリクエストは、現在取得したデータの最も古い時点からさらにさかのぼる
        oldest_timestamp = batch[0]['time']
        toTs = oldest_timestamp - 1  # 1時間ずらす

        # start_dateのタイムスタンプを超えないようにする
        toTs = max(toTs, start_timestamp)

        time.sleep(1)  # API制限対策として1秒スリープ

    df = pd.DataFrame(all_data)
    df.drop_duplicates(subset='time', inplace=True)
    df['date'] = pd.to_datetime(df['time'], unit='s')
    df = df[['date', 'open', 'high', 'low', 'close', 'volumefrom', 'volumeto']]
    df.rename(columns={'volumefrom': 'volume_ETH', 'volumeto': 'volume_USD'}, inplace=True)
    df.sort_values('date', inplace=True)

    # 指定期間のデータのみを抽出
    df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]

    return df.reset_index(drop=True)


# In[ ]:


import requests
import pandas as pd
import time
import os

def get_onchain_data_multiple_queries(query_categories):
    """
    複数のDuneクエリIDから順番にクエリを実行し、
    受け取ったオンチェーンデータをデータフレームに格納する関数。
    CSVファイルが既に存在する場合は、クエリを実行せずにCSVファイルからデータを読み込みます。

    Args:
        query_categories (dict): オンチェーンデータのカテゴリ名と対応するDuneクエリIDのリストを保持する辞書。

    Returns:
        pd.DataFrame: オンチェーンデータを含むDataFrame。
    """

    # CSVファイルのディレクトリ
    csv_dir = "/content/drive/MyDrive/Colab Notebooks/cryptoprice_prediction_tft/Ethereum_onchain_data"
    # 🔑 Dune API キーを入力
    API_KEY = "aAe1ISQlfKDsFsiBNcM5UHRW3LOjNukN"  # ここにあなたのAPIキーを入力してください
    headers = {
        "X-Dune-Api-Key": API_KEY,
        "Content-Type": "application/json"
    }

    # クエリの実行リクエスト
    def execute_dune_query(query_id):
        url = f"https://api.dune.com/api/v1/query/{query_id}/execute"
        response = requests.post(url, headers=headers)

        # デバッグ用: 中身を確認
        try:
            data = response.json()
        except Exception as e:
            print("Failed to parse JSON from Dune response")
            print("status:", response.status_code)
            print("text:", response.text[:500])
            raise

        if "execution_id" not in data:
            print("Dune API returned an error or unexpected payload:")
            print("status:", response.status_code)
            print("json:", data)
            # 必要に応じて例外を投げる
            raise RuntimeError(f"No execution_id returned for query_id={query_id}")

        execution_id = data["execution_id"]
        return execution_id

    # クエリの結果が出るまでポーリング
    def wait_for_result(execution_id):
        while True:
            url = f"https://api.dune.com/api/v1/execution/{execution_id}/status"
            res = requests.get(url, headers=headers).json()
            if res["state"] == "QUERY_STATE_COMPLETED":
                return
            elif res["state"] == "QUERY_STATE_FAILED":
                raise Exception("Query failed!")
            time.sleep(2)

    # 結果の取得
    def fetch_result(execution_id):
        url = f"https://api.dune.com/api/v1/execution/{execution_id}/results"
        res = requests.get(url, headers=headers).json()
        rows = res["result"]["rows"]
        df = pd.DataFrame(rows)
        return df

    # 全てのクエリを実行し、結果を結合
    category_dfs = {}
    for category, query_ids in query_categories.items():
        category_dataframes = []  # カテゴリ内のデータフレームを格納するリスト
        for query_id in query_ids:
            csv_file = os.path.join(csv_dir, f"{query_id}.csv")
            if os.path.exists(csv_file):
                print(f"Reading data for query ID {query_id} from CSV file: {csv_file}")
                df = pd.read_csv(csv_file)
            else:
                print(f"Fetching data for query ID {query_id} from Dune API")
                execution_id = execute_dune_query(query_id)
                wait_for_result(execution_id)
                df = fetch_result(execution_id)
                print(f"Saving data for query ID {query_id} to CSV file: {csv_file}")
                df.to_csv(csv_file, index=False)  # CSVファイルに保存

            # date列をdatetime型に変換し、時間単位に揃える
            df['date'] = pd.to_datetime(df['date']).dt.floor('H')

            if query_id in [5028220, 5265174, 5265270, 5265276, 5265285]:
                # 各cex_nameごとにtotal_flowを列として展開
                pivot_flow = df.pivot_table(index='date', columns='cex_name', values='total_flow', aggfunc='sum')

                # 各cex_nameごとにaddress_countを列として展開
                pivot_addr = df.pivot_table(index='date', columns='cex_name', values='address_count', aggfunc='sum')

                # 全体合計列の追加
                df = pd.DataFrame(index=pivot_flow.index)
                df['total_flow_sum'] = pivot_flow.sum(axis=1)
                df['address_count_sum'] = pivot_addr.sum(axis=1)

                # dateを列として戻す
                df = df.reset_index()

            category_dataframes.append(df)

        # カテゴリ内のデータフレームをconcatで結合
        if category_dataframes:
            category_df = pd.concat(category_dataframes, ignore_index=True)
            category_df.drop_duplicates(subset='date', inplace=True) # 重複データの削除
            category_df.sort_values('date', inplace=True) # 日付でソート
            category_dfs[category] = category_df.reset_index(drop=True)


    # カテゴリごとのDataFrameを1つに結合
    final_df = None
    for category, df in category_dfs.items():
        if final_df is None:
            final_df = df
        else:
            final_df = pd.merge(final_df, df, on='date', how='outer')


    # date列のタイムゾーン情報を削除 (もし存在する場合)
    if pd.api.types.is_datetime64tz_dtype(final_df['date']):
        final_df['date'] = final_df['date'].dt.tz_localize(None)

    return final_df


# In[ ]:


def aggregate_trade_history(input_path, output_path):
    """
    Binanceの取引履歴データを読み込み、必要なカラムを選択、
    タイムスタンプを変換し、金額を計算してParquetファイルとして保存する関数。

    Args:
        input_path (str): 入力CSVファイルのパス。
        output_path (str): 出力Parquetファイルのパス。
    """

    print(f'Processing {input_path} ...')

    df = pd.read_csv(input_path, header=None)
    df.columns = ['trade_id', 'price', 'qty', 'quote_qty', 'timestamp', 'is_buyer_maker', 'is_best_match']

    # タイムスタンプ変換
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    # 金額計算
    df['amount'] = df['quote_qty']

    # 不要なカラムを削除
    df = df[['price', 'qty', 'amount', 'is_buyer_maker']]

    # カラム名整形
    df.columns = ['price', 'qty', 'amount', 'is_buyer_maker']

    df['amount'] = df['price'] * df['qty']
    df['buy_qty'] = df['qty'] * (~df['is_buyer_maker'])
    df['sell_qty'] = df['qty'] * df['is_buyer_maker']

    df['buy_trade'] = ~df['is_buyer_maker']
    df['sell_trade'] = df['is_buyer_maker']

    # アグリゲート
    agg = df.resample('H').agg({
        'price': ['first', 'last', 'max', 'min', 'mean'],
        'qty': ['sum', 'count', 'mean'],
        'amount': 'sum',
        'buy_qty': ['sum', 'mean'],
        'sell_qty': ['sum', 'mean'],
        'buy_trade': 'sum',
        'sell_trade': 'sum'
    })

    agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
    agg = agg.dropna()
    agg['timestamp'] = agg.index
    agg.reset_index(drop=True, inplace=True)

    # 保存
    agg.to_parquet(output_path, index=False)
    print(f'Saved to {output_path}')

# 全Parquetを結合する関数
def load_all_trade_data(folder):
    files = sorted(glob.glob(os.path.join(folder, '*.parquet')))
    dfs = [pd.read_parquet(f) for f in files]
    df_all = pd.concat(dfs, ignore_index=True)
    df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])
    df_all.rename(columns={'timestamp': 'date'}, inplace=True)
    return df_all


def process_eth_trades_data(input_folder, output_folder):
    """
    ETHの取引履歴データを1時間単位で集計し、Parquetファイルとして保存する。
    出力フォルダにParquetファイルが存在する場合はそれらを読み込み、
    存在しない場合は入力フォルダからCSVファイルを読み込んでParquetファイルを作成する。

    Args:
        input_folder (str): 入力CSVファイルが格納されているフォルダのパス。
        output_folder (str): 出力Parquetファイルが保存されるフォルダのパス。

    Returns:
        pd.DataFrame: すべてのParquetファイルを結合したDataFrame。
    """

    # Google Driveをマウント
    drive.mount('/content/drive')

    # 出力フォルダが存在しなければ作成
    os.makedirs(output_folder, exist_ok=True)

    # 出力フォルダ内のParquetファイル一覧を取得
    parquet_files = sorted(glob.glob(os.path.join(output_folder, '*.parquet')))

    # Parquetファイルが存在する場合
    if parquet_files:
        print("Parquetファイルを読み込んで結合します。")
        df_all = load_all_trade_data(output_folder)  # load_all_hourly_data関数は既存のものを使用
    # Parquetファイルが存在しない場合
    else:
        print("CSVファイルを読み込んでParquetファイルを作成します。")
        # ファイル一覧を取得してソート
        csv_files = sorted(glob.glob(os.path.join(input_folder, '*.csv')))

        # 一括処理ループ
        for file_path in csv_files:
            file_name = os.path.basename(file_path)
            output_file = file_name.replace('.csv', '.parquet')
            output_path = os.path.join(output_folder, output_file)

            aggregate_trade_history(file_path, output_path)

        df_all = load_all_trade_data(output_folder)

    print("✅ 全ファイル処理完了")
    return df_all


# In[ ]:


# def aggregate_trade_history_hourly(df):
#     """
#     get_trade_historyで取得したデータを1時間単位で集計する関数。

#     Args:
#         df (pd.DataFrame): get_trade_historyで取得したデータフレーム。

#     Returns:
#         pd.DataFrame: 1時間単位で集計されたデータフレーム。
#     """

#     # 'timestamp'列をインデックスに設定
#     df = df.set_index('timestamp')

#     # 1時間単位で集計
#     aggregated_df = df.resample('1H').agg({
#         'price': 'mean',  # 価格の平均
#         'quantity': 'sum',  # 取引量の合計
#         'amount': 'sum',  # 取引金額の合計
#     })

#     # インデックスをリセット
#     aggregated_df = aggregated_df.reset_index()

#     return aggregated_df

# # 使用例
# trade_history_df = get_trade_history()  # get_trade_historyでデータを取得
# hourly_aggregated_df = aggregate_trade_history_hourly(trade_history_df)  # 1時間単位で集計
# print(hourly_aggregated_df)  # 集計結果を表示


# In[ ]:


def merge_data(df_price, df_onchain, df_trade, df_fear):
    df = pd.merge(df_price, df_onchain, on='date', how='inner')
    df = pd.merge(df, df_trade, on='date', how='inner')
    df = pd.merge(df, df_fear, on='date', how='inner')
    df = df.sort_values('date').reset_index(drop=True)
    return df


# In[ ]:


def create_regression_label(df, target_col='close', horizon=4):
    df['log_return_12h'] = np.log(df[target_col].shift(-horizon) / df[target_col])
    df['labels'] = np.where(df['log_return_12h'] > 0, 1, 0)
    df = df.drop('log_return_12h', axis=1)
    return df


# In[ ]:


def dynamic_cusum_filter(data, base_threshold=0.01, vol_factor=0.5):
    """
    ボラティリティに応じて閾値を動的に調整するCUSUMフィルタ
    """
    df = data.copy()
    rolling_volatility = df["price"].pct_change().rolling(window=50).std().fillna(0)
    df["threshold"] = base_threshold + vol_factor * rolling_volatility

    keep = [0]
    high, low = 0, 0

    for i in range(1, len(df)):
        change = df["price"].iloc[i] - df["price"].iloc[i - 1]
        threshold = df["threshold"].iloc[i]

        high = max(0, high + change)
        low = min(0, low + change)

        if high > threshold or low < -threshold:
            keep.append(i)
            high, low = 0, 0  # リセット

    return df.iloc[keep]
def apply_cusum_filter(price_series, quantile):
    t_events = []
    s_pos, s_neg = 0, 0

    x = pd.Series(price_series)
    diff = x.diff()

    threshold = np.quantile(np.abs(diff).dropna(), quantile)

    for i in range(1, len(price_series)):
        diff = price_series[i] - price_series[i - 1]
        s_pos = max(0, s_pos + diff)
        s_neg = min(0, s_neg + diff)
        if s_pos > threshold:
            t_events.append(price_series.index[i])
            s_pos = 0
        elif s_neg < -threshold:
            t_events.append(price_series.index[i])
            s_neg = 0
    # return pd.DatetimeIndex(t_events)
    return t_events


# In[ ]:


class StockDataset(torch.utils.data.Dataset):
    def __init__(self, time_series_data, labels, weights=None, sequence_length=20):
        self.time_series_data = time_series_data
        self.sequence_length = sequence_length
        self.labels = labels
        self.weights = weights


    def __len__(self):
        return len(self.time_series_data) - self.sequence_length + 1

    def __getitem__(self, idx):
        start_idx = idx
        end_idx = idx + self.sequence_length

        # 時系列データをスライス
        time_series_data = torch.tensor(self.time_series_data.iloc[start_idx:end_idx].values, dtype=torch.float32)
        volatility = torch.tensor(self.time_series_data['volatility'].iloc[start_idx:end_idx].values, dtype=torch.float32)
        # label = torch.tensor(self.labels[idx], dtype=torch.long)  # ラベルをテンソルに変換
        label = torch.tensor(self.labels.iloc[end_idx-1], dtype=torch.long)
        weight = torch.tensor(self.weights.iloc[end_idx-1], dtype=torch.float32)



        return time_series_data, label, weight


# In[ ]:


def mpSampleW_with_decay(t1, numCoEvents, close, molecule, decay_rate=0.01):
    """
    サンプルウェイトを計算し、時間経過とともに指数関数的に減衰させる。

    Args:
        t1 (pd.Series): イベントの終了時間を示すSeries。
        numCoEvents (pd.Series): 各時点での同時イベント数を示すSeries。
        close (pd.Series): 終値の時系列データを示すSeries。
        molecule (list): ウェイトを計算する時点のリスト。
        decay_rate (float): ウェイト減衰率。

    Returns:
        pd.Series: 計算されたサンプルウェイトを含むSeries。
    """
    # 対数リターンを計算
    ret = np.log(close).diff()

    wght = pd.Series(index=molecule)
    for tIn, tOut in t1.loc[wght.index].items():
        # 基本ウェイトの計算
        wght.loc[tIn] = (ret.loc[tIn:tOut] / numCoEvents.loc[tIn:tOut]).sum()

        # 時間経過による減衰を適用
        time_elapsed = (t1.index[-1] - tIn).days  # 現在からの経過日数
        decay_factor = np.exp(-decay_rate * time_elapsed)  # 指数関数的な減衰
        wght.loc[tIn] *= decay_factor

    # ウェイトの正規化
    wght = wght.abs()
    wght *= wght.shape[0] / wght.sum()

    return wght


# ラベルの独自性(ユニークネス)の推定
def mpNumCoEvents(closeIdx,t1,molecule):
    """
    バーごとの同時発生的な事象の数を計算する
    +molecule[0]はウェイトが計算される最初の日付
    +molecule[-1]はウェイトが計算される最後の日付
    t1[molecule].max()の前に始まる事象はすべて計算に影響する

    """
    # 期間[molecule[0],molecule[-1]]に及ぶ事象を見つける
    # クローズしていない事象はほかのウェイトに影響しなければならない
    t1=t1.fillna(closeIdx[-1])
    # 時点molecule[0]またはそのあとに終わる事象
    # 時点t1[molecule].max()またはその前に始まる事象
    # バーに及ぶ事象を数える
    iloc=closeIdx.searchsorted(np.array([t1.index[0],t1.max()]))
    count=pd.Series(0,index=closeIdx[iloc[0]:iloc[1]+1])
    for tIn,tOut in t1.items():
      count.loc[tIn:tOut]+=1
    return count.loc[molecule[0]:t1[molecule].max()]

# インディケーター行列の構築
def getIndMatrix(barIx,t1):
  # インディケーター行列を計算する
  indM=pd.DataFrame(0,index=barIx,columns=range(t1.shape[0]))
  for i,(t0,t1) in enumerate(t1.iteritems()):
    indM.loc[t0:t1,i]=1.
  return indM

# 平均独自性の計算
def getAvgUniqueness(indM):
  # インディケーター行列から平均独自性を計算する
  c=indM.sum(axis=1)# 同時発生性
  u=indM.div(c,axis=0)# 独自性
  avgU=u[u>0].mean()# 平均独自性
  return avgU

# 逐次ブートストラップからの抽出
def seqBootstrap(indM,sLength=None):
  # 逐次ブートストラップを通じてサンプルを生成する
  if sLength is None:
    sLength = indM.shape[1]
  phi=[]
  while len(phi)<sLength:
    avgU=pd.Series()
    for i in indM:
      indM_=indM[phi+[i]]
      avgU.loc[i]=getAvgUniqueness(indM_).iloc[-1]

    prob=avgU/avgU.sum()
    phi+=[np.random.choice(indM.columns,p=prob)]
  return phi


# In[ ]:


import requests
import pandas as pd

def get_fear_greed_index():
    """
    Fear & Greed Indexを取得し、特徴量化して価格データにマージする関数。

    Args:
        price_df (pd.DataFrame): 1時間足の価格情報を含むDataFrame。

    Returns:
        pd.DataFrame: Fear & Greed IndexをマージしたDataFrame。
    """
    # Fear & Greed Index API URL
    url = "https://api.alternative.me/fng/?limit=0&format=json"

    # データを取得
    response = requests.get(url)
    data = response.json()

    # DataFrame化
    df_fear = pd.DataFrame(data["data"])
    df_fear["date"] = pd.to_datetime(df_fear["timestamp"], unit="s")
    df_fear.set_index("date", inplace=True)

    # カラム名を分かりやすく
    df_fear = df_fear.rename(columns={"value": "fg_index", "value_classification": "fg_label"})

    # データ型変換
    df_fear["fg_index"] = df_fear["fg_index"].astype(int)

    # ラベルエンコーディング
    df_fear["fg_label_num"] = df_fear["fg_label"].astype("category").cat.codes

    # 1時間ごとにリサンプリングし、直前の値で埋める
    df_fear = df_fear.resample('1H').last().ffill()

    # 移動平均
    df_fear["fg_index_ma"] = df_fear["fg_index"].rolling(window=24).mean()


    # 乖離率
    df_fear["fg_index_diff"] = df_fear["fg_index"] - df_fear["fg_index_ma"]

    # 変化率
    df_fear["fg_index_change"] = df_fear["fg_index"].pct_change()

    # ラグ特徴量（1～2時点前）
    for col in ["fg_index", "fg_index_ma", "fg_index_diff", "fg_index_change"]:
        df_fear[f"{col}_lag1"] = df_fear[col].shift(1)
        df_fear[f"{col}_lag2"] = df_fear[col].shift(2)

    df_fear['date'] = df_fear.index
    df_fear.reset_index(drop=True, inplace=True)


    return df_fear


# In[ ]:


def split_and_prepare_data(df, split_date, decay_rate):

    df.index = pd.to_datetime(df['date'])
    df = df.drop('date', axis=1)
    # 訓練データとテストデータに分割する
    # 学習データを2003-01-01〜2020-12-31の期間としdf_trainに入力する
    df_train = df[: split_date]
    # 検証データを2021-01-01以降としてとしてdf_testに入力する
    df_test = df[split_date:]



    # # 絶対リターンの帰属による標本ウェイトの計算
    # closeIdx = df_train.index
    # numDays=7
    # t1 = pd.Series(df_train.index[numDays:],index=df_train.index[:-numDays])
    # molecule = pd.Series(df_train.index[:-numDays])

    # numCoEvents=mpNumCoEvents(closeIdx,t1,molecule)
    # df_train['weight']=mpSampleW_with_decay(t1,numCoEvents,df_train['close'],molecule,decay_rate)

    df_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_test.replace([np.inf, -np.inf], np.nan, inplace=True)



    train_labels = df_train[["labels_1h", "labels_3h", "labels_6h"]]
    test_labels = df_test[["labels_1h", "labels_3h", "labels_6h"]]
    # weights = df_train['weight']
    # train_returns = df_train["barrier_returns"]
    # test_returns = df_test["barrier_returns"]

    df_train = df_train.drop(["labels_1h", "labels_3h", "labels_6h"], axis=1)
    df_test = df_test.drop(["labels_1h", "labels_3h", "labels_6h"], axis=1)

    # df_train = df_train.drop(["labels", "weight", "barrier_returns"], axis=1)
    # df_test = df_test.drop(["labels", "barrier_returns"], axis=1)

    # 特徴量の標準化
    scaler = StandardScaler()
    # scaler = MinMaxScaler()

    features_to_scale = df_train.select_dtypes(include='number').columns.tolist()

    # Fit on training data and transform
    df_train[features_to_scale] = scaler.fit_transform(df_train[features_to_scale])
    # Transform testing data
    df_test[features_to_scale] = scaler.transform(df_test[features_to_scale])

    cusum_features = ["close", "dex_volume", "active_senders", "active_receivers", "address_count_sum", "contract_calls",
                                     "whale_tx_count", "sign_entropy_12","sign_entropy_24", "buy_sell_ratio"]

    df_train = features.add_cusum_features(df_train, cusum_features, quantile=0.95)
    df_test = features.add_cusum_features(df_test, cusum_features, quantile=0.95)



    return df_train, df_test, train_labels, test_labels


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
        print(f"Warning: Invalid sub_window_ratio ({sub_window_ratio}) for sequence_length ({sequence_length}). Skipping sub-window features.")
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
                        else:
                            features[f'{col}_sub_{window_name}_mean'] = np.nan
                            features[f'{col}_sub_{window_name}_std'] = np.nan
                            features[f'{col}_sub_{window_name}_min'] = np.nan
                            features[f'{col}_sub_{window_name}_max'] = np.nan
                            features[f'{col}_sub_{window_name}_last'] = np.nan
                            features[f'{col}_sub_{window_name}_diff'] = np.nan

            else:
                # numeric_seriesが空の場合、全ての統計量をNaNにする
                stats_keys = [
                    'mean', 'std', 'min', 'max', 'last', 'diff', 'skew', 'kurtosis', 'autocorr1',
                    'p10', 'p25', 'p50', 'p75', 'p90'
                ]
                for key in stats_keys:
                    features[f'{col}_{key}'] = np.nan
                if sub_window_indices:
                     sub_stats_keys = ['mean', 'std', 'min', 'max', 'last', 'diff']
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


def merge_external_data(
    df_eth_1h: pd.DataFrame,
    df_gli: pd.DataFrame | None = None,
    gli_col: str = "GLI_norm",
    df_vix: pd.DataFrame | None = None,
    vix_col: str = "VIX_close",
    df_btc_dom: pd.DataFrame | None = None,
    btc_dom_col: str = "btc_dominance",
    df_funding: pd.DataFrame | None = None,
    funding_col: str = "fundingRate",
) -> pd.DataFrame:
    """
    ETH 1h の DataFrame に、GLI / VIX / BTC dominance / Funding を
    reindex + ffill して統合する。

    前提:
      - df_eth_1h.index は DatetimeIndex（1h足、UTC or JST は揃っていること）
      - 外部DF（df_gli, df_vix, df_btc_dom, df_funding）は DatetimeIndex を持ち、
        それぞれの指標列が存在していること。

    各外部DFが None の場合は、その指標は追加されない。
    """

    df_merged = df_eth_1h.copy()
    target_index = df_merged.index

    # 安全のためインデックスを DatetimeIndex & sort
    if not isinstance(target_index, pd.DatetimeIndex):
        raise ValueError("df_eth_1h.index must be a DatetimeIndex")

    # -------- GLI --------
    if df_gli is not None:
        if not isinstance(df_gli.index, pd.DatetimeIndex):
            df_gli = df_gli.copy()
            df_gli.index = pd.to_datetime(df_gli.index)
        df_gli = df_gli.sort_index()

        if gli_col not in df_gli.columns:
            raise ValueError(f"{gli_col} not found in df_gli.columns")

        gli_series = df_gli[[gli_col]].reindex(target_index, method="ffill")
        df_merged[gli_col] = gli_series[gli_col]

    # -------- VIX --------
    if df_vix is not None:
        if not isinstance(df_vix.index, pd.DatetimeIndex):
            df_vix = df_vix.copy()
            df_vix.index = pd.to_datetime(df_vix.index)
        df_vix = df_vix.sort_index()

        if vix_col not in df_vix.columns:
            raise ValueError(f"{vix_col} not found in df_vix.columns")

        vix_series = df_vix[[vix_col]].reindex(target_index, method="ffill")
        df_merged[vix_col] = vix_series[vix_col]

    # -------- BTC dominance --------
    if df_btc_dom is not None:
        if not isinstance(df_btc_dom.index, pd.DatetimeIndex):
            df_btc_dom = df_btc_dom.copy()
            df_btc_dom.index = pd.to_datetime(df_btc_dom.index)
        df_btc_dom = df_btc_dom.sort_index()

        if btc_dom_col not in df_btc_dom.columns:
            raise ValueError(f"{btc_dom_col} not found in df_btc_dom.columns")

        dom_series = df_btc_dom[[btc_dom_col]].reindex(target_index, method="ffill")
        df_merged[btc_dom_col] = dom_series[btc_dom_col]

    # -------- Funding rate --------
    if df_funding is not None:
        if not isinstance(df_funding.index, pd.DatetimeIndex):
            df_funding = df_funding.copy()
            df_funding.index = pd.to_datetime(df_funding.index)
        df_funding = df_funding.sort_index()

        if funding_col not in df_funding.columns:
            raise ValueError(f"{funding_col} not found in df_funding.columns")

        funding_series = df_funding[[funding_col]].reindex(target_index, method="ffill")
        df_merged[funding_col] = funding_series[funding_col]

    return df_merged

