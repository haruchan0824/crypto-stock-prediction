"""Auto-extracted function-only legacy core from scripts/final.py."""
import os

import sys

import pandas as pd

import torch

from torch.utils.data import Dataset, DataLoader

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import importlib

from src import features

from src import model

from src import dataset

from src import optimization

from src import filtering

from src import tft_model  # Import tft_model

import requests

import pandas as pd

def _keyerror_excepthook(exc_type, exc_value, exc_traceback):
    if exc_type is KeyError:
        print(f"[ERROR] KeyError: {exc_value}")
        print("[ERROR] 利用可能カラム一覧:")
        for name, val in sorted(globals().items()):
            if isinstance(val, pd.DataFrame):
                print(f"  - {name}: {list(val.columns)}")
        return
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

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

import datetime as dt

import requests

import pandas as pd

def fetch_vix_yfinance(start_date, end_date=None, interval="1d"):
    """
    Yahoo Finance から VIX (^VIX) を取得する簡易関数。
    interval="1d" を基本にするのがおすすめ。

    戻り値:
        DataFrame(index=DatetimeIndex, columns=["VIX_close"])
    """
    import yfinance as yf

    if end_date is None:
        end_date = dt.datetime.utcnow().date()

    data = yf.download("^VIX", start=start_date, end=end_date, interval=interval)
    if data.empty:
        raise ValueError("No VIX data returned. Check dates or interval.")

    # Close のみ使う
    df_vix = data[["Close"]].rename(columns={"Close": "VIX_close"})
    df_vix.index = pd.to_datetime(df_vix.index)
    return df_vix

def fetch_binance_funding_rate(
    symbol="ETHUSDT",
    start_time=None,
    end_time=None,
    limit=1000,
):
    """
    Binance Futures API から funding rate 履歴を取得する。

    - デフォルトでは直近 limit 本を取得
    - start_time, end_time を指定する場合は UNIX ms（int）で渡す
      （使いやすくするなら、外側で datetime → int に変換）

    戻り値:
        DataFrame with columns: ["fundingTime", "fundingRate"]
    """
    base_url = "https://fapi.binance.com"
    endpoint = "/fapi/v1/fundingRate"

    params = {
        "symbol": symbol,
        "limit": limit,
    }
    if start_time is not None:
        params["startTime"] = int(start_time)
    if end_time is not None:
        params["endTime"] = int(end_time)

    resp = requests.get(base_url + endpoint, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if not data:
        return pd.DataFrame(columns=["fundingTime", "fundingRate"])

    df = pd.DataFrame(data)
    # time は ms
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(float)
    df = df.sort_values("fundingTime").reset_index(drop=True)
    return df[["fundingTime", "fundingRate"]]

def fetch_binance_funding_rate_range(symbol="ETHUSDT", days=180):
    """
    過去 days 日分の Funding Rate をざっくり取得して連結するラッパー。

    注意:
      - Binance API の仕様上、一度に取得できる本数(limit)が 1000 などで制限されるため、
        長期間を取る場合はループで過去にさかのぼる必要がある。
      - ここでは簡易的に「直近 days 日 ≒ days*3 本 (8hごと)」をまとめて取る形。

    戻り値:
        DataFrame(index=fundingTime, columns=["fundingRate"])
    """
    now = dt.datetime.utcnow()
    start = now - dt.timedelta(days=days)
    # Binance の funding は 8h ごとなので、最大本数を少し余裕を持たせる
    approx_points = days * 4  # 1日3本だけど余裕みて4
    df = fetch_binance_funding_rate(
        symbol=symbol,
        start_time=int(start.timestamp() * 1000),
        end_time=int(now.timestamp() * 1000),
        limit=min(1000, approx_points),
    )
    df = df.set_index("fundingTime")
    return df

def fetch_fred_series(series_id, api_key, start_date="2000-01-01", end_date=None):
    """
    FRED API から単一 series を取得するヘルパー関数。

    series_id の例:
      - "M2SL"   : 米国M2マネーサプライ
      - "WALCL"  : FRB総資産 (Fed balance sheet)
    """
    if end_date is None:
        end_date = dt.datetime.utcnow().date().strftime("%Y-%m-%d")

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    obs = data.get("observations", [])
    if not obs:
        return pd.DataFrame(columns=[series_id])

    df = pd.DataFrame(obs)
    df["date"] = pd.to_datetime(df["date"])
    # 一部 'nan' 文字列が混じるので float に変換
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.set_index("date")[["value"]].rename(columns={"value": series_id})
    return df

def build_global_liquidity_index(
    fred_api_key,
    start_date="2010-01-01",
    end_date=None,
    m2_series="M2SL",
    fed_bs_series="WALCL",
):
    df_m2 = fetch_fred_series(m2_series, fred_api_key, start_date, end_date)
    df_bs = fetch_fred_series(fed_bs_series, fred_api_key, start_date, end_date)

    # outer join してから ffill で埋める方が行が増える
    df = df_m2.join(df_bs, how="outer").sort_index()
    df[m2_series] = df[m2_series].ffill()
    df[fed_bs_series] = df[fed_bs_series].ffill()

    df["GLI_raw"] = df[m2_series] + df[fed_bs_series]

    # 全期間Z-score
    x = df["GLI_raw"]
    df["GLI_norm"] = (x - x.mean()) / (x.std() + 1e-8)

    return df

def fetch_btc_and_total_mcap_coingecko(days=365, vs_currency="usd"):
    """
    Coingecko の market_chart API を利用して、
    - BTC の時価総額
    - 全体の時価総額

    を取得し、そこから dominance を計算するスケルトン。

    ※ Coingecko API仕様は変更される可能性があるので、
      実際に使う前に最新ドキュメントを確認してください。
    """
    # BTC の market cap time series
    url_btc = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params_btc = {
        "vs_currency": vs_currency,
        "days": days,
    }
    r_btc = requests.get(url_btc, params=params_btc, timeout=10)
    r_btc.raise_for_status()
    data_btc = r_btc.json()
    # data_btc["market_caps"] : [[timestamp_ms, mcap], ...]
    btc_caps = pd.DataFrame(data_btc["market_caps"], columns=["ts", "btc_mcap"])
    btc_caps["ts"] = pd.to_datetime(btc_caps["ts"], unit="ms", utc=True)
    btc_caps = btc_caps.set_index("ts")

    # グローバル時価総額は /global/market_cap_chart のようなエンドポイントがある場合もあるが、
    # 仕様が変わりやすいので、一旦 BTC + 他主要コインを近似として使う、などが現実的。
    # ここではスケルトンとして、BTC単独から dominance を擬似的に作る例を示す。
    # （ちゃんとやるなら、ETH・他の topN も market_cap を取り total_mcap を計算する）
    total_mcap = btc_caps["btc_mcap"]  # ★仮に「全体 ≒ BTC」の場合は dominance ~1.0 になってしまう

    df = pd.DataFrame(index=btc_caps.index)
    df["btc_mcap"] = btc_caps["btc_mcap"]
    df["total_mcap"] = total_mcap
    df["btc_dominance"] = df["btc_mcap"] / (df["total_mcap"] + 1e-8)

    return df

def fetch_dune_query_results(query_id, api_key, params=None):
    """
    Dune API v2 の query result を取得する共通ヘルパー。
    - あらかじめ Dune 側で SQL を保存し query_id を控えておく。
    - gas fee, tx count, stablecoin supply などは SQL 側で日次/1h集計しておくと楽。

    戻り値:
        DataFrame (Duneが返す rows をそのまま DataFrame にしたもの)
    """
    headers = {
        "X-Dune-API-Key": api_key,
        "Content-Type": "application/json",
    }

    # Execution をトリガー（必要な場合）
    exec_url = f"https://api.dune.com/api/v1/query/{query_id}/execute"
    payload = {"query_parameters": params or {}}
    r_exec = requests.post(exec_url, json=payload, headers=headers, timeout=10)
    r_exec.raise_for_status()
    exec_id = r_exec.json()["execution_id"]

    # 結果が出るまでポーリング（簡易版）
    import time
    while True:
        res_url = f"https://api.dune.com/api/v1/execution/{exec_id}/results"
        r_res = requests.get(res_url, headers=headers, timeout=10)
        if r_res.status_code == 200:
            data = r_res.json()
            state = data.get("state")
            if state == "QUERY_STATE_COMPLETED":
                rows = data["result"]["rows"]
                df = pd.DataFrame(rows)
                return df
            elif state in ("QUERY_STATE_EXECUTING", "QUERY_STATE_PENDING"):
                time.sleep(2)
                continue
            else:
                raise RuntimeError(f"Dune query failed with state={state}")
        else:
            time.sleep(2)

def clean_gli(df_gli: pd.DataFrame) -> pd.DataFrame:
    df = df_gli.copy()
    df = df.sort_index()

    # 数値化（安全のため）
    for c in ["M2SL", "WALCL", "GLI_raw", "GLI_norm"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # M2, WALCL は前方埋めでギャップを埋める
    df[["M2SL", "WALCL", "GLI_raw"]] = df[["M2SL", "WALCL", "GLI_raw"]].ffill()

    # GLI_norm についても、計算済みのものが飛び飛びで入っている想定なので ffill
    df["GLI_norm"] = df["GLI_norm"].ffill()

    # 最初の方で GLI_norm が全NaNなら落とす
    df = df[df["GLI_norm"].notna()]

    return df

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

    # すべてのDatetimeIndexをUTCに変換して統一性を確保
    if target_index.tz is None:
        target_index = target_index.tz_localize('UTC')
    else:
        target_index = target_index.tz_convert('UTC')

    df_merged.index = target_index

    # -------- GLI --------
    if df_gli is not None:
        if not isinstance(df_gli.index, pd.DatetimeIndex):
            df_gli = df_gli.copy()
            df_gli.index = pd.to_datetime(df_gli.index)

        # GLIのインデックスをUTCに変換
        if df_gli.index.tz is None:
            df_gli.index = df_gli.index.tz_localize('UTC')
        else:
            df_gli.index = df_gli.index.tz_convert('UTC')

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

        # VIXのインデックスをUTCに変換
        if df_vix.index.tz is None:
            df_vix.index = df_vix.index.tz_localize('UTC')
        else:
            df_vix.index = df_vix.index.tz_convert('UTC')

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

        # BTC dominanceのインデックスをUTCに変換
        if df_btc_dom.index.tz is None:
            df_btc_dom.index = df_btc_dom.index.tz_localize('UTC')
        else:
            df_btc_dom.index = df_btc_dom.index.tz_convert('UTC')

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

        # Funding rateのインデックスをUTCに変換
        if df_funding.index.tz is None:
            df_funding.index = df_funding.index.tz_localize('UTC')
        else:
            df_funding.index = df_funding.index.tz_convert('UTC')

        df_funding = df_funding.sort_index()

        if funding_col not in df_funding.columns:
            raise ValueError(f"{funding_col} not found in df_funding.columns")

        funding_series = df_funding[[funding_col]].reindex(target_index, method="ffill")
        df_merged[funding_col] = funding_series[funding_col]

    return df_merged

import matplotlib.pyplot as plt

import seaborn as sns

import requests

import pandas as pd

import numpy as np

def winsorize_series(s: pd.Series, lower=0.01, upper=0.99):
    """
    上下パーセンタイルで値をクリップする（winsorize）。
    デフォルトは上下1%ずつ。

    s: pandas Series（連続特徴量）
    """
    low = s.quantile(lower)
    high = s.quantile(upper)
    return s.clip(lower=low, upper=high)

def winsorize_features(df, cont_cols, lower=0.01, upper=0.99):
    df = df.copy()
    for col in cont_cols:
        df[col] = winsorize_series(df[col], lower, upper)
    return df

import numpy as np, pandas as pd

from scipy.stats import ks_2samp

def _psi(a, b, bins=10):
    # Population Stability Index（分布ドリフト）
    a_finite = a[~np.isnan(a)]
    b_finite = b[~np.isnan(b)]

    # Check if there are any finite values left to compute quantiles
    if len(a_finite) == 0 or len(b_finite) == 0:
        return np.nan # Return NaN if no data to compare

    qa = np.quantile(a_finite, np.linspace(0,1,bins+1))
    qa[0]-=1e9; qa[-1]+=1e9
    ca,_ = np.histogram(a_finite, qa); cb,_ = np.histogram(b_finite, qa)

    # Ensure histograms are not all zeros to avoid division by zero
    if ca.sum() == 0 or cb.sum() == 0:
        return np.nan

    pa = (ca/ca.sum()+1e-6); pb = (cb/cb.sum()+1e-6)
    return float(np.sum((pa-pb)*np.log(pa/pb)))

def audit_cont_features(
    df: pd.DataFrame,
    cont_cols: list,
    train_mask: pd.Series,                   # True=train, False=val/test
    expect_standardized_suffix="_zn",        # *_zn は mean≈0,std≈1 を期待
    std_warn=5.0, std_crit=10.0,            # スケール異常の閾値（train基準）
    outlier_sigma=5.0,                       # |z|>5 の外れ率を測る
    psi_warn=0.1, psi_crit=0.25,            # ドリフトの閾値
    corr_high=0.95                           # 相関の高すぎるペア
):
    rep = []
    tr = df.loc[train_mask, cont_cols]
    va = df.loc[~train_mask, cont_cols]

    # train基準の統計
    # Handle cases where `tr` might be empty or all NaNs for a column
    tr_desc = tr.describe().T.assign(
        std_tr = tr.std(),
        skew_tr = tr.skew(),
        kurt_tr = tr.kurtosis()
    )

    # 標準化期待の列は z を計算（train基準）
    z = {}
    # Ensure there are non-zero std deviations to avoid division by zero
    mu = tr.mean()
    sd = tr.std().replace(0, 1.0)

    for c in cont_cols:
        # Skip if feature column is entirely NaN or empty in training data
        if c not in tr.columns or tr[c].dropna().empty:
            print(f"[WARN] Feature '{c}' in training data is empty or all NaN. Skipping audit for this feature.")
            rep.append(dict(
                feature=c, na_rate=1.0, finite_rate=0.0, mean_tr=np.nan, std_tr=np.nan,
                skew_tr=np.nan, kurt_tr=np.nan, outlier_rate="N/A", psi=np.nan, ks_p=np.nan,
                zn_mean_tr="", zn_std_tr="", flags="NaN,non-finite,EMPTY_TRAIN_DATA"))
            continue

        # Skip if feature column is entirely NaN or empty in validation data (for PSI/KS)
        if c not in va.columns or va[c].dropna().empty:
            print(f"[WARN] Feature '{c}' in validation data is empty or all NaN. PSI/KS will be NaN.")
            x_vt = np.array([]) # Represent as empty for _psi function
        else:
            x_vt = va[c].to_numpy()

        x_tr = tr[c].to_numpy()

        # Calculate z-scores using the (potentially adjusted) std dev
        z[c] = ((df[c] - mu[c]) / sd[c]).to_numpy()

        # 欠損・有限判定
        na_rate = float(df[c].isna().mean())
        finite_rate = float(np.isfinite(df[c]).mean())

        # スケールと外れ値（train基準）
        std_val = float(sd[c])
        z_abs = np.abs(z[c])
        outlier_rate = float(np.mean(z_abs > outlier_sigma))

        # 標準化チェック（*_zn 想定）
        std_expected = (expect_standardized_suffix and str(c).endswith(expect_standardized_suffix))
        mean_z_tr = float(((tr[c]-mu[c])/sd[c]).mean()) if std_expected and sd[c] != 0 else np.nan
        std_z_tr  = float(((tr[c]-mu[c])/sd[c]).std())  if std_expected and sd[c] != 0 else np.nan

        # ドリフト：PSI & KS
        psi = _psi(x_tr, x_vt) if len(va)>0 else np.nan # x_tr and x_vt are already arrays. _psi handles their finite parts.
        try:
            ks_p = float(ks_2samp(x_tr[~np.isnan(x_tr)], x_vt[~np.isnan(x_vt)]).pvalue) if len(va)>0 and not x_tr[~np.isnan(x_tr)].empty and not x_vt[~np.isnan(x_vt)].empty else np.nan
        except Exception:
            ks_p = np.nan

        rep.append(dict(
            feature=c,
            na_rate=na_rate,
            finite_rate=finite_rate,
            mean_tr=float(mu[c]),
            std_tr=std_val,
            skew_tr=float(tr[c].skew()),
            kurt_tr=float(tr[c].kurtosis()),
            outlier_rate=f"{outlier_rate:.4f}",
            psi=f"{psi:.3f}" if not np.isnan(psi) else np.nan,
            ks_p=f"{ks_p:.3g}" if not np.isnan(ks_p) else np.nan,
            zn_mean_tr=f"{mean_z_tr:.2f}" if std_expected and not np.isnan(mean_z_tr) else "",
            zn_std_tr=f"{std_z_tr:.2f}" if std_expected and not np.isnan(std_z_tr) else "",
            flags="") )

    rep = pd.DataFrame(rep)
    # フラグ付与（警告・重大）
    def _flag(row):
        flags=[]
        if row['na_rate']>0: flags.append("NaN")
        if row['finite_rate']<1.0: flags.append("non-finite")
        if row['std_tr']>=std_crit: flags.append("SCALE_CRIT")
        elif row['std_tr']>=std_warn: flags.append("SCALE_WARN")
        if str(row['outlier_rate']) != "N/A" and float(str(row['outlier_rate']))>0.01: flags.append("outliers>1%")
        if row['psi'] not in (np.nan, None) and float(row['psi'])>=psi_crit: flags.append("PSI_CRIT")
        elif row['psi'] not in (np.nan, None) and float(row['psi'])>=psi_warn: flags.append("PSI_WARN")
        # *_zn の期待ズレ
        if str(row.get('zn_std_tr',"")).strip():
            try:
                if abs(float(row['zn_mean_tr']))>0.1 or abs(float(row['zn_std_tr'])-1.0)>0.1:
                    flags.append("ZN_MISMATCH")
            except ValueError: # Handle 'N/A' or non-numeric strings
                pass
        return ",".join(flags)
    rep['flags'] = rep.apply(_flag, axis=1)

    # 高相関ペアの抽出（train基準）
    # Only compute correlation if `tr` is not empty
    if not tr.empty:
        corr = tr.corr().abs()
        pairs=[]
        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i+1,len(cols)):
                r = corr.iloc[i,j]
                if r>=corr_high:
                    pairs.append((cols[i], cols[j], float(r)))
        pairs_df = pd.DataFrame(pairs, columns=["feat_a","feat_b","|rho|"])
    else:
        pairs_df = pd.DataFrame(columns=["feat_a","feat_b","|rho|"])

    # 推奨アクションの要約
    print("=== Continuous Feature Audit ===")
    print(f"[WARN] NaN/non-finite あり: {rep[rep['flags'].str.contains('NaN|non-finite', na=False)].shape[0]} 列")
    print(f"[WARN] スケール警告(SCALE_WARN/CRIT): {rep[rep['flags'].str.contains('SCALE_', na=False)].shape[0]} 列")
    print(f"[WARN] 外れ値率>1%: {rep[rep['flags'].str.contains('outliers', na=False)].shape[0]} 列")
    print(f"[WARN] ドリフト(PSI_WARN/CRIT): {rep[rep['flags'].str.contains('PSI_', na=False)].shape[0]} 列")
    zn_mis = rep[rep['flags'].str.contains('ZN_MISMATCH', na=False)]
    if len(zn_mis):
        print(f"[WARN] *_zn なのに mean≈0,std≈1 を満たさない列: {zn_mis.feature.tolist()}")
    if len(pairs_df):
        print(f"[INFO] |ρ|≥{corr_high} の高相関ペア数: {len(pairs_df)}（代表1本に間引きを推奨）")

    # 重要列の抽出（要対応リスト）
    to_fix = rep[rep['flags']!=""].sort_values("flags")
    return rep.sort_values("flags"), pairs_df.sort_values("|rho|", ascending=False)

from sklearn.metrics import average_precision_score, roc_auc_score

from sklearn.feature_selection import mutual_info_classif

from scipy.stats import pointbiserialr

import numpy as np

import pandas as pd

def audit_flag_features_unsupervised(
    df: pd.DataFrame,
    flag_cols: list[str],
    train_mask: pd.Series | None = None,
    freq_for_stability: str = "M",
    verbose: bool = True
) -> pd.DataFrame:
    """
    ラベルを使わずに、フラグ（0/1）やカテゴリ特徴量の生成品質を監査する。
      - 発火率（全体/Train/ValTest）
      - 欠損率
      - 定数（全て0や全て1）チェック
      - チャタリング（遷移率）
      - 連続ON長（run-length統計）
      - 時系列安定性（発火率の期間標準偏差）

    Parameters
    ----------
    df : pd.DataFrame
        特徴量データ
    flag_cols : list[str]
        評価対象のフラグ・カテゴリ列名
    train_mask : pd.Series | None
        学習期間ブールマスク（Noneなら先頭70%をtrain）
    freq_for_stability : str
        発火率を集計する周期（例："M"=月次）
    """
    def _transition_rate(flag: pd.Series) -> float:
        f = flag.fillna(0).astype(int).values
        return float(np.mean(f[1:] != f[:-1])) if f.size > 1 else np.nan

    def _run_lengths(flag: pd.Series) -> dict:
        v = flag.fillna(0).astype(int).values
        runs, cur = [], 0
        for x in v:
            if x == 1:
                cur += 1
            elif cur > 0:
                runs.append(cur); cur = 0
        if cur > 0:
            runs.append(cur)
        if not runs:
            return {"mean": 0.0, "median": 0.0, "p90": 0.0}
        runs = np.asarray(runs)
        return {
            "mean": float(runs.mean()),
            "median": float(np.median(runs)),
            "p90": float(np.quantile(runs, 0.9))
        }

    def _period_stability(flag: pd.Series, idx: pd.DatetimeIndex, freq: str = "M") -> float:
        if not isinstance(idx, pd.DatetimeIndex):
            return np.nan
        g = flag.groupby(idx.to_period(freq)).mean()
        return float(g.std()) if g.size > 1 else 0.0

    # --- train/val 分割 ---
    if train_mask is None:
        n = len(df)
        train_mask = pd.Series(False, index=df.index)
        train_mask.iloc[: int(n * 0.7)] = True

    idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index, errors="coerce")
    rows = []

    for col in flag_cols:
        if col not in df.columns:
            rows.append({"feature": col, "exists": False})
            continue

        s = df[col].copy()
        uniq = pd.unique(s.dropna())
        const = len(uniq) <= 1
        rate_all = float((s == 1).mean())
        rate_tr = float((s[train_mask] == 1).mean())
        rate_vt = float((s[~train_mask] == 1).mean())
        na_rate = float(s.isna().mean())
        trans = _transition_rate(s)
        rl = _run_lengths(s)
        stab = _period_stability(s, idx, freq_for_stability)

        rows.append({
            "feature": col,
            "exists": True,
            "is_constant": bool(const),
            "na_rate": round(na_rate, 4),
            "fire_rate_all": round(rate_all, 4),
            "fire_rate_train": round(rate_tr, 4),
            "fire_rate_valtest": round(rate_vt, 4),
            "transition_rate": round(trans, 4) if trans == trans else np.nan,
            "run_mean": round(rl["mean"], 2),
            "run_median": round(rl["median"], 2),
            "run_p90": round(rl["p90"], 2),
            "stability_std_" + freq_for_stability: round(stab, 4),
        })

    out = pd.DataFrame(rows).sort_values(
        by=["exists", "is_constant", "fire_rate_all"], ascending=[False, True, False]
    ).reset_index(drop=True)

    if verbose:
        print("\n=== Flag/Categorical Feature Audit (Unsupervised) ===")
        with pd.option_context("display.max_rows", 200, "display.max_columns", 200, "display.width", 160):
            print(out.head(50))
        print("\n[Guide]")
        print("- fire_rate_all: 状態フラグ 0.2–0.6 / イベントフラグ 0.05–0.3 が目安")
        print("- transition_rate が高いほどチャタリング多 → enter/exitを広げる")
        print("- run_p90 が極小なら短期ノイズ化している可能性あり")
        print("- stability_std が大きすぎる場合、期間ごとの発火率変動が大きい（閾値の再fitを検討）")

    return out

import numpy as np

def classify_all_sequences_stock_3cat(
    X_3d_numpy: np.ndarray,
    feature_names: list,
    *,
    ma_short_candidates=None,
    ma_long_candidates=None,
    price_candidates=None,
    trend_strength_th: float = 0.0008,
    price_above_long_th: float = 0.0008,
    price_below_long_th: float = 0.0008,
    debug: bool = True,
    eps: float = 1e-12,
    min_abs_long_ma: float = 1e-6,   # ★ 追加: long MA が0近傍なら判定不能
) -> np.ndarray:
    """
    3カテゴリ: ['Uptrend','Downtrend','Range']
    注意:
      - X_3dのclose/MAが「標準化済み」の場合、この比率ベース判定は壊れやすい。
        その場合は「正規化前の生値」で戦略判定するのが推奨。
    """
    n_sequences, seq_len, n_features = X_3d_numpy.shape
    strategy_labels = np.full(n_sequences, "Range", dtype=object)

    if ma_short_candidates is None:
        ma_short_candidates = ["ma_20", "MA_20", "SMA_20", "ema_20", "EMA_20", "MA_t_6"]
    if ma_long_candidates is None:
        ma_long_candidates = ["ma_120", "MA_120", "SMA_120", "ema_120", "EMA_120", "MA_t_72"]
    if price_candidates is None:
        price_candidates = ["close", "Close", "adj_close", "Adj Close", "PRICE"]

    def find_first(cands):
        for nm in cands:
            if nm in feature_names:
                return feature_names.index(nm), nm
        return None, None

    ma_s_idx, ma_s_name = find_first(ma_short_candidates)
    ma_l_idx, ma_l_name = find_first(ma_long_candidates)
    px_idx, px_name = find_first(price_candidates)

    if debug:
        print("[stock strategy 3cat] found:",
              {"ma_short": ma_s_name, "ma_long": ma_l_name, "price": px_name})

    if px_idx is None or ma_s_idx is None or ma_l_idx is None:
        if debug:
            print("[stock strategy 3cat] missing key features -> all Range")
        return strategy_labels

    # last step をまとめて取り出し（ベクトル化）
    p  = X_3d_numpy[:, -1, px_idx].astype(float)
    ms = X_3d_numpy[:, -1, ma_s_idx].astype(float)
    ml = X_3d_numpy[:, -1, ma_l_idx].astype(float)

    finite = np.isfinite(p) & np.isfinite(ms) & np.isfinite(ml)

    # ★ 正規化済みっぽい検知（目安）
    # close/MA が平均0付近で標準偏差~1に近いなら z-score の可能性が高い
    if debug:
        p_mu, p_sd = np.nanmean(p[finite]), np.nanstd(p[finite])
        ml_mu, ml_sd = np.nanmean(ml[finite]), np.nanstd(ml[finite])
        if (abs(p_mu) < 1.0 and 0.3 < p_sd < 3.0) and (abs(ml_mu) < 1.0 and 0.3 < ml_sd < 3.0):
            print("[WARN] close/MA look normalized (mean~0,std~1). "
                  "Ratio-based strategy thresholds may be invalid. "
                  "Prefer computing strategy on RAW (unnormalized) prices/MAs.")

    # 判定不能条件
    bad = (~finite) | (np.abs(p) < eps) | (np.abs(ml) < min_abs_long_ma)

    # 指標
    strength = (ms - ml) / (np.abs(p) + eps)
    rel_to_long = (p - ml) / (np.abs(ml) + eps)

    # 例: AND条件で厳格化（Range増）
    up_core = (strength > trend_strength_th) & (rel_to_long >= price_above_long_th)
    dn_core = (strength < -trend_strength_th) & (rel_to_long <= -price_below_long_th)


    # ラベル決定（両方true/両方falseはRange）
    labels = np.full(n_sequences, "Range", dtype=object)
    labels[(~bad) & up_core & (~dn_core)] = "Uptrend"
    labels[(~bad) & dn_core & (~up_core)] = "Downtrend"
    strategy_labels[:] = labels

    if debug:
        unique, counts = np.unique(strategy_labels, return_counts=True)
        print("[stock strategy 3cat] counts:", dict(zip(unique, counts)))

    return strategy_labels

def assign_stock_3strategy_from_sequences(
    X_3d: np.ndarray,
    feature_names: list,
    *,
    ma_short_candidates=None,
    ma_long_candidates=None,
    price_candidates=None,
    trend_strength_th: float = 0.003,
    price_above_long_th: float = 0.003,
    price_below_long_th: float = 0.003,
    debug: bool = True,
):
    """3戦略名 + strategy_type_id (0:Down,1:Range,2:Up) を返す"""
    strategy_name = classify_all_sequences_stock_3cat(
        X_3d_numpy=X_3d,
        feature_names=feature_names,
        ma_short_candidates=ma_short_candidates,
        ma_long_candidates=ma_long_candidates,
        price_candidates=price_candidates,
        trend_strength_th=trend_strength_th,
        price_above_long_th=price_above_long_th,
        price_below_long_th=price_below_long_th,
        debug=debug,
    )

    strategy_type_id = np.full(strategy_name.shape[0], -1, dtype=int)
    for k, v in STOCK_STRATEGY_NAME_TO_ID.items():
        strategy_type_id[strategy_name == k] = v

    # 念のため unknown が残るなら Range に寄せる（安全側）
    if np.any(strategy_type_id < 0):
        if debug:
            print("[WARN] unknown strategy found -> force Range")
        strategy_type_id[strategy_type_id < 0] = STOCK_STRATEGY_NAME_TO_ID["Range"]

    return strategy_name, strategy_type_id

def merge_labels_dict_to_array(labels_dict, strategy_labels):
    """
    labels_dict: create_strategy_specific_binary_labels_simple が返した dict
                 {strategy_name: label_array_for_that_strategy}
    strategy_labels: shape (N_seq,), 各シーケンスの戦略名
    戻り値: labels_per_seq (N_seq,)  それぞれのシーケンスに対応するラベル（-1/0/1）
    """
    n_seq = len(strategy_labels)
    labels_per_seq = np.full(n_seq, -1, dtype=float)

    unique_strats = np.unique(strategy_labels)
    for s in unique_strats:
        if s not in labels_dict:
            continue
        idx = np.where(strategy_labels == s)[0]
        lab_s = labels_dict[s]
        if len(idx) != len(lab_s):
            raise ValueError(f"Strategy {s}: index数とlabel数が一致しません")
        labels_per_seq[idx] = lab_s

    return labels_per_seq

def build_diagnostics_df(
    df_processed,
    X_3d_numpy,
    feature_names_3d,
    strategy_labels,
    labels_per_seq,
    horizon,
    cont_features_all,
    cat_features_all,
):
    n_seq, seq_length, num_features = X_3d_numpy.shape

    # ★ df_processed ベースで future return 計算
    future_scores_all = filtering.calculate_future_score_from_processed(
        df_processed=df_processed,
        sequence_length=seq_length,
        horizon=horizon,
        col_close="close",
    )

    # end_index は「各シーケンスの終端時刻」
    end_pos = np.arange(seq_length - 1, seq_length - 1 + n_seq)
    end_index = df_processed.index[end_pos]

    df_diag = pd.DataFrame({
        "seq_id": np.arange(n_seq),
        "log_return_24h": future_scores_all,
        "label": labels_per_seq,
        "strategy_type": strategy_labels,
    })
    df_diag["valid_label"] = df_diag["label"] != -1

    # Set the DatetimeIndex here for correct time-based slicing later
    df_diag.index = end_index
    df_diag.index.name = 'date'

    # ATR24 の計算と df_diag への格納
    # df_processed から high, low, close を抽出し、talib.ATR を使用
    # ATR は通常、一定期間の平均真の範囲を計算するため、 rolling window を考慮
    # talib.ATR は NaN を返すため、df_diag のインデックスに沿って結合する
    import talib
    if 'high' in df_processed.columns and 'low' in df_processed.columns and 'close' in df_processed.columns:
        # ATR24 を計算。期間は24時間 (1日) を想定
        df_processed_sorted = df_processed.sort_index()
        atr_values = talib.ATR(
            df_processed_sorted['high'].astype(float),
            df_processed_sorted['low'].astype(float),
            df_processed_sorted['close'].astype(float),
            timeperiod=24 # 24時間 ATR
        )
        # df_diag のインデックスに合わせて ATR 値をマージ
        df_diag['ATR24'] = atr_values.reindex(df_diag.index)
        # ATR 計算で生じるNaNを処理
        df_diag['ATR24'] = df_diag['ATR24'].fillna(0) # または df_diag['ATR24'].ffill() など
    else:
        print("Warning: 'high', 'low', or 'close' columns not found in df_processed for ATR calculation.")
        df_diag['ATR24'] = np.nan # カラムがない場合はNaNで埋める

    df_diag['volume'] = df_processed['volume_ETH'].reindex(df_diag.index)
    df_diag['volume'] = df_diag['volume'].fillna(0)



    # ここは以前と同じロジックでOK（ATR24/volume は診断用なので original_df でも df_processed でも可）
    # もし df_processed に high/low/volume が入っていれば df_processed から計算してもよい

    # --- Start of existing feature extraction logic (remains mostly same) ---
    # 連続・カテゴリ特徴量を最終タイムステップから展開（以前出したコードと同じ）
    last_step_features = X_3d_numpy[:, -1, :]
    feature_to_index = {name: idx for idx, name in enumerate(feature_names_3d)}

    for ft in cont_features_all:
        if ft in feature_to_index:
            df_diag[ft] = last_step_features[:, feature_to_index[ft]]
        else:
            df_diag[ft] = np.nan

    for ft in cat_features_all:
        if ft in feature_to_index:
            df_diag[ft] = last_step_features[:, feature_to_index[ft]]
        else:
            df_diag[ft] = np.nan
    # --- End of existing feature extraction logic ---

    df_diag['date'] = df_diag.index
    df_diag = df_diag.reset_index(drop=True)

    return df_diag

def assign_regime_v3_compressed(df):
    df = df.copy()

    # ===========================
    # 0. strategy_type_id の作成
    # ===========================
    if "strategy_type" not in df.columns:
        raise ValueError("df に 'strategy_type' 列が存在しません。")

    # 株式版と同一の整数ID体系で付与（上で定義した STOCK_STRATEGY_NAME_TO_ID を使用）
    # unknown/想定外は -1 にする
    df["strategy_type_id"] = df["strategy_type"].map(STOCK_STRATEGY_NAME_TO_ID).fillna(-1).astype("int64")

    # 互換のため strategy2id も返す（株式版と同一のID体系）
    strategy2id = dict(STOCK_STRATEGY_NAME_TO_ID)

    # ===========================
    # ① Volatility regime (V)
    # ===========================
    median_atr = df["ATR24"].rolling(24 * 30, min_periods=24 * 7).median()

    df["V_regime"] = np.where(
        df["ATR24"] <= median_atr, "V_low", "V_high"
    )

    # ===========================
    # ② Trend regime (T)
    # ===========================
    df["T_regime"] = np.where(
        df["trend_strength_6_24"] > 0,
        "T_up",
        "T_down"
    )

    # ===========================
    # ③ Activity regime (A)
    # ※ 圧縮版では composite には使用しない
    # ===========================
    df["A_regime"] = np.where(
        df["tx_count_zn"] > 0,
        "A_on",
        "A_off"
    )

    # ===========================
    # ④ Macro regime (M)
    # ===========================
    macro_score = (
        df["VIX_zn_sm"]
        + df["btc_dom_zn_sm"]
        + df["funding_zn_sm"]
    )

    df["M_regime"] = np.where(
        macro_score < 0,
        "M_on",    # risk-on
        "M_off"    # risk-off
    )

    # ===========================
    # ⑤ Final composite regime（圧縮版）
    # ===========================
    df["regime_v3_compressed"] = (
        df["V_regime"] + "_" +
        df["T_regime"] + "_" +
        df["M_regime"]
    )

    return df, strategy2id

import datetime as dt

import requests

import pandas as pd

def generate_time_based_wfa_splits(
    df,
    date_col="date",
    train_span_days=270,
    test_span_days=30,
    min_train_rows=110,
    min_test_rows=40,
):
    df = df.sort_values(date_col).copy()
    dates = pd.to_datetime(df[date_col].values)

    # 明示的にすべての日付オブジェクトがUTCタイムゾーンアウェアであることを確認
    if dates.tz is None:
        dates = dates.tz_localize('UTC')
    else:
        dates = dates.tz_convert('UTC') # 既にtz-awareだがUTCでない可能性がある場合にUTCに変換

    min_date = dates.min()
    max_date = dates.max()

    splits = []

    # 最初の test_start は train_span_days 経過後
    current_test_start = min_date + pd.Timedelta(days=train_span_days)

    while True:
        # これらは current_test_start からタイムゾーンアウェアネスを継承するはず
        train_start = current_test_start - pd.Timedelta(days=train_span_days)
        train_end   = current_test_start - pd.Timedelta(seconds=1)

        test_start  = current_test_start
        test_end    = current_test_start + pd.Timedelta(days=test_span_days)

        # 念のため、比較境界もUTCであることを確認
        if train_start.tz is None: train_start = train_start.tz_localize('UTC')
        if train_end.tz is None: train_end = train_end.tz_localize('UTC')
        if test_start.tz is None: test_start = test_start.tz_localize('UTC')
        if test_end.tz is None: test_end = test_end.tz_localize('UTC')

        # test_end がデータ終端を超えたら終了
        if test_start >= max_date:
            break
        if test_end > max_date:
            test_end = max_date

        # 各期間のサンプル数をチェック
        mask_train = (dates >= train_start) & (dates <= train_end)
        mask_test  = (dates >= test_start)  & (dates <= test_end)

        n_train = mask_train.sum()
        n_test  = mask_test.sum()

        if (n_train >= min_train_rows) and (n_test >= min_test_rows):
            splits.append(
                {
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                }
            )

        # 次の fold に進める（ここでは test_span_days ずつシフト）
        current_test_start = current_test_start + pd.Timedelta(days=test_span_days)
        if current_test_start >= max_date:
            break
    print("[DEBUG] splits generated:", len(splits))
    if splits:
        print("[DEBUG] first split:", splits[0])
        print("[DEBUG] last split:", splits[-1])

    return splits

import pandas as pd

import pandas as pd

import numpy as np

def ensure_date_column(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df = df.copy()
    # 1) date列が無いが DatetimeIndex の場合 → date列を作る
    if date_col not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df[date_col] = df.index
        elif df.index.name == date_col:
            # index名が date なら reset_index
            df = df.reset_index()
        else:
            raise KeyError(
                f"'{date_col}' not in columns and index is not DatetimeIndex "
                f"(index.name={df.index.name}). Please provide date column."
            )
    # 2) 型を datetime に統一
    df[date_col] = pd.to_datetime(df[date_col])
    return df

def make_regime_aware_wfa_folds_debug(
    df_diag: pd.DataFrame,
    feature_cols,
    label_col: str = "label",
    date_col: str = "date",
    regime_col: str = "regime_id",
    valid_col: str = "valid_label",
    strategy_type: str | None = None,
    train_span_days: int = 365,
    test_span_days: int = 30,
    min_train_rows: int = 500,
    min_test_rows: int = 200,
    *,
    regime_mode: str = "filter",   # "filter" / "none" / "weight"（※weightはfold作成だけなら情報付与）
    require_both_classes_train: bool = True,
    require_both_classes_test: bool = True,
    return_df: bool = False,
):
    """
    regime_mode:
      - "filter": 既存ロジック（trainをtest_regimeに絞る）※落ちやすい
      - "none"  : 絞らない（推奨）
      - "weight": 絞らず、test_regime一致フラグをfold_metaに入れる（学習側でweightに使う想定）
    """
    df = df_diag.copy()

    df = ensure_date_column(df_diag, date_col=date_col)

    # # date_col が index 名になっている場合のケア
    # if date_col == df.index.name:
    #     df = df.reset_index(names=[date_col])



    # --- 有効ラベル・戦略フィルタ ---
    mask = df[label_col].isin([0, 1])
    if valid_col in df.columns:
        mask &= df[valid_col].astype(bool)
    if strategy_type is not None and "strategy_type" in df.columns:
        mask &= (df["strategy_type"] == strategy_type)

    # features + date + regime が NaN でないものを使用
    df = df[mask].dropna(subset=list(feature_cols) + [date_col, regime_col]).copy()
    if df.empty:
        fold_meta_df = pd.DataFrame()
        diag_df = pd.DataFrame([{
            "fold_idx": None, "status": "error",
            "skip_reason": "no_valid_samples_after_filtering"
        }])
        return (fold_meta_df, df, diag_df) if return_df else (fold_meta_df, diag_df)

    df = df.sort_values(date_col)

    # --- 時間ベースの WFA スプリット生成 ---
    splits = generate_time_based_wfa_splits(
        df,
        date_col=date_col,
        train_span_days=train_span_days,
        test_span_days=test_span_days,
        min_train_rows=min_train_rows,
        min_test_rows=min_test_rows,
    )
    if not splits:
        fold_meta_df = pd.DataFrame()
        diag_df = pd.DataFrame([{
            "fold_idx": None, "status": "error",
            "skip_reason": "no_wfa_splits_generated"
        }])
        return (fold_meta_df, df, diag_df) if return_df else (fold_meta_df, diag_df)

    fold_records = []
    diag_records = []

    for i, sp in enumerate(splits):
        train_start = sp["train_start"]
        train_end   = sp["train_end"]
        test_start  = sp["test_start"]
        test_end    = sp["test_end"]

        mask_train_time = (df[date_col] >= train_start) & (df[date_col] <= train_end)
        mask_test_time  = (df[date_col] >= test_start)  & (df[date_col] <= test_end)

        df_train_time = df[mask_train_time]
        df_test_time  = df[mask_test_time]

        rec = {
            "fold_idx": i,
            "train_start": train_start, "train_end": train_end,
            "test_start": test_start, "test_end": test_end,
            "n_train_time": len(df_train_time),
            "n_test_time": len(df_test_time),
            "train_nunique_y_time": df_train_time[label_col].nunique() if len(df_train_time) else 0,
            "test_nunique_y_time": df_test_time[label_col].nunique() if len(df_test_time) else 0,
            "status": None,
            "skip_reason": None,
        }

        if df_train_time.empty or df_test_time.empty:
            rec["status"] = "skip"
            rec["skip_reason"] = "empty_train_or_test_time_window"
            diag_records.append(rec)
            continue

        test_regimes = df_test_time[regime_col].unique()

        # ---- regime の扱い ----
        if regime_mode == "filter":
            df_train = df_train_time[df_train_time[regime_col].isin(test_regimes)].copy()
        else:
            df_train = df_train_time.copy()

        df_test = df_test_time.copy()

        rec["n_train_after_regime"] = len(df_train)
        rec["n_test_after_regime"] = len(df_test)
        rec["train_nunique_y_after_regime"] = df_train[label_col].nunique() if len(df_train) else 0
        rec["test_nunique_y_after_regime"] = df_test[label_col].nunique() if len(df_test) else 0
        rec["n_test_regimes"] = len(test_regimes)

        # ---- 行数条件 ----
        if len(df_train) < min_train_rows or len(df_test) < min_test_rows:
            rec["status"] = "skip"
            rec["skip_reason"] = "min_rows_not_met"
            diag_records.append(rec)
            continue

        # ---- 両クラス条件 ----
        if require_both_classes_train and df_train[label_col].nunique() < 2:
            rec["status"] = "skip"
            rec["skip_reason"] = "train_missing_class"
            diag_records.append(rec)
            continue

        if require_both_classes_test and df_test[label_col].nunique() < 2:
            rec["status"] = "skip"
            rec["skip_reason"] = "test_missing_class"
            diag_records.append(rec)
            continue

        # ---- 採用 ----
        rec["status"] = "ok"
        rec["skip_reason"] = ""
        diag_records.append(rec)

        fold_records.append(
            {
                "fold_idx": i,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "n_train": len(df_train),
                "n_test": len(df_test),
                "test_regimes": ",".join(map(str, test_regimes)),
                "regime_mode": regime_mode,
            }
        )

    fold_meta_df = pd.DataFrame(fold_records)
    diag_df = pd.DataFrame(diag_records)

    if return_df:
        return fold_meta_df, df, diag_df
    else:
        return fold_meta_df, diag_df

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score

def regime_aware_wfa_auc_with_rf(
    df_diag,
    feature_cols,
    label_col="label",
    date_col="date",
    regime_col="regime_id",
    valid_col="valid_label",
    strategy_type=None,
    train_span_days=365,
    test_span_days=30,
    min_train_rows=500,
    min_test_rows=200,
    n_estimators=300,
    random_state=42,
):
    """
    ★ 新版: Regime-aware WFA RandomForest AUC
      - スプリット生成は make_regime_aware_wfa_folds に委譲
      - fold_meta_df を使って学習＆AUC計算
    """

    df = ensure_date_column(df_diag, date_col=date_col)

    # ① fold定義 ＋ フィルタ済み df を取得
    fold_meta_df, df = make_regime_aware_wfa_folds_debug( # Correctly unpack df, and ignore the diagnostic df with _
        df_diag=df_diag,
        feature_cols=feature_cols,
        label_col=label_col,
        date_col=date_col,
        regime_col=regime_col,
        valid_col=valid_col,
        strategy_type=strategy_type,
        train_span_days=train_span_days,
        test_span_days=test_span_days,
        min_train_rows=min_train_rows,
        min_test_rows=min_test_rows,
        regime_mode="none",          # ★ここが重要
        return_df=False,              # Change to True to return df
    )

    if fold_meta_df.empty:
        print("No valid folds for RF WFA.")
        return None

    print("fold_meta_df:", fold_meta_df.shape)
    print(df["status"].value_counts())
    print(df["skip_reason"].value_counts().head(20))


    records = []

    for row in fold_meta_df.itertuples(index=False):
        train_start = row.train_start
        train_end   = row.train_end
        test_start  = row.test_start
        test_end    = row.test_end
        test_regimes_str = row.test_regimes

        mask_train_time = (df[date_col] >= train_start) & (df[date_col] <= train_end)
        mask_test_time  = (df[date_col] >= test_start)  & (df[date_col] <= test_end)

        df_train_time = df[mask_train_time]
        df_test_time  = df[mask_test_time]

        # test_regimes で train をフィルタ（念のため再現）
        if isinstance(test_regimes_str, str):
            test_regimes = test_regimes_str.split(",")
        else:
            # This case should ideally not happen if fold_meta_df is well-formed.
            # Fallback to getting unique regimes from the actual test data if it occurs.
            test_regimes = list(df_test_time[regime_col].unique())

        df_train = df_train_time[df_train_time[regime_col].isin(test_regimes)].copy()
        df_test  = df_test_time.copy()

        # 念のためガード（理論上は make_regime_aware_wfa_folds 側で保証済み）
        if df_train.shape[0] < min_train_rows or df_test.shape[0] < min_test_rows:
            continue
        if df_train[label_col].nunique() < 2 or df_test[label_col].nunique() < 2:
            continue

        X_train = df_train[feature_cols].values
        y_train = df_train[label_col].values
        X_test  = df_test[feature_cols].values
        y_test  = df_test[label_col].values

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)

        prob_train = clf.predict_proba(X_train)[:, 1]
        prob_test  = clf.predict_proba(X_test)[:, 1]
        auc_train  = roc_auc_score(y_train, prob_train)
        auc_test   = roc_auc_score(y_test, prob_test)

        records.append(
            {
                "fold_idx": row.fold_idx,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "n_train": df_train.shape[0],
                "n_test": df_test.shape[0],
                "test_regimes": test_regimes_str,
                "auc_train": auc_train,
                "auc_test": auc_test,
            }
        )

        print(
            f"[RF Fold {row.fold_idx}] "
            f"{train_start} ~ {train_end} -> {test_start} ~ {test_end} | "
            f"n_train={df_train.shape[0]}, n_test={df_test.shape[0]}, "
            f"regimes={set(test_regimes)}, "
            f"AUC(train)={auc_train:.3f}, AUC(test)={auc_test:.3f}"
        )

    if not records:
        print("No valid folds with both classes present (RF).")
        return None

    results_df = pd.DataFrame(records)
    return results_df

def smooth_macro_indicators(
    df: pd.DataFrame,
    macro_cols: list,
    window: int = 24*3,   # ETH 1h → 3日分
    min_periods: int = None,
    suffix: str = "_sm"
):
    """
    外部マクロ（低頻度データ）を rolling-median により滑らかにする。

    Parameters:
        df : DataFrame
        macro_cols : smoothing する列のリスト
        window : rolling window（1h データで24*3 = 3日）
        min_periods : 最低サンプル数（Noneならwindowの半分）
        suffix : 平滑化後列に付与するサフィックス

    Returns:
        df_sm : 新しい列を追加した DataFrame
    """
    df = df.copy()

    if min_periods is None:
        min_periods = window // 2  # 半分あれば安定して算出できる

    for col in macro_cols:
        if col not in df.columns:
            print(f"[WARN] {col} not found, skip")
            continue

        # ロール中値
        df[col + suffix] = (
            df[col]
            .rolling(window=window, min_periods=min_periods)
            .median()
        )

        # 前方埋め（外部データは欠損が出やすい）
        df[col + suffix] = df[col + suffix].ffill().bfill()

    return df

import torch

import torch.nn.functional as F

def binary_focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    logits: (B,) or (B,1) などの生ロジット
    targets: (B,) 0/1 のラベル（float にキャストしておく）

    alpha: 正例の重み（class imbalance 対応）
    gamma: hard example をどれだけ強調するか
    """

    # BCE with logits を要素ごとに計算
    bce_loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )  # shape: (B,)

    # p_t = 正しく分類された確率
    # （cross entropy の微分から exp(-loss) で書ける）
    p_t = torch.exp(-bce_loss)

    # 正例にだけ alpha を効かせる（targets=1 → alpha, 0 → (1-alpha)）
    alpha_factor = targets * alpha + (1.0 - targets) * (1.0 - alpha)

    # (1 - p_t)^gamma で easy なサンプルの重みを下げる
    modulating_factor = (1.0 - p_t) ** gamma

    loss = alpha_factor * modulating_factor * bce_loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss

import pandas as pd

import numpy as np

def _ensure_regime_id(df: pd.DataFrame, regime_col: str) -> pd.DataFrame:
    """
    regime_col が存在しない場合、よくある候補列から regime_id を自動生成する。
    生成された列名は regime_col（デフォルト 'regime_id'）に揃える。
    """
    df = df.copy()
    if regime_col in df.columns:
        return df

    # よくある候補（あなたの実装文脈に合わせて増やしてOK）
    candidates = ["regime_v3_compressed", "regime", "regime_label", "regime_name"]
    src = next((c for c in candidates if c in df.columns), None)

    if src is None:
        raise KeyError(
            f"'{regime_col}' is missing and no candidate regime column found. "
            f"Available columns include: {list(df.columns)[:30]} ..."
        )

    # factorize で 0..K-1 にする（NaN は -1 になるので dropna で落ちる）
    codes, _ = pd.factorize(df[src], sort=True)
    df[regime_col] = codes.astype(int)
    return df

def ensure_date_column(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    - date_col が無い場合は DatetimeIndex から作る
    - 文字列/naive/tz-aware 混在でも比較できるように UTC に揃える
    """
    df = df.copy()

    if date_col not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df[date_col] = df.index
        elif df.index.name == date_col:
            df = df.reset_index()
        else:
            raise KeyError(
                f"'{date_col}' not in columns and index is not DatetimeIndex "
                f"(index.name={df.index.name})."
            )

    # UTC に統一（naive → UTC とみなす / tz-aware → UTC変換）
    # ※ generate_time_based_wfa_splits 側も UTC を前提にしているため揃える
    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    if df[date_col].isna().any():
        # date が NaT の行は後段 dropna で落とすのでここでは警告のみ
        pass

    return df

def make_regime_aware_wfa_folds_debug(
    df_diag: pd.DataFrame,
    feature_cols,
    label_col: str = "label",
    date_col: str = "date",
    regime_col: str = "regime_id",
    valid_col: str = "valid_label",
    strategy_type: str | None = None,
    train_span_days: int = 270,
    test_span_days: int = 60,
    min_train_rows: int = 110,
    min_test_rows: int = 40,
    *,
    # ★ デフォルトを "weight" 推奨（filter は fold が死にやすい）
    regime_mode: str = "none",  # "filter" / "none" / "weight"
    require_both_classes_train: bool = True,
    require_both_classes_test: bool = False,

    # ★ fold数確保のための救済策
    auto_expand_test: bool = True,
    max_expand_test_days: int = 120,      # test_end を最大どこまで伸ばすか（例: +120日）
    expand_step_days: int = 7,            # 何日刻みで伸ばすか（週刻みが扱いやすい）

    auto_expand_train: bool = False,      # 通常は不要（strategy 切り出し時のみ有効化検討）
    max_expand_train_days: int = 365,     # train_start を最大どこまで前倒すか

    return_df: bool = False,
):
    """
    fold が落ちる最大要因
      - test_missing_class: テスト窓が短くて片側クラスしか出ない
      - train_missing_class: regime filter で train の片側クラスが消える

    対策:
      - regime_mode は原則 weight/none（filterしない）
      - test_missing_class は test_end を自動延長して救済（上限あり）
      - train_missing_class が regime filter 由来なら filter を解除して救済
    """
    df = df_diag.copy()
    df = ensure_date_column(df, date_col=date_col)  # ★ df_diag ではなく df を渡す
    df = _ensure_regime_id(df, regime_col=regime_col)

    # --- 有効ラベル・戦略フィルタ ---
    mask = df[label_col].isin([0, 1])
    if valid_col in df.columns:
        mask &= df[valid_col].astype(bool)
    if strategy_type is not None and "strategy_type" in df.columns:
        mask &= (df["strategy_type"] == strategy_type)

    # features + date + regime が NaN でないものを使用
    df = df[mask].dropna(subset=list(feature_cols) + [date_col, regime_col]).copy()
    if df.empty:
        fold_meta_df = pd.DataFrame()
        diag_df = pd.DataFrame([{
            "fold_idx": None, "status": "error",
            "skip_reason": "no_valid_samples_after_filtering"
        }])
        return (fold_meta_df, df, diag_df) if return_df else (fold_meta_df, diag_df)

    df = df.sort_values(date_col)

    # --- 時間ベース split ---
    splits = generate_time_based_wfa_splits(
        df,
        date_col=date_col,
        train_span_days=train_span_days,
        test_span_days=test_span_days,
        min_train_rows=min_train_rows,
        min_test_rows=min_test_rows,
    )
    if not splits:
        fold_meta_df = pd.DataFrame()
        diag_df = pd.DataFrame([{
            "fold_idx": None, "status": "error",
            "skip_reason": "no_wfa_splits_generated"
        }])
        return (fold_meta_df, df, diag_df) if return_df else (fold_meta_df, diag_df)

    fold_records = []
    diag_records = []

    # 便利関数：両クラス揃ってるか
    def has_both_classes(s: pd.Series) -> bool:
        u = pd.Series(s).dropna().unique()
        return len(u) >= 2

    for i, sp in enumerate(splits):
        train_start = sp["train_start"]
        train_end   = sp["train_end"]
        test_start  = sp["test_start"]
        test_end    = sp["test_end"]

        rec = {
            "fold_idx": i,
            "train_start": train_start, "train_end": train_end,
            "test_start": test_start, "test_end": test_end,
            "status": None,
            "skip_reason": None,
        }

        # --- time window 抽出（UTCで比較）---
        mask_train_time = (df[date_col] >= train_start) & (df[date_col] <= train_end)
        mask_test_time  = (df[date_col] >= test_start)  & (df[date_col] <= test_end)

        df_train_time = df.loc[mask_train_time]
        df_test_time  = df.loc[mask_test_time]

        rec["n_train_time"] = len(df_train_time)
        rec["n_test_time"]  = len(df_test_time)
        rec["train_nunique_y_time"] = df_train_time[label_col].nunique() if len(df_train_time) else 0
        rec["test_nunique_y_time"]  = df_test_time[label_col].nunique() if len(df_test_time) else 0

        if df_train_time.empty or df_test_time.empty:
            rec["status"] = "skip"
            rec["skip_reason"] = "empty_train_or_test_time_window"
            diag_records.append(rec)
            continue

        # --- test_missing_class 救済: test_end を延長 ---
        if require_both_classes_test and (df_test_time[label_col].nunique() < 2) and auto_expand_test:
            extended = False
            max_end = test_end + pd.Timedelta(days=max_expand_test_days)
            cand_end = test_end

            while cand_end < max_end:
                cand_end = cand_end + pd.Timedelta(days=expand_step_days)
                mask_test_time2 = (df[date_col] >= test_start) & (df[date_col] <= cand_end)
                df_test_time2 = df.loc[mask_test_time2]
                if len(df_test_time2) >= min_test_rows and df_test_time2[label_col].nunique() >= 2:
                    df_test_time = df_test_time2
                    rec["test_end"] = cand_end
                    rec["test_end_expanded_days"] = int((cand_end - test_end).days)
                    rec["n_test_time"] = len(df_test_time)
                    rec["test_nunique_y_time"] = df_test_time[label_col].nunique()
                    extended = True
                    break

            if not extended:
                rec["status"] = "skip"
                rec["skip_reason"] = "test_missing_class"
                diag_records.append(rec)
                continue

        # --- train_missing_class 救済（必要なら train_start を前倒し）---
        if require_both_classes_train and (df_train_time[label_col].nunique() < 2) and auto_expand_train:
            extended = False
            min_start = train_start - pd.Timedelta(days=max_expand_train_days)
            cand_start = train_start

            while cand_start > min_start:
                cand_start = cand_start - pd.Timedelta(days=expand_step_days)
                mask_train_time2 = (df[date_col] >= cand_start) & (df[date_col] <= train_end)
                df_train_time2 = df.loc[mask_train_time2]
                if len(df_train_time2) >= min_train_rows and df_train_time2[label_col].nunique() >= 2:
                    df_train_time = df_train_time2
                    rec["train_start"] = cand_start
                    rec["train_start_expanded_days"] = int((train_start - cand_start).days)
                    rec["n_train_time"] = len(df_train_time)
                    rec["train_nunique_y_time"] = df_train_time[label_col].nunique()
                    extended = True
                    break

            if not extended:
                rec["status"] = "skip"
                rec["skip_reason"] = "train_missing_class"
                diag_records.append(rec)
                continue

        # --- regime 情報 ---
        test_regimes = df_test_time[regime_col].unique()
        rec["n_test_regimes"] = len(test_regimes)
        rec["test_regimes"] = ",".join(map(str, test_regimes))

        # --- regime_mode 適用 ---
        if regime_mode == "filter":
            df_train = df_train_time[df_train_time[regime_col].isin(test_regimes)].copy()
            rec["regime_mode_applied"] = "filter"
        else:
            df_train = df_train_time.copy()
            rec["regime_mode_applied"] = regime_mode

        df_test = df_test_time.copy()

        rec["n_train_after_regime"] = len(df_train)
        rec["n_test_after_regime"]  = len(df_test)
        rec["train_nunique_y_after_regime"] = df_train[label_col].nunique() if len(df_train) else 0
        rec["test_nunique_y_after_regime"]  = df_test[label_col].nunique() if len(df_test) else 0

        # --- min rows ---
        if len(df_train) < min_train_rows or len(df_test) < min_test_rows:
            rec["status"] = "skip"
            rec["skip_reason"] = "min_rows_not_met"
            diag_records.append(rec)
            continue

        # --- 両クラス条件（filterで落ちるなら filter解除で救済）---
        if require_both_classes_train and df_train[label_col].nunique() < 2:
            if rec["regime_mode_applied"] == "filter":
                # filter が原因なら解除して救済（foldを残すのが目的）
                df_train = df_train_time.copy()
                rec["regime_mode_applied"] = "none_fallback_from_filter"
                rec["n_train_after_regime"] = len(df_train)
                rec["train_nunique_y_after_regime"] = df_train[label_col].nunique()

            if df_train[label_col].nunique() < 2:
                rec["status"] = "skip"
                rec["skip_reason"] = "train_missing_class"
                diag_records.append(rec)
                continue

        if require_both_classes_test and df_test[label_col].nunique() < 2:
            rec["status"] = "skip"
            rec["skip_reason"] = "test_missing_class"
            diag_records.append(rec)
            continue

        # --- OK fold ---
        rec["status"] = "ok"
        fold_records.append({
            "fold_idx": i,
            "train_start": rec["train_start"],
            "train_end":   train_end,
            "test_start":  test_start,
            "test_end":    rec["test_end"],
            "n_train": len(df_train),
            "n_test":  len(df_test),
            "test_regimes": rec["test_regimes"],
            "regime_mode_applied": rec["regime_mode_applied"],
            # weight用フラグ（学習側で使うなら）
            "train_in_test_regime_flag": 1 if regime_mode in ("weight",) else 0,
        })

        diag_records.append(rec)

    fold_meta_df = pd.DataFrame(fold_records)
    diag_df = pd.DataFrame(diag_records)

    if return_df:
        return fold_meta_df, df, diag_df
    else:
        return fold_meta_df, diag_df

import numpy as np

import pandas as pd

import torch

from torch import nn

from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score

from tft_model import TFT

from typing import Dict, Optional, Tuple

import numpy as np

import pandas as pd

def _mk_diag(
    fold_row,
    status: str,
    reason: str,
    **kwargs
) -> dict:
    out = {
        "fold_idx": int(fold_row["fold_idx"]),
        "status": status,          # "ok" or "skip" or "error"
        "skip_reason": reason,     # 例: "train_ds_empty"
    }
    out.update(kwargs)
    return out

def _fit_normalizer_on_train(df, cols, train_start, train_end):
    subset = df.loc[train_start:train_end, cols]
    mean = subset.mean()
    std = subset.std().replace(0, 1.0)
    return mean, std

def _apply_normalizer(df, cols, mean, std):
    df_new = df.copy()
    for c in cols:
        if c in mean.index and c in std.index:
            df_new[c] = (df_new[c] - mean[c]) / std[c]
    return df_new

def _pick_thr_by_target_rate(
    x: np.ndarray,
    *,
    target_rate: float,
    mode: str,
    prefer: str = "closest",   # "closest" | "ge" | "le"
    max_unique_scan: int = 5000,
) -> float:
    """
    x: 1D finite array
    mode:
      - "up"    : positive if x >= thr
      - "down"  : positive if x <= thr
      - "range" : positive if |x| <= thr  (x is abs already or raw? -> here raw OK if mode range uses abs inside)
    prefer:
      - "closest": |rate-target|最小
      - "ge": rate>=target の中で最も近い（なければclosest）
      - "le": rate<=target の中で最も近い（なければclosest）
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan

    if mode == "range":
        vals = np.abs(x)
    else:
        vals = x

    # unique candidates
    uniq = np.unique(vals)
    if uniq.size == 0:
        return np.nan

    # 安全のためユニークが多すぎるときは間引く（計算量対策）
    if uniq.size > max_unique_scan:
        # 均等にサンプリングして探索（それでも十分安定します）
        idx = np.linspace(0, uniq.size - 1, max_unique_scan).astype(int)
        uniq = uniq[idx]

    # rate computation
    if mode == "up":
        # positives: x >= thr
        rates = np.array([(x >= thr).mean() for thr in uniq], dtype=np.float64)
    elif mode == "down":
        rates = np.array([(x <= thr).mean() for thr in uniq], dtype=np.float64)
    elif mode == "range":
        rates = np.array([(np.abs(x) <= thr).mean() for thr in uniq], dtype=np.float64)
    else:
        raise ValueError(f"unknown mode={mode}")

    # NOTE: 上の range 行は lint 回避のため下で再代入（実行時は次行が使われます）
    # This line has been corrected. It was causing a SyntaxError due to incorrect bracket usage.
    # if mode == "range":
    #     rates = np.array([(np.abs(x) <= thr).mean() for thr in uniq], dtype=np.float64)

    # pick
    diff = np.abs(rates - target_rate)

    if prefer == "ge":
        mask = rates >= target_rate
        if mask.any():
            j = np.argmin(diff + (~mask) * 1e9)
            return float(uniq[j])
        # fallback
    elif prefer == "le":
        mask = rates <= target_rate
        if mask.any():
            j = np.argmin(diff + (~mask) * 1e9)
            return float(uniq[j])
        # fallback

    j = int(np.argmin(diff))
    return float(uniq[j])

def fit_thresholds_on_train_3cat(
    *,
    future_ret_full: np.ndarray,            # length = n_seq, seq_id で引く
    strategy_type_id_full: np.ndarray,      # length = n_seq
    train_ids: np.ndarray,                  # seq_id array
    pos_rate_up: float = 0.20,
    pos_rate_down: float = 0.20,
    pos_rate_range: float = 0.30,
    strategy_id_map: Optional[Dict[int, str]] = None,
    min_samples_per_strategy: int = 50,
) -> Dict[int, float]:
    """
    train_ids に属するサンプルだけで閾値を学習（Fold内比率固定化）。

    戦略ごとに1つの thr:
      - Uptrend   : fr >= thr_up が positive（pos_rate_up 近傍）
      - Downtrend : fr <= thr_down が positive（pos_rate_down 近傍）
      - Range     : |fr| <= thr_range が positive（pos_rate_range 近傍）
    """
    if strategy_id_map is None:
        strategy_id_map = {0: "Downtrend", 1: "Range", 2: "Uptrend"}

    future_ret_full = np.asarray(future_ret_full, dtype=np.float64)
    strategy_type_id_full = np.asarray(strategy_type_id_full)
    train_ids = np.asarray(train_ids, dtype=int)

    n_seq = len(future_ret_full)
    train_ids = train_ids[(train_ids >= 0) & (train_ids < n_seq)]

    thr: Dict[int, float] = {}

    for sid, sname in strategy_id_map.items():
        idx = train_ids[strategy_type_id_full[train_ids] == sid]
        if idx.size == 0:
            thr[sid] = np.nan
            continue

        fr = future_ret_full[idx]
        fr = fr[np.isfinite(fr)]

        if fr.size < max(5, min_samples_per_strategy):
            # 母数が小さいと探索が不安定なので quantile fallback
            if fr.size == 0:
                thr[sid] = np.nan
                continue
            if sname == "Uptrend":
                thr[sid] = float(np.quantile(fr, 1.0 - pos_rate_up))
            elif sname == "Downtrend":
                thr[sid] = float(np.quantile(fr, pos_rate_down))
            else:
                thr[sid] = float(np.quantile(np.abs(fr), pos_rate_range))
            continue

        if sname == "Uptrend":
            thr[sid] = _pick_thr_by_target_rate(
                fr, target_rate=pos_rate_up, mode="up", prefer="closest"
            )
        elif sname == "Downtrend":
            thr[sid] = _pick_thr_by_target_rate(
                fr, target_rate=pos_rate_down, mode="down", prefer="closest"
            )
        else:  # Range
            thr[sid] = _pick_thr_by_target_rate(
                fr, target_rate=pos_rate_range, mode="range", prefer="closest"
            )

    return thr

def apply_thresholds_3cat(
    *,
    future_ret_full: np.ndarray,            # length = n_seq
    strategy_type_id_full: np.ndarray,      # length = n_seq
    thresholds: Dict[int, float],
    strategy_id_map: Optional[Dict[int, str]] = None,
) -> np.ndarray:
    """
    thresholds を全サンプルに適用して 0/1 ラベルを返す（-1無し）。
    ※ NaN future_ret は 0 に倒す（評価や学習の安定のため）
    """
    if strategy_id_map is None:
        strategy_id_map = {0: "Downtrend", 1: "Range", 2: "Uptrend"}

    future_ret_full = np.asarray(future_ret_full, dtype=np.float64)
    strategy_type_id_full = np.asarray(strategy_type_id_full)

    n_seq = len(future_ret_full)
    labels = np.zeros(n_seq, dtype=np.float32)

    finite_mask = np.isfinite(future_ret_full)
    fr = future_ret_full.copy()
    fr[~finite_mask] = np.nan  # NaNは後で0ラベル固定

    for sid, sname in strategy_id_map.items():
        thr = thresholds.get(sid, np.nan)
        idx = np.where(strategy_type_id_full == sid)[0]
        if idx.size == 0:
            continue

        if not np.isfinite(thr):
            labels[idx] = 0.0
            continue

        fr_sid = fr[idx]
        ok = np.isfinite(fr_sid)

        out = np.zeros(idx.size, dtype=np.float32)
        if ok.any():
            v = fr_sid[ok]
            if sname == "Uptrend":
                out[ok] = (v >= thr).astype(np.float32)
            elif sname == "Downtrend":
                out[ok] = (v <= thr).astype(np.float32)
            else:  # Range
                out[ok] = (np.abs(v) <= thr).astype(np.float32)

        # 非finiteは0のまま
        labels[idx] = out

    return labels

def build_foldwise_labels_3cat_trainfit(
    *,
    df_diag: pd.DataFrame,
    train_ids: np.ndarray,
    future_ret_col: str = "log_return_h5",
    strategy_id_col: str = "strategy_type_id",
    pos_rate_up: float = 0.20,
    pos_rate_down: float = 0.20,
    pos_rate_range: float = 0.30,
) -> Tuple[np.ndarray, Dict[int, float]]:
    """
    df_diag から seq_id ベースで full配列を作り、
    train_idsのみでthr推定→全サンプルに適用した labels_per_seq を返す。

    重要:
      - df_diag が欠損行/フィルタ行を含んでも seq_id で整列するのでズレない
      - X_3d_numpy の n_seq と一致する labels_per_seq が出る
    """
    if "seq_id" not in df_diag.columns:
        raise ValueError("df_diag must have 'seq_id' column")
    if future_ret_col not in df_diag.columns:
        raise ValueError(f"{future_ret_col} not in df_diag")
    if strategy_id_col not in df_diag.columns:
        raise ValueError(f"{strategy_id_col} not in df_diag")

    seq_ids = df_diag["seq_id"].to_numpy(dtype=int)
    n_seq = int(seq_ids.max()) + 1

    # full arrays by seq_id
    future_ret_full = np.full(n_seq, np.nan, dtype=np.float64)
    strategy_type_id_full = np.full(n_seq, -999, dtype=np.int64)

    fr = df_diag[future_ret_col].to_numpy(dtype=np.float64)
    st = df_diag[strategy_id_col].to_numpy(dtype=np.int64)

    # 代入（同じseq_idが複数行あるなら最後で上書きされる。基本は1行のはず）
    future_ret_full[seq_ids] = fr
    strategy_type_id_full[seq_ids] = st

    thresholds = fit_thresholds_on_train_3cat(
        future_ret_full=future_ret_full,
        strategy_type_id_full=strategy_type_id_full,
        train_ids=train_ids,
        pos_rate_up=pos_rate_up,
        pos_rate_down=pos_rate_down,
        pos_rate_range=pos_rate_range,
        strategy_id_map={0: "Downtrend", 1: "Range", 2: "Uptrend"},
    )

    labels_per_seq = apply_thresholds_3cat(
        future_ret_full=future_ret_full,
        strategy_type_id_full=strategy_type_id_full,
        thresholds=thresholds,
        strategy_id_map={0: "Downtrend", 1: "Range", 2: "Uptrend"},
    )

    return labels_per_seq.astype(np.float32), thresholds

class CryptoBinaryDataset(Dataset):
    """
    時系列データとそれに対応するバイナリターゲット、グループID、時間インデックスを保持するカスタムデータセット。
    TFTモデルへの入力形式に合わせることを目的とする。
    """
    def __init__(
            self,
            dataframe,
            encoder_len,
            decoder_len,
            real_feature_cols,
            categorical_feature_cols,
            target_col,
            group_id_col,
            time_idx_col,
            train_end_timestamp=None,
            is_train: bool = False,          # ★ 追加：train / val/test 判定用
        ):
        """
        Args:
            dataframe (pd.DataFrame): 時系列データを含むDataFrame。
                                       カラムは features, target, group_id, time_idx を含む必要がある。
            encoder_len (int): エンコーダ部分の時系列長。
            decoder_len (int): デコーダ部分の時系列長。
            real_feature_cols (list): 実数特徴量のカラム名のリスト。
            categorical_feature_cols (list): カテゴリ特徴量のカラム名のリスト。
            target_col (str): ターゲットのカラム名。
            group_id_col (str): グループIDのカラム名（例: 時系列ID）。
            time_idx_col (str): 時間インデックスのカラム名。
            train_end_timestamp (pd.Timestamp, optional): 学習期間の最終タイムスタンプ。時間減衰重み付けに使用。
                                                        デフォルトは None。
        """
        if dataframe is None or dataframe.empty:
            raise ValueError("Input DataFrame cannot be None or empty.")

        self.dataframe = dataframe.copy() # 元のDataFrameを変更しないようにコピー
        self.encoder_len = encoder_len
        self.decoder_len = decoder_len
        self.seq_len = encoder_len + decoder_len # 全体のシーケンス長
        self.real_feature_cols = real_feature_cols
        self.categorical_feature_cols = categorical_feature_cols
        self.target_col = target_col
        self.group_id_col = group_id_col
        self.time_idx_col = time_idx_col
        self.train_end_timestamp = train_end_timestamp # 学習期間の最終タイムスタンプを保持
        self.is_train = is_train

        if "strategy_type" not in self.dataframe.columns:
            raise ValueError("strategy_type カラムが dataframe に存在しません。戦略別 weight / oversampling のために必要です。")

        # time_idx が0から始まる連続した整数であることを確認
        # 各 group_id ごとに time_idx を再インデックス付け
        # print("Debug(Dataset): Re-indexing time_idx for each group...") # Debug comment
        self.dataframe[self.time_idx_col] = self.dataframe.groupby(self.group_id_col).cumcount()
        # print("Debug(Dataset): Re-indexing complete.") # Debug comment


        # 各グループの開始インデックスと終了インデックスを計算
        self.group_indices = {}
        self.sequences = [] # 各シーケンスのメタデータを格納
        # 最小必要な長さを計算
        min_sequence_length = self.encoder_len + self.decoder_len

        # print(f"Debug(Dataset): Calculating group indices and filtering groups smaller than {min_sequence_length}...") # Debug comment
        for group_id, group_df in self.dataframe.groupby(self.group_id_col):
            if len(group_df) >= min_sequence_length:
                num_possible_sequences = len(group_df) - min_sequence_length + 1

                for i in range(num_possible_sequences):
                    seq_start_iloc_in_group = i
                    seq_end_iloc_in_group = seq_start_iloc_in_group + self.seq_len - 1

                    sequence_end_time = group_df.iloc[seq_end_iloc_in_group].name

                    # ★ このシーケンスの代表戦略（最後の時点の strategy_type を採用）
                    seq_strategy_type = group_df["strategy_type"].iloc[seq_end_iloc_in_group]

                    self.sequences.append(
                        {
                            "group_id": group_id,
                            "start_iloc_in_group": seq_start_iloc_in_group,
                            "end_iloc_in_group": seq_end_iloc_in_group,
                            "sequence_end_time": sequence_end_time,
                            "strategy_type": seq_strategy_type,  # ★ 追加
                            "target": group_df[self.target_col]
                                .iloc[
                                    seq_start_iloc_in_group + self.encoder_len :
                                    seq_start_iloc_in_group + self.encoder_len + self.decoder_len
                                ]
                                .values,
                        }
                    )

        if not self.sequences:
            print("Warning: No sequences were generated with the given parameters.")

        # ★★ 戦略ごとの oversampling（窓の重複抽出）: train のときだけ適用 ★★
        if self.is_train:
            oversampled = []
            for seq in self.sequences:
                s_type = seq.get("strategy_type", None)
                factor = STRATEGY_OVERSAMPLE_FACTORS.get(s_type, 1)
                for _ in range(factor):
                    oversampled.append(seq.copy())
            self.sequences = oversampled


    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_info = self.sequences[idx]
        group_id = seq_info['group_id']
        start_iloc_in_group = seq_info['start_iloc_in_group']
        # end_iloc_in_group = seq_info['end_iloc_in_group'] # このインデックスを含む # Debug comment

        # グループデータ全体を取得し、ilocでシーケンスをスライス
        group_df = self.dataframe[self.dataframe[self.group_id_col] == group_id].reset_index(drop=True) # グループを抽出してilocのためにインデックスをリセット

        # シーケンス全体 (encoder + decoder)
        sequence_df = group_df.iloc[start_iloc_in_group : start_iloc_in_group + self.seq_len].copy() # 終了インデックスはilocに含まれないため +self.seq_len

        # encoder および decoder 部分を分離
        encoder_df = sequence_df.iloc[:self.encoder_len]
        decoder_df = sequence_df.iloc[self.encoder_len:] # decoder_len の長さになるはず

        # 特徴量とターゲットを抽出
        encoder_real_input = encoder_df[self.real_feature_cols].values
        encoder_categorical_input = encoder_df[self.categorical_feature_cols].values
        decoder_real_input = decoder_df[self.real_feature_cols].values
        decoder_categorical_input = decoder_df[self.categorical_feature_cols].values
        target = sequence_df[self.target_col].iloc[self.encoder_len:self.encoder_len+self.decoder_len].values # デコーダ部分のターゲット

        # ★ 戦略IDを数値化して追加
        strategy_name = seq_info["strategy_type"]         # 例："Downtrend"
        strategy_id = STRATEGY_ID_MAP[strategy_name]      # 数値に変換（0〜4）


        # time_idx を抽出
        encoder_time_idx = encoder_df[self.time_idx_col].values
        decoder_time_idx = decoder_df[self.time_idx_col].values

        # numpy array を tensor に変換
        encoder_real_input = torch.tensor(encoder_real_input, dtype=torch.float32)
        encoder_categorical_input = torch.tensor(encoder_categorical_input, dtype=torch.int64) # カテゴリはLongTensor

        target = torch.tensor(target, dtype=torch.float32).unsqueeze(-1) # BCEWithLogitsLossはfloatを期待, (B, T, 1) もしくは (B, T) にするため unsqueeze

        encoder_time_idx = torch.tensor(encoder_time_idx, dtype=torch.int64)
        decoder_time_idx = torch.tensor(decoder_time_idx, dtype=torch.int64)


        # 静的特徴量は現在サポートしないため空のテンソル
        static_real_input = torch.empty(0, dtype=torch.float32)
        static_categorical_input = torch.empty(0, dtype=torch.int64)

        # train_end_timestamp をバッチに含める
        # None の可能性があるため文字列に変換するか、適切な形式で渡す
        # ここでは文字列に変換して default_collate が処理できるようにする
        train_end_ts_str = str(self.train_end_timestamp) if self.train_end_timestamp is not None else "None"

        # sequence_end_time も Timestamp なので文字列に変換してバッチに含める
        sequence_end_time_str = str(seq_info['sequence_end_time'])


        # バッチに含めるデータを辞書として返す
        batch_data = {
            "encoder_real_input": encoder_real_input, # (encoder_len, num_real_features)
            "encoder_categorical_input": encoder_categorical_input, # (encoder_len, num_cat_features)
            "decoder_real_input": decoder_real_input, # (decoder_len, num_real_features)
            "decoder_categorical_input": decoder_categorical_input, # (decoder_len, num_cat_features)
            "target": target, # (decoder_len, 1) or (decoder_len,)
            "encoder_time_idx": encoder_time_idx, # (encoder_len,)
            "decoder_time_idx": decoder_time_idx, # (decoder_len,)
            "static_real_input": static_real_input, # (num_static_real_features,) - currently empty
            "static_categorical_input": static_categorical_input, # (num_static_cat_features,) - currently empty
            "group_id": torch.tensor(group_id, dtype=torch.int64), # スカラー
            "sequence_end_time": sequence_end_time_str, # ★ 修正: 文字列として渡す
            "train_end_timestamp": train_end_ts_str, # ★ 修正: 文字列として渡す
            "strategy_type": seq_info["strategy_type"],
            "strategy_id": strategy_id,                   # ← ★ここが重要
        }

        return batch_data

def add_regime_id(df: pd.DataFrame,
                  regime_col: str = "regime_v3_compressed",
                  id_col: str = "regime_id") -> pd.DataFrame:
    """
    圧縮レジーム文字列を int ID に変換して、TFT のカテゴリ特徴量として使う。
    """
    df = df.copy()
    if regime_col not in df.columns:
        raise ValueError(f"{regime_col} が df.columns にありません")

    # カテゴリをコード化（0..n-1）
    df[id_col] = df[regime_col].astype("category").cat.codes.astype(int)
    return df

def prepare_tft_base_frame(df_diag: pd.DataFrame,
                           group_id_value: int = 0,
                           date_col: str = "date") -> pd.DataFrame:
    """
    ランダムフォレスト用の df_diag から、
    TFT / CryptoBinaryDataset 用のベース DataFrame を作る。

    - index: DatetimeIndex（すでにそうなっている前提）
    - group_id: 単一銘柄なので 0 で固定
    - time_idx: 0..N-1 の連番
    """
    df = df_diag.copy()

    # index が DatetimeIndex でない場合、date_col をインデックスに設定する
    if not isinstance(df.index, pd.DatetimeIndex):
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
        else:
            raise ValueError(f"df_diag.index は DatetimeIndex であるか、'{date_col}' カラムが必要です")

    df = df.sort_index()
    df["group_id"] = group_id_value
    df["time_idx"] = np.arange(len(df), dtype=int)
    return df

def make_tft_dataset_for_period(
    df_base: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    cont_features: list,
    cat_features: list,
    encoder_len: int,
    decoder_len: int,
    is_train: bool = False,
    target_col: str = "target",
    group_id_col: str = "group_id",
    time_idx_col: str = "time_idx",
    train_end_timestamp: pd.Timestamp | None = None,
) -> CryptoBinaryDataset | None:
    """
    df_base（全期間）から [start_ts, end_ts] 周辺を切り出して CryptoBinaryDataset を作る。

    変更点：
      - エンコーダ部分の履歴を確保するために、
        start_ts より encoder_len 分だけ過去にマージンを取って抽出する。
      - これにより、テスト区間が短くても「シーケンス長不足」で
        fold がスキップされるケースを減らす。
    """
    # 1) エンコーダ分の履歴マージンを取る（1h 足前提で hours として扱う）
    #    必要であれば encoder_len + decoder_len にしても良いです。
    history_margin = encoder_len
    extended_start = start_ts - pd.Timedelta(hours=history_margin)

    mask = (df_base.index >= extended_start) & (df_base.index <= end_ts)
    df_slice = df_base.loc[mask].copy().sort_index()

    if df_slice.empty:
        print(
            f"[WARN] 拡張窓 [{extended_start} - {end_ts}] で有効データが 0 件なので "
            f"この区間はスキップします (orig: [{start_ts} - {end_ts}])"
        )
        return None

    # 2) シーケンス長チェック用
    min_required_len = encoder_len + decoder_len
    group_sizes = df_slice.groupby(group_id_col).size()
    valid_groups = group_sizes[group_sizes >= min_required_len].index

    if len(valid_groups) == 0:
        print(
            f"[make_tft_dataset_for_period] 拡張窓 {extended_start} - {end_ts} : "
            f"必要シーケンス長 {min_required_len} を満たす group_id が無し → None "
            f"(orig: {start_ts} - {end_ts})"
        )
        return None

    df_slice = df_slice[df_slice[group_id_col].isin(valid_groups)].copy()

    # 3) CryptoBinaryDataset を構築
    #    （コンストラクタの引数順は、これまで使っていたものに合わせている）
    dataset = CryptoBinaryDataset(
        df_slice,           # dataframe (positional)
        encoder_len,        # positional
        decoder_len,        # positional
        cont_features,      # positional
        cat_features,       # positional
        target_col,         # positional
        group_id_col,       # positional
        time_idx_col,       # positional
        train_end_timestamp=train_end_timestamp,  # keyword (optional)
        is_train=is_train,
    )

    if len(dataset) == 0:
        print(
            f"[make_tft_dataset_for_period] 拡張窓 {extended_start} - {end_ts} : "
            f"CryptoBinaryDataset の長さが 0 → None "
            f"(orig: {start_ts} - {end_ts})"
        )
        return None

    return dataset

def train_one_tft_fold(
    df_base: pd.DataFrame,
    fold_row: pd.Series,
    cont_features: list,
    cat_features: list,
    base_config: dict,
    encoder_len: int = 24,
    decoder_len: int = 1,  # ★一点予測
    batch_size: int = 64,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    device: torch.device | str = "cuda",
    target_col: str = "target",
    eval_strategy_type: str | None = None,
    # ===== 追加：fold内 train-fit ラベル作成 =====
    future_ret_col: str = "log_return_h6",        # ★6時間後log return列名（あなたのdfに合わせて変更）
    strategy_id_col: str = "strategy_type_id",    # ★0/1/2の戦略ID列名
    # target_positive_rate（再導入）
    target_positive_rate: float = 0.20,  # Up/Downのpositive率
    pos_rate_range: float = 0.30,        # Rangeはabs(fr)の小さい側をpositiveにする率
    # ===== 追加：fold内正規化 =====
    fold_normalize: bool = True,
) -> dict:

    try:

        if decoder_len != 1:
            raise ValueError(f"decoder_len must be 1 for point forecast. got={decoder_len}")

        device = torch.device(device)

        train_start = pd.to_datetime(fold_row["train_start"])
        train_end   = pd.to_datetime(fold_row["train_end"])
        test_start  = pd.to_datetime(fold_row["test_start"])
        test_end    = pd.to_datetime(fold_row["test_end"])

        val_span_hours = 24 * 7
        val_cut = train_end - pd.Timedelta(hours=val_span_hours)

        # ============
        # 0) df_base validate
        # ============
        for col in cont_features + cat_features:
            if col not in df_base.columns:
                raise ValueError(f"Missing feature col in df_base: {col}")

        if future_ret_col not in df_base.columns:
            raise ValueError(f"Missing future_ret_col in df_base: {future_ret_col}")

        if strategy_id_col not in df_base.columns:
            raise ValueError(f"Missing strategy_id_col in df_base: {strategy_id_col}")

        # ============
        # 1) fold内 train-fit 閾値→ラベル作成
        # ============
        # train-fitは leakage回避のため「train_in（=val_cutまで）」で閾値学習
        df_work = df_base.copy()

        # 閾値学習に使うインデックス（train_in期間）
        train_fit_mask = (df_work.index >= train_start) & (df_work.index <= val_cut)
        train_fit_idx = np.where(train_fit_mask)[0] # FIXED: Removed .values

        fr_all = df_work[future_ret_col].to_numpy(dtype=np.float32)
        sid_all = df_work[strategy_id_col].to_numpy()

        finite_train_fit_idx = train_fit_idx[np.isfinite(fr_all[train_fit_idx])]
        if finite_train_fit_idx.size < 100:
            print(f"[Fold {fold_row['fold_idx']}] Too few finite future_ret in train-fit ({finite_train_fit_idx.size}) → skip")
            return None

        thresholds = fit_thresholds_on_train_3cat(
            future_ret_full=fr_all,
            strategy_type_id_full=sid_all,
            train_ids=finite_train_fit_idx,
            pos_rate_up=target_positive_rate,
            pos_rate_down=target_positive_rate,
            pos_rate_range=pos_rate_range,
            strategy_id_map={0: "Downtrend", 1: "Range", 2: "Uptrend"},
        )

        labels_all = apply_thresholds_3cat(
            future_ret_full=fr_all,
            strategy_type_id_full=sid_all,
            thresholds=thresholds,
            strategy_id_map={0: "Downtrend", 1: "Range", 2: "Uptrend"},
        ).astype(np.float32)

        df_work[target_col] = labels_all

        # ============
        # 2) fold内正規化（train_inでfit→全体へ適用）
        # ============
        if fold_normalize:
            mean, std = _fit_normalizer_on_train(
                df_work, cont_features, train_start=train_start, train_end=val_cut
            )
            df_work = _apply_normalizer(df_work, cont_features, mean, std)

        # ============
        # 3) train/val/test dataset
        # ============
        train_ds = make_tft_dataset_for_period(
            df_work,
            start_ts=train_start,
            end_ts=val_cut,
            cont_features=cont_features,
            cat_features=cat_features,
            encoder_len=encoder_len,
            decoder_len=decoder_len,  # ★1
            is_train=True,
            train_end_timestamp=train_end,
            target_col=target_col,
        )
        if train_ds is None or len(train_ds) == 0:
            return _mk_diag(
                fold_row, "skip", "train_ds_empty",
                train_start=str(train_start), train_end=str(val_cut),
            )

        val_ds = make_tft_dataset_for_period(
            df_work,
            start_ts=val_cut,
            end_ts=train_end,
            cont_features=cont_features,
            cat_features=cat_features,
            encoder_len=encoder_len,
            decoder_len=decoder_len,  # ★1
            is_train=False,
            target_col=target_col,
        )
        if val_ds is None or len(val_ds) == 0:
            return _mk_diag(
                fold_row, "skip", "val_ds_empty",
                val_start=str(val_cut), val_end=str(train_end),
            )

        test_ds = make_tft_dataset_for_period(
            df_work,
            start_ts=test_start,
            end_ts=test_end,
            cont_features=cont_features,
            cat_features=cat_features,
            encoder_len=encoder_len,
            decoder_len=decoder_len,  # ★1
            is_train=False,
            target_col=target_col,
        )
        if test_ds is None or len(test_ds) == 0:
            return _mk_diag(
                fold_row, "skip", "test_ds_empty",
                test_start=str(test_start), test_end=str(test_end),
            )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

        # ============
        # 4) config
        # ============
        cfg = dict(base_config)
        cfg["encode_length"] = encoder_len
        cfg["seq_length"] = encoder_len + decoder_len  # 24+1
        cfg["time_varying_real_variables_encoder"] = len(cont_features)
        cfg["time_varying_real_variables_decoder"] = len(cont_features)
        cfg["time_varying_categoical_variables"]   = len(cat_features)

        vocab_sizes = []
        for col in cat_features:
            unique_vals = df_work.loc[train_start:train_end, col].dropna().unique()
            if len(unique_vals) == 0:
                vocab_sizes.append(1)
            else:
                vocab_sizes.append(int(np.max(unique_vals)) + 1)
        cfg["time_varying_embedding_vocab_sizes"] = vocab_sizes

        model = TFT(cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # ============
        # 5) pos_ratio（train_dsから）
        # ============
        y_full = []
        for s in getattr(train_ds, "sequences", []):
            if target_col not in s:
                continue
            t = np.array(s[target_col])
            if t.size == 0:
                continue
            y_full.append(float(np.ravel(t)[-1]))
        y_full = np.array(y_full, dtype=float)

        if y_full.size == 0:
            return _mk_diag(fold_row, "skip", "train_label_all_nan")
        uniq = np.unique(y_full)
        if uniq.size < 2:
            # 全部0 or 全部1
            return _mk_diag(
                fold_row, "skip", "train_label_one_class",
                uniq=uniq.tolist(), pos_ratio=float(y_full.mean()), n_train_seq=int(len(train_ds)),
            )

        pos_ratio = float(np.mean(y_full))
        if pos_ratio <= 0.0 or pos_ratio >= 1.0:
            print(f"[Fold {fold_row['fold_idx']}] pos_ratio={pos_ratio:.4f} degenerate → skip fold")
            return None

        alpha = 0.5
        gamma = 1.5
        print(f"[Fold {fold_row['fold_idx']}] thresholds={thresholds}")
        print(f"[Fold {fold_row['fold_idx']}] pos_ratio={pos_ratio:.4f}, alpha={alpha:.2f}, gamma={gamma}")

        # ============
        # 6) train loop (あなたの既存ロジックを維持)
        # ============
        best_val_loss = np.inf
        best_state = None
        patience = 5
        no_improve = 0

        for epoch in range(num_epochs):
            model.train()
            train_loss_sum = 0.0

            for batch in train_loader:
                enc_real = batch["encoder_real_input"].to(device)
                dec_real = batch["decoder_real_input"].to(device)

                enc_cat = batch["encoder_categorical_input"]
                if enc_cat.size(-1) > 0:
                    enc_cat = enc_cat.to(device)
                else:
                    enc_cat = torch.empty(enc_real.size(0), enc_real.size(1), 0, dtype=torch.int64, device=device)

                dec_cat = batch["decoder_categorical_input"]
                if dec_cat.size(-1) > 0:
                    dec_cat = dec_cat.to(device)
                else:
                    dec_cat = torch.empty(dec_real.size(0), dec_real.size(1), 0, dtype=torch.int64, device=device)

                y = batch["target"].to(device)  # (B,1)期待
                strat_list = batch.get("strategy_type", None)
                strategy_ids = batch["strategy_id"].to(device)

                if strat_list is not None:
                    if not isinstance(strat_list, (list, tuple)):
                        strat_list = list(strat_list)
                    strat_weights = [STRATEGY_WEIGHT_MAP.get(s, 1.0) for s in strat_list]
                    strat_weights = torch.tensor(strat_weights, dtype=torch.float32, device=device)
                else:
                    strat_weights = torch.ones(enc_real.size(0), dtype=torch.float32, device=device)

                optimizer.zero_grad()

                out, *_ = model(
                    x_enc_real=enc_real,
                    x_dec_real=dec_real,
                    x_enc_cat=enc_cat,
                    x_dec_cat=dec_cat,
                    strategy_ids=strategy_ids,
                )

                logits_last = out[:, -1, 0]
                y_last = y[:, -1].float().view(-1)

                loss_per_sample = binary_focal_loss_with_logits(
                    logits=logits_last,
                    targets=y_last,
                    alpha=alpha,
                    gamma=gamma,
                    reduction="none",
                ).view(-1)

                loss = (loss_per_sample * strat_weights).mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss_sum += float(loss.item())

            avg_train_loss = train_loss_sum / max(1, len(train_loader))

            # ---- val ----
            model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    enc_real = batch["encoder_real_input"].to(device)
                    dec_real = batch["decoder_real_input"].to(device)

                    enc_cat = batch["encoder_categorical_input"]
                    if enc_cat.size(-1) > 0:
                        enc_cat = enc_cat.to(device)
                    else:
                        enc_cat = torch.empty(enc_real.size(0), enc_real.size(1), 0, dtype=torch.int64, device=device)

                    dec_cat = batch["decoder_categorical_input"]
                    if dec_cat.size(-1) > 0:
                        dec_cat = dec_cat.to(device)
                    else:
                        dec_cat = torch.empty(dec_real.size(0), dec_real.size(1), 0, dtype=torch.int64, device=device)

                    y = batch["target"].to(device)

                    strat_list = batch.get("strategy_type", None)
                    strategy_ids = batch["strategy_id"].to(device)

                    if strat_list is not None:
                        if not isinstance(strat_list, (list, tuple)):
                            strat_list = list(strat_list)
                        strat_weights = [STRATEGY_WEIGHT_MAP.get(s, 1.0) for s in strat_list]
                        strat_weights = torch.tensor(strat_weights, dtype=torch.float32, device=device)
                    else:
                        strat_weights = torch.ones(enc_real.size(0), dtype=torch.float32, device=device)

                    out, *_ = model(
                        x_enc_real=enc_real,
                        x_dec_real=dec_real,
                        x_enc_cat=enc_cat,
                        x_dec_cat=dec_cat,
                        strategy_ids=strategy_ids,
                    )

                    logits_last = out[:, -1, 0]
                    y_last = y[:, -1].float().view(-1)

                    val_loss_per_sample = binary_focal_loss_with_logits(
                        logits=logits_last,
                        targets=y_last,
                        alpha=alpha,
                        gamma=gamma,
                        reduction="none",
                    ).view(-1)

                    val_loss = (val_loss_per_sample * strat_weights).mean()
                    val_loss_sum += float(val_loss.item())

            avg_val_loss = val_loss_sum / max(1, len(val_loader))
            print(
                f"[Fold {fold_row['fold_idx']}] "
                f"Epoch {epoch+1}/{num_epochs} "
                f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f}"
            )

            if avg_val_loss < best_val_loss - 1e-4:
                best_val_loss = avg_val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  EarlyStopping: {patience} epochs no improvement.")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # ============
        # 7) AUC eval（あなたの既存の eval_auc をそのまま使ってOK）
        # ============
        def eval_auc(dataloader, strategy_type_name: str | None = None):
            model.eval()
            all_logits, all_y, all_ts = [], [], []

            with torch.no_grad():
                for batch in dataloader:
                    enc_real = batch["encoder_real_input"].to(device)
                    dec_real = batch["decoder_real_input"].to(device)

                    enc_cat = batch["encoder_categorical_input"]
                    if enc_cat.size(-1) > 0:
                        enc_cat = enc_cat.to(device)
                    else:
                        enc_cat = torch.empty(enc_real.size(0), enc_real.size(1), 0, dtype=torch.int64, device=device)

                    dec_cat = batch["decoder_categorical_input"]
                    if dec_cat.size(-1) > 0:
                        dec_cat = dec_cat.to(device)
                    else:
                        dec_cat = torch.empty(dec_real.size(0), dec_real.size(1), 0, dtype=torch.int64, device=device)

                    y = batch["target"].to(device)
                    strategy_ids = batch["strategy_id"].to(device)

                    out, *_ = model(
                        x_enc_real=enc_real,
                        x_dec_real=dec_real,
                        x_enc_cat=enc_cat,
                        x_dec_cat=dec_cat,
                        strategy_ids=strategy_ids,
                    )

                    logits_last = out[:, -1, 0]
                    y_last = y[:, -1].view(-1)

                    all_logits.append(logits_last.detach().cpu().numpy())
                    all_y.append(y_last.detach().cpu().numpy())
                    all_ts.extend(batch["sequence_end_time"])

            if len(all_logits) == 0:
                return (np.nan, np.nan, np.nan, np.nan)

            logits = np.concatenate(all_logits)
            y_true = np.concatenate(all_y)

            if np.unique(y_true).size < 2:
                auc_all = np.nan
                auc_all_inv = np.nan
            else:
                probs = 1 / (1 + np.exp(-logits))
                auc_all = roc_auc_score(y_true, probs)
                auc_all_inv = roc_auc_score(y_true, 1.0 - probs)

            if strategy_type_name is None:
                return (auc_all, auc_all_inv, np.nan, np.nan)

            ts_dt = pd.to_datetime(all_ts)
            df_pred = pd.DataFrame({"ts": ts_dt, "logits": logits, "y": y_true}).set_index("ts")
            df_join = df_pred.join(df_work[["strategy_type"]], how="left")  # ★df_work（fold内加工後）

            df_strat = df_join[df_join["strategy_type"] == strategy_type_name]
            if df_strat.empty or np.unique(df_strat["y"].values).size < 2:
                return (auc_all, auc_all_inv, np.nan, np.nan)

            probs_strat = 1 / (1 + np.exp(-df_strat["logits"].values))
            auc_strat = roc_auc_score(df_strat["y"].values, probs_strat)
            auc_strat_inv = roc_auc_score(df_strat["y"].values, 1.0 - probs_strat)
            return (auc_all, auc_all_inv, auc_strat, auc_strat_inv)

        auc_train_all, auc_train_all_inv, auc_train_strat, auc_train_strat_inv = eval_auc(
            train_loader, strategy_type_name=eval_strategy_type
        )
        auc_test_all, auc_test_all_inv, auc_test_strat, auc_test_strat_inv = eval_auc(
            test_loader, strategy_type_name=eval_strategy_type
        )

        return {
            "fold_idx": fold_row["fold_idx"],
            "status": "ok",
            "skip_reason": "",
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "thresholds": thresholds,
            "auc_train": auc_train_all,
            "auc_test": auc_test_all,
            "auc_train2": auc_train_all_inv,
            "auc_test2": auc_test_all_inv,
            "auc_train_strategy": auc_train_strat,
            "auc_test_strategy": auc_test_strat,
            "auc_train_strategy2": auc_train_strat_inv,
            "auc_test_strategy2": auc_test_strat_inv,
        }
    except Exception as e:
        return _mk_diag(
            fold_row, "error", "exception",
            error_type=type(e).__name__,
            error_msg=str(e),
            traceback=traceback.format_exc(limit=3),
        )

def run_tft_wfa_with_diagnostics(
    df_diag: pd.DataFrame,
    fold_meta_df: pd.DataFrame,
    cont_features: list,
    cat_features_base: list,
    base_config: dict,
    regime_col: str = "regime_v3_compressed",
    add_regime_embedding: bool = True,
    encoder_len: int = 24,
    decoder_len: int = 1,
    batch_size: int = 64,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    device: str | torch.device = "cuda",
    label_col: str = "target",
    eval_strategy_type: str | None = None,
    verbose: bool = True,
):
    """
    返り値:
      - results_df: status='ok' の fold のみ（AUC等）
      - diag_df   : 全fold（ok/skip/error）＋スキップ理由
      - summary_df: skip_reason 集計
    """

    # ----------------------------
    # 0) 前処理（あなたの元コード踏襲）
    # ----------------------------
    df = df_diag.copy()
    if add_regime_embedding:
        df = add_regime_id(df, regime_col=regime_col, id_col="regime_id")
        cat_features = cat_features_base + ["regime_id"]
    else:
        cat_features = list(cat_features_base)

    df_base = prepare_tft_base_frame(df)

    # label_col + valid_label で絞る
    df_base = df_base[(df_base[label_col].isin([0, 1])) & (df_base["valid_label"])].copy()

    # 列存在チェック
    for col in cont_features + cat_features + [label_col]:
        if col not in df_base.columns:
            raise ValueError(f"{col} が df_base.columns に存在しません")

    # ----------------------------
    # 1) fold loop（全部回収）
    # ----------------------------
    all_records = []
    for _, row in fold_meta_df.iterrows():
        fold_idx = int(row["fold_idx"])
        if verbose:
            print(f"\n========== TFT WFA Fold {fold_idx} ==========")

        res = train_one_tft_fold(
            df_base=df_base,
            fold_row=row,
            cont_features=cont_features,
            cat_features=cat_features,
            base_config=base_config,
            encoder_len=encoder_len,
            decoder_len=decoder_len,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
            target_col=label_col,
            eval_strategy_type=eval_strategy_type,
        )

        # train_one_tft_fold を上で改修した前提（dictが返る）
        if res is None:
            # 念のための保険（旧挙動が残っていても回収する）
            all_records.append({
                "fold_idx": fold_idx,
                "status": "skip",
                "skip_reason": "returned_none",
            })
            if verbose:
                print(f"[Fold {fold_idx}] SKIP: returned_none")
            continue

        # fold meta も付けておくと診断が楽
        for k in ["train_start", "train_end", "test_start", "test_end"]:
            if k in row:
                res.setdefault(k, str(pd.to_datetime(row[k])))

        all_records.append(res)

        if verbose and res.get("status") == "ok":
            print(
                f"[Fold {fold_idx}] AUC train={res.get('auc_train', np.nan):.3f}, "
                f"test={res.get('auc_test', np.nan):.3f}"
            )
        elif verbose:
            print(f"[Fold {fold_idx}] {res.get('status')} reason={res.get('skip_reason')}")

    diag_df = pd.DataFrame(all_records)

    # ----------------------------
    # 2) 表示用の集計
    # ----------------------------
    if diag_df.empty:
        summary_df = pd.DataFrame()
        results_df = pd.DataFrame()
        return results_df, diag_df, summary_df

    # ok のみ
    results_df = diag_df[diag_df["status"] == "ok"].copy()

    # skip/error の集計
    summary_df = (
        diag_df.assign(skip_reason=diag_df["skip_reason"].fillna(""))
              .groupby(["status", "skip_reason"], dropna=False)
              .size()
              .reset_index(name="n_folds")
              .sort_values(["status", "n_folds"], ascending=[True, False])
              .reset_index(drop=True)
    )

    # 追加で “どのfoldが落ちたか” を見やすく
    if "skip_reason" in diag_df.columns:
        diag_df = diag_df.sort_values(["status", "skip_reason", "fold_idx"]).reset_index(drop=True)

    return results_df, diag_df, summary_df

from torch.utils.data import DataLoader

import torch

import numpy as np

from sklearn.metrics import roc_auc_score

import pytorch_lightning as pl

from tft_model import TFTBinaryClassifier

def tft_trainer_fn(
    train_loader,
    val_loader,
    base_config: dict,
):
    """
    1 fold 分の TFT 学習＋ AUC(train,val) を返すトレーナー関数。

    Parameters
    ----------
    train_loader : DataLoader
        CryptoBinaryDataset などを使った学習用ローダ
    val_loader : DataLoader
        検証用ローダ（同形式）
    base_config : dict
        既存の TFT 設定。例：
        {
            "model": {...  # モデル用ハイパーパラメータ},
            "max_epochs": 15,
            "gradient_clip_val": 1.0,
            ...
        }

    Returns
    -------
    auc_train : float
    auc_val : float
    """

    # -------------------------
    # 1. 入力次元をバッチから自動取得
    # -------------------------
    first_batch = next(iter(train_loader))
    x_cont = first_batch["x_cont"]          # (B, seq_len, n_cont)
    x_cat = first_batch.get("x_cat", None)  # ない場合は None
    regime_id = first_batch.get("regime_id", None)

    n_cont = x_cont.shape[-1]
    n_cat = x_cat.shape[-1] if x_cat is not None else 0
    n_regimes = int(regime_id.max().item() + 1) if regime_id is not None else 0

    # -------------------------
    # 2. モデルインスタンス生成
    # -------------------------
    model_kwargs = base_config.get("model", {})
    model = TFTBinaryClassifier(
        n_cont=n_cont,
        n_cat=n_cat,
        n_regimes=n_regimes,
        **model_kwargs,
    )

    # -------------------------
    # 3. Trainer 準備
    # -------------------------
    max_epochs = base_config.get("max_epochs", 15)
    grad_clip = base_config.get("gradient_clip_val", 1.0)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=grad_clip,
        enable_checkpointing=False,
        logger=False,
        deterministic=True,
    )

    # -------------------------
    # 4. 学習
    # -------------------------
    trainer.fit(model, train_loader, val_loader)

    # -------------------------
    # 5. AUC 計算ユーティリティ
    # -------------------------
    def collect_probs_and_labels(loader):
        model.eval()
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                # バッチから y とモデル入力を取り出し
                y = batch["y"].to(model.device)  # (B,)
                # forward は batch dict をそのまま渡す想定
                # もし forward(self, x_cont, x_cat, regime_id, ...) ならその形に書き換えてください
                logits = model(batch)            # (B,) のロジットを返す想定
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                all_probs.append(probs)
                all_targets.append(y.detach().cpu().numpy())

        all_probs = np.concatenate(all_probs)
        all_targets = np.concatenate(all_targets)
        return all_probs, all_targets

    # train / val で AUC
    prob_train, y_train = collect_probs_and_labels(train_loader)
    prob_val, y_val = collect_probs_and_labels(val_loader)

    auc_train = roc_auc_score(y_train, prob_train)
    auc_val = roc_auc_score(y_val, prob_val)

    print(f"[TFT] AUC(train)={auc_train:.4f}, AUC(val)={auc_val:.4f}")

    return auc_train, auc_val

def regime_aware_wfa_with_tft(
    df_diag: pd.DataFrame,
    base_config: dict,
    regime_col: str = "regime_v3_compressed",
    date_col: str = "date",
    valid_col: str = "valid_label",
    strategy_type=None,
    cont_features: list | None = None,
    cat_features: list | None = None,
    train_span_days: int = 270,
    test_span_days: int = 60,
    min_train_rows: int = 110,
    min_test_rows: int = 40,
    encoder_len: int = 24,
    decoder_len: int = 1,
    batch_size: int = 64,
    num_epochs: int = 1,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-4,
    device: str | torch.device = "cuda",
    label_col: str = "label",
    eval_strategy_type: str | None = None,   # ★ 追加
):
    """
    RF 版 regime_aware_wfa_auc_with_rf とほぼ同じインターフェースで
    TFT WFA を実行するラッパー。
    """

    df = df_diag.copy()

    if date_col == df.index.name:
        df = df.reset_index(names=[date_col])

    mask = df[label_col].isin([0, 1])
    if valid_col in df.columns:
        mask &= df[valid_col].astype(bool)
    if strategy_type is not None and "strategy_type" in df.columns:
        mask &= (df["strategy_type"] == strategy_type)

    df = df[mask].copy()

    # label_col を "target" にマップする処理
    _label_col_for_folds = label_col # Default to original label_col
    if label_col != "target":
        if label_col in df.columns:
            df = df.rename(columns={label_col: "target"})
            _label_col_for_folds = "target"
        elif "target" in df.columns:
            print(f"Warning: '{label_col}' not found, but 'target' column exists. Using 'target' as label column.")
            _label_col_for_folds = "target"
        else:
            raise ValueError(f"{label_col} も 'target' も df_diag にありません")
    else: # label_col is already "target"
        if label_col not in df.columns:
             raise ValueError(f"{label_col} が df_diag にありません")


    # 特徴量リストのデフォルト
    if cont_features is None:
        cont_features = CONT_FEATURES_ALL  # あなたの最終版のリスト
    if cat_features is None:
        cat_features = CAT_FEATURES_ALL    # あなたの最終版のリスト

    # ① RF と共通の fold 生成
    feature_cols = cont_features + cat_features

    fold_meta_df, diag_df = make_regime_aware_wfa_folds_debug(
        df_diag=df_diag_v3,
        feature_cols=feature_cols,
        label_col="target",      # 実列名に合わせる
        date_col="date",         # 無ければ DatetimeIndex から作る
        regime_col="regime_id",
        valid_col="valid_label",
        train_span_days=270,
        test_span_days=30,
        min_train_rows=110,
        min_test_rows=40,
        require_both_classes_train=True,
        require_both_classes_test=False,
        regime_mode="none",    # ←重要
        auto_expand_test=True,   # ←重要
        max_expand_test_days=120,
        expand_step_days=7,
        return_df=False,
    )

    # ② その fold 定義を使って TFT WFA を実行
    results_df, diag_df, summary_df = run_tft_wfa_with_diagnostics(
        df_diag=df_diag_v3,
        fold_meta_df=fold_meta_df,
        cont_features=CONT_FEATURES_FOR_AUC,
        cat_features_base=CAT_FEATURES_FOR_AUC,
        base_config=base_config,
        encoder_len=24,
        decoder_len=1,
        label_col="target",
        device="cuda",
        verbose=True,
    )
    # tft_results = run_tft_wfa(
    #     df_diag=df,
    #     fold_meta_df=fold_meta_df,
    #     cont_features=cont_features,
    #     cat_features_base=cat_features,
    #     base_config=base_config,
    #     regime_col=regime_col,
    #     add_regime_embedding=True,
    #     encoder_len=encoder_len,
    #     decoder_len=decoder_len,
    #     batch_size=batch_size,
    #     num_epochs=num_epochs,
    #     learning_rate=learning_rate,
    #     weight_decay=weight_decay,
    #     device=device,
    #     label_col=_label_col_for_folds,
    #     eval_strategy_type=eval_strategy_type,   # ★ ここで渡す
    # )

    return results_df, diag_df, summary_df, fold_meta_df

import numpy as np

import pandas as pd

import torch

from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score

from tft_model import TFT  # Ensure TFT is available

import numpy as np

import pandas as pd

from sklearn.metrics import roc_auc_score

import lightgbm as lgb

def _mk_diag(fold_row, status: str, reason: str, **kwargs) -> dict:
    out = {
        "fold_idx": int(fold_row["fold_idx"]),
        "status": status,      # ok / skip / error
        "skip_reason": reason,
    }
    out.update(kwargs)
    return out

def run_lgbm_one_fold_with_diagnostics(
    *,
    df_base: pd.DataFrame,
    fold_row: pd.Series | dict,
    cont_features: list[str],
    cat_features: list[str],
    # TFTと合わせる（TFTは decoder_len=1 で point forecast）
    target_col: str = "target",
    eval_strategy_type: str | None = None,

    # ===== TFTと同じ fold内ラベル生成パラメータ =====
    future_ret_col: str = "log_return_h6",
    strategy_id_col: str = "strategy_type_id",
    target_positive_rate: float = 0.20,
    pos_rate_range: float = 0.30,

    # ===== TFTと同じ fold内正規化 =====
    fold_normalize: bool = True,

    # ===== LGBM params =====
    lgbm_params: dict | None = None,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 100,
    verbose_eval: int = 200,
) -> dict | None:
    """
    TFT側の fold と同じ train/val/test 期間で、LGBMを学習。
    - 閾値学習は leakage 回避のため train_in（= val_cutまで）で実施（TFTと同じ）
    - 正規化も train_in で fit（TFTと同じ）
    - val は train_end の直前 7日分（TFTと同じ val_span_hours=24*7）
    """
    try:
        # ---------
        # fold range
        # ---------
        train_start = pd.to_datetime(fold_row["train_start"])
        train_end   = pd.to_datetime(fold_row["train_end"])
        test_start  = pd.to_datetime(fold_row["test_start"])
        test_end    = pd.to_datetime(fold_row["test_end"])

        # TFT側: val_span_hours = 24*7
        val_span_hours = 24 * 7
        val_cut = train_end - pd.Timedelta(hours=val_span_hours)

        # ---------
        # validate columns
        # ---------
        for col in cont_features + cat_features:
            if col not in df_base.columns:
                raise ValueError(f"Missing feature col: {col}")

        if future_ret_col not in df_base.columns:
            raise ValueError(f"Missing future_ret_col: {future_ret_col}")
        if strategy_id_col not in df_base.columns:
            raise ValueError(f"Missing strategy_id_col: {strategy_id_col}")

        # ---------
        # 1) fold内 train-fit で閾値学習 → ラベル生成（TFTと同じ流れ）
        # ---------
        df_work = df_base.copy()

        # train-fitは leakage回避のため「train_in（=val_cutまで）」で閾値学習
        train_fit_mask = (df_work.index >= train_start) & (df_work.index <= val_cut)
        train_fit_idx = np.where(train_fit_mask)[0]

        fr_all = df_work[future_ret_col].to_numpy(dtype=np.float32)
        sid_all = df_work[strategy_id_col].to_numpy()

        finite_train_fit_idx = train_fit_idx[np.isfinite(fr_all[train_fit_idx])]
        if finite_train_fit_idx.size < 100:
            return _mk_diag(fold_row, "skip", "too_few_finite_future_ret_in_train_fit",
                            n_finite=int(finite_train_fit_idx.size))

        # TFT側の戦略ID map（3cat）に合わせる
        strategy_id_map = {0: "Downtrend", 1: "Range", 2: "Uptrend"}

        thresholds = fit_thresholds_on_train_3cat(
            future_ret_full=fr_all,
            strategy_type_id_full=sid_all,
            train_ids=finite_train_fit_idx,
            pos_rate_up=target_positive_rate,
            pos_rate_down=target_positive_rate,
            pos_rate_range=pos_rate_range,
            strategy_id_map=strategy_id_map,
        )

        labels_all = apply_thresholds_3cat(
            future_ret_full=fr_all,
            strategy_type_id_full=sid_all,
            thresholds=thresholds,
            strategy_id_map=strategy_id_map,
        ).astype(np.float32)

        df_work[target_col] = labels_all

        # ---------
        # 2) fold内正規化（train_inでfit→全体へ適用）（TFTと同じ）
        # ---------
        if fold_normalize:
            mean, std = _fit_normalizer_on_train(
                df_work, cont_features, train_start=train_start, train_end=val_cut
            )
            df_work = _apply_normalizer(df_work, cont_features, mean, std)

        # ---------
        # 3) train / val / test の行データを作る
        #    ※TFTはsequenceだが、baselineは「その時刻の特徴量」だけを使う
        # ---------
        def _slice(start, end):
            return df_work.loc[start:end].copy()

        df_train = _slice(train_start, val_cut)
        df_val   = _slice(val_cut, train_end)
        df_test  = _slice(test_start, test_end)

        # 空 or one-class はスキップ（TFT同様に安全側）
        if df_train.empty or df_val.empty or df_test.empty:
            return _mk_diag(
                fold_row, "skip", "empty_split",
                n_train=int(len(df_train)), n_val=int(len(df_val)), n_test=int(len(df_test))
            )

        y_train = df_train[target_col].to_numpy()
        y_val   = df_val[target_col].to_numpy()
        y_test  = df_test[target_col].to_numpy()

        if np.unique(y_train).size < 2:
            return _mk_diag(fold_row, "skip", "train_label_one_class",
                            uniq=np.unique(y_train).tolist(), pos_ratio=float(np.mean(y_train)))
        if np.unique(y_val).size < 2:
            return _mk_diag(fold_row, "skip", "val_label_one_class",
                            uniq=np.unique(y_val).tolist(), pos_ratio=float(np.mean(y_val)))
        if np.unique(y_test).size < 2:
            return _mk_diag(fold_row, "skip", "test_label_one_class",
                            uniq=np.unique(y_test).tolist(), pos_ratio=float(np.mean(y_test)))

        # 特徴量行列
        feature_cols = cont_features + cat_features
        X_train = df_train[feature_cols]
        X_val   = df_val[feature_cols]
        X_test  = df_test[feature_cols]

        # cat_features を LightGBM に伝える（intのままでもOK）
        cat_feature_names = [c for c in cat_features if c in X_train.columns]

        # ---------
        # 4) LightGBM 学習
        # ---------
        if lgbm_params is None:
            lgbm_params = dict(
                objective="binary",
                metric="auc",
                learning_rate=0.03,
                num_leaves=63,
                min_data_in_leaf=50,
                feature_fraction=0.9,
                bagging_fraction=0.9,
                bagging_freq=1,
                lambda_l2=1.0,
                verbose=-1,
            )

        dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_feature_names, free_raw_data=False)
        dvalid = lgb.Dataset(X_val,   label=y_val,   categorical_feature=cat_feature_names, free_raw_data=False)

        model = lgb.train(
            params=lgbm_params,
            train_set=dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=bool(verbose_eval)),
                lgb.log_evaluation(period=verbose_eval) if verbose_eval else lgb.log_evaluation(period=0),
            ],
        )

        # ---------
        # 5) AUC計算（TFTと同じく全体 + 戦略別も返す）
        # ---------
        p_train = model.predict(X_train, num_iteration=model.best_iteration)
        p_test  = model.predict(X_test,  num_iteration=model.best_iteration)

        auc_train = roc_auc_score(y_train, p_train)
        auc_test  = roc_auc_score(y_test,  p_test)

        # 戦略別AUC（eval_strategy_type が指定されている場合）
        def _auc_by_strategy(df_part: pd.DataFrame, probs: np.ndarray) -> float | None:
            if eval_strategy_type is None:
                return None
            if "strategy_type" not in df_part.columns:
                return None
            mask = (df_part["strategy_type"] == eval_strategy_type)
            if not mask.any():
                return None
            yy = df_part.loc[mask, target_col].to_numpy()
            pp = probs[mask.to_numpy()]
            if np.unique(yy).size < 2:
                return None
            return float(roc_auc_score(yy, pp))

        auc_train_strat = _auc_by_strategy(df_train, p_train)
        auc_test_strat  = _auc_by_strategy(df_test,  p_test)

        return {
            "fold_idx": int(fold_row["fold_idx"]),
            "status": "ok",
            "skip_reason": "",
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "val_cut": val_cut,
            "thresholds": thresholds,
            "best_iter": int(model.best_iteration or 0),
            "auc_train": float(auc_train),
            "auc_test": float(auc_test),
            "auc_train_strategy": auc_train_strat if auc_train_strat is None else float(auc_train_strat),
            "auc_test_strategy": auc_test_strat if auc_test_strat is None else float(auc_test_strat),
            "n_train_rows": int(len(df_train)),
            "n_val_rows": int(len(df_val)),
            "n_test_rows": int(len(df_test)),
            "pos_train": float(np.mean(y_train)),
            "pos_test": float(np.mean(y_test)),
        }

    except Exception as e:
        return _mk_diag(
            fold_row, "error", "exception",
            error_type=type(e).__name__,
            error_msg=str(e),
        )

def run_lgbm_wfa_with_diagnostics(
    *,
    df_diag: pd.DataFrame,
    fold_meta_df: pd.DataFrame,
    cont_features: list[str],
    cat_features_base: list[str],
    regime_col: str = "regime_v3_compressed",
    add_regime_embedding: bool = True,   # TFTと同じ引数形にする（LGBMでは regime をcatに足すだけ）
    label_col: str = "target",
    eval_strategy_type: str | None = None,

    # TFT互換：fold内ラベル生成
    future_ret_col: str = "log_return_h6",
    strategy_id_col: str = "strategy_type_id",
    target_positive_rate: float = 0.20,
    pos_rate_range: float = 0.30,

    fold_normalize: bool = True,

    # LGBM
    lgbm_params: dict | None = None,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 100,
    verbose_eval: int = 200,
) -> pd.DataFrame:
    """
    TFTの run_tft_wfa_with_diagnostics と同じ fold_meta_df を入力して回す。
    df_diag は TFT側で使っている元データ（index=ts, 特徴量列を含む）を想定。
    """
    # LGBMで使うcat_features
    cat_features = list(cat_features_base)
    if add_regime_embedding and (regime_col in df_diag.columns) and (regime_col not in cat_features):
        cat_features.append(regime_col)

    # df_base は index を時系列にしておく（TFT側前提に合わせる）
    df_base = df_diag.copy()
    if not isinstance(df_base.index, pd.DatetimeIndex):
        # date列があれば index化（あなたの既存コードに合わせて必要なら調整）
        if "date" in df_base.columns:
            df_base["date"] = pd.to_datetime(df_base["date"])
            df_base = df_base.set_index("date")
        else:
            raise ValueError("df_diag must have DatetimeIndex or a 'date' column.")

    df_base = df_base.sort_index()

    results = []
    for _, row in fold_meta_df.iterrows():
        out = run_lgbm_one_fold_with_diagnostics(
            df_base=df_base,
            fold_row=row,
            cont_features=cont_features,
            cat_features=cat_features,
            target_col=label_col,
            eval_strategy_type=eval_strategy_type,
            future_ret_col=future_ret_col,
            strategy_id_col=strategy_id_col,
            target_positive_rate=target_positive_rate,
            pos_rate_range=pos_rate_range,
            fold_normalize=fold_normalize,
            lgbm_params=lgbm_params,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
        )
        if out is None:
            # TFT側に合わせて None はスキップ扱いに寄せる
            out = _mk_diag(row, "skip", "returned_none")
        results.append(out)

    return pd.DataFrame(results)

import numpy as np

import pandas as pd

def _first_existing_col(df: pd.DataFrame, candidates: list[str], *, required=True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of these columns exist: {candidates}")
    return None

def fold_by_fold_tft_vs_lgbm(
    *,
    tft_results_df: pd.DataFrame,
    lgbm_results_df: pd.DataFrame,
    fold_meta_df: pd.DataFrame | None = None,
    # 列名ゆれ吸収
    auc_candidates: list[str] = None,
    iter_candidates: list[str] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    TFT結果とLGBM結果を fold_idx で突き合わせて比較表を作成。
    返り値: (compare_df, summary_dict)
    """
    auc_candidates = auc_candidates or ["auc_test", "test_auc", "AUC_test", "auc"]
    iter_candidates = iter_candidates or ["best_iter", "best_iteration", "best_step", "n_estimators"]

    if "fold_idx" not in tft_results_df.columns:
        raise KeyError("tft_results_df must contain 'fold_idx'")
    if "fold_idx" not in lgbm_results_df.columns:
        raise KeyError("lgbm_results_df must contain 'fold_idx'")

    # AUC列
    tft_auc_col = _first_existing_col(tft_results_df, auc_candidates, required=True)
    lgbm_auc_col = _first_existing_col(lgbm_results_df, auc_candidates, required=True)

    # iter列（任意）
    tft_iter_col = _first_existing_col(tft_results_df, iter_candidates, required=False)
    lgbm_iter_col = _first_existing_col(lgbm_results_df, iter_candidates, required=False)

    # 参考で残したい列
    keep_tft = ["fold_idx", tft_auc_col]
    keep_lgbm = ["fold_idx", lgbm_auc_col]

    for c in ["status", "skip_reason", "error_type", "error_msg", "pos_test", "pos_train"]:
        if c in tft_results_df.columns:
            keep_tft.append(c)
        if c in lgbm_results_df.columns:
            keep_lgbm.append(c)

    if tft_iter_col is not None:
        keep_tft.append(tft_iter_col)
    if lgbm_iter_col is not None:
        keep_lgbm.append(lgbm_iter_col)

    tft_small = tft_results_df[keep_tft].copy()
    lgbm_small = lgbm_results_df[keep_lgbm].copy()

    # リネーム
    tft_small = tft_small.rename(columns={
        tft_auc_col: "tft_auc",
        (tft_iter_col or "___none___"): "tft_best_iter"
    })
    lgbm_small = lgbm_small.rename(columns={
        lgbm_auc_col: "lgbm_auc",
        (lgbm_iter_col or "___none___"): "lgbm_best_iter"
    })

    # iter列が無かった場合でも列を作る
    if "tft_best_iter" not in tft_small.columns:
        tft_small["tft_best_iter"] = np.nan
    if "lgbm_best_iter" not in lgbm_small.columns:
        lgbm_small["lgbm_best_iter"] = np.nan

    # マージ
    comp = pd.merge(tft_small, lgbm_small, on="fold_idx", how="outer", suffixes=("_tft", "_lgbm"))

    # fold期間も付けたいなら
    if fold_meta_df is not None and "fold_idx" in fold_meta_df.columns:
        meta_cols = [c for c in ["fold_idx","train_start","train_end","test_start","test_end","n_train","n_test"] if c in fold_meta_df.columns]
        comp = pd.merge(fold_meta_df[meta_cols], comp, on="fold_idx", how="right")

    # ΔAUC・勝敗
    comp["delta_auc"] = comp["tft_auc"] - comp["lgbm_auc"]
    comp["tft_win"] = np.where(comp[["tft_auc","lgbm_auc"]].notna().all(axis=1), comp["delta_auc"] > 0, np.nan)

    # pos_test 差（ある場合）
    tft_pos_test_col = _first_existing_col(comp, ["pos_test_tft", "pos_test"], required=False)
    lgbm_pos_test_col = _first_existing_col(comp, ["pos_test_lgbm", "pos_test"], required=False)

    # 注意: suffixの付き方で pos_test_tft / pos_test_lgbm になっているはず
    if "pos_test_tft" in comp.columns and "pos_test_lgbm" in comp.columns:
        comp["delta_pos_test"] = comp["pos_test_tft"] - comp["pos_test_lgbm"]
    else:
        comp["delta_pos_test"] = np.nan

    # 見やすい列順
    front = [c for c in ["fold_idx","train_start","train_end","test_start","test_end"] if c in comp.columns]
    core = ["tft_auc","lgbm_auc","delta_auc","tft_win","tft_best_iter","lgbm_best_iter","delta_pos_test"]
    rest = [c for c in comp.columns if c not in front + core]
    comp = comp[front + core + rest].sort_values("fold_idx")

    # summary
    valid = comp[["tft_auc","lgbm_auc"]].notna().all(axis=1)
    v = comp.loc[valid].copy()

    summary = {}
    if len(v) > 0:
        summary = {
            "n_folds_compared": int(len(v)),
            "tft_auc_mean": float(v["tft_auc"].mean()),
            "lgbm_auc_mean": float(v["lgbm_auc"].mean()),
            "delta_auc_mean": float(v["delta_auc"].mean()),
            "delta_auc_median": float(v["delta_auc"].median()),
            "tft_win_rate": float((v["delta_auc"] > 0).mean()),
            "best_fold": int(v.sort_values("delta_auc", ascending=False).iloc[0]["fold_idx"]),
            "worst_fold": int(v.sort_values("delta_auc", ascending=True).iloc[0]["fold_idx"]),
        }

    return comp, summary

import json

from pathlib import Path

from typing import Dict, List, Optional, Any

import pandas as pd

def make_run_dir(root: str | Path = "./artifacts", run_name: Optional[str] = None) -> Path:
    """
    artifacts/run_YYYYMMDD_HHMMSS のように保存先を作る。
    run_name を指定すれば artifacts/run_name/ に保存。
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    if run_name is None:
        run_name = pd.Timestamp.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def _can_use_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        return False

def save_df(df: pd.DataFrame, path_base: Path, *, index: bool = True) -> Dict[str, str]:
    """
    path_base から .parquet or .csv を作成。
    返り値: {"format": "...", "path": "..."}
    """
    path_base.parent.mkdir(parents=True, exist_ok=True)

    df_to_save = df.copy()
    if 'thresholds' in df_to_save.columns:
        # Ensure all elements are explicitly JSON strings or 'null' string representation
        df_to_save['thresholds'] = df_to_save['thresholds'].apply(
            lambda x: json.dumps(x) if isinstance(x, dict) else json.dumps(None) # Convert non-dict to JSON null
        ).astype(str) # Ensure the entire series is of string dtype

    if _can_use_parquet():
        path = path_base.with_suffix(".parquet")
        df_to_save.to_parquet(path, index=index)
        return {"format": "parquet", "path": str(path)}
    else:
        path = path_base.with_suffix(".csv")
        df_to_save.to_csv(path, index=index)
        return {"format": "csv", "path": str(path)}

def load_df(path_base: Path) -> pd.DataFrame:
    """
    path_base(.parquet/.csv) を自動判別して読み込む。
    """
    pq = path_base.with_suffix(".parquet")
    csv = path_base.with_suffix(".csv")

    df = None
    if pq.exists():
        df = pd.read_parquet(pq)
    elif csv.exists():
        df = pd.read_csv(csv)

    if df is not None and 'thresholds' in df.columns:
        # Convert 'thresholds' column back from JSON string to dictionary
        df['thresholds'] = df['thresholds'].apply(lambda x: json.loads(x) if pd.notna(x) and x != 'null' else None)

    if df is None:
        raise FileNotFoundError(f"Neither {pq} nor {csv} exists.")

    return df

def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)

def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def validate_fold_meta_df(fold_meta_df: pd.DataFrame) -> None:
    req = ["fold_idx", "train_start", "train_end", "test_start", "test_end"]
    missing = [c for c in req if c not in fold_meta_df.columns]
    if missing:
        raise ValueError(f"fold_meta_df missing columns: {missing}")

    if fold_meta_df["fold_idx"].duplicated().any():
        dups = fold_meta_df.loc[fold_meta_df["fold_idx"].duplicated(), "fold_idx"].tolist()
        raise ValueError(f"fold_meta_df fold_idx duplicated: {dups[:10]} ...")

    # datetime化（timezoneが混ざる場合があるので安全に）
    for c in ["train_start", "train_end", "test_start", "test_end"]:
        fold_meta_df[c] = pd.to_datetime(fold_meta_df[c], utc=True)

def validate_comp_df(comp_df: pd.DataFrame) -> None:
    req = ["fold_idx", "tft_auc", "lgbm_auc", "delta_auc", "tft_win"]
    missing = [c for c in req if c not in comp_df.columns]
    if missing:
        raise ValueError(f"comp_df missing columns: {missing}")

def validate_df_diag_v3(
    df_diag_v3: pd.DataFrame,
    *,
    date_col: str = "date",
    label_col: str = "target",
    strategy_id_col: str = "strategy_type_id",
    regime_col: Optional[str] = None,
) -> None:
    req = [date_col, label_col, strategy_id_col]
    if regime_col is not None:
        req.append(regime_col)
    missing = [c for c in req if c not in df_diag_v3.columns]
    if missing:
        raise ValueError(f"df_diag_v3 missing columns: {missing}")

    # dateはUTCに寄せる（WFAで型不一致を起こしがち）
    df_diag_v3[date_col] = pd.to_datetime(df_diag_v3[date_col], utc=True)

def save_artifacts_for_reuse(
    *,
    run_dir: Path,
    df_diag_v3: pd.DataFrame,
    fold_meta_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    tft_results_df: Optional[pd.DataFrame] = None,  # Add tft_results_df
    lgbm_results_df: Optional[pd.DataFrame] = None, # Add lgbm_results_df
    # 追跡用メタ
    cont_features: List[str],
    cat_features: List[str],
    base_config: dict,
    extra_meta: Optional[dict] = None,
    # 列名情報
    date_col: str = "date",
    label_col: str = "target",
    strategy_id_col: str = "strategy_type_id",
    regime_col: str = "regime_v3_compressed",
) -> dict:
    """
    artifactsを run_dir に保存し、manifest（保存パス一覧）を返す。
    """
    # validation（保存前に最低限の破損チェック）
    validate_df_diag_v3(df_diag_v3, date_col=date_col, label_col=label_col, strategy_id_col=strategy_id_col, regime_col=regime_col)
    validate_fold_meta_df(fold_meta_df.copy())
    validate_comp_df(comp_df)

    manifest = {"run_dir": str(run_dir), "files": {}}

    # DataFrames
    manifest["files"]["df_diag_v3"] = save_df(df_diag_v3, run_dir / "df_diag_v3", index=False)
    manifest["files"]["fold_meta_df"] = save_df(fold_meta_df, run_dir / "fold_meta_df", index=False)
    manifest["files"]["comp_df"] = save_df(comp_df, run_dir / "comp_df", index=False)

    if tft_results_df is not None:
        manifest["files"]["tft_results_df"] = save_df(tft_results_df, run_dir / "tft_results_df", index=False)
    if lgbm_results_df is not None:
        manifest["files"]["lgbm_results_df"] = save_df(lgbm_results_df, run_dir / "lgbm_results_df", index=False)

    # metadata
    meta = {
        "date_col": date_col,
        "label_col": label_col,
        "strategy_id_col": strategy_id_col,
        "regime_col": regime_col,
        "cont_features": cont_features,
        "cat_features": cat_features,
        "base_config": base_config,
    }
    if extra_meta:
        meta.update(extra_meta)

    save_json(meta, run_dir / "meta.json")
    manifest["files"]["meta_json"] = {"format": "json", "path": str(run_dir / "meta.json")}

    # manifest itself
    save_json(manifest, run_dir / "manifest.json")
    manifest["files"]["manifest_json"] = {"format": "json", "path": str(run_dir / "manifest.json")}

    return manifest

def load_artifacts_for_reuse(
    run_dir: str | Path,
) -> dict:
    """
    run_dir から df_diag_v3 / fold_meta_df / comp_df / meta を読み込み、整合性チェックして返す。
    """
    run_dir = Path(run_dir)
    meta = load_json(run_dir / "meta.json")

    df_diag_v3 = load_df(run_dir / "df_diag_v3")
    fold_meta_df = load_df(run_dir / "fold_meta_df")
    comp_df = load_df(run_dir / "comp_df")

    tft_results_df = None
    tft_path = run_dir / "tft_results_df"
    if tft_path.with_suffix(".parquet").exists() or tft_path.with_suffix(".csv").exists():
        tft_results_df = load_df(tft_path)

    lgbm_results_df = None
    lgbm_path = run_dir / "lgbm_results_df"
    if lgbm_path.with_suffix(".parquet").exists() or lgbm_path.with_suffix(".csv").exists():
        lgbm_results_df = load_df(lgbm_path)

    # validate / normalize
    validate_df_diag_v3(
        df_diag_v3,
        date_col=meta["date_col"],
        label_col=meta["label_col"],
        strategy_id_col=meta["strategy_id_col"],
        regime_col=meta.get("regime_col"),
    )
    validate_fold_meta_df(fold_meta_df)
    validate_comp_df(comp_df)

    return {
        "df_diag_v3": df_diag_v3,
        "fold_meta_df": fold_meta_df,
        "comp_df": comp_df,
        "tft_results_df": tft_results_df, # Add to return
        "lgbm_results_df": lgbm_results_df, # Add to return
        "meta": meta,
        "run_dir": str(run_dir),
    }

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _filter_ok_only(df: pd.DataFrame, model_prefix: str) -> pd.Series:
    """
    comp_df から 'ok' fold のマスクを作る（status列が無い場合は全てok扱い）。
    model_prefix: "tft" or "lgbm"
    """
    col = f"status_{model_prefix}"
    if col in df.columns:
        return df[col].fillna("ok").eq("ok")
    if "status" in df.columns:
        return df["status"].fillna("ok").eq("ok")
    return pd.Series(True, index=df.index)

def plot_fold_by_fold_auc_comparison(
    comp_df: pd.DataFrame,
    out_path: str,
    *,
    title: str = "Fold-by-fold ROC-AUC (TFT vs LightGBM)",
    ok_only: bool = True,
) -> None:
    """
    comp_df columns required: fold_idx, tft_auc, lgbm_auc (from fold_by_fold_tft_vs_lgbm output)
    """
    dfp = comp_df.copy()

    # fold順に並べる
    dfp = dfp.sort_values("fold_idx")

    # ok fold のみに絞る（両方okのfoldだけ）
    if ok_only:
        ok_tft = _filter_ok_only(dfp, "tft")
        ok_lgbm = _filter_ok_only(dfp, "lgbm")
        dfp = dfp[ok_tft & ok_lgbm].copy()

    # 数値化
    for c in ["tft_auc", "lgbm_auc"]:
        dfp[c] = pd.to_numeric(dfp[c], errors="coerce")

    # 欠損除外
    dfp = dfp.dropna(subset=["tft_auc", "lgbm_auc"])

    x = dfp["fold_idx"].astype(int).to_numpy()
    y_tft = dfp["tft_auc"].to_numpy()
    y_lgbm = dfp["lgbm_auc"].to_numpy()

    # プロット
    plt.figure(figsize=(12, 5))
    plt.plot(x, y_lgbm, marker="o", label="LightGBM")
    plt.plot(x, y_tft, marker="o", label="TFT")
    plt.axhline(0.5, linestyle="--", linewidth=1)

    plt.title(title)
    plt.xlabel("fold_idx")
    plt.ylabel("ROC-AUC (test)")
    plt.legend()
    plt.tight_layout()

    _ensure_dir(os.path.dirname(out_path) or ".")
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_auc_distribution_box_violin(
    comp_df: pd.DataFrame,
    out_box_path: str,
    out_violin_path: str,
    *,
    title_prefix: str = "AUC distribution (ok folds)",
    ok_only: bool = True,
) -> None:
    """
    Boxplot + Violinplot を別々に保存。
    comp_df columns required: tft_auc, lgbm_auc
    """
    dfp = comp_df.copy()

    if ok_only:
        ok_tft = _filter_ok_only(dfp, "tft")
        ok_lgbm = _filter_ok_only(dfp, "lgbm")
        dfp = dfp[ok_tft & ok_lgbm].copy()

    dfp["tft_auc"] = pd.to_numeric(dfp["tft_auc"], errors="coerce")
    dfp["lgbm_auc"] = pd.to_numeric(dfp["lgbm_auc"], errors="coerce")
    dfp = dfp.dropna(subset=["tft_auc", "lgbm_auc"])

    tft = dfp["tft_auc"].to_numpy()
    lgbm = dfp["lgbm_auc"].to_numpy()

    # --- Boxplot ---
    plt.figure(figsize=(7, 5))
    plt.boxplot([lgbm, tft], labels=["LightGBM", "TFT"], showmeans=True)
    plt.axhline(0.5, linestyle="--", linewidth=1)
    plt.title(f"{title_prefix} - boxplot")
    plt.ylabel("ROC-AUC (test)")
    plt.tight_layout()
    _ensure_dir(os.path.dirname(out_box_path) or ".")
    plt.savefig(out_box_path, dpi=200)
    plt.close()

    # --- Violinplot ---
    plt.figure(figsize=(7, 5))
    parts = plt.violinplot([lgbm, tft], showmeans=True, showextrema=True, showmedians=True)
    plt.xticks([1, 2], ["LightGBM", "TFT"])
    plt.axhline(0.5, linestyle="--", linewidth=1)
    plt.title(f"{title_prefix} - violin")
    plt.ylabel("ROC-AUC (test)")
    plt.tight_layout()
    _ensure_dir(os.path.dirname(out_violin_path) or ".")
    plt.savefig(out_violin_path, dpi=200)
    plt.close()

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _to_dt(x):
    return pd.to_datetime(x, errors="coerce")

def _pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _ok_mask_from_comp(comp_df: pd.DataFrame, ok_only: bool = True) -> pd.Series:
    """
    comp_df から「比較可能なfold（両方AUCあり）」かつ（任意で）status==ok のマスク
    """
    m = comp_df[["tft_auc", "lgbm_auc"]].notna().all(axis=1)
    if not ok_only:
        return m

    status_cols = [c for c in ["status_tft", "status_lgbm", "status"] if c in comp_df.columns]
    if not status_cols:
        return m

    for c in status_cols:
        m = m & comp_df[c].fillna("ok").eq("ok")
    return m

def enrich_fold_regime_stats(
    comp_df: pd.DataFrame,
    fold_meta_df: pd.DataFrame | None,
    df: pd.DataFrame | None,
    *,
    date_col: str = "date",
    regime_col: str = "regime",
) -> pd.DataFrame:
    """
    優先順位:
      (1) comp_df に regime 列があればそれを利用（fold単位）
      (2) fold_meta_df に regime 列があればマージ
      (3) df と fold_meta_df の test期間から regime 分布をfoldごとに集計
    返り値: comp_df に以下の列を付与したもの
      - fold_regime_mode : test期間で最頻出のregime
      - regime_dist_*    : test期間の各regimeの割合（列が増える）
    """
    out = comp_df.copy()

    # (1) comp_df に regime がある場合
    if regime_col in out.columns:
        # fold単位でユニークならそのまま mode とみなす
        out["fold_regime_mode"] = out[regime_col]
        return out

    # (2) fold_meta_df に regime がある場合
    if fold_meta_df is not None and regime_col in fold_meta_df.columns and "fold_idx" in fold_meta_df.columns:
        tmp = fold_meta_df[["fold_idx", regime_col]].copy()
        out = out.merge(tmp, on="fold_idx", how="left")
        out["fold_regime_mode"] = out[regime_col]
        return out

    # (3) df と fold_meta_df から集計
    if df is None or fold_meta_df is None:
        # 作れないのでそのまま返す
        return out

    if "fold_idx" not in fold_meta_df.columns:
        return out

    # 日付列の扱い：indexがdatetimeならそれも許容
    df2 = df.copy()
    if date_col in df2.columns:
        df2[date_col] = _to_dt(df2[date_col])
        df2 = df2.dropna(subset=[date_col])
        df2 = df2.sort_values(date_col)
    else:
        # indexがdatetimeである想定
        if not isinstance(df2.index, pd.DatetimeIndex):
            return out
        df2 = df2.sort_index()

    if regime_col not in df2.columns:
        # regime がデータに無いなら作れない
        return out

    # foldごとに test期間の regime 分布を作る
    rows = []
    for _, r in fold_meta_df.iterrows():
        f = int(r["fold_idx"])
        if "test_start" not in r or "test_end" not in r:
            continue

        ts = _to_dt(r["test_start"])
        te = _to_dt(r["test_end"])
        if pd.isna(ts) or pd.isna(te):
            continue

        if date_col in df2.columns:
            m = (df2[date_col] >= ts) & (df2[date_col] <= te)
            sub = df2.loc[m, regime_col]
        else:
            sub = df2.loc[ts:te, regime_col]

        sub = sub.dropna()
        if sub.empty:
            continue

        vc = sub.value_counts(normalize=True)  # 割合
        mode = vc.index[0]
        row = {"fold_idx": f, "fold_regime_mode": mode}
        for k, v in vc.items():
            row[f"regime_dist_{k}"] = float(v)
        rows.append(row)

    if rows:
        reg_df = pd.DataFrame(rows)
        out = out.merge(reg_df, on="fold_idx", how="left")

    return out

def save_winlose_regime_distribution(
    comp_df_enriched: pd.DataFrame,
    out_path: str,
    *,
    ok_only: bool = True,
    title: str = "Regime distribution (Win folds vs Lose folds)",
) -> None:
    df = comp_df_enriched.copy()
    df = df.sort_values("fold_idx")

    m = _ok_mask_from_comp(df, ok_only=ok_only) & df["tft_win"].notna()
    df = df[m].copy()

    # regime_dist_* があるならそれを使う。なければ fold_regime_mode の頻度で代用。
    dist_cols = [c for c in df.columns if c.startswith("regime_dist_")]

    if dist_cols:
        # win/lose それぞれで平均（=各foldの割合の平均）を作る
        win = df[df["tft_win"].astype(bool)]
        lose = df[~df["tft_win"].astype(bool)]

        win_mean = win[dist_cols].mean(axis=0).fillna(0.0) if len(win) else pd.Series(0.0, index=dist_cols)
        lose_mean = lose[dist_cols].mean(axis=0).fillna(0.0) if len(lose) else pd.Series(0.0, index=dist_cols)

        # 列名整形
        regimes = [c.replace("regime_dist_", "") for c in dist_cols]
        W = win_mean.to_numpy()
        L = lose_mean.to_numpy()

        # stacked bar
        plt.figure(figsize=(9, 5))
        bottom_w = 0.0
        bottom_l = 0.0
        x = np.array([0, 1])
        labels = ["Win", "Lose"]

        for i, reg in enumerate(regimes):
            plt.bar(x[0], W[i], bottom=bottom_w, label=str(reg))
            plt.bar(x[1], L[i], bottom=bottom_l)
            bottom_w += W[i]
            bottom_l += L[i]

        plt.xticks(x, labels)
        plt.ylim(0, 1.0)
        plt.ylabel("Mean regime proportion in test window")
        plt.title(title)
        plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.0))
        plt.tight_layout()

        _ensure_dir(os.path.dirname(out_path) or ".")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        return

    # dist_colsが無い → fold_regime_mode で頻度比較
    if "fold_regime_mode" not in df.columns:
        # 作れない
        print("[WARN] regime info not found. Skip regime distribution plot.")
        return

    win = df[df["tft_win"].astype(bool)]["fold_regime_mode"].dropna().astype(str)
    lose = df[~df["tft_win"].astype(bool)]["fold_regime_mode"].dropna().astype(str)

    if win.empty and lose.empty:
        print("[WARN] fold_regime_mode is empty. Skip regime distribution plot.")
        return

    all_regs = sorted(set(win.unique()).union(set(lose.unique())))
    win_rate = np.array([(win == r).mean() if len(win) else 0.0 for r in all_regs])
    lose_rate = np.array([(lose == r).mean() if len(lose) else 0.0 for r in all_regs])

    x = np.arange(len(all_regs))
    w = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - w/2, win_rate, width=w, label="Win folds")
    plt.bar(x + w/2, lose_rate, width=w, label="Lose folds")
    plt.xticks(x, all_regs, rotation=30, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Proportion of folds")
    plt.title(title + " (by fold mode)")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()

    _ensure_dir(os.path.dirname(out_path) or ".")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def save_winlose_posrate_distribution(
    comp_df: pd.DataFrame,
    out_path: str,
    *,
    ok_only: bool = True,
    title: str = "pos_test distribution (Win folds vs Lose folds)",
) -> None:
    df = comp_df.copy()
    df = df.sort_values("fold_idx")

    m = _ok_mask_from_comp(df, ok_only=ok_only) & df["tft_win"].notna()
    df = df[m].copy()

    # 添付コードの merge 後は pos_test_tft / pos_test_lgbm になりやすい
    tft_pos_col = _pick_first_existing(df, ["pos_test_tft", "pos_test"])
    lgbm_pos_col = _pick_first_existing(df, ["pos_test_lgbm", "pos_test"])

    if tft_pos_col is None:
        print("[WARN] pos_test column not found. Skip pos_rate plot.")
        return

    df[tft_pos_col] = pd.to_numeric(df[tft_pos_col], errors="coerce")
    if lgbm_pos_col is not None:
        df[lgbm_pos_col] = pd.to_numeric(df[lgbm_pos_col], errors="coerce")

    win = df[df["tft_win"].astype(bool)]
    lose = df[~df["tft_win"].astype(bool)]

    data = [win[tft_pos_col].dropna().to_numpy(), lose[tft_pos_col].dropna().to_numpy()]
    labels = [f"TFT {tft_pos_col} (Win)", f"TFT {tft_pos_col} (Lose)"]

    # 可能なら LGBM も追加
    if lgbm_pos_col is not None and lgbm_pos_col != tft_pos_col:
        data += [win[lgbm_pos_col].dropna().to_numpy(), lose[lgbm_pos_col].dropna().to_numpy()]
        labels += [f"LGBM {lgbm_pos_col} (Win)", f"LGBM {lgbm_pos_col} (Lose)"]

    # 空対策
    if all(len(d) == 0 for d in data):
        print("[WARN] pos_test arrays are empty. Skip pos_rate plot.")
        return

    plt.figure(figsize=(10, 5))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel("pos_test (positive rate in test)")
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    _ensure_dir(os.path.dirname(out_path) or ".")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def save_fold_timeline_plot(
    comp_df: pd.DataFrame,
    out_path: str,
    *,
    ok_only: bool = True,
    title: str = "WFA test window timeline (Win/Lose by TFT)",
) -> None:
    df = comp_df.copy()

    # 必要列
    need = ["fold_idx", "test_start", "test_end", "tft_win", "tft_auc", "lgbm_auc"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        print(f"[WARN] timeline plot needs columns: {missing}. Skip timeline plot.")
        return

    df["test_start"] = _to_dt(df["test_start"])
    df["test_end"] = _to_dt(df["test_end"])
    df = df.dropna(subset=["test_start", "test_end"])

    m = _ok_mask_from_comp(df, ok_only=ok_only) & df["tft_win"].notna()
    df = df[m].copy().sort_values("fold_idx")

    if df.empty:
        print("[WARN] no valid folds for timeline plot.")
        return

    # matplotlib の broken_barh 用に数値化（日付→matplotlib内部のfloat）
    # ここでは pandas の toordinal ベースで簡易化（表示は日付のまま）
    # → 純matplotlibで十分
    starts = df["test_start"].map(pd.Timestamp.toordinal).to_numpy()
    ends = df["test_end"].map(pd.Timestamp.toordinal).to_numpy()
    widths = (ends - starts + 1).astype(float)

    y = np.arange(len(df))  # foldを縦に積む
    win = df["tft_win"].astype(bool).to_numpy()

    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    # 勝ち/負けで2回に分けて描画（色はデフォルトでOK、区別は凡例で）
    for i in range(len(df)):
        ax.broken_barh([(starts[i], widths[i])], (y[i] - 0.4, 0.8))

    # 勝ちfoldにマーカー
    for i in range(len(df)):
        if win[i]:
            ax.scatter(starts[i] + widths[i]/2, y[i], marker="o", s=30)

    ax.set_yticks(y)
    ax.set_yticklabels(df["fold_idx"].astype(int).tolist())
    ax.set_xlabel("Date (ordinal)")
    ax.set_ylabel("fold_idx")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)

    # x軸を日付っぽく見せる（端だけでも）
    x_min = int(starts.min())
    x_max = int((starts + widths).max())
    tick_pos = np.linspace(x_min, x_max, 6).astype(int)
    tick_lab = [pd.Timestamp.fromordinal(int(t)).strftime("%Y-%m-%d") for t in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, rotation=20, ha="right")

    plt.tight_layout()
    _ensure_dir(os.path.dirname(out_path) or ".")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def run_winning_fold_diagnostics_and_save(
    tft_results_df: pd.DataFrame,
    lgbm_results_df: pd.DataFrame,
    fold_meta_df: pd.DataFrame | None,
    *,
    df_raw: pd.DataFrame | None = None,
    date_col: str = "date",
    regime_col: str = "regime",
    out_dir: str = "./figures",
    ok_only: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    返り値:
      - comp_df_enriched（regime集計を付与した比較表）
      - summary（fold_by_fold_tft_vs_lgbm の summary）
    """
    _ensure_dir(out_dir)

    # 添付コードの比較関数（既に定義済み）を使う
    comp_df, summary = fold_by_fold_tft_vs_lgbm(
        tft_results_df=tft_results_df,
        lgbm_results_df=lgbm_results_df,
        fold_meta_df=fold_meta_df,
    )

    # regime 付与
    comp_df_enriched = enrich_fold_regime_stats(
        comp_df=comp_df,
        fold_meta_df=fold_meta_df,
        df=df_raw,
        date_col=date_col,
        regime_col=regime_col,
    )

    # 保存：regime
    save_winlose_regime_distribution(
        comp_df_enriched,
        out_path=os.path.join(out_dir, "winlose_regime_distribution.png"),
        ok_only=ok_only,
    )

    # 保存：pos_rate（pos_test）
    save_winlose_posrate_distribution(
        comp_df_enriched,  # comp_dfでも良いが、enrichedで統一
        out_path=os.path.join(out_dir, "winlose_pos_test_distribution.png"),
        ok_only=ok_only,
    )

    # 保存：タイムライン
    save_fold_timeline_plot(
        comp_df_enriched,
        out_path=os.path.join(out_dir, "wfa_test_window_timeline.png"),
        ok_only=ok_only,
    )

    return comp_df_enriched, summary

import numpy as np

import pandas as pd

from typing import Dict, List, Tuple, Optional

def _first_existing_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns exist: {candidates}")

def _summarize_auc(results_df: pd.DataFrame) -> Dict[str, float]:
    """
    run_tft_wfa_with_diagnostics の戻り値から AUC を要約。
    想定列: auc_test / status など（無ければあるものを使う）
    """
    auc_col = _first_existing_col(results_df, ["auc_test", "test_auc", "AUC_test", "auc"])
    if "status" in results_df.columns:
        ok = results_df.query("status == 'ok'").copy()
    else:
        ok = results_df.copy()

    out = {
        "n_folds_total": float(len(results_df)),
        "n_folds_ok": float(len(ok)),
        "auc_mean": float(ok[auc_col].mean()) if len(ok) else np.nan,
        "auc_median": float(ok[auc_col].median()) if len(ok) else np.nan,
        "auc_std": float(ok[auc_col].std(ddof=1)) if len(ok) > 1 else np.nan,
    }
    return out

def build_feature_groups(
    cont_all: List[str],
    cat_all: List[str],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    CONT_GROUPS = {
        "price_tech": [
            "ATR24","ret_6h","volatility_change_6h","trend_strength_6_24",
            "ret_6h_lag1","volatility_change_6h_lag1","trend_strength_6_24_lag1",
            "delta_ret_6h","delta_volatility_change_6h",
        ],
        "onchain_existing": [
            "active_receivers_regime_score_zn","whale_tx_count_slope_s6_zn","dex_volume_event_score_zn",
            "active_receivers_regime_score_zn_lag1","whale_tx_count_slope_s6_zn_lag1","dex_volume_event_score_zn_lag1",
            "delta_whale_tx_count_slope_s6_zn","delta_dex_volume_event_score_zn",
        ],
        "onchain_dune": [
            "velocity_stablecoin_24h_zn","delta_velocity_stablecoin_24h",
            "dex_volume_usd_zn","delta_dex_volume_usd",
            "median_gas_price_zn","delta_median_gas_price",
            "tx_count_zn","delta_tx_count",
        ],
        "external_vix": ["VIX_zn","delta_VIX"],
    }

    CAT_GROUPS = {
        "flags_basic": [
            "buy_dominant_flag","momentum_6h_cusum_up_flag","spread_spike_flag",
            "rsi_overbought_flag","rsi_oversold_flag",
        ],
        "flags_stablecoin": [
            "velocity_stablecoin_24h_regime_up_flag","velocity_stablecoin_24h_regime_down_flag",
            "velocity_stablecoin_24h_cusum_up_flag","velocity_stablecoin_24h_cusum_down_flag",
        ],
        "flags_dex": [
            "dex_volume_regime_up_flag","dex_volume_regime_down_flag",
            "dex_volume_cusum_up_flag","dex_volume_cusum_down_flag",
        ],
        "flags_gas": [
            "gas_fee_regime_up_flag","gas_fee_regime_down_flag",
            "gas_fee_cusum_up_flag","gas_fee_cusum_down_flag",
        ],
        "flags_tx": [
            "tx_regime_up_flag","tx_regime_down_flag",
            "tx_cusum_up_flag","tx_cusum_down_flag",
        ],
        "flags_vix": ["VIX_regime_up_flag","VIX_regime_down_flag"],
        "strategy_id": ["strategy_type_id"],
    }

    # 実際に存在する列だけに絞る（typo耐性）
    cont_set = set(cont_all)
    cat_set  = set(cat_all)

    CONT_GROUPS = {k: [c for c in v if c in cont_set] for k, v in CONT_GROUPS.items()}
    CAT_GROUPS  = {k: [c for c in v if c in cat_set]  for k, v in CAT_GROUPS.items()}

    return CONT_GROUPS, CAT_GROUPS

def build_lofo_feature_sets(
    cont_all: List[str],
    cat_all: List[str],
    cont_groups: Dict[str, List[str]],
    cat_groups: Dict[str, List[str]],
) -> List[Tuple[str, List[str], List[str]]]:
    sets = []
    sets.append(("FULL", cont_all, cat_all))

    for gname, cols in cont_groups.items():
        if len(cols) == 0:
            continue
        cont_kept = [c for c in cont_all if c not in cols]
        sets.append((f"DROP_CONT::{gname}", cont_kept, cat_all))

    for gname, cols in cat_groups.items():
        if len(cols) == 0:
            continue
        cat_kept = [c for c in cat_all if c not in cols]
        sets.append((f"DROP_CAT::{gname}", cont_all, cat_kept))

    return sets

def run_tft_group_lofo_ablation(
    *,
    df_diag: pd.DataFrame,
    fold_meta_df: pd.DataFrame,
    cont_all: List[str],
    cat_all: List[str],
    base_config: dict,
    regime_col: str = "regime_v3_compressed",
    add_regime_embedding: bool = True,
    encoder_len: int = 24,
    decoder_len: int = 1,
    batch_size: int = 64,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    label_col: str = "target",
    eval_strategy_type: Optional[str] = None,
    verbose: bool = True,
    fold_filter: Optional[List[int]] = None,
    # ★追加：外部からfeature_setsを渡せるように
    feature_sets_override: Optional[List[Tuple[str, List[str], List[str]]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:

    fm = fold_meta_df.copy()
    if fold_filter is not None:
        fm = fm[fm["fold_idx"].isin(fold_filter)].copy().reset_index(drop=True)
    if len(fm) == 0:
        raise ValueError("fold_meta_df becomes empty after fold_filter")

    # ★差し替えがあるならそれを使う
    if feature_sets_override is not None:
        feature_sets = feature_sets_override
    else:
        cont_groups, cat_groups = build_feature_groups(cont_all, cat_all)
        feature_sets = build_lofo_feature_sets(cont_all, cat_all, cont_groups, cat_groups)

    raw_results: Dict[str, pd.DataFrame] = {}
    summary_rows = []

    for name, cont_feats, cat_feats in feature_sets:
        if verbose:
            print(f"\n[ABLAT] {name}: cont={len(cont_feats)} cat={len(cat_feats)} folds={len(fm)}")

        results_df_wfa, _, _ = run_tft_wfa_with_diagnostics(
            df_diag=df_diag,
            fold_meta_df=fm,
            cont_features=cont_feats,
            cat_features_base=cat_feats,
            base_config=base_config,
            regime_col=regime_col,
            add_regime_embedding=add_regime_embedding,
            encoder_len=encoder_len,
            decoder_len=decoder_len,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
            label_col=label_col,
            eval_strategy_type=eval_strategy_type,
            verbose=verbose,
        )

        raw_results[name] = results_df_wfa

        s = _summarize_auc(results_df_wfa)
        s.update({"setting": name, "n_cont": float(len(cont_feats)), "n_cat": float(len(cat_feats))})
        summary_rows.append(s)

    summary_df = pd.DataFrame(summary_rows)

    if "FULL" not in raw_results:
        raise RuntimeError("FULL result missing")
    full_res = raw_results["FULL"]
    auc_col = _first_existing_col(full_res, ["auc_test", "test_auc", "AUC_test", "auc"])

    full_fold = full_res[["fold_idx", auc_col]].copy()
    full_fold = full_fold.rename(columns={auc_col: "FULL_auc"}).set_index("fold_idx")
    if "status" in full_res.columns:
        full_fold["FULL_status"] = full_res["status"].values

    fold_delta_rows = []
    for name, df in raw_results.items():
        if name == "FULL":
            continue

        tmp = df[["fold_idx", auc_col]].copy()
        tmp = tmp.rename(columns={auc_col: f"{name}_auc"}).set_index("fold_idx")
        if "status" in df.columns:
            tmp[f"{name}_status"] = df["status"].values

        merged = full_fold.join(tmp, how="inner")
        merged["setting"] = name
        merged["delta_vs_full"] = merged[f"{name}_auc"] - merged["FULL_auc"]
        fold_delta_rows.append(merged.reset_index())

    fold_delta_df = pd.concat(fold_delta_rows, axis=0, ignore_index=True) if fold_delta_rows else pd.DataFrame()

    full_mean = float(summary_df.loc[summary_df["setting"] == "FULL", "auc_mean"].iloc[0])
    summary_df["delta_auc_mean_vs_full"] = summary_df["auc_mean"] - full_mean
    summary_df = summary_df.sort_values("delta_auc_mean_vs_full", ascending=True).reset_index(drop=True)

    return summary_df, fold_delta_df, raw_results

from typing import List, Dict, Tuple, Optional

import pandas as pd

import numpy as np

def build_individual_lofo_feature_sets(
    *,
    cont_all: List[str],
    cat_all: List[str],
    cont_groups: Dict[str, List[str]],
    cat_groups: Dict[str, List[str]],
    # 優先順（ここに書いた順に回す）
    group_priority: List[str] = None,
    include_cat: bool = False,   # まずはFalse推奨（連続だけ掃除）
) -> List[Tuple[str, List[str], List[str]]]:
    """
    戻り値: (setting_name, cont_feats, cat_feats) のリスト
    - FULL も含める
    - それ以外は DROP_ONE::<group>::<feature> の1本落とし
    """
    if group_priority is None:
        group_priority = ["price_tech", "onchain_existing", "onchain_dune", "external_vix"]

    # group_priorityに無い群も後ろに追加
    all_groups = list(dict.fromkeys(group_priority + list(cont_groups.keys()) + list(cat_groups.keys())))

    feature_sets: List[Tuple[str, List[str], List[str]]] = []
    # FULL
    feature_sets.append(("FULL", cont_all, cat_all))

    # 連続特徴（優先順）
    for g in all_groups:
        feats = cont_groups.get(g, [])
        for f in feats:
            cont_feats = [x for x in cont_all if x != f]
            feature_sets.append((f"DROP_ONE::{g}::{f}", cont_feats, cat_all))

    # カテゴリ（必要になったら）
    if include_cat:
        for g in all_groups:
            feats = cat_groups.get(g, [])
            for f in feats:
                cat_feats = [x for x in cat_all if x != f]
                feature_sets.append((f"DROP_ONE_CAT::{g}::{f}", cont_all, cat_feats))

    return feature_sets

def select_win_folds_from_full(
    full_results_df: pd.DataFrame,
    *,
    auc_col_candidates=("auc_test", "test_auc", "AUC_test", "auc"),
    mode: str = "threshold",   # "threshold" or "top_quantile"
    threshold: float = 0.65,
    top_quantile: float = 0.5, # 上位50%を勝ちfoldにする等
    require_status_ok: bool = True,
) -> List[int]:
    # AUC列特定
    auc_col = None
    for c in auc_col_candidates:
        if c in full_results_df.columns:
            auc_col = c
            break
    if auc_col is None:
        raise ValueError(f"AUC col not found in FULL df. candidates={auc_col_candidates}")

    df = full_results_df.copy()

    if require_status_ok and "status" in df.columns:
        df = df[df["status"] == "ok"].copy()

    df = df.dropna(subset=[auc_col])

    if len(df) == 0:
        return []

    if mode == "threshold":
        win = df[df[auc_col] >= threshold]["fold_idx"].astype(int).tolist()
        return sorted(win)

    if mode == "top_quantile":
        thr = float(df[auc_col].quantile(1.0 - top_quantile))
        win = df[df[auc_col] >= thr]["fold_idx"].astype(int).tolist()
        return sorted(win)

    raise ValueError(f"Unknown mode: {mode}")

from typing import List, Tuple, Dict, Optional

import pandas as pd

def run_tft_individual_lofo(
    *,
    df_diag: pd.DataFrame,
    fold_meta_df: pd.DataFrame,
    cont_all: List[str],
    cat_all: List[str],
    base_config: dict,
    # TFT params
    regime_col: str = "regime_v3_compressed",
    add_regime_embedding: bool = True,
    encoder_len: int = 24,
    decoder_len: int = 1,
    batch_size: int = 64,
    num_epochs: int = 5,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    label_col: str = "target",
    eval_strategy_type: Optional[str] = None,
    verbose: bool = True,
    # win fold selection
    win_mode: str = "threshold",    # "threshold" or "top_quantile"
    win_threshold: float = 0.65,
    win_top_quantile: float = 0.5,
    # LOFO scope
    group_priority: List[str] = None,
    include_cat: bool = False,
    # ★追加：LOFOを回すfold範囲
    lofo_fold_scope: str = "all",   # "all" or "win_only"
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame], List[int]]:
    """
    Returns:
      - summary_df        : setting別(=落とした特徴)の auc_mean / delta_vs_full 等
      - fold_delta_df     : fold別×setting別の delta_vs_full
      - raw_results       : setting -> results_df_wfa
      - win_folds         : FULLの結果から抽出した勝ちfold
    """

    # 1) FULLだけ先に回す（勝ちfold抽出用）
    full_feature_sets = [("FULL", cont_all, cat_all)]
    full_summary, _, full_raw = run_tft_group_lofo_ablation(
        df_diag=df_diag,
        fold_meta_df=fold_meta_df,
        cont_all=cont_all,
        cat_all=cat_all,
        base_config=base_config,
        regime_col=regime_col,
        add_regime_embedding=add_regime_embedding,
        encoder_len=encoder_len,
        decoder_len=decoder_len,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        label_col=label_col,
        eval_strategy_type=eval_strategy_type,
        verbose=verbose,
        fold_filter=None,
        feature_sets_override=full_feature_sets,
    )
    full_res = full_raw["FULL"]

    # 2) 勝ちfold抽出（FULLを基準）
    win_folds = select_win_folds_from_full(
        full_res,
        mode=win_mode,
        threshold=win_threshold,
        top_quantile=win_top_quantile,
        require_status_ok=True,
    )
    if len(win_folds) == 0:
        win_folds = select_win_folds_from_full(full_res, mode="top_quantile", top_quantile=0.5)

    if verbose:
        print(f"\n[IND-LOFO] win_folds={win_folds} (n={len(win_folds)})")

    # 3) 個別1本DROPのfeature_sets生成
    cont_groups, cat_groups = build_feature_groups(cont_all, cat_all)
    ind_feature_sets = build_individual_lofo_feature_sets(
        cont_all=cont_all,
        cat_all=cat_all,
        cont_groups=cont_groups,
        cat_groups=cat_groups,
        group_priority=group_priority,
        include_cat=include_cat,
    )

    # 4) LOFOを回すfold範囲
    if lofo_fold_scope not in ("all", "win_only"):
        raise ValueError("lofo_fold_scope must be 'all' or 'win_only'")

    fold_filter = None if lofo_fold_scope == "all" else win_folds

    # 5) 個別LOFO（all folds か win_only）
    summary_df, fold_delta_df, raw_results = run_tft_group_lofo_ablation(
        df_diag=df_diag,
        fold_meta_df=fold_meta_df,
        cont_all=cont_all,
        cat_all=cat_all,
        base_config=base_config,
        regime_col=regime_col,
        add_regime_embedding=add_regime_embedding,
        encoder_len=encoder_len,
        decoder_len=decoder_len,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        label_col=label_col,
        eval_strategy_type=eval_strategy_type,
        verbose=verbose,
        fold_filter=fold_filter,
        feature_sets_override=ind_feature_sets,
    )

    return summary_df, fold_delta_df, raw_results, win_folds

import os, json

import pandas as pd

from typing import Dict, Any, Optional

def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)

def save_lofo_results(
    out_dir: str,
    *,
    group_summary_df: pd.DataFrame,
    group_fold_delta_df: pd.DataFrame,
    ind_summary_df: pd.DataFrame,
    ind_fold_delta_df: pd.DataFrame,
    win_folds: list,
    meta: Optional[Dict[str, Any]] = None,
    save_format: str = "parquet",   # "parquet" or "csv"
) -> None:
    ensure_dir(out_dir)

    def _save_df(df: pd.DataFrame, name: str):
        if save_format == "parquet":
            df.to_parquet(os.path.join(out_dir, f"{name}.parquet"), index=False)
        elif save_format == "csv":
            df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
        else:
            raise ValueError("save_format must be parquet or csv")

    _save_df(group_summary_df, "group_lofo_summary")
    _save_df(group_fold_delta_df, "group_lofo_fold_delta")
    _save_df(ind_summary_df, "ind_lofo_summary")
    _save_df(ind_fold_delta_df, "ind_lofo_fold_delta")

    with open(os.path.join(out_dir, "win_folds.json"), "w", encoding="utf-8") as f:
        json.dump([int(x) for x in win_folds], f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta or {}, f, ensure_ascii=False, indent=2)

def load_lofo_results(in_dir: str) -> Dict[str, Any]:
    def _load_df(name: str) -> pd.DataFrame:
        p_parq = os.path.join(in_dir, f"{name}.parquet")
        p_csv  = os.path.join(in_dir, f"{name}.csv")
        if os.path.exists(p_parq):
            return pd.read_parquet(p_parq)
        if os.path.exists(p_csv):
            return pd.read_csv(p_csv)
        raise FileNotFoundError(f"missing {name}.parquet/csv in {in_dir}")

    group_summary_df = _load_df("group_lofo_summary")
    group_fold_delta_df = _load_df("group_lofo_fold_delta")
    ind_summary_df = _load_df("ind_lofo_summary")
    ind_fold_delta_df = _load_df("ind_lofo_fold_delta")

    with open(os.path.join(in_dir, "win_folds.json"), "r", encoding="utf-8") as f:
        win_folds = json.load(f)

    meta_path = os.path.join(in_dir, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    return {
        "group_lofo_summary_df": group_summary_df,
        "group_lofo_fold_delta_df": group_fold_delta_df,
        "ind_lofo_summary_df": ind_summary_df,
        "ind_lofo_fold_delta_df": ind_fold_delta_df,
        "win_folds": win_folds,
        "meta": meta,
    }

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from typing import Optional

def _ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

def _savefig(path: str, dpi: int = 200) -> None:
    _ensure_parent(path)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()

def _extract_feature_name_from_setting(setting: str) -> str:
    """
    build_individual_lofo_feature_sets の命名が不明でも崩れにくいようにざっくり抽出。
    例:
      "DROP__rsi_14" -> "rsi_14"
      "LOFO_DROP: rsi_14" -> "rsi_14"
      "NO_price_tech" -> "price_tech"
    """
    s = str(setting)
    for sep in ["DROP__", "DROP_", "DROP:", "DROP ", "NO_", "NO:", "ABLATE_", "ABLATE:", "SETTING="]:
        if sep in s:
            s = s.split(sep, 1)[1].strip()
    return s.strip()

def plot_group_lofo_delta_auc_bar(
    group_summary_df: pd.DataFrame,
    out_path: str,
    top_k: Optional[int] = 20,
    title: str = "群LOFO：ΔAUC（平均, ablated - FULL）",
) -> pd.DataFrame:
    df = group_summary_df.copy()
    if "setting" not in df.columns or "delta_auc_mean_vs_full" not in df.columns:
        raise KeyError("group_summary_df must contain ['setting','delta_auc_mean_vs_full']")

    # FULLは除外（差0なので）
    df = df[df["setting"] != "FULL"].copy()
    df["group"] = df["setting"].apply(_extract_feature_name_from_setting)
    df["delta_auc"] = pd.to_numeric(df["delta_auc_mean_vs_full"], errors="coerce")

    # 重要度はより負（落とすとAUCが下がる）
    df = df.sort_values("delta_auc", ascending=True)
    if top_k is not None:
        df = df.head(top_k)

    plt.figure(figsize=(10, max(4, 0.35 * len(df))))
    y = np.arange(len(df))
    plt.barh(y, df["delta_auc"].to_numpy())
    plt.yticks(y, df["group"].tolist())
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("ΔAUC（ablated - FULL）")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", alpha=0.25)
    _savefig(out_path)

    return df[["setting","group","delta_auc"]]

def plot_individual_lofo_delta_auc_ranking(
    ind_summary_df: pd.DataFrame,
    out_path: str,
    top_k: int = 30,
    title: str = "個別LOFO：ΔAUCランキング（より負＝重要）",
) -> pd.DataFrame:
    df = ind_summary_df.copy()
    if "setting" not in df.columns or "delta_auc_mean_vs_full" not in df.columns:
        raise KeyError("ind_summary_df must contain ['setting','delta_auc_mean_vs_full']")

    df = df[df["setting"] != "FULL"].copy()
    df["feature"] = df["setting"].apply(_extract_feature_name_from_setting)
    df["delta_auc"] = pd.to_numeric(df["delta_auc_mean_vs_full"], errors="coerce")
    df = df.sort_values("delta_auc", ascending=True).head(top_k)

    plt.figure(figsize=(10, max(5, 0.32 * len(df))))
    y = np.arange(len(df))
    plt.barh(y, df["delta_auc"].to_numpy())
    plt.yticks(y, df["feature"].tolist())
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("ΔAUC（ablated - FULL）")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", alpha=0.25)
    _savefig(out_path)

    return df[["setting","feature","delta_auc"]]

def plot_winning_fold_contrib_compare(
    ind_fold_delta_df: pd.DataFrame,
    win_folds: list,
    out_path: str,
    top_k: int = 25,
    title: str = "勝ちfoldにおける特徴量寄与比較（Win vs Non-win, 平均ΔAUC）",
) -> pd.DataFrame:
    """
    ind_fold_delta_df 必須列:
      - fold_idx
      - setting
      - delta_vs_full  (ablated_auc - FULL_auc)
    """
    df = ind_fold_delta_df.copy()
    req = {"fold_idx", "setting", "delta_vs_full"}
    miss = req - set(df.columns)
    if miss:
        raise KeyError(f"ind_fold_delta_df missing columns: {miss}")

    df = df[df["setting"] != "FULL"].copy()
    df["feature"] = df["setting"].apply(_extract_feature_name_from_setting)
    df["fold_idx"] = pd.to_numeric(df["fold_idx"], errors="coerce").astype("Int64")
    df["delta"] = pd.to_numeric(df["delta_vs_full"], errors="coerce")

    win_set = set(int(x) for x in win_folds)
    df["is_win_fold"] = df["fold_idx"].apply(lambda x: int(x) in win_set if pd.notna(x) else False)

    # win / nonwin の平均との差
    win_mean = df[df["is_win_fold"]].groupby("feature")["delta"].mean().rename("win_mean")
    non_mean = df[~df["is_win_fold"]].groupby("feature")["delta"].mean().rename("nonwin_mean")

    comp = pd.concat([win_mean, non_mean], axis=1).dropna()
    comp["diff_win_minus_nonwin"] = comp["win_mean"] - comp["nonwin_mean"]

    # 「勝ちfoldでより効く（より負方向に寄る）」= win_mean が nonwin より小さい → diff が負
    # ここでは “勝ちfoldで重要になった順”として diff 昇順を採用
    comp = comp.sort_values("diff_win_minus_nonwin", ascending=True).head(top_k)

    y = np.arange(len(comp))
    h = 0.38
    plt.figure(figsize=(12, max(5, 0.35 * len(comp))))
    plt.barh(y - h/2, comp["nonwin_mean"].to_numpy(), height=h, label="Non-win folds")
    plt.barh(y + h/2, comp["win_mean"].to_numpy(), height=h, label="Winning folds")
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.yticks(y, comp.index.tolist())
    plt.title(title)
    plt.xlabel("平均ΔAUC（ablated - FULL）")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", alpha=0.25)
    plt.legend()
    _savefig(out_path)

    return comp.reset_index().rename(columns={"index": "feature"})

import numpy as np

import pandas as pd

def _ensure_utc_ts(s):
    return pd.to_datetime(s, utc=True)

def summarize_fold_context(
    *,
    df: pd.DataFrame,
    fold_meta_df: pd.DataFrame,
    date_col: str = "date",
    regime_col: str = "regime_id",
    strategy_id_col: str = "strategy_type_id",
    # 6時間足前提なら 24*7=168 が “7日”
    val_span_hours: int = 24 * 7,
) -> pd.DataFrame:
    """
    各foldについて、train/testの
    - regime分布（比率）
    - strategy_type_id分布（比率）
    - pos_test（もしあれば）
    を要約して返す
    """
    d = df.copy()
    if date_col not in d.columns:
        raise KeyError(f"df must contain date_col='{date_col}'")
    d[date_col] = pd.to_datetime(d[date_col], utc=True)
    d = d.sort_values(date_col)

    d = add_regime_id(d, regime_col=regime_col, id_col="regime_id")

    if regime_col not in d.columns:
        raise KeyError(f"df must contain regime_col='{regime_col}'")
    if strategy_id_col not in d.columns:
        raise KeyError(f"df must contain strategy_id_col='{strategy_id_col}'")

    out_rows = []
    for _, r in fold_meta_df.iterrows():
        fold_idx = int(r["fold_idx"])
        train_start = _ensure_utc_ts(r["train_start"])
        train_end   = _ensure_utc_ts(r["train_end"])
        test_start  = _ensure_utc_ts(r["test_start"])
        test_end    = _ensure_utc_ts(r["test_end"])

        # TFTと同じく val_cut を置くなら
        val_cut = train_end - pd.Timedelta(hours=val_span_hours)

        train_mask = (d[date_col] >= train_start) & (d[date_col] <= val_cut)
        test_mask  = (d[date_col] >= test_start)  & (d[date_col] <= test_end)

        d_tr = d.loc[train_mask]
        d_te = d.loc[test_mask]

        row = {
            "fold_idx": fold_idx,
            "train_start": train_start, "train_end": train_end,
            "test_start": test_start, "test_end": test_end,
            "n_train_time": int(len(d_tr)),
            "n_test_time": int(len(d_te)),
        }

        # regime比率（train/test）
        tr_reg = d_tr[regime_col].value_counts(normalize=True, dropna=False)
        te_reg = d_te[regime_col].value_counts(normalize=True, dropna=False)
        for k, v in tr_reg.items():
            row[f"train_regime_{k}_ratio"] = float(v)
        for k, v in te_reg.items():
            row[f"test_regime_{k}_ratio"] = float(v)

        # strategy_id比率（train/test）
        tr_sid = d_tr[strategy_id_col].value_counts(normalize=True, dropna=False)
        te_sid = d_te[strategy_id_col].value_counts(normalize=True, dropna=False)
        for k, v in tr_sid.items():
            row[f"train_sid_{k}_ratio"] = float(v)
        for k, v in te_sid.items():
            row[f"test_sid_{k}_ratio"] = float(v)

        out_rows.append(row)

    fold_ctx_df = pd.DataFrame(out_rows).fillna(0.0)
    return fold_ctx_df

def compare_wins_losses(
    *,
    comp_df: pd.DataFrame,
    fold_ctx_df: pd.DataFrame,
    win_col: str = "tft_win",
    top_k: int = 12,
) -> pd.DataFrame:
    """
    wins vs losses の平均との差（wins - losses）を大きい順に返す
    """
    merged = pd.merge(comp_df[["fold_idx", win_col, "delta_auc"]], fold_ctx_df, on="fold_idx", how="inner")
    wins = merged.loc[merged[win_col] == 1.0]
    losses = merged.loc[merged[win_col] == 0.0]

    # 比率系の列を対象（*_ratio）
    ratio_cols = [c for c in merged.columns if c.endswith("_ratio")]

    diffs = []
    for c in ratio_cols:
        diff = wins[c].mean() - losses[c].mean()
        diffs.append((c, diff, wins[c].mean(), losses[c].mean()))

    diff_df = pd.DataFrame(diffs, columns=["feature", "mean_diff(wins-losses)", "wins_mean", "losses_mean"])
    diff_df = diff_df.sort_values("mean_diff(wins-losses)", ascending=False)

    # 重要度が高い順に上位を返す（負側も見たいので両端）
    top_pos = diff_df.head(top_k)
    top_neg = diff_df.tail(top_k).sort_values("mean_diff(wins-losses)")
    return pd.concat([top_pos, top_neg], axis=0)

import numpy as np

import pandas as pd

from typing import Dict, List, Tuple, Optional

def _first_existing_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns exist: {candidates}")

def _summarize_auc(results_df: pd.DataFrame) -> Dict[str, float]:
    """
    run_tft_wfa_with_diagnostics の戻り値から AUC を要約。
    想定列: auc_test / status など（無ければあるものを使う）
    """
    auc_col = _first_existing_col(results_df, ["auc_test", "test_auc", "AUC_test", "auc"])
    if "status" in results_df.columns:
        ok = results_df.query("status == 'ok'").copy()
    else:
        ok = results_df.copy()

    out = {
        "n_folds_total": float(len(results_df)),
        "n_folds_ok": float(len(ok)),
        "auc_mean": float(ok[auc_col].mean()) if len(ok) else np.nan,
        "auc_median": float(ok[auc_col].median()) if len(ok) else np.nan,
        "auc_std": float(ok[auc_col].std(ddof=1)) if len(ok) > 1 else np.nan,
    }
    return out

def build_feature_groups(
    cont_all: List[str],
    cat_all: List[str],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    CONT_GROUPS = {
        "price_tech": [
            "ATR24","ret_6h","volatility_change_6h","trend_strength_6_24",
            "ret_6h_lag1","volatility_change_6h_lag1","trend_strength_6_24_lag1",
            "delta_ret_6h","delta_volatility_change_6h",
        ],
        "onchain_existing": [
            "active_receivers_regime_score_zn","whale_tx_count_slope_s6_zn","dex_volume_event_score_zn",
            "active_receivers_regime_score_zn_lag1","whale_tx_count_slope_s6_zn_lag1","dex_volume_event_score_zn_lag1",
            "delta_whale_tx_count_slope_s6_zn","delta_dex_volume_event_score_zn",
        ],
        "onchain_dune": [
            "velocity_stablecoin_24h_zn","delta_velocity_stablecoin_24h",
            "dex_volume_usd_zn","delta_dex_volume_usd",
            "median_gas_price_zn","delta_median_gas_price",
            "tx_count_zn","delta_tx_count",
        ],
        "external_vix": ["VIX_zn","delta_VIX"],
    }

    CAT_GROUPS = {
        "flags_basic": [
            "buy_dominant_flag","momentum_6h_cusum_up_flag","spread_spike_flag",
            "rsi_overbought_flag","rsi_oversold_flag",
        ],
        "flags_stablecoin": [
            "velocity_stablecoin_24h_regime_up_flag","velocity_stablecoin_24h_regime_down_flag",
            "velocity_stablecoin_24h_cusum_up_flag","velocity_stablecoin_24h_cusum_down_flag",
        ],
        "flags_dex": [
            "dex_volume_regime_up_flag","dex_volume_regime_down_flag",
            "dex_volume_cusum_up_flag","dex_volume_cusum_down_flag",
        ],
        "flags_gas": [
            "gas_fee_regime_up_flag","gas_fee_regime_down_flag",
            "gas_fee_cusum_up_flag","gas_fee_cusum_down_flag",
        ],
        "flags_tx": [
            "tx_regime_up_flag","tx_regime_down_flag",
            "tx_cusum_up_flag","tx_cusum_down_flag",
        ],
        "flags_vix": ["VIX_regime_up_flag","VIX_regime_down_flag"],
        "strategy_id": ["strategy_type_id"],
    }

    # 実際に存在する列だけに絞る（typo耐性）
    cont_set = set(cont_all)
    cat_set  = set(cat_all)

    CONT_GROUPS = {k: [c for c in v if c in cont_set] for k, v in CONT_GROUPS.items()}
    CAT_GROUPS  = {k: [c for c in v if c in cat_set]  for k, v in CAT_GROUPS.items()}

    return CONT_GROUPS, CAT_GROUPS

def build_lofo_feature_sets(
    cont_all: List[str],
    cat_all: List[str],
    cont_groups: Dict[str, List[str]],
    cat_groups: Dict[str, List[str]],
) -> List[Tuple[str, List[str], List[str]]]:
    sets = []
    sets.append(("FULL", cont_all, cat_all))

    for gname, cols in cont_groups.items():
        if len(cols) == 0:
            continue
        cont_kept = [c for c in cont_all if c not in cols]
        sets.append((f"DROP_CONT::{gname}", cont_kept, cat_all))

    for gname, cols in cat_groups.items():
        if len(cols) == 0:
            continue
        cat_kept = [c for c in cat_all if c not in cols]
        sets.append((f"DROP_CAT::{gname}", cont_all, cat_kept))

    return sets

def run_tft_group_lofo_ablation(
    *,
    df_diag: pd.DataFrame,
    fold_meta_df: pd.DataFrame,
    cont_all: List[str],
    cat_all: List[str],
    base_config: dict,
    # TFT params
    regime_col: str = "regime_v3_compressed",
    add_regime_embedding: bool = True,
    encoder_len: int = 24,
    decoder_len: int = 1,
    batch_size: int = 64,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    label_col: str = "target",
    eval_strategy_type: Optional[str] = None,
    verbose: bool = True,
    # 抽出したいfold（例：勝ちfold限定）
    fold_filter: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    戻り値:
      - summary_df: 各条件（FULL, DROP_...）のAUC要約 + ΔAUC
      - fold_delta_df: fold別のAUC（FULLとの差分）一覧（解析用）
      - raw_results: 各条件の run_tft_wfa_with_diagnostics 生結果（辞書）
    """
    # fold限定
    fm = fold_meta_df.copy()
    if fold_filter is not None:
        fm = fm[fm["fold_idx"].isin(fold_filter)].copy().reset_index(drop=True)
    if len(fm) == 0:
        raise ValueError("fold_meta_df becomes empty after fold_filter")

    # グループ生成
    cont_groups, cat_groups = build_feature_groups(cont_all, cat_all)
    feature_sets = build_lofo_feature_sets(cont_all, cat_all, cont_groups, cat_groups)

    # 実行
    raw_results: Dict[str, pd.DataFrame] = {}
    summary_rows = []

    for name, cont_feats, cat_feats in feature_sets:
        if verbose:
            print(f"\n[ABLAT] {name}: cont={len(cont_feats)} cat={len(cat_feats)} folds={len(fm)}")

        res_df = run_tft_wfa_with_diagnostics(
            df_diag=df_diag,
            fold_meta_df=fm,
            cont_features=cont_feats,
            cat_features_base=cat_feats,
            base_config=base_config,
            regime_col=regime_col,
            add_regime_embedding=add_regime_embedding,
            encoder_len=encoder_len,
            decoder_len=decoder_len,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
            label_col=label_col,
            eval_strategy_type=eval_strategy_type,
            verbose=verbose,
        )

        raw_results[name] = res_df

        s = _summarize_auc(res_df)
        s.update({
            "setting": name,
            "n_cont": float(len(cont_feats)),
            "n_cat": float(len(cat_feats)),
        })
        summary_rows.append(s)

    summary_df = pd.DataFrame(summary_rows)

    # FULL基準のΔAUC
    if "FULL" not in raw_results:
        raise RuntimeError("FULL result missing")
    full_res = raw_results["FULL"]
    auc_col = _first_existing_col(full_res, ["auc_test", "test_auc", "AUC_test", "auc"])

    # fold別delta表
    def _get_fold_auc(df: pd.DataFrame, label: str) -> pd.DataFrame:
        out = df[["fold_idx"]].copy()
        out[label] = df[auc_col].astype(float).values
        if "status" in df.columns:
            out[label + "_status"] = df["status"].values
        return out

    full_fold = _get_fold_auc(full_res, "FULL_auc").set_index("fold_idx")

    fold_delta_rows = []
    for name, df in raw_results.items():
        d = _get_fold_auc(df, f"{name}_auc").set_index("fold_idx")
        merged = full_fold.join(d, how="inner")
        # OK foldのみで見たい場合はここでフィルタ可（今はそのまま）
        merged["setting"] = name
        merged["delta_vs_full"] = merged[f"{name}_auc"] - merged["FULL_auc"]
        fold_delta_rows.append(merged.reset_index())

    fold_delta_df = pd.concat(fold_delta_rows, axis=0, ignore_index=True)

    # summary_df に ΔAUC（平均差）を付与
    full_mean = float(summary_df.loc[summary_df["setting"] == "FULL", "auc_mean"].iloc[0])
    summary_df["delta_auc_mean_vs_full"] = summary_df["auc_mean"] - full_mean

    # 見やすい順にソート（落としたら悪化＝重要 → deltaがマイナス大）
    summary_df = summary_df.sort_values("delta_auc_mean_vs_full", ascending=True).reset_index(drop=True)

    return summary_df, fold_delta_df, raw_results

import numpy as np

import pandas as pd

from sklearn.metrics import roc_auc_score

import lightgbm as lgb

def _auc_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if np.unique(y_true).size < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_pred))

def run_lgbm_wfa_baseline_same_fold_features_labels(
    *,
    df_diag: pd.DataFrame,
    fold_meta_df: pd.DataFrame,
    feature_cols: list,
    label_col: str = "target",         # ← TFTと同じラベル列
    date_col: str = "date",            # ← df_diagの日時列（無ければ index を使う）
    valid_col: str = "valid_label",    # ← TFTと同じ有効フラグ
    # val を train の末尾から切り出す（TFTの val_cut と同趣旨）
    val_span_days: int | None = 7,     # 例: crypto なら 7日、株なら 30日など。Noneでvalなし
    # LGBM params（必要なら上書き）
    lgbm_params: dict | None = None,
    # 戦略別AUCも計算したい場合（df_diagに strategy_type or strategy_type_id がある前提）
    eval_strategy_col: str | None = None,   # 例: "strategy_type"  or "strategy_type_id"
    eval_strategy_value=None,               # 例: "Uptrend" or 2
    verbose: bool = True,
):
    """
    同一 fold / 同一特徴量 / 同一label で LightGBM baseline を回すWFA。

    fold_meta_df は少なくとも以下列を持つ想定:
      ["fold_idx","train_start","train_end","test_start","test_end"]

    戻り値:
      results_df: foldごとのAUC等
      diag_df: skip理由（foldごと）
    """

    df = df_diag.copy()

    # date_col が無い場合は index を date として扱う（DatetimeIndex前提）
    if date_col not in df.columns:
        if df.index.name == date_col or isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={df.index.name: date_col})
        else:
            raise KeyError(f"'{date_col}' not in columns, and index is not DatetimeIndex.")

    # フィルタ（TFTと合わせる）
    mask = df[label_col].isin([0, 1])
    if valid_col in df.columns:
        mask &= df[valid_col].astype(bool)

    # feature + date の NaN を落とす（TFTと同じ土俵にする）
    use_cols = list(feature_cols) + [label_col, date_col]
    df = df.loc[mask, use_cols + ([eval_strategy_col] if eval_strategy_col else [])].dropna(subset=use_cols).copy()
    df = df.sort_values(date_col)

    if df.empty:
        raise ValueError("No valid rows after filtering (label/valid/NaN).")

    # LGBM default params
    params = dict(
        n_estimators=5000,
        learning_rate=0.02,
        num_leaves=63,
        min_data_in_leaf=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.0,
        objective="binary",
        metric="auc",
        random_state=42,
        n_jobs=-1,
    )
    if lgbm_params:
        params.update(lgbm_params)

    results = []
    diag = []

    # LightGBMはカテゴリは「category dtype」か categorical_feature 指定で扱える
    # 文字列列があれば category へ（ID列も categoryでOK）
    X_df_all = df[feature_cols].copy()
    for c in feature_cols:
        if X_df_all[c].dtype == "object":
            X_df_all[c] = X_df_all[c].astype("category")

    y_all = df[label_col].astype(int).values
    t_all = pd.to_datetime(df[date_col])

    # fold loop
    for _, row in fold_meta_df.iterrows():
        fold_idx = int(row["fold_idx"])
        train_start = pd.to_datetime(row["train_start"])
        train_end   = pd.to_datetime(row["train_end"])
        test_start  = pd.to_datetime(row["test_start"])
        test_end    = pd.to_datetime(row["test_end"])

        # train/test time mask
        mask_train = (t_all >= train_start) & (t_all <= train_end)
        mask_test  = (t_all >= test_start)  & (t_all <= test_end)

        idx_train_full = np.where(mask_train)[0]
        idx_test = np.where(mask_test)[0]

        if idx_train_full.size == 0 or idx_test.size == 0:
            diag.append(dict(fold_idx=fold_idx, status="skip", skip_reason="empty_train_or_test",
                             train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end))
            continue

        # val cut (trainの末尾をvalにする)
        if val_span_days is None:
            idx_train = idx_train_full
            idx_val = np.array([], dtype=int)
        else:
            val_cut = train_end - pd.Timedelta(days=int(val_span_days))
            mask_train_in = (t_all >= train_start) & (t_all <= val_cut)
            mask_val = (t_all > val_cut) & (t_all <= train_end)
            idx_train = np.where(mask_train_in)[0]
            idx_val = np.where(mask_val)[0]

            # valが空ならフォールバック（train_fullをtrainに）
            if idx_train.size == 0 or idx_val.size == 0:
                idx_train = idx_train_full
                idx_val = np.array([], dtype=int)

        y_train = y_all[idx_train]
        y_test = y_all[idx_test]

        # クラス欠損チェック（最低限 test は両クラス必要）
        if np.unique(y_train).size < 2:
            diag.append(dict(fold_idx=fold_idx, status="skip", skip_reason="train_missing_class",
                             train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end))
            continue
        if np.unique(y_test).size < 2:
            diag.append(dict(fold_idx=fold_idx, status="skip", skip_reason="test_missing_class",
                             train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end))
            continue

        X_train = X_df_all.iloc[idx_train]
        X_test  = X_df_all.iloc[idx_test]

        # val ありなら early stopping
        callbacks = []
        eval_set = None
        if idx_val.size > 0:
            X_val = X_df_all.iloc[idx_val]
            y_val = y_all[idx_val]
            # val側のクラスが片側だけの場合、early stoppingが不安定なので無しで学習
            if np.unique(y_val).size >= 2:
                eval_set = [(X_val, y_val)]
                callbacks = [lgb.early_stopping(stopping_rounds=200, verbose=bool(verbose))]
            else:
                eval_set = None
                callbacks = []

        model = lgb.LGBMClassifier(**params)

        if verbose:
            n_tr, n_val, n_te = idx_train.size, idx_val.size, idx_test.size
            pr = y_train.mean()
            print(f"\n========== LGBM WFA Fold {fold_idx} ==========")
            print(f"[Fold {fold_idx}] n_train={n_tr}, n_val={n_val}, n_test={n_te}, pos_ratio_train={pr:.4f}")

        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_metric="auc",
            callbacks=callbacks,
        )

        # predict proba
        p_train = model.predict_proba(X_train)[:, 1]
        p_test  = model.predict_proba(X_test)[:, 1]

        auc_train = _auc_safe(y_train, p_train)
        auc_test  = _auc_safe(y_test, p_test)

        out = dict(
            fold_idx=fold_idx,
            train_start=train_start, train_end=train_end,
            test_start=test_start, test_end=test_end,
            n_train=int(idx_train.size),
            n_val=int(idx_val.size),
            n_test=int(idx_test.size),
            auc_train_all=auc_train,
            auc_test_all=auc_test,
        )

        # 戦略別AUC（同一test集合の中からさらに抽出）
        if eval_strategy_col is not None and eval_strategy_value is not None and eval_strategy_col in df.columns:
            strat_test = df.iloc[idx_test][eval_strategy_col].values
            m = (strat_test == eval_strategy_value)
            if m.sum() >= 10 and np.unique(y_test[m]).size >= 2:
                out["auc_test_strategy"] = _auc_safe(y_test[m], p_test[m])
                out["n_test_strategy"] = int(m.sum())
            else:
                out["auc_test_strategy"] = np.nan
                out["n_test_strategy"] = int(m.sum())

        results.append(out)

        if verbose:
            print(f"[Fold {fold_idx}] AUC train(all)={auc_train:.3f}, test(all)={auc_test:.3f}")

    results_df = pd.DataFrame(results).sort_values("fold_idx").reset_index(drop=True)
    diag_df = pd.DataFrame(diag).sort_values("fold_idx").reset_index(drop=True)

    return results_df, diag_df

import numpy as np

import pandas as pd

from typing import Dict, Optional, Tuple

def fit_thresholds_on_train_3cat(
    *,
    future_ret: np.ndarray,                 # shape (N_seq,)
    strategy_type_id: np.ndarray,           # shape (N_seq,) int
    train_ids: np.ndarray,                  # seq_id list/array (int)
    pos_rate_up: float = 0.20,
    pos_rate_down: float = 0.20,
    pos_rate_range: float = 0.30,
    strategy_id_map: Optional[Dict[int, str]] = None,
) -> Dict[int, float]:
    if strategy_id_map is None:
        strategy_id_map = {0: "Downtrend", 1: "Range", 2: "Uptrend"}

    train_ids = np.asarray(train_ids)
    thr: Dict[int, float] = {}

    train_ids = train_ids[(train_ids >= 0) & (train_ids < len(future_ret))]

    for sid, sname in strategy_id_map.items():
        idx = train_ids[strategy_type_id[train_ids] == sid]
        if len(idx) == 0:
            thr[sid] = np.nan
            continue

        fr = future_ret[idx]
        fr = fr[np.isfinite(fr)]
        if len(fr) == 0:
            thr[sid] = np.nan
            continue

        if sname == "Uptrend":
            thr[sid] = float(np.quantile(fr, 1.0 - pos_rate_up))
        elif sname == "Downtrend":
            thr[sid] = float(np.quantile(fr, pos_rate_down))
        else:  # Range
            thr[sid] = float(np.quantile(np.abs(fr), pos_rate_range))

    return thr

def apply_thresholds_3cat(
    *,
    future_ret: np.ndarray,                 # (N_seq,)
    strategy_type_id: np.ndarray,           # (N_seq,)
    thresholds: Dict[int, float],
    strategy_id_map: Optional[Dict[int, str]] = None,
) -> np.ndarray:
    if strategy_id_map is None:
        strategy_id_map = {0: "Downtrend", 1: "Range", 2: "Uptrend"}

    labels = np.zeros(len(future_ret), dtype=np.float32)
    fr_all = np.where(np.isfinite(future_ret), future_ret, 0.0).astype(np.float32)

    for sid, sname in strategy_id_map.items():
        thr = thresholds.get(sid, np.nan)
        idx = np.where(strategy_type_id == sid)[0]
        if len(idx) == 0:
            continue

        fr = fr_all[idx]
        if not np.isfinite(thr):
            labels[idx] = 0.0
            continue

        if sname == "Uptrend":
            labels[idx] = (fr >= thr).astype(np.float32)
        elif sname == "Downtrend":
            labels[idx] = (fr <= thr).astype(np.float32)
        else:  # Range
            labels[idx] = (np.abs(fr) <= thr).astype(np.float32)

    return labels

def build_foldwise_labels_3cat_trainfit(
    *,
    df_diag: pd.DataFrame,
    train_ids: np.ndarray,
    future_ret_col: str = "future_ret",      # ★あなたの列名に合わせて変更
    strategy_id_col: str = "strategy_type_id",
    pos_rate_up: float = 0.20,
    pos_rate_down: float = 0.20,
    pos_rate_range: float = 0.30,
    strategy_id_map: Optional[Dict[int, str]] = None,
) -> Tuple[np.ndarray, Dict[int, float]]:
    if strategy_id_map is None:
        strategy_id_map = {0: "Downtrend", 1: "Range", 2: "Uptrend"}

    if future_ret_col not in df_diag.columns:
        raise ValueError(f"{future_ret_col} not in df_diag")
    if strategy_id_col not in df_diag.columns:
        raise ValueError(f"{strategy_id_col} not in df_diag")

    future_ret = df_diag[future_ret_col].to_numpy(dtype=np.float32)
    strategy_type_id = df_diag[strategy_id_col].to_numpy()

    thresholds = fit_thresholds_on_train_3cat(
        future_ret=future_ret,
        strategy_type_id=strategy_type_id,
        train_ids=train_ids,
        pos_rate_up=pos_rate_up,
        pos_rate_down=pos_rate_down,
        pos_rate_range=pos_rate_range,
        strategy_id_map=strategy_id_map,
    )

    labels_per_seq = apply_thresholds_3cat(
        future_ret=future_ret,
        strategy_type_id=strategy_type_id,
        thresholds=thresholds,
        strategy_id_map=strategy_id_map,
    )

    return labels_per_seq, thresholds

def _ensure_date_col(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df2 = df.copy()
    if date_col not in df2.columns:
        if df2.index.name == date_col:
            df2 = df2.reset_index()
        else:
            # index が datetime なら date_col を生やす
            if isinstance(df2.index, pd.DatetimeIndex):
                df2 = df2.copy()
                df2[date_col] = df2.index
            else:
                raise KeyError(f"'{date_col}' not found in columns and index is not DatetimeIndex.")
    df2[date_col] = pd.to_datetime(df2[date_col])
    return df2

def get_ids_in_period(df: pd.DataFrame, date_col: str, start_ts, end_ts) -> np.ndarray:
    start_ts = pd.to_datetime(start_ts)
    end_ts = pd.to_datetime(end_ts)
    mask = (df[date_col] >= start_ts) & (df[date_col] <= end_ts)
    return df.index[mask].to_numpy()

import numpy as np

import pandas as pd

from sklearn.metrics import roc_auc_score

import lightgbm as lgb

def run_lgbm_wfa_foldwise_label(
    df_diag: pd.DataFrame,
    fold_meta_df: pd.DataFrame,
    feature_cols: list,
    *,
    date_col: str = "date",
    future_ret_col: str = "future_ret",     # ←あなたの列名に合わせる
    strategy_id_col: str = "strategy_type_id",
    valid_col: str = "valid_label",
    # foldwise label（train-fit）用：戦略別の正例率目標
    pos_rate_up: float = 0.25,
    pos_rate_down: float = 0.25,
    pos_rate_range: float = 0.25,
    # trainの最後をvalへ
    val_span_days: int = 30,
    lgb_params: dict | None = None,
    num_boost_round: int = 5000,
    early_stopping_rounds: int = 200,
    random_state: int = 42,
):
    df = df_diag.copy()

    # --- 必須列チェック ---
    need = [date_col, future_ret_col, strategy_id_col] + list(feature_cols)
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"df is missing columns: {miss}")

    # --- date 正規化（既にUTCならそのまま） ---
    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    df = df.dropna(subset=[date_col])

    # --- valid フィルタ（あれば） ---
    if valid_col in df.columns:
        df = df[df[valid_col].astype(bool)].copy()

    # --- feature/future_ret/strategy_id の欠損除去（重要） ---
    df = df.dropna(subset=list(feature_cols) + [future_ret_col, strategy_id_col]).copy()
    df = df.sort_values(date_col)

    # ここで df が小さすぎるなら、そもそも情報が落ちてる
    if df.empty:
        return pd.DataFrame(), pd.DataFrame([{
            "fold_idx": -1, "status": "error", "skip_reason": "df_empty_after_filter"
        }])

    # --- 戦略ID→目標正例率 ---
    # （あなたのID割当：Down=0, Range=1, Up=2 の想定）
    pos_rate_by_sid = {
        0: float(pos_rate_down),
        1: float(pos_rate_range),
        2: float(pos_rate_up),
    }

    if lgb_params is None:
        lgb_params = dict(
            objective="binary",
            metric="auc",
            learning_rate=0.03,
            num_leaves=64,
            min_data_in_leaf=200,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            lambda_l2=1.0,
            seed=random_state,
            verbosity=-1,
        )

    ok_rows = []
    diag_rows = []

    # ===== fold loop =====
    for _, row in fold_meta_df.iterrows():
        f = int(row["fold_idx"])
        ts, te = pd.to_datetime(row["train_start"], utc=True), pd.to_datetime(row["train_end"], utc=True)
        ss, se = pd.to_datetime(row["test_start"],  utc=True), pd.to_datetime(row["test_end"],  utc=True)

        # --- まず date で直接切る（indexは使わない） ---
        m_train = (df[date_col] >= ts) & (df[date_col] <= te)
        m_test  = (df[date_col] >= ss) & (df[date_col] <= se)

        df_train_full = df.loc[m_train].copy()
        df_test = df.loc[m_test].copy()

        if df_train_full.empty or df_test.empty:
            diag_rows.append(dict(
                fold_idx=f, status="skip", skip_reason="empty_split_time_mask",
                n_train=len(df_train_full), n_val=0, n_test=len(df_test),
                train_start=ts, train_end=te, test_start=ss, test_end=se,
            ))
            continue

        # --- val 分割：train の末尾 val_span_days を val にする ---
        val_cut = te - pd.Timedelta(days=val_span_days)
        df_train = df_train_full[df_train_full[date_col] <= val_cut].copy()
        df_val   = df_train_full[df_train_full[date_col] >  val_cut].copy()

        if df_train.empty or df_val.empty:
            diag_rows.append(dict(
                fold_idx=f, status="skip", skip_reason="empty_train_or_val_after_cut",
                n_train=len(df_train), n_val=len(df_val), n_test=len(df_test),
                train_start=ts, train_end=te, test_start=ss, test_end=se,
            ))
            continue

        # ===== foldwise label（train-fit）=====
        # 戦略ごとに train の future_ret 分布から閾値を決めて 0/1 を作る
        def _make_labels(_df_part: pd.DataFrame, thresholds_by_sid: dict[int, float]) -> np.ndarray:
            y = np.full(len(_df_part), -1, dtype=int)
            sids = _df_part[strategy_id_col].astype(int).to_numpy()
            fr   = _df_part[future_ret_col].astype(float).to_numpy()
            for sid, thr in thresholds_by_sid.items():
                idx = np.where(sids == sid)[0]
                if idx.size == 0:
                    continue
                if sid == 2:      # Uptrend: future_ret >= thr を1
                    y[idx] = (fr[idx] >= thr).astype(int)
                elif sid == 0:    # Downtrend: future_ret <= thr を1
                    y[idx] = (fr[idx] <= thr).astype(int)
                else:             # Range: |future_ret| <= thr を1
                    y[idx] = (np.abs(fr[idx]) <= thr).astype(int)
            return y

        thresholds = {}
        for sid, pos_rate in pos_rate_by_sid.items():
            fr_train_sid = df_train.loc[df_train[strategy_id_col].astype(int) == sid, future_ret_col].astype(float)
            if fr_train_sid.empty:
                thresholds[sid] = np.nan
                continue
            if sid == 2:   # Up: 上側(1-pos_rate)分位
                thresholds[sid] = np.quantile(fr_train_sid.values, 1.0 - pos_rate)
            elif sid == 0: # Down: 下側(pos_rate)分位
                thresholds[sid] = np.quantile(fr_train_sid.values, pos_rate)
            else:          # Range: |ret| の下側(pos_rate)分位
                thresholds[sid] = np.quantile(np.abs(fr_train_sid.values), pos_rate)

        y_train = _make_labels(df_train, thresholds)
        y_val   = _make_labels(df_val, thresholds)
        y_test  = _make_labels(df_test, thresholds)

        # -1 を落とす（ラベル不可＝該当戦略が無い等）
        def _filter_xy(_df_part, y):
            m = (y >= 0)
            return _df_part.loc[m, feature_cols], y[m]

        Xtr, ytr = _filter_xy(df_train, y_train)
        Xva, yva = _filter_xy(df_val,   y_val)
        Xte, yte = _filter_xy(df_test,  y_test)

        if len(Xtr)==0 or len(Xva)==0 or len(Xte)==0:
            diag_rows.append(dict(
                fold_idx=f, status="skip", skip_reason="empty_after_label_filter",
                n_train=len(Xtr), n_val=len(Xva), n_test=len(Xte),
                thresholds=thresholds,
            ))
            continue

        # AUC計算には両クラス必要
        if (np.unique(ytr).size < 2) or (np.unique(yte).size < 2):
            diag_rows.append(dict(
                fold_idx=f, status="skip", skip_reason="missing_class_after_label",
                n_train=len(Xtr), n_val=len(Xva), n_test=len(Xte),
                ytr_counts=dict(pd.Series(ytr).value_counts()),
                yte_counts=dict(pd.Series(yte).value_counts()),
                thresholds=thresholds,
            ))
            continue

        # ===== LGBM 学習 =====
        dtrain = lgb.Dataset(Xtr, label=ytr)
        dvalid = lgb.Dataset(Xva, label=yva, reference=dtrain)

        booster = lgb.train(
            params=lgb_params,
            train_set=dtrain,
            valid_sets=[dvalid],
            num_boost_round=num_boost_round,
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )

        p_tr = booster.predict(Xtr, num_iteration=booster.best_iteration)
        p_te = booster.predict(Xte, num_iteration=booster.best_iteration)

        auc_tr = roc_auc_score(ytr, p_tr)
        auc_te = roc_auc_score(yte, p_te)

        ok_rows.append(dict(
            fold_idx=f,
            auc_train_all=float(auc_tr),
            auc_test_all=float(auc_te),
            n_train=len(Xtr), n_val=len(Xva), n_test=len(Xte),
            thresholds=thresholds,
            best_iter=int(booster.best_iteration),
        ))

    results_df = pd.DataFrame(ok_rows).sort_values("fold_idx").reset_index(drop=True)
    diag_df    = pd.DataFrame(diag_rows).sort_values("fold_idx").reset_index(drop=True)
    return results_df, diag_df

import joblib

import os

import joblib

import os

import pandas as pd

import numpy as np

import numpy as np # Ensure numpy is imported

import torch

from torch.utils.data import Dataset

import numpy as np

import pandas as pd

class CryptoBinaryDataset(Dataset):
    """
    時系列データとそれに対応するバイナリターゲット、グループID、時間インデックスを保持するカスタムデータセット。
    TFTモデルへの入力形式に合わせることを目的とする。
    """
    def __init__(self, dataframe, encoder_len, decoder_len, real_feature_cols, categorical_feature_cols, target_col, group_id_col, time_idx_col, train_end_timestamp=None):
        """
        Args:
            dataframe (pd.DataFrame): 時系列データを含むDataFrame。
                                       カラムは features, target, group_id, time_idx を含む必要がある。
            encoder_len (int): エンコーダ部分の時系列長。
            decoder_len (int): デコーダ部分の時系列長。
            real_feature_cols (list): 実数特徴量のカラム名のリスト。
            categorical_feature_cols (list): カテゴリ特徴量のカラム名のリスト。
            target_col (str): ターゲットのカラム名。
            group_id_col (str): グループIDのカラム名（例: 時系列ID）。
            time_idx_col (str): 時間インデックスのカラム名。
            train_end_timestamp (pd.Timestamp, optional): 学習期間の最終タイムスタンプ。時間減衰重み付けに使用。
                                                        デフォルトは None。
        """
        if dataframe is None or dataframe.empty:
            raise ValueError("Input DataFrame cannot be None or empty.")

        self.dataframe = dataframe.copy() # 元のDataFrameを変更しないようにコピー
        self.encoder_len = encoder_len
        self.decoder_len = decoder_len
        self.seq_len = encoder_len + decoder_len # 全体のシーケンス長
        self.real_feature_cols = real_feature_cols
        self.categorical_feature_cols = categorical_feature_cols
        self.target_col = target_col
        self.group_id_col = group_id_col
        self.time_idx_col = time_idx_col
        self.train_end_timestamp = train_end_timestamp # 学習期間の最終タイムスタンプを保持

        # time_idx が0から始まる連続した整数であることを確認
        # 各 group_id ごとに time_idx を再インデックス付け
        print("Debug(Dataset): Re-indexing time_idx for each group...")
        self.dataframe[self.time_idx_col] = self.dataframe.groupby(self.group_id_col).cumcount()
        print("Debug(Dataset): Re-indexing complete.")


        # 各グループの開始インデックスと終了インデックスを計算
        self.group_indices = {}
        self.sequences = [] # 各シーケンスのメタデータを格納
        # 最小必要な長さを計算
        min_sequence_length = self.encoder_len + self.decoder_len

        print(f"Debug(Dataset): Calculating group indices and filtering groups smaller than {min_sequence_length}...")
        for group_id, group_df in self.dataframe.groupby(self.group_id_col):
            if len(group_df) >= min_sequence_length:
                # このグループから抽出できるシーケンスの数を計算
                # 各シーケンスは encoder_len + decoder_len の長さを持つ
                # time_idx が 0 から始まるとして、最後のシーケンスの開始 time_idx は len(group_df) - (encoder_len + decoder_len)
                # つまり、開始 time_idx は 0 から len(group_df) - min_sequence_length まで
                num_possible_sequences = len(group_df) - min_sequence_length + 1 # +1 は0からのインデックスを考慮

                for i in range(num_possible_sequences):
                    # シーケンスの開始 time_idx
                    seq_start_time_idx = i
                    # このシーケンスに対応するDataFrame内の開始・終了行インデックスを取得 (iloc)
                    seq_start_iloc_in_group = i # グループ内のiloc開始位置
                    seq_end_iloc_in_group = seq_start_iloc_in_group + self.seq_len -1 # グループ内のiloc終了位置 (含む)


                    # 元のDataFrameにおけるiloc開始・終了位置を計算 (ilocは0から始まる整数インデックス)
                    original_df_iloc_start = group_df.iloc[seq_start_iloc_in_group].name # name()で元のDataFrameのindexを取得
                    original_df_iloc_end = group_df.iloc[seq_end_iloc_in_group].name

                    # シーケンスの終了時刻を取得 (デコーダー部分の最後のタイムスタンプ)
                    sequence_end_time = group_df.iloc[seq_end_iloc_in_group].name # time_idx が seq_len - 1 の行のインデックス（datetime）

                    self.sequences.append({
                        'group_id': group_id,
                        'start_iloc_in_group': seq_start_iloc_in_group,
                        'end_iloc_in_group': seq_end_iloc_in_group, # このインデックスを含む
                        'sequence_end_time': sequence_end_time, # シーケンスの最終時刻
                        # ターゲットはデコーダの最初のステップのターゲットを使用する前提
                        'target': group_df[self.target_col].iloc[seq_start_iloc_in_group + self.encoder_len : seq_start_iloc_in_group + self.encoder_len + self.decoder_len].values
                    })
            else:
                # デバッグプリントを追加
                 print(f"Debug(Dataset): Skipping group {group_id} due to insufficient length ({len(group_df)} < {min_sequence_length}).")

        print(f"Debug(Dataset): Finished processing groups. Generated {len(self.sequences)} sequences.")

        if not self.sequences:
             print("Warning: No sequences were generated with the given parameters.")


    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_info = self.sequences[idx]
        group_id = seq_info['group_id']
        start_iloc_in_group = seq_info['start_iloc_in_group']
        end_iloc_in_group = seq_info['end_iloc_in_group'] # このインデックスを含む

        # グループデータ全体を取得し、ilocでシーケンスをスライス
        group_df = self.dataframe[self.dataframe[self.group_id_col] == group_id].reset_index(drop=True) # グループを抽出してilocのためにインデックスをリセット

        # シーケンス全体 (encoder + decoder)
        sequence_df = group_df.iloc[start_iloc_in_group : start_iloc_in_group + self.seq_len].copy() # 終了インデックスはilocに含まれないため +self.seq_len

        # encoder および decoder 部分を分離
        encoder_df = sequence_df.iloc[:self.encoder_len]
        decoder_df = sequence_df.iloc[self.encoder_len:] # decoder_len の長さになるはず

        # 特徴量とターゲットを抽出
        encoder_real_input = encoder_df[self.real_feature_cols].values
        encoder_categorical_input = encoder_df[self.categorical_feature_cols].values
        decoder_real_input = decoder_df[self.real_feature_cols].values
        decoder_categorical_input = decoder_df[self.categorical_feature_cols].values
        target = sequence_df[self.target_col].iloc[self.encoder_len:self.encoder_len+self.decoder_len].values # デコーダ部分のターゲット


        # time_idx を抽出
        encoder_time_idx = encoder_df[self.time_idx_col].values
        decoder_time_idx = decoder_df[self.time_idx_col].values

        # numpy array を tensor に変換
        encoder_real_input = torch.tensor(encoder_real_input, dtype=torch.float32)
        encoder_categorical_input = torch.tensor(encoder_categorical_input, dtype=torch.int64) # カテゴリはLongTensor

        target = torch.tensor(target, dtype=torch.float32).unsqueeze(-1) # BCEWithLogitsLossはfloatを期待, (B, T, 1) or (B, T) にするため unsqueeze

        encoder_time_idx = torch.tensor(encoder_time_idx, dtype=torch.int64)
        decoder_time_idx = torch.tensor(decoder_time_idx, dtype=torch.int64)


        # 静的特徴量は現在サポートしないため空のテンソル
        static_real_input = torch.empty(0, dtype=torch.float32)
        static_categorical_input = torch.empty(0, dtype=torch.int64)

        # train_end_timestamp をバッチに含める
        # None の可能性があるため文字列に変換するか、適切な形式で渡す
        # ここでは文字列に変換して default_collate が処理できるようにする
        train_end_ts_str = str(self.train_end_timestamp) if self.train_end_timestamp is not None else "None"

        # sequence_end_time も Timestamp なので文字列に変換してバッチに含める
        sequence_end_time_str = str(seq_info['sequence_end_time'])


        # バッチに含めるデータを辞書として返す
        batch_data = {
            "encoder_real_input": encoder_real_input, # (encoder_len, num_real_features)
            "encoder_categorical_input": encoder_categorical_input, # (encoder_len, num_cat_features)
            "decoder_real_input": decoder_real_input, # (decoder_len, num_real_features)
            "decoder_categorical_input": decoder_categorical_input, # (decoder_len, num_cat_features)
            "target": target, # (decoder_len, 1) or (decoder_len,)
            "encoder_time_idx": encoder_time_idx, # (encoder_len,)
            "decoder_time_idx": decoder_time_idx, # (decoder_len,)
            "static_real_input": static_real_input, # (num_static_real_features,) - currently empty
            "static_categorical_input": static_categorical_input, # (num_static_cat_features,) - currently empty
            "group_id": torch.tensor(group_id, dtype=torch.int64), # スカラー
            "sequence_end_time": sequence_end_time_str, # ★ 修正: 文字列として渡す
            "train_end_timestamp": train_end_ts_str # ★ 修正: 文字列として渡す
        }

        return batch_data

def create_dataframe_for_dataset(X_list, y_list, original_indices_list, integrated_clusters, feature_names_3d, seq_length, original_indices_filtered, target_strategy_name):
    """
    Creates a pandas DataFrame suitable for CryptoBinaryDataset from a list of sequences.
    Adds necessary columns like 'group_id', 'time_idx', 'target', 'original_cluster_id', 'strategy_name'.

    Args:
        X_list (list of np.ndarray): List of 3D numpy arrays, each representing a sequence
                                      of features for a time series (num_timesteps, num_features).
        y_list (list of np.ndarray): List of 1D numpy arrays, each representing the target
                                      label for the corresponding sequence (single value or replicated).
        original_indices_list (list of pd.DatetimeIndex or np.ndarray): List of original indices (timestamps or numeric time_idx)
                                                                         for each sequence. Each element is the index sequence for one full sequence.
        integrated_clusters (np.ndarray): 1D numpy array of cluster IDs for ALL filtered sequences,
                                          aligned with original_indices_filtered. These IDs are from the strategy-specific clustering.
        feature_names_3d (list): List of feature names corresponding to the last dimension of X arrays (from X_3d_numpy).
        seq_length (int): The expected length of each sequence (encode_len + decoder_len).
        original_indices_filtered (pd.Index or np.ndarray): The full original indices after filtering,
                                                              aligned with integrated_clusters and integrated_strategy_names.
        target_strategy_name (str): The specific strategy name this DataFrame is being created for.
                                    Used to assign the strategy name column and for debugging.


    Returns:
        pd.DataFrame: DataFrame with columns ['group_id', 'time_idx', 'target', 'original_cluster_id', 'strategy_name'] + feature_names_3d,
                      or None if no valid sequences are processed or required data is missing.
    """
    processed_data = []
    group_id_counter = 0

    # Check if essential inputs are valid
    if feature_names_3d is None or len(feature_names_3d) == 0:
         print("Error(create_dataframe_for_dataset): feature_names_3d is missing or empty.")
         return None

    # Dynamically determine the number of features from the first sequence in the list
    num_features_in_data = 0
    current_feature_names_for_df = [] # Initialize an empty list for feature names to use in DataFrame

    if X_list and isinstance(X_list[0], np.ndarray) and X_list[0].ndim > 1:
        num_features_in_data = X_list[0].shape[-1]
        # Adjust the feature names list based on the actual number of features in the data
        if num_features_in_data != len(feature_names_3d):
            print(f"Debug(create_dataframe): Warning: feature_names_3d length ({len(feature_names_3d)}) does not match actual feature count in X_list ({num_features_in_data}). Adjusting feature names list.")
            # Use the first `num_features_in_data` names from feature_names_3d
            current_feature_names_for_df = feature_names_3d[:num_features_in_data]
        else:
            current_feature_names_for_df = feature_names_3d # Use the provided list if lengths match
    else:
        print("Error(create_dataframe_for_dataset): X_list is empty or not in expected format. Cannot determine feature count.")
        return None

    # Ensure the number of features in the input data is consistent with the (potentially adjusted) feature name list
    if num_features_in_data != len(current_feature_names_for_df):
         print(f"Error(create_dataframe_for_dataset): Feature count mismatch after adjusting feature names. Data has {num_features_in_data} features, but list has {len(current_feature_names_for_df)} names. Cannot proceed.")
         return None


    # Ensure original_indices_filtered is a pandas Index/DatetimeIndex for efficient lookup
    original_indices_full_pd = None
    if isinstance(original_indices_filtered, (pd.Index, pd.DatetimeIndex)):
         original_indices_full_pd = original_indices_filtered
    elif isinstance(original_indices_filtered, np.ndarray):
         try:
              if pd.api.types.is_datetime64_any_dtype(original_indices_filtered):
                   original_indices_full_pd = pd.DatetimeIndex(original_indices_filtered)
              else:
                   original_indices_full_pd = pd.Index(original_indices_filtered)
         except Exception as e:
              print(f"Debug(create_dataframe): Error converting original_indices_filtered np.ndarray to pandas Index: {e}")
              return None
    else:
         print(f"Debug(create_dataframe): original_indices_filtered is not a pandas Index/DatetimeIndex or np.ndarray. Got {type(original_indices_filtered)}.")
         return None


    # Create a mapping from timestamp/index to its position in the full filtered index for integrated_clusters lookup
    # This map is crucial for aligning sequence start indices with the full integrated arrays
    if original_indices_full_pd is not None:
        try:
            # Use a faster approach for creating the map if indices are unique
            # If not unique, this might not work as expected for sequences
            # For time series, indices *should* be unique.
            full_index_to_pos = pd.Series(range(len(original_indices_full_pd)), index=original_indices_full_pd)
            # print(f"Debug(create_dataframe): Created lookup map for {len(full_index_to_pos)} original indices.")
        except Exception as e:
            print(f"Debug(create_dataframe): Error creating lookup map for original_indices_full_pd: {e}")
            # Fallback to a list conversion if map creation fails (e.g., unhashable types, though DatetimeIndex should be hashable)
            try:
                 original_indices_full_list = original_indices_full_pd.tolist()
                 full_index_to_pos = None # Indicate using list lookup
                 print("Debug(create_dataframe): Falling back to list lookup for original indices.")
            except Exception as e_list:
                 print(f"Debug(create_dataframe): Error converting original_indices_full_pd to list for lookup: {e_list}")
                 return None # Cannot proceed without a valid lookup mechanism
    else:
        print("Debug(create_dataframe): original_indices_full_pd is None. Cannot create lookup map.")
        return None


    print(f"Debug(create_dataframe): Processing {len(X_list)} sequences for strategy '{target_strategy_name}'...")

    # Ensure X_list, y_list, and original_indices_list have the same length
    if not (len(X_list) == len(y_list) == len(original_indices_list)):
         print(f"Debug(create_dataframe): Input list lengths mismatch: X={len(X_list)}, y={len(y_list)}, indices={len(original_indices_list)}. Skipping DataFrame creation.")
         return None


    for i, (X_seq, y_seq, indices_seq) in enumerate(zip(X_list, y_list, original_indices_list)):
        # Add debug print for sequence index and shapes
        # print(f"Debug(create_dataframe): Processing sequence {i}/{len(X_list)-1}. X_seq shape: {X_seq.shape}, y_seq shape: {y_seq.shape if isinstance(y_seq, np.ndarray) else 'N/A'}, indices_seq length: {len(indices_seq)}")


        # Ensure the sequence length matches the expected seq_length
        # X_seq shape is (num_timesteps, num_features)
        # indices_seq length is num_timesteps
        # y_seq shape is (predict_len,) or (1,) if target is single value per sequence
        # CryptoBinaryDataset expects y_seq to be (predict_len,) with replicated target

        # --- Added check to skip sequences with target -1 ---
        if y_seq.ndim > 0 and len(y_seq) > 0 and y_seq[0] == -1:
            # print(f"Debug(create_dataframe): Skipping sequence {i}: Target is -1.")
            continue
        # --- End added check ---

        # Check if X_seq is a valid numpy array and has the correct number of dimensions
        if not isinstance(X_seq, np.ndarray) or X_seq.ndim != 2:
             print(f"Debug(create_dataframe): Skipping sequence {i}: X_seq is not a 2D numpy array. Shape/Type: {X_seq.shape if isinstance(X_seq, np.ndarray) else type(X_seq)}.")
             continue

        # Check sequence length consistency
        if X_seq.shape[0] != seq_length or len(indices_seq) != seq_length:
            print(f"Debug(create_dataframe): Skipping sequence {i}: Length mismatch. X_seq.shape[0]={X_seq.shape[0]}, len(indices_seq)={len(indices_seq)}, expected seq_length={seq_length}.")
            continue # Skip this sequence if lengths are inconsistent

        # Check feature count consistency for the current sequence
        if X_seq.shape[-1] != num_features_in_data:
             print(f"Debug(create_dataframe): Skipping sequence {i}: Feature count mismatch. X_seq.shape[-1]={X_seq.shape[-1]}, expected {num_features_in_data}. This suggests an issue in how X_list was prepared.")
             continue


        # --- Get the original cluster ID for this sequence ---
        # This should be looked up using the *first* original index of the sequence
        # within the full original_indices_filtered array, which aligns with integrated_clusters.
        original_cluster_id = -3 # Default to indicate not found or processed

        if len(indices_seq) > 0:
            first_original_index = indices_seq[0]

            # Look up the position of this original index in the full filtered index
            if full_index_to_pos is not None: # Using dictionary or Series lookup
                position_in_full = full_index_to_pos.get(first_original_index)
                if position_in_full is None: # Handle case where get returns None
                     position_in_full = -1 # Indicate not found
                else: # Convert Series output (can be Series if index is not unique) to scalar int
                    if isinstance(position_in_full, pd.Series):
                         if not position_in_full.empty:
                             position_in_full = position_in_full.iloc[0] # Take the first position if duplicates exist
                         else:
                             position_in_full = -1 # Indicate not found if Series is empty after get
                    position_in_full = int(position_in_full) # Ensure it's an integer position

            else: # Using list lookup fallback
                try:
                     position_in_full = original_indices_full_list.index(first_original_index)
                except ValueError:
                     position_in_full = -1 # Not found


            # Use the found position to get the cluster ID from the integrated clusters
            if position_in_full != -1 and position_in_full < len(integrated_clusters):
                original_cluster_id = integrated_clusters[position_in_full]
            else:
                # If original index not found in full filtered indices, assign default cluster ID
                original_cluster_id = -3
                print(f"Debug(create_dataframe): Warning: Sequence {i} (first index {first_original_index}) not found in full filtered indices. Assigning cluster -3.")

        else:
            # print(f"Debug(create_dataframe): Skipping sequence {i}: Empty original_indices_list for sequence.")
            continue # Skip if original indices list is empty

        # --- End Get cluster ID ---


        # Debugging: Print sequence info
        # print(f"Debug(create_dataframe): Processing sequence {i}: First original index={first_original_index}, Cluster ID={original_cluster_id}, Strategy Name={target_strategy_name}.")

        sequence_data_rows = [] # List to hold data for each timestep in the current sequence

        # Structure data for DataFrame
        # Each timestep in the sequence becomes a row in the DataFrame
        # Use current_feature_names_for_df which is aligned with X_seq.shape[-1]
        sequence_processing_successful = True # Flag to track if this sequence is processed successfully
        # Correct the range to iterate up to X_seq.shape[0] (which should be seq_length)
        for t in range(X_seq.shape[0]): # Use X_seq.shape[0] instead of seq_length
            # Add debug print for timestep index
            # print(f"Debug(create_dataframe): Processing sequence {i}, timestep {t}/{X_seq.shape[0]-1}.")
            timestep_data = {
                'group_id': group_id_counter,
                'time_idx': t, # Relative time index within the sequence
                # Target is the same for all timesteps in the sequence (replicated by CryptoBinaryDataset)
                # Target should be a single value (0 or 1) as -1 targets are skipped
                'target': y_seq[0] if y_seq.ndim > 0 else y_seq, # Take the first (or only) target value
                'original_cluster_id': original_cluster_id, # Add cluster ID for context
                'strategy_name': target_strategy_name # Assign the target strategy name directly
            }

            # Add feature values for this timestep using current_feature_names_for_df
            # The check for feature dimension mismatch should now be handled by the dynamic adjustment above.
            # Add debug print here to verify the lengths match after adjustment
            # if X_seq.shape[-1] != len(current_feature_names_for_df):
            #      print(f"Debug(create_dataframe): Still mismatch after adjustment? X_seq.shape[-1]={X_seq.shape[-1]}, len(current_feature_names_for_df)={len(current_feature_names_for_df)}. Sequence {i}, Timestep {t}.")
            #      sequence_processing_successful = False # Mark as unsuccessful
            #      break # Exit inner timestep loop


            # --- REVISED FEATURE EXTRACTION LOOP ---
            # Ensure we are iterating up to the number of features in the data (num_features_in_data)
            # and using the corresponding name from current_feature_names_for_df
            if X_seq.shape[-1] != num_features_in_data:
                 print(f"Error(create_dataframe): Feature count mismatch for sequence {i}, timestep {t}. X_seq.shape[-1]={X_seq.shape[-1]}, expected {num_features_in_data}. Skipping sequence.")
                 sequence_processing_successful = False
                 break # Exit timestep loop if feature count is inconsistent

            if len(current_feature_names_for_df) != num_features_in_data:
                 print(f"Error(create_dataframe): Internal feature name list length mismatch for sequence {i}, timestep {t}. len(current_feature_names_for_df)={len(current_feature_names_for_df)}, expected {num_features_in_data}. Skipping sequence.")
                 sequence_processing_successful = False
                 break # Exit timestep loop if internal list is inconsistent


            for j in range(num_features_in_data): # Iterate based on the actual number of features in the data
                 feature_name = current_feature_names_for_df[j] # Get the corresponding feature name

                 try:
                     # Access the feature value at timestep t for feature index j
                     # X_seq is (num_timesteps, num_features)
                     feature_value = X_seq[t, j]
                     timestep_data[feature_name] = feature_value

                 except IndexError as e:
                     # This catch should ideally not be needed if checks above are correct, but keep as safeguard
                     print(f"Debug(create_dataframe): IndexError accessing X_seq[{t}, {j}] for sequence {i}, timestep {t}, feature {j} ('{feature_name}'). X_seq shape: {X_seq.shape}. Error: {e}")
                     sequence_processing_successful = False # Mark as unsuccessful
                     break # Exit inner feature loop

            if not sequence_processing_successful:
                 break # Exit inner timestep loop if feature loop failed


            sequence_data_rows.append(timestep_data) # Append data for this timestep

        # After processing all timesteps for the current sequence
        if sequence_processing_successful and sequence_data_rows: # Only append if the entire sequence was processed without issues and is not empty
             processed_data.extend(sequence_data_rows) # Add all timestep rows for this sequence to the main list
             group_id_counter += 1 # Increment group_id only if the sequence was fully processed


    if not processed_data:
        print("Warning (create_dataframe): No valid sequences found after filtering (e.g., all targets were -1 or processing errors). Returning None.")
        return None

    # Create DataFrame
    df = pd.DataFrame(processed_data)

    # --- Categorical Feature Mapping and Clamping ---
    # Define categorical feature names that need specific processing
    # This list should be consistent with the one used in the training cell (8T7A9fQxlrAj/ss65vZTb6uon)
    # Use a comprehensive list that covers all possible categorical features
    # ADD 'original_cluster_id' to this list
    categorical_feature_names_list_full = [
        "close_cusum", "dex_volume_cusum", "active_senders_cusum", "active_receivers_cusum",
        "address_count_sum_cusum", "contract_calls_cusum", "whale_tx_count_cusum",
        "sign_entropy_12_cusum","sign_entropy_24_cusum",  "buy_sell_ratio_cusum",
        "MA_6_24_cross_flag", "MA_12_48_cross_flag", "MA_24_72_cross_flag",
        "MA_slope_6_24_change_flag", "MA_12_48_change_flag",
        "MA_slope_pct_change_6_24_change_flag", "MA_slope_pct_change_12_48_change_flag",
        "MA_slope_pct_change_24_72_change_flag","volatility_change_flag",
        "MA_6_24_72_trend_flag", # Added this feature
        "hour", "day_of_week", "day", # Time-based features
        "original_cluster_id" # ADDED: Treat cluster ID as a categorical feature
    ]


    # Define features that need -1, 0, 1 -> 0, 1, 2 mapping
    # These are typically binary flag features that might have -1 as a state
    # 'original_cluster_id' does not need this mapping, as its IDs are >= -3
    features_to_map_neg1_to_0 = [
        "close_cusum", "dex_volume_cusum", "active_senders_cusum", "active_receivers_cusum",
        "address_count_sum_cusum", "contract_calls_cusum", "whale_tx_count_cusum",
        "sign_entropy_12_cusum","sign_entropy_24_cusum",  "buy_sell_ratio_cusum",
        "MA_6_24_cross_flag", "MA_12_48_cross_flag", "MA_24_72_cross_flag",
        "MA_slope_6_24_change_flag", "MA_12_48_change_flag",
        "MA_slope_pct_change_6_24_change_flag", "MA_slope_pct_change_12_48_change_flag",
        "MA_slope_pct_change_24_72_change_flag","volatility_change_flag",
        "MA_6_24_72_trend_flag" # Added this feature to the mapping list
    ]

    # Define features that need clamping and their max expected values (vocab_size - 1)
    # The actual vocab size should be determined from the data or known properties,
    # but clamping provides a safeguard.
    # Max values should be consistent with how these features were generated.
    # 'original_cluster_id' needs special handling for its range (-3, -1, 0, 1, ...)
    # It might be better to map -3 and -1 to a reserved category index (e.g., 0) and shift others.
    # Or, treat -3 and -1 as distinct categories if they are meaningful.
    # Let's map -3 and -1 to index 0, and shift all other cluster IDs (0, 1, 2...) by +1 -> (1, 2, 3...).
    # This ensures all categorical IDs are >= 0.
    features_to_clamp = {
        "hour": 23,       # 0-23
        "day_of_week": 6, # 0-6
        "day": 365        # Assuming day of year 0-365. Note: Needs adjustment for leap years if applicable.
    }

    # Apply mapping and clamping to the DataFrame columns
    print("\nDebug(create_dataframe): Applying categorical feature mapping and ensuring non-negative integers...")
    # Iterate through the comprehensive list of categorical feature names
    for col in categorical_feature_names_list_full:
        # Only process the column if it exists in the created DataFrame
        if col in df.columns:
            original_dtype = df[col].dtype # Store original dtype for debug
            try:
                # Ensure column is numeric before mapping/clamping
                # Use errors='coerce' to turn non-numeric values into NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')

                # Handle potential NaNs that might be created by coerce
                # Mapping/clamping should ideally operate on non-NaN values
                # Let's fill NaNs temporarily for mapping/clamping if necessary,
                # or ensure the logic handles them. The replace/clip methods
                # generally handle NaNs by leaving them as NaN.

                if col == 'original_cluster_id':
                     # Map -3 and -1 to category 0, and shift all other cluster IDs (0, 1, 2...) by +1 -> (1, 2, 3...).
                     # This ensures all categorical IDs are >= 0.
                     # Use a temporary placeholder for NaNs if any exist before mapping
                     df[col] = df[col].fillna(-999).astype(float) # Handle NaNs temporarily
                     # Use boolean indexing or a vectorized approach for clarity and potential performance
                     # Create a mask for values that should be mapped to 0
                     mask_map_to_0 = (df[col] == -3.0) | (df[col] == -1.0)
                     # Create a mask for values that should be shifted by +1
                     mask_shift_by_1 = (df[col] >= 0.0)

                     # Apply the mapping
                     df[col][mask_map_to_0] = 0.0
                     df[col][mask_shift_by_1] = df[col][mask_shift_by_1] + 1.0

                     # Convert to integer. Fill placeholder (-999) with 0 after mapping
                     df[col] = df[col].replace(-999, 0.0).astype(np.int64)
                     # print(f"Debug(create_dataframe): Applied cluster ID mapping (-3,-1->0, >=0 -> +1) and filled NaNs with 0 for '{col}'.")


                elif col in features_to_map_neg1_to_0:
                    # Apply -1 -> 0, 0 -> 1, 1 -> 2 mapping for specific flags
                    # Ensure float type for replace, then convert to int64
                    # Replace NaNs with a temporary value (e.g., -999) before mapping, then fill after
                    df[col] = df[col].fillna(-999).astype(float).replace({-1.0: 0.0, 0.0: 1.0, 1.0: 2.0})
                    # Convert to integer. NaNs (originally -999) will remain NaN if not handled.
                    df[col] = df[col].astype(np.int64, errors='ignore')
                    # After mapping, explicitly set any remaining NaNs (originally -999) to a placeholder value (e.g., 0)
                    df[col] = df[col].fillna(0).astype(np.int64)

                    # print(f"Debug(create_dataframe): Applied -1,0,1 -> 0,1,2 mapping and filled NaNs with 0 for '{col}'.")

                elif col in features_to_clamp:
                     # Apply clamping
                     clamp_max = features_to_clamp[col]
                     # Ensure numeric and then clamp, coercing errors to NaN
                     df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
                     df[col] = np.clip(df[col], 0.0, float(clamp_max))
                     # Convert to integer after clamping. NaNs will likely remain NaN if not handled explicitly.
                     df[col] = df[col].astype(np.int64, errors='ignore')
                     # Fill NaNs with 0 after clamping
                     df[col] = df[col].fillna(0).astype(np.int64)
                    # print(f"Debug(create_dataframe): Applied clamping [0, {clamp_max}] and filled NaNs with 0 to '{col}'.")
                else:
                    # Ensure other listed categorical features are numeric and then integer type
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
                    df[col] = df[col].astype(np.int64, errors='ignore')
                     # Fill NaNs with 0
                    df[col] = df[col].fillna(0).astype(np.int64)
                    # print(f"Debug(create_dataframe): Ensured '{col}' is numeric, int64, and filled NaNs with 0.")

                # Debug: Print stats after processing
                # Use value_counts to get counts for all unique values, including 0 if present
                # value_counts = df[col].value_counts(dropna=False).sort_index() # Include NaN counts if any remain
                # min_val_after = df[col].min()
                # max_val_after = df[col].max()

                # print(f"  Debug(create_dataframe): Processed '{col}': Min={min_val_after}, Max={max_val_after}")
                # print(f"  Debug(create_dataframe): Value Counts for '{col}':\n{value_counts}")
                # # Check if minimum value is less than 0
                # if min_val_after < 0:
                #     print(f"  !!! Warning: Negative value ({min_val_after}) found in '{col}' after processing.")


            except Exception as e:
                print(f"Error(create_dataframe): Failed to map/clamp categorical feature '{col}'. Original dtype was {original_dtype}. Error: {e}")
                # If mapping/clamping fails for a column, it might still be included
                # but with potentially incorrect values or dtype.
                # A more robust approach might be to drop the column or raise an error.
                # For now, log the error and continue.
                pass # Continue processing other columns

        else:
            # This case indicates a discrepancy between feature_names and categorical_feature_names_list
            # If a column is listed as categorical but not in the DataFrame, it's likely a configuration error.
            # print(f"Warning(create_dataframe): Categorical feature '{col}' listed in categorical_feature_names_list but not found in DataFrame columns.")
            pass # Silently skip features not found in the DataFrame


    # --- End of Categorical Feature Mapping and Clamping ---


    # Ensure correct data types for standard columns
    df['group_id'] = df['group_id'].astype(int)
    df['time_idx'] = df['time_idx'].astype(int)
    # Target should match the type expected by BCEWithLogitsLoss (float)
    # Target should be 0 or 1 after filtering -1, so float is fine
    df['target'] = df['target'].astype(float)
    # original_cluster_id has been mapped to >= 0 integers
    df['original_cluster_id'] = df['original_cluster_id'].astype(np.int64) # Ensure int64 for embedding lookup
    df['strategy_name'] = df['strategy_name'].astype(str) # Store as string

    # Set a dummy time index for the DataFrame index, as CryptoBinaryDataset uses time_idx column
    # Using range is simple and sufficient here
    df.index = pd.RangeIndex(start=0, stop=len(df), step=1)


    print(f"\nDebug(create_dataframe): Successfully created DataFrame with shape {df.shape}")

    return df

def save_metrics(strategy_name, results, output_dir="metrics_logs"):
    """評価指標をCSVに保存"""
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{strategy_name}_metrics.csv")

    fields = [
        "accuracy", "precision", "recall", "f1", "roc_auc",
        "optimal_threshold_val", "best_f05_on_val", "num_test_samples"
    ]

    # CSVに追記（ヘッダは最初のみ）
    write_header = not os.path.exists(file_path)
    with open(file_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()

        writer.writerow({
            "accuracy": results.get("accuracy", float("nan")),
            "precision": results.get("precision", float("nan")),
            "recall": results.get("recall", float("nan")),
            "f1": results.get("f1", float("nan")),
            "roc_auc": results.get("roc_auc", float("nan")),
            "optimal_threshold_val": results.get("optimal_threshold_val", float("nan")),
            "best_f05_on_val": results.get("best_f05_on_val", float("nan")),
            "num_test_samples": results.get("num_test_samples", 0)
        })

    print(f"✅ Metrics saved to {file_path}")

def load_metrics(strategy_name, output_dir="metrics_logs"):
    """保存された評価指標を読み込み"""
    file_path = os.path.join(output_dir, f"{strategy_name}_metrics.csv")
    if not os.path.exists(file_path):
        print(f"⚠️ No metrics file found for {strategy_name}")
        return None
    df = pd.read_csv(file_path)
    print(f"📊 Loaded metrics for {strategy_name}:")
    print(df.tail(5))  # 最新5行表示
    return df

import torch

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

class TemperatureScaler(nn.Module):
    def __init__(self, T_init=1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(float(T_init)))

    def forward(self, logits):
        # logits: (N,) 前提。スカラーTで割るだけ
        return logits / self.temperature.clamp_min(1e-6)

def fit_temperature(logits_calib: torch.Tensor, y_calib: torch.Tensor, lr=0.01, max_iter=50):
    # 1Dを強制
    z = logits_calib.detach().view(-1)          # (N,)
    y = y_calib.detach().view(-1).float()       # (N,)

    scaler = TemperatureScaler(1.0)
    opt = optim.LBFGS([scaler.temperature], lr=lr, max_iter=max_iter, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        zc = scaler(z)                           # (N,)
        loss = F.binary_cross_entropy_with_logits(zc, y)  # 形状一致
        loss.backward()
        return loss

    opt.step(closure)
    return scaler

def predict_proba_with_temperature(logits, scaler: TemperatureScaler):
    z = scaler(logits.reshape(-1,1))
    return torch.sigmoid(z).cpu().numpy().ravel()

def predict_proba(model, batch, scaler):
    model.eval()
    enc_real = batch["encoder_real_input"].to(device)
    enc_cat  = batch["encoder_categorical_input"]
    dec_real = batch["decoder_real_input"].to(device)
    dec_cat  = batch["decoder_categorical_input"]

    enc_cat = enc_cat.to(device) if enc_cat.size(-1) > 0 else torch.empty(enc_cat.shape[0], enc_cat.shape[1], 0, dtype=torch.int64, device=device)
    dec_cat = dec_cat.to(device) if dec_cat.size(-1) > 0 else torch.empty(dec_cat.shape[0], dec_cat.shape[1], 0, dtype=torch.int64, device=device)

    out, *_ = model(
        x_enc_real=enc_real, x_dec_real=dec_real,
        x_enc_cat=enc_cat,   x_dec_cat=dec_cat
    )
    logits_last = out[:, -1, 0]
    # 校正
    probs = predict_proba_with_temperature(logits_last, scaler)   # σ(z/T)
    return probs  # ndarray shape (B,)

def _safe_clip(p, eps=1e-8):
    return np.clip(p, eps, 1.0 - eps)

def _safe_logit(p, eps=1e-8):
    p = _safe_clip(p, eps)
    return np.log(p) - np.log(1.0 - p)

def apply_calibration_to_logits(
    logits_np,
    calib_mode=None,
    temp_scaler=None,
    device="cpu"
):
    """
    logits_np: shape (N,)
    戻り値: 校正後確率 (N,)
    """
    if calib_mode == "temperature" and temp_scaler is not None:
        # PyTorchの温度スケーラ（logits -> logits/T -> sigmoid）
        import torch
        z = torch.from_numpy(logits_np).to(device)
        with torch.no_grad():
            zc = temp_scaler(z).cpu().numpy()
        return 1.0 / (1.0 + np.exp(-zc))

    # elif calib_mode == "platt" and platt is not None:
    #     return platt.predict_proba(logits_np.reshape(-1, 1))[:, 1]

    # elif calib_mode == "isotonic" and iso is not None:
    #     return iso.predict_proba(logits_np.ravel())

    else:
        # 校正なし（フォールバック）
        return 1.0 / (1.0 + np.exp(-logits_np))

import numpy as np

from dataclasses import dataclass

from sklearn.isotonic import IsotonicRegression

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import brier_score_loss

def _safe_clip(p):
    return np.clip(p, 1e-8, 1-1e-8)

def _safe_logit(p):
    p = _safe_clip(p)
    return np.log(p/(1-p))

def expected_calibration_error(y_true, p_pred, n_bins=15):
    """ECE (lower is better)"""
    y = np.asarray(y_true).astype(int).ravel()
    p = _safe_clip(np.asarray(p_pred).ravel())
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i+1])
        if m.any():
            conf = p[m].mean()
            acc  = (y[m] == (p[m] >= 0.5)).mean()
            ece += (m.mean()) * abs(acc - conf)
    return float(ece)

class TemperatureScaler(torch.nn.Module):
    """Learn temperature T > 0 for logits."""
    def __init__(self):
        super().__init__()
        self.logT = torch.nn.Parameter(torch.zeros(1))  # T = exp(logT)

    def forward(self, z):
        T = torch.exp(self.logT) + 1e-8
        return z / T

def fit_temperature(logits_t, y_t, max_iter=500, lr=0.05, device=None):
    """Fit T by minimizing NLL on calibration set."""
    if device is None:
        device = logits_t.device
    model = TemperatureScaler().to(device)
    opt = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=max_iter)

    y = y_t.float().to(device)
    z = logits_t.to(device)

    def closure():
        opt.zero_grad()
        zT = model(z)
        loss = F.binary_cross_entropy_with_logits(zT, y, reduction="mean")
        loss.backward()
        return loss

    opt.step(closure)
    return model

class CalibratorBundle:
    mode: str                 # "temperature" / "platt" / "isotonic"
    temp: TemperatureScaler   # or None
    platt: LogisticRegression # or None
    iso: IsotonicRegression   # or None

def apply_calibration_from_bundle(logits_np, bundle: CalibratorBundle, device=None):
    """Return calibrated probabilities from raw logits (numpy 1D)."""
    if bundle.mode == "temperature":
        z = torch.tensor(logits_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            zT = bundle.temp(z)
            p  = torch.sigmoid(zT).cpu().numpy()
        return _safe_clip(p)
    elif bundle.mode == "platt":
        return _safe_clip(bundle.platt.predict_proba(logits_np.reshape(-1,1))[:,1])
    elif bundle.mode == "isotonic":
        return _safe_clip(bundle.iso.predict(logits_np))
    else:
        # no calibration
        return _safe_clip(1/(1+np.exp(-logits_np)))

def fit_best_calibrator(calib_logits, calib_labels, n_bins_ece=15, device=None):
    """
    試す校正:
      1) Temperature scaling（PyTorchで最小化）
      2) Platt scaling（LogisticRegression）
      3) Isotonic regression
    ECEが最小のものを返す
    """
    y = np.asarray(calib_labels).astype(int).ravel()
    z_np = np.asarray(calib_logits).ravel()
    assert y.size == z_np.size and y.size > 0

    # 1) Temperature
    z_t = torch.tensor(z_np, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)
    temp_model = fit_temperature(z_t, y_t, device=device)
    with torch.no_grad():
        p_temp = torch.sigmoid(temp_model(z_t)).cpu().numpy()
    ece_temp = expected_calibration_error(y, p_temp, n_bins=n_bins_ece)

    # 2) Platt
    pl = LogisticRegression(max_iter=1000)
    pl.fit(z_np.reshape(-1,1), y)
    p_platt = pl.predict_proba(z_np.reshape(-1,1))[:,1]
    ece_platt = expected_calibration_error(y, p_platt, n_bins=n_bins_ece)

    # 3) Isotonic
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(z_np, y)
    p_iso = iso.predict(z_np)
    ece_iso = expected_calibration_error(y, p_iso, n_bins=n_bins_ece)

    scores = [("temperature", ece_temp), ("platt", ece_platt), ("isotonic", ece_iso)]
    scores.sort(key=lambda x: x[1])
    best = scores[0][0]

    if best == "temperature":
        return CalibratorBundle(mode="temperature", temp=temp_model, platt=None, iso=None), ece_temp
    elif best == "platt":
        return CalibratorBundle(mode="platt", temp=None, platt=pl, iso=None), ece_platt
    else:
        return CalibratorBundle(mode="isotonic", temp=None, platt=None, iso=iso), ece_iso

import numpy as np

from sklearn.metrics import precision_score, recall_score, fbeta_score

def optimize_abstention_margin_on_val(
    probs_cal_val: np.ndarray,        # 校正後確率 (N,)
    y_val: np.ndarray,                # ラベル0/1  (N,)
    tau: float,                       # 既に最適化済みの閾値 τ*
    coverage_floor: float = 0.25,     # 使いたい最小カバレッジ（例）
    margins: np.ndarray = None,       # 試す帯幅グリッド
    min_used: int = 50                # 最低評価件数
):
    if margins is None:
        margins = np.linspace(0.00, 0.20, 41)  # 0〜0.2 を 0.005 刻みなど

    best = None  # (precision, f05, recall, coverage, margin)
    N = len(y_val)
    for m in margins:
        # 使うサンプル＝しきい値から十分離れている
        use_mask = np.abs(probs_cal_val - tau) >= m
        used = int(use_mask.sum())
        coverage = used / max(1, N)

        if used < min_used or coverage < coverage_floor:
            continue

        y_used = y_val[use_mask]
        y_pred = (probs_cal_val[use_mask] >= tau).astype(int)

        # 片側クラスのみはスキップ
        if np.unique(y_pred).size < 2 or np.unique(y_used).size < 2:
            continue

        prec = precision_score(y_used, y_pred, zero_division=0)
        rec  = recall_score(y_used, y_pred, zero_division=0)
        f05  = fbeta_score(y_used, y_pred, beta=0.5, zero_division=0)

        cand = (prec, f05, rec, coverage, float(m))
        if best is None:
            best = cand
        else:
            # 優先: Precision → F0.5 → Recall → Coverage（高い方）
            if (cand[0] > best[0] + 1e-12 or
                (abs(cand[0]-best[0])<=1e-12 and cand[1] > best[1] + 1e-12) or
                (abs(cand[0]-best[0])<=1e-12 and abs(cand[1]-best[1])<=1e-12 and cand[2] > best[2] + 1e-12) or
                (abs(cand[0]-best[0])<=1e-12 and abs(cand[1]-best[1])<=1e-12 and abs(cand[2]-best[2])<=1e-12 and cand[3] > best[3] + 1e-12)):
                best = cand

    if best is None:
        # フォールバック：m=0（除外なし）
        return 0.0, {"precision": np.nan, "recall": np.nan, "f05": np.nan, "coverage": 1.0}

    prec, f05, rec, cov, m_star = best
    return m_star, {"precision": prec, "recall": rec, "f05": f05, "coverage": cov}

def weighted_bce_loss(pred, target, weights=None, valid_mask=None):
    """
    pred   : (B,)  logits
    target : (B,)  {0,1} or {-1,0,1}（-1は無効）
    weights: (B,)  任意（時間減衰×クラス重みなど）
             ※ 多時刻(B,T)を渡す設計は別関数に分けるのが安全
    valid_mask: (B,) bool（省略時は target!=-1 を使う）
    """
    # 形状/型の整形
    pred   = pred.reshape(-1)
    target = target.reshape(-1).float()

    # valid マスクの決定
    if valid_mask is None:
        valid_mask = (target != -1)
    else:
        valid_mask = valid_mask.reshape(-1).bool()

    # 有効サンプル抽出
    pred_v   = pred[valid_mask]
    target_v = target[valid_mask]

    if pred_v.numel() == 0:
        # 有効サンプルが無い場合は 0 を返す
        return torch.tensor(0.0, device=pred.device)

    # 重み処理
    weights_v = None
    if weights is not None:
        w = torch.as_tensor(weights, device=pred.device, dtype=torch.float32)
        # 想定外の2D重みが来た場合の防御（最後時刻のみ学習前提）
        if w.ndim == 2:
            # 呼び出し側が(B,T)を渡してきた場合は最後時刻を利用
            w = w[:, -1]
        w = w.reshape(-1)

        # 非有限値置換＆下限クリップ（数値安定）
        w = torch.where(torch.isfinite(w), w, torch.zeros_like(w))
        w = torch.clamp(w, min=1e-6)

        weights_v = w[valid_mask]
        # 平均1で正規化（実効LRの暴れ防止）
        weights_v = weights_v / (weights_v.mean() + 1e-8)

    # BCE with logits（要：reduction='none'）
    # pred_v, target_v と weights_v は同じ長さ
    elem = F.binary_cross_entropy_with_logits(
        pred_v, target_v, reduction='none'
    )


    if weights_v is not None:
        elem = elem * weights_v

    return elem.mean()

def mixup_batch_real_and_label(enc_real, dec_real, target, valid_value=0, invalid_value=-1, alpha=0.2):
    """
    enc_real: (B, enc_len, n_enc_real)
    dec_real: (B, dec_len, n_dec_real)
    target  : (B, dec_len)  # 先頭1ステップを使う前提（複製されているならOK）
    return: mixed_enc_real, mixed_dec_real, mixed_target_single, valid_mask
    """
    if alpha is None or alpha <= 0.0:
        # Mixup 無効時（恒等）
        t_single = target[:, 0].float()
        valid_mask = (t_single != invalid_value)
        return enc_real, dec_real, t_single, valid_mask

    lam = np.random.beta(alpha, alpha)
    B = enc_real.size(0)
    device = enc_real.device
    perm = torch.randperm(B, device=device)

    # 実数入力のみ線形補間
    enc_real_mixed = lam * enc_real + (1.0 - lam) * enc_real[perm]
    dec_real_mixed = lam * dec_real + (1.0 - lam) * dec_real[perm]

    # ラベル（ソフト化）
    t1 = target[:, 0].float()        # (B,)
    t2 = target[perm, 0].float()     # (B,)
    mixed_target = lam * t1 + (1.0 - lam) * t2

    # 無効ラベル（-1）が混在する組は除外（どちらかが -1 なら無効）
    valid_mask = (t1 != invalid_value) & (t2 != invalid_value)

    return enc_real_mixed, dec_real_mixed, mixed_target, valid_mask

import torch

import torch.nn.functional as F

def focal_loss_with_weights(
    logits,
    targets,
    weights=None,      # 時間重み × クラス重みなど（任意）
    valid_mask=None,
    alpha=0.5,         # 正例の強調度
    gamma=1.5          # 難サンプル強調度
):
    """
    logits: (B,) or (N,) - 出力ロジット（生値）
    targets: (B,) or (N,) - {0,1} or {-1,0,1}
    weights: (B,) 任意のサンプル重み（時間減衰やクラス重みを掛け合わせたもの）
    """
    device = logits.device
    logits = logits.reshape(-1)
    targets = targets.reshape(-1).float()

    # 無効値マスク処理
    if valid_mask is None:
        valid_mask = (targets != -1)
    else:
        valid_mask = valid_mask.reshape(-1).bool()

    logits = logits[valid_mask]
    targets = targets[valid_mask]
    if logits.numel() == 0:
        return torch.tensor(0.0, device=device)

    # シグモイド確率
    probs = torch.sigmoid(logits)
    pt = probs * targets + (1 - probs) * (1 - targets)  # p_t

    # focal loss 本体
    focal_weight = (1 - pt) ** gamma
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

    loss_elem = alpha_t * focal_weight * bce

    # サンプル重み適用
    if weights is not None:
        weights = torch.as_tensor(weights, device=device, dtype=torch.float32).reshape(-1)[valid_mask]
        weights = weights / (weights.mean() + 1e-8)
        loss_elem = loss_elem * weights

    return loss_elem.mean()

import torch

import numpy as np

import pandas as pd

import torch

from sklearn.metrics import accuracy_score, precision_score, fbeta_score, recall_score, f1_score, confusion_matrix, roc_auc_score

import torch.nn as nn

import torch.nn.functional as F # Import F for BCEWithLogitsLoss or FocalLoss

from torch.utils.data import DataLoader # Import DataLoader

from sklearn.metrics import classification_report # Import classification_report

import copy # Import copy for model state

import math # Import math for cosine scheduler

from torch.optim.lr_scheduler import OneCycleLR

import math

import matplotlib.pyplot as plt

import json

import csv

import os

class BinaryFocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # targets ∈ {0,1}, 形状は logits と同じ。-1 は前段でマスクして除外しておく
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        p = torch.sigmoid(logits)
        p_t = p*targets + (1-p)*(1-targets)
        alpha_t = self.alpha*targets + (1-self.alpha)*(1-targets)
        loss = alpha_t * (1 - p_t).pow(self.gamma) * bce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

import torch

import numpy as np

import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, fbeta_score, recall_score, f1_score, confusion_matrix, roc_auc_score

import torch.nn as nn

import torch.nn.functional as F # Import F for BCEWithLogitsLoss or FocalLoss

from torch.utils.data import DataLoader # Import DataLoader

from sklearn.metrics import classification_report # Import classification_report

import copy # Import copy for model state

import math # Import math for cosine scheduler

from torch.optim.lr_scheduler import OneCycleLR

import math

import matplotlib.pyplot as plt

import json

import csv

import os

class BinaryFocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # targets ∈ {0,1}, 形状は logits と同じ。-1 は前段でマスクして除外しておく
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        p = torch.sigmoid(logits)
        p_t = p*targets + (1-p)*(1-targets)
        alpha_t = self.alpha*targets + (1-self.alpha)*(1-targets)
        loss = alpha_t * (1 - p_t).pow(self.gamma) * bce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

import json

import matplotlib.pyplot as plt

import torch

import numpy as np

import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, fbeta_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset

import copy

import math

from torch.optim.lr_scheduler import LambdaLR

import os # Import os for saving results

import joblib # Import joblib for saving results

def compute_weighted_bce_loss(logits, targets, w_pos, w_neg):
    """
    logits: (B, T) or (B, T, 1) のロジット（T はデコーダ長など）
    targets: 同形状の {0,1} or {-1,0,1}（-1 は無視）
    """
    # 形状合わせ
    if logits.dim() == 3 and logits.size(-1) == 1:
        logits = logits.squeeze(-1)
    if targets.dim() == 3 and targets.size(-1) == 1:
        targets = targets.squeeze(-1)

    # マスク（-1 を無視）
    valid_mask = (targets >= 0)  # True: 0/1, False: -1

    # 有効なロジットとターゲットのみを抽出
    # view(-1)で1次元に平坦化し、valid_mask[valid_mask]でmask==Trueの位置の要素を取得
    valid_logits = logits[valid_mask]
    valid_targets = targets[valid_mask] # valid_maskはtargetsと同じ形状である必要あり

    # 有効なターゲットが一つもない場合は損失0を返す
    if valid_targets.numel() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)


    # 生ロス（要素ごと）、有効な要素に対してのみ計算
    # BCEWithLogitsLossはfloat型を期待
    bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
    element_loss = bce_loss(valid_logits, valid_targets.float()) # shape: (num_valid_elements,)


    # サンプル重み（陽性= w_pos, 陰性= w_neg）
    # valid_targets は {0,1}
    weight = torch.where(valid_targets == 1,
                         torch.as_tensor(w_pos, device=targets.device, dtype=element_loss.dtype),
                         torch.as_tensor(w_neg, device=targets.device, dtype=element_loss.dtype))

    # 重み付き平均
    weighted_loss = (element_loss * weight)
    denom = weight.sum().clamp_min(1e-8)  # ゼロ割回避
    return weighted_loss.sum() / denom
