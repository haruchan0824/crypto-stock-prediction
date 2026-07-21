# 面接準備ノート: 金融時系列予測における TFT vs LightGBM

## 1. Project Summary

### このプロジェクトは何か
- このプロジェクトは、金融時系列予測タスクにおいて **Temporal Fusion Transformer（TFT）** と **LightGBM** を比較したものです。
- 対象ドメインは以下の 2 つです。
  - **暗号資産（Crypto）予測**
  - **株式（Stock）予測**
- 目的は単に平均スコアを出すことではなく、fold ごとの挙動や特徴量依存を理解することです。

### なぜこのプロジェクトを行ったか
- 金融 ML では、ある split でよく見えるモデルが、別のレジームや別の資産クラスでは通用しないことがあります。
- そのため、TFT のようなシーケンスモデルが、LightGBM のような強い tabular baseline を本当に上回るのかを検証したかったです。
- また、どちらが勝つかだけではなく、**どのような条件でうまく機能するか**も理解したかったです。

### 比較したモデル
- **TFT**: 時系列文脈と時間方向の特徴量相互作用を捉えることを意図したシーケンスモデル
- **LightGBM**: 特徴量エンジニアリング済みデータに対して強く、解釈もしやすい勾配ブースティングモデル

### 対象ドメイン
- **Crypto**: walk-forward validation と LOFO 的な解釈を実施
- **Stocks**: 整合した fold 設定で fold 単位比較と LOFO 解釈を実施

## 2. One-Minute Explanation

> このプロジェクトでは、金融時系列予測に対して Temporal Fusion Transformer と LightGBM を、暗号資産と株式の 2 ドメインで比較しました。目的は、より複雑なシーケンスモデルが、強い tabular baseline を公正な時系列評価の下で本当に上回るのかを見ることでした。Crypto では結果はほぼ拮抗で、TFT の mean AUC は 0.601、LightGBM は 0.603 でした。つまり、ここでのポイントは明確な勝敗というよりレジーム依存性でした。一方 Stocks では、TFT の mean AUC は 0.5815、LightGBM は 0.5127 で、TFT が 5 fold 中 4 fold で勝ちました。さらに LOFO を用いて特徴量依存も見たことで、モデル選択はドメイン・レジーム・特徴量相互作用に依存する、というのが全体の学びでした。

## 3. Three-Minute Explanation

> このプロジェクトは、TFT と LightGBM を暗号資産と株式の 2 つの金融時系列設定で比較したものです。2 モデルを選んだ理由は、前提としているモデリング思想が大きく異なるからです。TFT はシーケンスモデルで、時系列文脈や時間的な相互作用を扱うことを狙っています。一方 LightGBM は、特徴量エンジニアリングが十分に効いているときに非常に強いベースラインです。
>
> Crypto では、時系列の順序を守るために walk-forward validation を使いました。未来の情報が学習に混ざる leakage を避けたかったためです。評価指標は AUC を使いました。方向性の分類問題で、閾値に依存しない ranking quality を見たかったからです。Crypto の結果は平均ではかなり近く、TFT の mean AUC は 0.601、LightGBM は 0.603 でした。TFT は fold の過半数では勝っていて、この点から一貫した優位ではなく、レジーム依存的な競争力があると解釈しました。
>
> そこで、平均スコアだけでなく、fold-by-fold の結果、AUC 分布、win-vs-non-win 分析を見ました。これにより、TFT は一部 fold では強く勝つ一方で、別の fold では明確に負けることが分かりました。さらに LOFO によって、どの特徴量群や特徴量が有効そうか、あるいは冗長そうかを調べました。
>
> Stock 実験では、TFT と LightGBM を同じ fold 構成で比較しました。ここでは TFT がより明確に平均で優位で、mean AUC は 0.5815 対 0.5127、さらに 5 fold 中 4 fold で TFT が勝ちました。fold ごとの表を見ると、最も大きな相対差は fold 2 で、TFT が負けたのは fold 3 だけでした。LOFO の結果からは、TFT が市場・マクロ・時間文脈を含むより豊かな特徴量群から恩恵を受けている一方、LightGBM はより要約的な handcrafted feature に強く依存しているように見えました。
>
> つまり結論は、ドメインによって答えが変わるということです。複雑なモデルが自動的に優れているわけではありませんが、アーキテクチャと特徴量の相互作用は、金融設定によって大きく変わり得ます。

## 4. Five-Minute Explanation

> このプロジェクトは、「すでに強い tabular baseline として LightGBM があるとき、より複雑なシーケンスモデルである TFT に移る価値が本当にあるのか」という実務的な問いから始まりました。
>
> TFT と LightGBM を選んだ理由は、両者がよい対比になっているからです。TFT は時間方向の構造や、時変な特徴量重要度を捉えることを目的としています。一方で LightGBM は、有用な情報が handcrafted feature に十分要約されている場合に非常に強いことが多いです。金融では、強い baseline と比較すると改善が消えることも多いため、この比較には意味があります。
>
> 評価は時系列を尊重する形で行いました。Crypto では walk-forward validation を使い、各 fold で過去データだけで学習し、その先の未見期間で評価しました。ランダム split は、未来情報の混入や不自然な train-test の重なりを生みやすいので避けました。Stock では、TFT と LightGBM が同じ fold 構成で比較されるように aligned fold-based evaluation を使いました。
>
> 指標には AUC を使いました。方向性の分類問題であり、単一閾値に依存しない ranking quality を見たかったからです。また、平均スコアだけではなく、全体比較・fold-by-fold の挙動・LOFO ベースの解釈を分けて見るようにしました。
>
> Crypto では、平均結果はほぼ引き分けでした。TFT の mean AUC は 0.601、LightGBM は 0.603、mean delta は -0.0015 です。ただし TFT は約 52% の fold で勝っていました。つまり、単純に LightGBM が優れていたと言うより、TFT は一部の条件ではしっかり勝つが、別の条件では負ける、というレジーム依存的な挙動をしていたと解釈できます。ここから学んだのは、シーケンスモデルは競争力を持ち得るが、必ずしも一貫して superior ではないということです。
>
> さらに、LOFO を使って特徴量依存を見ました。Crypto では、on-chain 特徴量が意味のあるシグナルを持っていそうだ、という示唆が得られました。ただし、LOFO は因果的な重要度ではなく、あくまで性能感度分析として解釈すべきだと考えています。
>
> Stock ではストーリーが異なりました。TFT はより明確な平均優位を示し、mean AUC は 0.5815 対 0.5127、標準偏差は 0.0511 対 0.0155、そして 5 fold 中 4 fold で TFT が勝ちました。最も大きい改善は fold 2 で、fold 3 だけは TFT が負けています。したがって、Stock の結果は「この設定では TFT に平均的な優位がある」と言えますが、「TFT が常に優れている」とまでは言えません。
>
> Stock の LOFO 結果は、この違いの解釈に役立ちました。TFT の上位 LOFO 特徴量には、`fred_DGS10_ret1`、`ATR_14`、`volume`、`log_volume`、`ret_1_rolling_mean_20`、`trend_60_120`、`yf_^GSPC_close_ret1` など、市場・マクロ・時間文脈に関わる入力が含まれていました。一方 LightGBM は、`ma_120`、`ATR_14`、`ret_1_rolling_mean_20`、`log_volume` のような、より要約的な handcrafted feature に依存しているように見えました。さらに、`log_ret_1`、`mkt_trend_regime_id`、`log_ret_5`、`ma_120` などは除外した方が LightGBM の性能が上がるケースもあり、特徴量が冗長、あるいはそのアーキテクチャに合っていない可能性も示されました。
>
> 最終的な結論はクロスドメインです。Crypto では LightGBM が非常に競争力を保ち、TFT はレジーム依存的でした。Stocks では TFT の平均的な優位がより明確でした。したがって、「どちらのモデルが常に勝つか」ではなく、「ドメイン・レジーム・特徴量空間とアーキテクチャの相互作用によって答えが変わる」が、このプロジェクトの一番 defensible なまとめです。今後は、対象ドメインを増やすこと、Stock 側の artifact を拡充すること、そして ensemble や regime-switching を検討することが自然な次のステップです。

## 5. Experiment Design Notes

### なぜ TFT と LightGBM を比較したのか
- 明確に異なる前提を持つ 2 つの強いモデルだからです。
- TFT は sequential modeling がどこまで効くかを試せます。
- LightGBM は、特徴量エンジニアリングが十分な場合にそれだけで十分かを試せます。
- TFT が LightGBM に勝つなら、それは弱い baseline 相手ではなく、強い baseline 相手の結果なので意味があります。

### なぜ walk-forward / fold-based evaluation を使ったのか
- 金融データは時間順を持つからです。
- ランダム split は leakage や一般化性能の過大評価を招く可能性があります。
- Walk-forward validation は、常に未来データで評価するため、実運用に近いです。
- Stock では aligned folds を使うことで、公平な比較にしました。

### なぜ AUC を使ったのか
- タスクが方向性の分類問題だからです。
- AUC は threshold-independent です。
- 単一閾値ではなく、fold をまたいだ ranking quality を比較するのに向いています。

### どうやって leakage を避けたか
- 評価では時間順を保ちました。
- Crypto では、学習は過去データ、テストはその先の未見期間です。
- Stock 比較でも、時間をランダムに混ぜず、fold 単位で整合させています。
- [TODO] 面接で聞かれたら、特徴量生成時の timestamp 制約をより具体的に説明できるようにしておく。

### なぜ LOFO を使ったのか
- 特徴量や特徴量群を外したときに性能がどう変わるかを見られるからです。
- どの入力が有用、冗長、あるいは有害かを診断するのに役立ちます。
- 同じ特徴量でも、モデルによって使い方が違う可能性を比較しやすいです。
- ただし、**因果的な重要度ではなく、性能感度分析**として説明すべきです。

## 6. Crypto Findings

### Main result
- **TFT mean AUC:** 0.601
- **LightGBM mean AUC:** 0.603
- **Mean delta (TFT - LGBM):** -0.0015
- **TFT win rate:** ~52%

### Key interpretation points
- Crypto の平均結果は **かなり近い**。
- Mean AUC では LightGBM がわずかに上。
- ただし TFT は fold の過半数で勝っている。
- つまり **一貫した優位** ではなく、**レジーム依存の競争力** があったと解釈するのが自然。
- 平均値だけでなく fold レベルの分析が重要だった。
- LOFO からは **on-chain 特徴量が有効そう** という示唆が得られた。

### Cautions / limitations
- 平均差は非常に小さい。
- 勝率だけを主結果として扱うべきではない。強い負け fold があれば平均は相殺される。
- LOFO を決定的な重要度ランキングとして扱うべきではない。
- Crypto の結論は、「どちらが支配的か」よりも「どのように fold ごとに振る舞うか」の分析として強い。

## 7. Stock Findings

### Main full-fold result
- **Total folds analyzed:** 5
- **Mean TFT AUC:** 0.5815
- **Std TFT AUC:** 0.0511
- **Mean LightGBM AUC:** 0.5127
- **Std LightGBM AUC:** 0.0155
- **Mean delta AUC (TFT - LGBM):** 0.0689
- **TFT wins:** 4 out of 5 folds

### Fold-by-fold nuance
- TFT は **5 fold 中 4 fold** で LightGBM を上回った。
- **Best fold by delta:** fold 2
- **Worst fold by delta:** fold 3
- Fold 詳細:

| Fold | TFT AUC | LightGBM AUC | Delta AUC |
| --- | ---: | ---: | ---: |
| 0 | 0.585650 | 0.505614 | 0.080036 |
| 1 | 0.584992 | 0.515848 | 0.069143 |
| 2 | 0.613839 | 0.504185 | 0.109654 |
| 3 | 0.486788 | 0.496575 | -0.009787 |
| 4 | 0.636347 | 0.541082 | 0.095265 |

### LOFO interpretation
- TFT の上位 LOFO 特徴量:
  - `fred_DGS10_ret1`
  - `ATR_14`
  - `volume`
  - `log_volume`
  - `ret_1_rolling_mean_20`
  - `trend_60_120`
  - `log_ret_5`
  - `log_ret_1`
  - `yf_^GSPC_close_ret1`
  - `ma_120`
- LightGBM は、より静的・要約的な特徴量への依存が強いことが示唆された:
  - `ma_120`
  - `ATR_14`
  - `ret_1_rolling_mean_20`
  - `log_volume`
  - `fred_term_spread_10y_2y`
  - `fred_DGS10_ret1`
  - `yf_^VIX_close_ret1`
  - `yf_^GSPC_close_ret1`
- LightGBM では、以下のように除外した方が性能が上がる特徴量もあった:
  - `log_ret_1`
  - `mkt_trend_regime_id`
  - `log_ret_5`
  - `ma_120`
- 実務的な解釈:
  - TFT は、市場・マクロ・時間文脈を含むより豊かな特徴量群から恩恵を受けやすいように見える。
  - LightGBM は、summary-style な handcrafted feature の定義により強く依存する強い tabular baseline として振る舞っているように見える。

### Cautions / limitations
- Stock の結果は、**この設定では平均的に優位** と言うべきであって、TFT の普遍的優位を主張すべきではない。
- 報告されている比較は **5 fold** である。
- Fold 3 は、その優位が一様ではなかったことを示している。
- Stock 側は、Crypto 側ほど図表 artifact が揃っていない。

## 8. Cross-Domain Comparison

### Similarities between crypto and stock
- 両実験とも同じ 2 モデルを比較している。
- どちらも fold ベースで、時系列順を尊重した評価思想を使っている。
- 平均指標だけでは不十分であることを示している。
- どちらも LOFO 的な分析が特徴量感度の理解に役立っている。

### Differences between crypto and stock
- **Crypto** では平均結果はほぼ拮抗。
- **Stocks** では TFT の平均優位がより明確。
- Crypto では主な解釈はレジーム依存性。
- Stocks では、TFT がこの設定で平均的に優位だったという解釈が中心。
- Stocks の方が、アーキテクチャごとの特徴量相互作用の違いがより見えやすい。

### What this suggests about model choice and feature interaction
- モデル選択は **ドメイン依存** である可能性が高い。
- 特徴量の有効性も、モデルアーキテクチャに依存し得る。
- シーケンスモデルがあるからといって、強い tabular baseline が不要になるわけではない。
- 特徴量を増やしても、すべてのモデルが同じように恩恵を受けるわけではない。
- モデル選択は、アーキテクチャ単体ではなく **性能と特徴量相互作用の両方** で見るべき。

## 9. Key Technical Talking Points

- TFT と LightGBM を比較したのは、sequence modeling と strong tabular modeling の対比になるから。
- LightGBM は弱い baseline ではなく、真剣なベンチマークとして扱った。
- 金融データでは random split が危険なので、時系列順を保った評価を使った。
- Crypto の主結果は「TFT が勝った」ではなく、「平均ではほぼ拮抗」だったこと。
- Crypto では TFT は一貫して superior というより、レジーム依存的だった。
- Stocks では、報告した 5-fold 設定で TFT に明確な平均優位があった。
- 平均 AUC だけでなく、fold レベルの解釈を重視した。
- AUC を使ったのは、方向性分類に適していて threshold-independent だから。
- LOFO は、アーキテクチャごとの特徴量感度を比較するのに役立った。
- LOFO は因果的な feature importance ではなく、感度分析として説明する。
- Crypto の LOFO は on-chain signal の有効性を示唆した。
- Stock の LOFO は、TFT が richer な market / macro / temporal-context feature から恩恵を受けやすいことを示唆した。
- 一部特徴量は LightGBM では除外した方が良く、冗長性や model-feature mismatch を示唆した。
- このプロジェクトの大きな学びの 1 つは、モデル選択がドメインによって変わること。
- もう 1 つは、アーキテクチャの良し悪しだけでなく、評価設計と特徴量相互作用が重要だということ。

## 10. Limitations and Future Work

### What is not solved yet
- このプロジェクトは、どの金融ドメインでも 1 つのモデルが最良だと示したわけではない。
- Stock 結果は、限定的な 5-fold サマリに基づく。
- Stock 側の可視化や artifact は、Crypto 側ほど充実していない。
- LOFO は感度分析であり、因果的証拠ではない。
- [TODO] 面接では、前処理や特徴量タイムスタンプ整合の詳細を説明できるように補強する。

### What the next improvements would be
- Stock 側にも Crypto と同様の可視化 artifact を追加する。
- より多くのドメイン、より長い期間に評価を拡張する。
- regime-switching や architecture-routing を検討する。
- TFT と LightGBM の ensemble を試す。
- LOFO を使って弱い・有害な特徴量をより体系的に削減する。
