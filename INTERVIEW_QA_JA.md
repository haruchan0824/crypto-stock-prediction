# 模擬面接 Q&A: 金融時系列予測における TFT vs LightGBM

このドキュメントは面接準備用です。回答は、簡潔で事実ベース、かつ口頭で説明しやすい形にしています。

## 1. Project Overview

### Q1. このプロジェクトは何ですか？
**Short answer:**
このプロジェクトは、**Temporal Fusion Transformer（TFT）** と **LightGBM** を、**暗号資産** と **株式** の金融時系列予測で比較したものです。目的は平均性能だけでなく、fold ごとの挙動や特徴量感度まで含めて評価することでした。

**Extended answer:**
単なる benchmark table にしたくなかったので、Crypto では walk-forward validation、Stocks では aligned folds、さらに LOFO 分析まで入れて、各モデルがどの特徴量に依存しているかも見ました。

**Interview intent:**
プロジェクト全体を高いレベルで分かりやすく説明できるか。

### Q2. このプロジェクトの主な結論は何ですか？
**Short answer:**
結論は **ドメイン依存** でした。Crypto では TFT と LightGBM は平均でほぼ同等でしたが、Stocks では報告した設定の下で TFT がより明確な平均優位を示しました。

**Extended answer:**
この対比が一番重要でした。つまり、モデル選択はモデルの複雑さだけではなく、ドメイン・評価設定・特徴量空間の相互作用で決まる、ということです。

**Interview intent:**
過度に誇張せず、結果を要約できるか。

### Q3. 単にモデルを学習してスコアを出しただけのプロジェクトと何が違いますか？
**Short answer:**
単なる leaderboard 的な比較ではなく、**全体比較・fold-by-fold の挙動・LOFO ベースの解釈** を分けて見た点です。つまり、最終スコアだけでなく、なぜそうなったかを説明できるようにしました。

**Extended answer:**
金融 ML では、平均スコアだけだと不安定性・レジーム依存・特徴量の冗長性が隠れてしまいます。そのため、モデルの「振る舞い」を見る設計にしました。

**Interview intent:**
単純な benchmark を超えた思考ができているか。

## 2. Motivation and Problem Setting

### Q4. なぜこの問題を選んだのですか？
**Short answer:**
金融時系列予測は、データがノイジー・非定常・レジーム依存で、モデルの頑健性を試すには良い題材だからです。また、複雑なモデルが強い baseline を必ずしも上回らない点も技術的に面白いと考えました。

**Interview intent:**
問題設定の技術的な難しさを理解しているか。

### Q5. なぜ Crypto と Stocks を同じプロジェクトで扱ったのですか？
**Short answer:**
同じモデル比較がドメインをまたいでも成立するかを見たかったからです。両方を見ることで、結論が安定的なのか、それともドメイン依存なのかを確認できました。

**Extended answer:**
実際に、その違いは大きな学びでした。Crypto はほぼ拮抗、Stocks は TFT により有利、という結果になりました。

**Interview intent:**
複数ドメイン構成の意図を説明できるか。

### Q6. なぜ回帰ではなく、このタスク定式化を選んだのですか？
**Short answer:**
評価は方向性の分類として扱い、AUC を主指標にしました。その方が、単一の閾値に依存せず ranking quality を比較しやすかったからです。

**Extended answer:**
もちろん回帰が不適切という意味ではありません。ただ、このプロジェクトでは fold 間比較を分かりやすくするために分類設定を選びました。

**Interview intent:**
モデリング上の選択とトレードオフを説明できるか。

## 3. Model Choice

### Q7. なぜ TFT と LightGBM を比較したのですか？
**Short answer:**
2 つのモデルが強く、かつ前提が異なるからです。TFT は時系列文脈を扱うシーケンスモデルで、LightGBM は engineered feature に強い tabular baseline です。

**Extended answer:**
この比較には意味があります。TFT が弱い baseline に勝つだけでは説得力が弱いですが、LightGBM 相手なら比較として十分に厳しいからです。

**Interview intent:**
比較設定の妥当性を理解しているか。

### Q8. なぜ LightGBM は良い baseline なのですか？
**Short answer:**
LightGBM は強く、高速で、解釈もしやすく、特徴量エンジニアリングがうまく機能しているときは非常に手強いからです。本気の applied ML なら入っていてほしい baseline だと思っています。

**Interview intent:**
baseline の扱いが適切か。

### Q9. TFT は LightGBM が取りこぼす何を捉えると考えましたか？
**Short answer:**
TFT は、時間方向の文脈や、より複雑な特徴量相互作用を捉えやすいと考えました。特に、特徴量と結果の関係が static ではなく sequence として変化する場合に有利かもしれません。

**Extended answer:**
ただし、ここは慎重に言うべきで、プロジェクトが直接そのメカニズムを証明したわけではありません。特に Stocks で平均優位が見えた、という実証結果がある、という言い方に留めます。

**Interview intent:**
inductive bias を、根拠のない断言なしに説明できるか。

### Q10. なぜ他のモデルは比較しなかったのですか？
**Short answer:**
幅広く並べるより、比較の深さを優先したからです。TFT と LightGBM だけでも十分に強いアーキテクチャ対比になり、評価と解釈により多くの時間を使いたかったです。

**Extended answer:**
もちろん次のステップとしてモデルを増やすのはありですが、このプロジェクトでは shallow な model zoo より、defensible な比較を優先しました。

**Interview intent:**
スコープ設計を説明できるか。

## 4. Evaluation Design

### Q11. なぜ Crypto では walk-forward validation を使ったのですか？
**Short answer:**
金融データは時系列順を持つため、walk-forward validation の方が実運用に近いからです。シャッフル分割ではなく、常に未来の未見データで評価できる点が重要でした。

**Interview intent:**
時系列評価の基本を理解しているか。

### Q12. なぜ Stocks では aligned folds を使ったのですか？
**Short answer:**
TFT と LightGBM の比較を公平にするためです。同じ fold 境界で比較することで、モデル差を fold 設計の差ではなく、モデルそのものの差として見やすくしました。

**Interview intent:**
実験設計の公平性を説明できるか。

### Q13. なぜ AUC を主指標にしたのですか？
**Short answer:**
AUC は方向性の分類問題に適していて、threshold-independent だからです。単一の運用閾値に縛られず、fold をまたいだ ranking quality を比較できます。

**Extended answer:**
ただし、実運用では AUC だけで十分とは言いません。もし実際の意思決定に使うなら、calibration や threshold 最適化、下流の utility 指標も見たいです。

**Interview intent:**
指標選択とその限界を理解しているか。

### Q14. どうやって leakage を避けましたか？
**Short answer:**
最大の対策は、時系列順を保った評価です。Crypto では過去で学習して未来で評価し、Stocks でも時間をランダムに混ぜず fold 単位で整合させました。

**Extended answer:**
ここで必要以上に言い過ぎないことも重要です。前処理や timestamp alignment の厳密な実装詳細を聞かれたら、手元のパイプラインに基づいて [TODO] と明示するのが正直です。

**Interview intent:**
leakage 対策を誠実に説明できるか。

### Q15. なぜ fold レベルの分析が重要なのですか？
**Short answer:**
平均値だけでは不安定性が隠れるからです。非定常な金融データでは、同じ mean AUC でも fold ごとの振る舞いが大きく違うことがあります。

**Interview intent:**
平均性能だけでなく分散や安定性を考えられるか。

### Q16. なぜ fold によって性能が変動したのですか？
**Short answer:**
最も自然な説明は、fold ごとにデータ生成条件が変わっていたからです。金融時系列では、ボラティリティ、トレンド、マクロ環境、市場活動が変わることで、どのモデルや特徴量が効くかも変わります。

**Extended answer:**
ただし、これは解釈であって証明ではありません。fold ごとの差は不安定性やレジーム依存性の証拠にはなりますが、因果を断定するものではありません。

**Interview intent:**
変動を過剰に断定せずに解釈できるか。

## 5. Results Interpretation

### Q17. Crypto 実験では何が起きましたか？
**Short answer:**
Crypto は平均ではほぼ拮抗でした。TFT の mean AUC は **0.601**、LightGBM は **0.603**、mean delta は **-0.0015**、そして TFT は約 **52%** の fold で勝ちました。

**Extended answer:**
重要なのは、勝率と平均 AUC が少し違うストーリーを示していたことです。つまり、TFT は一部 fold でしっかり勝つ一方、別の fold では明確に負けていた可能性があります。

**Interview intent:**
Crypto の結果を正確かつニュアンス込みで説明できるか。

### Q18. Stock 実験では何が起きましたか？
**Short answer:**
Stocks では、報告した 5-fold 設定において TFT が平均で上回りました。TFT の mean AUC は **0.5815**、LightGBM は **0.5127** で、TFT は **5 fold 中 4 fold** で勝ちました。

**Extended answer:**
ただし、これは「この設定での明確な平均優位」と言うべきで、普遍的な結論として言い切るべきではありません。fold 単位の差も依然として重要です。

**Interview intent:**
Stock の結果を誇張せず説明できるか。

### Q19. なぜ Crypto の結果を regime-sensitive と表現するのですか？
**Short answer:**
平均結果は非常に近いのに、TFT は勝つ fold と負ける fold が分かれていたからです。そのパターンは、競争力はあるが条件依存性が高いモデルと整合的です。

**Interview intent:**
「レジーム依存」を実務的に説明できるか。

### Q20. なぜ Stocks では TFT の方が良かったと考えますか？
**Short answer:**
慎重に言うなら、Stocks では TFT が市場・マクロ・時間文脈を含むより豊かな特徴量組み合わせから恩恵を受けていたように見えます。これは LOFO パターンに支えられた解釈であって、メカニズムの証明ではありません。

**Interview intent:**
確実ではない点を含めて、妥当な仮説を述べられるか。

### Q21. Stocks の fold-by-fold のニュアンスはどう説明しますか？
**Short answer:**
TFT は **5 fold 中 4 fold** で勝ち、最大の改善は **fold 2**、最も弱い結果は **fold 3** でした。つまり、Stock の結果は平均では強いですが、すべての fold で一様だったわけではありません。

**Interview intent:**
fold レベルの具体的なニュアンスを覚えて説明できるか。

## 6. LOFO / Feature Importance

### Q22. なぜ単純な feature importance ではなく LOFO を使ったのですか？
**Short answer:**
LOFO は、特徴量や特徴量群を除外したときに性能がどう変わるかを見るため、モデル性能とより直接的につながっています。そのため、アーキテクチャごとの特徴量感度を比較するのに向いています。

**Extended answer:**
もちろん、LOFO が常に他の importance より優れていると言いたいわけではありません。このプロジェクトでは「どの入力が有用・冗長・有害か」を見る実務的な問いに合っていたため採用しました。

**Interview intent:**
なぜ LOFO を選んだかを説明できるか。

### Q23. 除外した方が性能が上がる特徴量はどう解釈しますか？
**Short answer:**
その特徴量が冗長、ノイズ、あるいは現在のモデル設定にうまく合っていない可能性があると解釈します。つまり、特徴量は多ければ多いほど良いわけではない、ということです。

**Interview intent:**
負の importance を妥当に解釈できるか。

### Q24. Crypto の LOFO から何が分かりましたか？
**Short answer:**
Crypto の LOFO からは、**on-chain 特徴量が意味のあるシグナルを持っていそうだ** という示唆が得られました。ただし、LOFO は因果的説明ではなく感度分析として扱うべきです。

**Interview intent:**
特徴量分析を Crypto の結果と結びつけて説明できるか。

### Q25. Stocks の LOFO から何が分かりましたか？
**Short answer:**
TFT は市場・マクロ・時間文脈を含む richer な特徴量群から恩恵を受けやすく、LightGBM はより要約的な handcrafted feature に依存する strong tabular baseline のように振る舞っていました。また、一部特徴量は LightGBM では除外した方が良い結果になりました。

**Extended answer:**
たとえば `log_ret_1`、`mkt_trend_regime_id`、`log_ret_5`、`ma_120` などは、LightGBM で除外時に改善が見られました。これは冗長性や feature-architecture mismatch の可能性を示します。

**Interview intent:**
ニュアンスを保ちながら特徴量解釈を話せるか。

## 7. Crypto vs Stock Comparison

### Q26. Crypto と Stocks を直接比較して何を学びましたか？
**Short answer:**
同じアーキテクチャ比較でも、ドメインによって結論が変わることを学びました。Crypto では平均でほぼ拮抗、Stocks では TFT を支持する証拠がより強く出ました。

**Interview intent:**
クロスドメインの学びを抽出できるか。

### Q27. これはモデル選択について何を示していますか？
**Short answer:**
モデル選択は「どのモデルが偉いか」ではなく、ドメインに即して行うべきだということです。複雑なモデルが自動的に優れているわけではなく、特徴量との相互作用もドメインごとに変わります。

**Interview intent:**
実務的なモデル選択の考え方を持っているか。

### Q28. 面接官に「つまり TFT の方が良いんですね」と言われたらどう答えますか？
**Short answer:**
それは言い過ぎだと答えます。Stocks では報告した設定で TFT に明確な平均優位がありましたが、Crypto では平均でほぼ拮抗だったため、より正確には **ドメイン依存** という結論です。

**Interview intent:**
過度な一般化を避けられるか。

## 8. Limitations and Future Work

### Q29. このプロジェクトの主な弱みは何ですか？
**Short answer:**
主な制約は、Stock の結果が比較的小さな **5-fold** サマリに基づいていること、Stock 側の artifact が Crypto 側ほど揃っていないこと、そして LOFO が因果的 importance を与えるものではないことです。また、詳細実装が文書化されていない点については言い過ぎないようにしています。

**Interview intent:**
限界を正直に話せるか。

### Q30. 次に何を改善しますか？
**Short answer:**
Stock 側の可視化 artifact を追加し、より多くのドメインと長い期間で評価し、LOFO で弱い特徴量をより体系的に削減したいです。さらに、regime-switching や ensemble も試したいです。

**Interview intent:**
現実的な次ステップを提案できるか。

### Q31. このプロジェクトをどう productionize しますか？
**Short answer:**
まずは再現可能な offline pipeline、固定された train-validation スケジュール、データドリフトや fold 単位劣化の監視から始めます。その上で、モデル versioning、定期再学習、AUC 以外の業務指標も加えます。

**Extended answer:**
ただし、現状のリポジトリがそのまま production-ready だとは言いません。実運用には、より厳密な feature timestamp 保証、自動 backfill、意思決定閾値の監視なども必要です。実装詳細の一部は [TODO] です。

**Interview intent:**
研究寄りのプロジェクトを production 観点につなげて話せるか。

## 9. Engineering / Reproducibility / Repository Design

### Q32. リポジトリは再現性のためにどう整理されていますか？
**Short answer:**
スクリプト、設定、ソースコード、図、ドキュメントを分けていて、モデリング本体とレポーティングを分離しています。README には `tft_vs_lgbm_compare.csv`、`tft_vs_lgbm_summary.json`、`lofo_group_agg.csv`、`lofo_feature_agg.csv` のような想定 artifact も記載しています。

**Interview intent:**
モデリングだけでなく再現性も意識しているか。

### Q33. 文書にない実装詳細を面接で聞かれたらどう答えますか？
**Short answer:**
文書化されている範囲を正確に答え、不明な部分は [TODO] と明示します。面接では、自信ありげに間違うより、正確に不足を認める方が大事だと考えています。

**Interview intent:**
不確実性への向き合い方が適切か。

### Q34. README と interview-prep 資料はどう使い分けていますか？
**Short answer:**
README は外向けの project report、interview-prep 系は口頭説明向けの内部資料です。この分離によって、公開ドキュメントは整えつつ、面接用には短く defensible な説明を準備できます。

**Interview intent:**
ドキュメント戦略を説明できるか。

### Q35. このプロジェクト全体の一文要約をすると？
**Short answer:**
強い tabular baseline と sequence model は、金融ドメインによって全く異なる振る舞いを示し得るため、モデル選択はアーキテクチャだけでなく、時系列評価・fold レベル分析・特徴量相互作用に基づいて行うべき、というのがこのプロジェクトの最も defensible な結論です。

**Interview intent:**
最後を正確で守りやすい一文で締められるか。
