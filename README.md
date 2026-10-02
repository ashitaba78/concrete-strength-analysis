# Concrete Compressive Strength Analysis

## 概要

コンクリートの配合条件および材齢から圧縮強度を分析・予測する、
Pythonによる材料データ分析のポートフォリオです。

公開されている材料データを用いて、

- データ読み込み・確認
- 基本統計量
- 可視化
- 相関分析
- 重回帰分析
- 多重共線性の確認
- 線形回帰による予測
- Random Forestによる予測
- モデル性能比較
- Feature Importanceによる特徴量評価
- Excel・PNGレポートの自動生成

までを1本のPythonプログラムで実装しました。

---

## 分析の目的

材料開発における、

**材料組成・配合条件 → 材料特性**

というデータ分析の基本的な流れをPythonで実装することを目的としています。

目的変数としてコンクリート圧縮強度（Strength）を使用し、
8種類の配合条件・材齢との関係を分析しました。

---

## 使用データ

UCI Machine Learning Repositoryで公開されている  
**Concrete Compressive Strength Dataset** を使用しています。

- データ件数：1,030件
- 説明変数：8項目
- 目的変数：Concrete Compressive Strength
- 欠損値：なし

### 説明変数

- Cement
- Blast Furnace Slag
- Fly Ash
- Water
- Superplasticizer
- Coarse Aggregate
- Fine Aggregate
- Age

### 目的変数

- Strength（MPa）

Dataset:

https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength

Citation:

Yeh, I. (1998). Concrete Compressive Strength [Dataset].  
UCI Machine Learning Repository.  
https://doi.org/10.24432/C5PK67

---

## 使用技術

- Python
- pandas
- matplotlib
- statsmodels
- scikit-learn
- openpyxl
- xlrd

---

## プロジェクト構成

```text
portfolio/
│
├─ README.md
├─ requirements.txt
│
├─ data/
│   └─ Concrete_Data.xls
│
├─ src/
│   └─ concrete_analysis.py
│
└─ output/
    ├─ analysis_result.xlsx
    ├─ 01_strength_histogram.png
    ├─ 02_cement_vs_strength.png
    ├─ 03_strength_correlation.png
    ├─ 04_predictor_correlation_matrix.png
    ├─ 05_actual_vs_predicted_linear.png
    ├─ 06_actual_vs_predicted_random_forest.png
    └─ 07_random_forest_feature_importance.png
```

---

## 分析内容

### 1. データ確認

pandasを使用してExcelデータを読み込み、

- データ件数
- データ型
- 欠損値
- 基本統計量

を確認しました。

全1,030件について欠損値はありませんでした。

---

### 2. 圧縮強度の基本統計

Strengthの基本統計量は以下の通りでした。

| 指標 | 結果 |
|---|---:|
| データ件数 | 1,030 |
| 平均値 | 35.82 MPa |
| 中央値 | 34.44 MPa |
| 最小値 | 2.33 MPa |
| 最大値 | 82.60 MPa |
| 標準偏差 | 16.71 MPa |

---

### 3. 相関分析

Strengthとの単純相関では、主に以下の結果が得られました。

| 変数 | 相関係数 |
|---|---:|
| Cement | 0.498 |
| Superplasticizer | 0.366 |
| Age | 0.329 |
| Water | -0.290 |

Cement、Superplasticizer、Ageでは正の相関、
Waterでは負の相関が確認されました。

ただし、相関関係は因果関係を意味するものではないため、
複数の説明変数を同時に考慮する重回帰分析も実施しました。

---

## 重回帰分析

8変数を説明変数、Strengthを目的変数として
OLSによる重回帰分析を実施しました。

### モデル全体

| 指標 | 結果 |
|---|---:|
| R² | 0.615 |
| 調整済みR² | 0.612 |
| F値 | 204.3 |
| F検定 | p < .001 |

8変数を用いたモデルは統計的に有意であり、
Strengthの変動の約61.5%を説明しました。

標準化回帰係数ではCement、Blast Furnace Slag、Ageなどに
比較的大きな正の係数が確認されました。

一方、Waterには負の係数が確認されました。

---

## 多重共線性

説明変数についてVIFを計算したところ、
CementやBlast Furnace Slagなど複数の変数で
VIFが5を上回りました。

このため、各回帰係数の解釈には
説明変数間の多重共線性を考慮する必要があると判断しました。

また、説明変数同士の単純相関を確認したところ、
絶対値0.5以上となった主な組み合わせは、

- Water × Superplasticizer：r = -0.657

でした。

単純な2変数間相関だけではなく、
複数の説明変数の組み合わせによる重複が存在する可能性が
VIFから示唆されました。

### 説明変数間の相関行列

![Predictor Correlation Matrix]
(output/04_predictor_correlation_matrix.png)


---

## 機械学習による予測

データを、

- 学習用：80%
- テスト用：20%

に分割し、未知データに対する予測性能を評価しました。

データ分割には `random_state=42` を使用しています。

### Linear Regression

| 指標 | 結果 |
|---|---:|
| R² | 0.628 |
| MAE | 7.745 MPa |
| RMSE | 9.797 MPa |

### Random Forest

| 指標 | 結果 |
|---|---:|
| R² | 0.879 |
| MAE | 3.798 MPa |
| RMSE | 5.594 MPa |

今回のテストデータでは、
Random Forestが線形回帰より高い予測性能を示しました。


### 実測値と予測値

![Actual vs Predicted - Random Forest]
(output/06_actual_vs_predicted_random_forest.png)

これは、材料配合条件と圧縮強度の関係に、
単純な線形モデルだけでは捉えにくい
非線形関係や変数間の相互作用が含まれている可能性を示唆します。

なお、今回は1回のTrain/Test分割による評価であり、
モデル性能を確定的に評価するものではありません。

---

## Random Forest Feature Importance

### 特徴量重要度の可視化

![Random Forest Feature Importance]
(output/07_random_forest_feature_importance.png)

Random Forestの特徴量重要度は以下の結果となりました。

| 変数 | Importance |
|---|---:|
| Age | 0.333 |
| Cement | 0.327 |
| Water | 0.123 |
| BlastFurnaceSlag | 0.076 |
| Superplasticizer | 0.057 |
| FineAggregate | 0.036 |
| CoarseAggregate | 0.028 |
| FlyAsh | 0.020 |

今回構築したRandom Forestモデルでは、
AgeとCementの重要度が特に高い結果となりました。

ただしFeature Importanceは因果関係を表すものではなく、
材料としての物理的重要性を直接示すものでもありません。

---

## 自動出力

`concrete_analysis.py` を1回実行すると、
分析結果をExcelおよびPNG画像として自動生成します。

Excelファイルには、

- 元データ
- 欠損値確認
- 基本統計量
- 材齢別集計
- 相関行列
- Strengthとの相関
- 説明変数間相関
- 重回帰分析
- Beta・VIF
- モデル比較
- 実測値・予測値
- Feature Importance

をシート別に出力します。

---

## 実行方法

必要なライブラリをインストールします。

```text
py -m pip install -r requirements.txt
```

その後、プロジェクトフォルダで以下を実行します。

```text
py src\concrete_analysis.py
```

分析結果は `output` フォルダに保存されます。

---

## 今後の改善

今後は以下を検討します。

- Cross Validationによるモデル性能の安定性評価
- Random Forestのハイパーパラメータ調整
- Permutation Importanceによる特徴量評価
- 残差分析
- 他の回帰モデルとの比較

---

## 補足

本プロジェクトは、公開データを用いた自主学習・自主分析です。

材料データの前処理、統計解析、可視化、
機械学習による物性予測という一連のデータ分析フローを
Pythonで実装することを目的として作成しました。
