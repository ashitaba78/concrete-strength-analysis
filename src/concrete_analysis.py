from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


# =========================================================
# 1. フォルダ・ファイルの場所を設定
# =========================================================

# このPythonファイルがある場所
script_dir = Path(__file__).resolve().parent

# portfolioフォルダ
project_dir = script_dir.parent

# 元データ
data_file = project_dir / "data" / "Concrete_Data.xls"

# 出力先
output_dir = project_dir / "output"

# outputフォルダがなければ作成
output_dir.mkdir(exist_ok=True)


# =========================================================
# 2. 元データが存在するか確認
# =========================================================

if not data_file.exists():
    raise FileNotFoundError(
        f"元データが見つかりません。\n"
        f"確認した場所: {data_file}"
    )


# =========================================================
# 3. Excelデータを読み込む
# =========================================================

df = pd.read_excel(data_file)


# =========================================================
# 4. 列名を分かりやすく変更
# =========================================================

df.columns = [
    "Cement",
    "BlastFurnaceSlag",
    "FlyAsh",
    "Water",
    "Superplasticizer",
    "CoarseAggregate",
    "FineAggregate",
    "Age",
    "Strength"
]


# =========================================================
# 5. 説明変数と目的変数を設定
# =========================================================

predictor_columns = [
    "Cement",
    "BlastFurnaceSlag",
    "FlyAsh",
    "Water",
    "Superplasticizer",
    "CoarseAggregate",
    "FineAggregate",
    "Age"
]

X = df[predictor_columns]

y = df["Strength"]


# =========================================================
# 6. 欠損値を確認
# =========================================================

missing_values = (
    df.isnull()
    .sum()
    .reset_index()
)

missing_values.columns = [
    "Variable",
    "Missing_Count"
]


# =========================================================
# 7. 基本統計量
# =========================================================

basic_statistics = (
    df.describe()
    .T
    .reset_index()
)

basic_statistics = basic_statistics.rename(
    columns={"index": "Variable"}
)


# =========================================================
# 8. 材齢ごとの圧縮強度集計
# =========================================================

age_summary = (
    df.groupby("Age")["Strength"]
    .agg(
        [
            "count",
            "mean",
            "min",
            "max",
            "std"
        ]
    )
    .reset_index()
)


# =========================================================
# 9. 全変数の相関行列
# =========================================================

correlation_matrix = df.corr(
    numeric_only=True
)


# =========================================================
# 10. Strengthとの相関
# =========================================================

strength_correlation = (
    correlation_matrix["Strength"]
    .drop("Strength")
)


strength_correlation_result = pd.DataFrame(
    {
        "Variable": strength_correlation.index,
        "Correlation": strength_correlation.values,
        "AbsoluteCorrelation": strength_correlation.abs().values
    }
)


strength_correlation_result = (
    strength_correlation_result.sort_values(
        "AbsoluteCorrelation",
        ascending=False
    )
)


# =========================================================
# 11. 説明変数どうしの相関行列
# =========================================================

predictor_correlation_matrix = X.corr()


# =========================================================
# 12. 説明変数どうしの相関ペア
# =========================================================

correlation_pairs = []


for i in range(len(predictor_columns)):

    for j in range(
        i + 1,
        len(predictor_columns)
    ):

        variable_1 = predictor_columns[i]
        variable_2 = predictor_columns[j]

        correlation_value = (
            predictor_correlation_matrix.loc[
                variable_1,
                variable_2
            ]
        )

        correlation_pairs.append(
            {
                "Variable_1": variable_1,
                "Variable_2": variable_2,
                "Correlation": correlation_value,
                "AbsoluteCorrelation": abs(
                    correlation_value
                )
            }
        )


correlation_pairs_result = pd.DataFrame(
    correlation_pairs
)


correlation_pairs_result = (
    correlation_pairs_result.sort_values(
        "AbsoluteCorrelation",
        ascending=False
    )
)


# =========================================================
# 13. 通常の重回帰分析
# =========================================================

X_with_const = sm.add_constant(X)


ols_model = sm.OLS(
    y,
    X_with_const
)


ols_result = ols_model.fit()


# =========================================================
# 14. 説明変数・目的変数を標準化
# =========================================================

X_standardized = (
    (X - X.mean())
    / X.std(ddof=0)
)


y_standardized = (
    (y - y.mean())
    / y.std(ddof=0)
)


# =========================================================
# 15. 標準化したデータで重回帰
# =========================================================

X_standardized_with_const = sm.add_constant(
    X_standardized
)


standardized_model = sm.OLS(
    y_standardized,
    X_standardized_with_const
)


standardized_result = standardized_model.fit()


beta = (
    standardized_result.params
    .drop("const")
)


# =========================================================
# 16. VIF・Toleranceを計算
# =========================================================

X_vif = sm.add_constant(X)

vif_rows = []


for i, variable in enumerate(
    X_vif.columns
):

    if variable == "const":
        continue

    vif_value = (
        variance_inflation_factor(
            X_vif.values,
            i
        )
    )

    tolerance_value = (
        1 / vif_value
    )

    vif_rows.append(
        {
            "Variable": variable,
            "VIF": vif_value,
            "Tolerance": tolerance_value
        }
    )


vif_result = pd.DataFrame(
    vif_rows
)


# =========================================================
# 17. 重回帰結果を表にまとめる
# =========================================================

confidence_intervals = (
    ols_result.conf_int()
)


regression_result = pd.DataFrame(
    {
        "Variable": predictor_columns,

        "B": [
            ols_result.params[variable]
            for variable in predictor_columns
        ],

        "Beta": [
            beta[variable]
            for variable in predictor_columns
        ],

        "Std_Error": [
            ols_result.bse[variable]
            for variable in predictor_columns
        ],

        "t_value": [
            ols_result.tvalues[variable]
            for variable in predictor_columns
        ],

        "p_value": [
            ols_result.pvalues[variable]
            for variable in predictor_columns
        ],

        "CI_2.5%": [
            confidence_intervals.loc[
                variable,
                0
            ]
            for variable in predictor_columns
        ],

        "CI_97.5%": [
            confidence_intervals.loc[
                variable,
                1
            ]
            for variable in predictor_columns
        ]
    }
)


regression_result = regression_result.merge(
    vif_result,
    on="Variable"
)


regression_result["Absolute_Beta"] = (
    regression_result["Beta"].abs()
)


regression_result = (
    regression_result.sort_values(
        "Absolute_Beta",
        ascending=False
    )
)


# =========================================================
# 18. 学習用・テスト用データに分割
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# =========================================================
# 19. 線形回帰モデル
# =========================================================

linear_model = LinearRegression()


linear_model.fit(
    X_train,
    y_train
)


linear_prediction = (
    linear_model.predict(
        X_test
    )
)


# =========================================================
# 20. Random Forestモデル
# =========================================================

random_forest_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)


random_forest_model.fit(
    X_train,
    y_train
)


random_forest_prediction = (
    random_forest_model.predict(
        X_test
    )
)


# =========================================================
# 21. 線形回帰の評価
# =========================================================

linear_r2 = r2_score(
    y_test,
    linear_prediction
)


linear_mae = mean_absolute_error(
    y_test,
    linear_prediction
)


linear_mse = mean_squared_error(
    y_test,
    linear_prediction
)


linear_rmse = (
    linear_mse ** 0.5
)


# =========================================================
# 22. Random Forestの評価
# =========================================================

rf_r2 = r2_score(
    y_test,
    random_forest_prediction
)


rf_mae = mean_absolute_error(
    y_test,
    random_forest_prediction
)


rf_mse = mean_squared_error(
    y_test,
    random_forest_prediction
)


rf_rmse = (
    rf_mse ** 0.5
)


# =========================================================
# 23. モデル比較表
# =========================================================

model_comparison = pd.DataFrame(
    {
        "Model": [
            "Linear Regression",
            "Random Forest"
        ],

        "R2": [
            linear_r2,
            rf_r2
        ],

        "MAE_MPa": [
            linear_mae,
            rf_mae
        ],

        "RMSE_MPa": [
            linear_rmse,
            rf_rmse
        ]
    }
)


# =========================================================
# 24. 実測値・予測値をまとめる
# =========================================================

prediction_result = pd.DataFrame(
    {
        "Original_Index": y_test.index,

        "Actual_Strength": (
            y_test.values
        ),

        "Linear_Prediction": (
            linear_prediction
        ),

        "RandomForest_Prediction": (
            random_forest_prediction
        )
    }
)


prediction_result[
    "Linear_Error"
] = (
    prediction_result[
        "Linear_Prediction"
    ]
    - prediction_result[
        "Actual_Strength"
    ]
)


prediction_result[
    "RandomForest_Error"
] = (
    prediction_result[
        "RandomForest_Prediction"
    ]
    - prediction_result[
        "Actual_Strength"
    ]
)


# =========================================================
# 25. Random Forest Feature Importance
# =========================================================

importance_result = pd.DataFrame(
    {
        "Variable": predictor_columns,

        "Importance": (
            random_forest_model
            .feature_importances_
        )
    }
)


importance_result = (
    importance_result.sort_values(
        "Importance",
        ascending=False
    )
)


# =========================================================
# 26. モデル全体の統計情報
# =========================================================

ols_model_summary = pd.DataFrame(
    {
        "Metric": [
            "Observations",
            "R-squared",
            "Adjusted R-squared",
            "F-statistic",
            "F-test p-value"
        ],

        "Value": [
            len(df),
            ols_result.rsquared,
            ols_result.rsquared_adj,
            ols_result.fvalue,
            ols_result.f_pvalue
        ]
    }
)


# =========================================================
# 27. Excelに分析結果をまとめる
# =========================================================

excel_file = (
    output_dir
    / "analysis_result.xlsx"
)


with pd.ExcelWriter(
    excel_file,
    engine="openpyxl"
) as writer:

    df.to_excel(
        writer,
        sheet_name="Data",
        index=False
    )

    missing_values.to_excel(
        writer,
        sheet_name="Missing_Values",
        index=False
    )

    basic_statistics.to_excel(
        writer,
        sheet_name="Basic_Statistics",
        index=False
    )

    age_summary.to_excel(
        writer,
        sheet_name="Age_Summary",
        index=False
    )

    correlation_matrix.to_excel(
        writer,
        sheet_name="Correlation_Matrix"
    )

    strength_correlation_result.to_excel(
        writer,
        sheet_name="Strength_Correlation",
        index=False
    )

    predictor_correlation_matrix.to_excel(
        writer,
        sheet_name="Predictor_Correlation"
    )

    correlation_pairs_result.to_excel(
        writer,
        sheet_name="Correlation_Pairs",
        index=False
    )

    ols_model_summary.to_excel(
        writer,
        sheet_name="OLS_Model_Summary",
        index=False
    )

    regression_result.to_excel(
        writer,
        sheet_name="Regression",
        index=False
    )

    model_comparison.to_excel(
        writer,
        sheet_name="Model_Comparison",
        index=False
    )

    prediction_result.to_excel(
        writer,
        sheet_name="Predictions",
        index=False
    )

    importance_result.to_excel(
        writer,
        sheet_name="Feature_Importance",
        index=False
    )


# =========================================================
# 28. Strengthのヒストグラム
# =========================================================

plt.figure(
    figsize=(8, 6)
)

plt.hist(
    df["Strength"],
    bins=20
)

plt.title(
    "Distribution of Concrete Strength"
)

plt.xlabel(
    "Strength (MPa)"
)

plt.ylabel(
    "Frequency"
)

plt.tight_layout()

plt.savefig(
    output_dir
    / "01_strength_histogram.png",
    dpi=150
)

plt.close()


# =========================================================
# 29. CementとStrengthの散布図
# =========================================================

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    df["Cement"],
    df["Strength"]
)

plt.title(
    "Cement vs Concrete Strength"
)

plt.xlabel(
    "Cement (kg/m3)"
)

plt.ylabel(
    "Strength (MPa)"
)

plt.tight_layout()

plt.savefig(
    output_dir
    / "02_cement_vs_strength.png",
    dpi=150
)

plt.close()


# =========================================================
# 30. Strengthとの相関棒グラフ
# =========================================================

correlation_plot = (
    strength_correlation_result.sort_values(
        "AbsoluteCorrelation",
        ascending=True
    )
)


plt.figure(
    figsize=(8, 6)
)

plt.barh(
    correlation_plot["Variable"],
    correlation_plot[
        "AbsoluteCorrelation"
    ]
)

plt.title(
    "Correlation with Concrete Strength"
)

plt.xlabel(
    "Absolute Correlation"
)

plt.ylabel(
    "Variable"
)

plt.tight_layout()

plt.savefig(
    output_dir
    / "03_strength_correlation.png",
    dpi=150
)

plt.close()


# =========================================================
# 31. 説明変数どうしの相関行列
# =========================================================

plt.figure(
    figsize=(10, 8)
)


image = plt.imshow(
    predictor_correlation_matrix,
    vmin=-1,
    vmax=1
)


plt.colorbar(
    image
)


plt.xticks(
    range(
        len(predictor_columns)
    ),
    predictor_columns,
    rotation=45,
    ha="right"
)


plt.yticks(
    range(
        len(predictor_columns)
    ),
    predictor_columns
)


for i in range(
    len(predictor_columns)
):

    for j in range(
        len(predictor_columns)
    ):

        value = (
            predictor_correlation_matrix
            .iloc[i, j]
        )

        plt.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center"
        )


plt.title(
    "Correlation Matrix of Predictor Variables"
)

plt.tight_layout()

plt.savefig(
    output_dir
    / "04_predictor_correlation_matrix.png",
    dpi=150
)

plt.close()


# =========================================================
# 32. 線形回帰：実測値 vs 予測値
# =========================================================

linear_minimum = min(
    y_test.min(),
    linear_prediction.min()
)


linear_maximum = max(
    y_test.max(),
    linear_prediction.max()
)


plt.figure(
    figsize=(7, 7)
)


plt.scatter(
    y_test,
    linear_prediction
)


plt.plot(
    [
        linear_minimum,
        linear_maximum
    ],
    [
        linear_minimum,
        linear_maximum
    ]
)


plt.title(
    "Actual vs Predicted - Linear Regression"
)

plt.xlabel(
    "Actual Strength (MPa)"
)

plt.ylabel(
    "Predicted Strength (MPa)"
)

plt.tight_layout()

plt.savefig(
    output_dir
    / "05_actual_vs_predicted_linear.png",
    dpi=150
)

plt.close()


# =========================================================
# 33. Random Forest：実測値 vs 予測値
# =========================================================

rf_minimum = min(
    y_test.min(),
    random_forest_prediction.min()
)


rf_maximum = max(
    y_test.max(),
    random_forest_prediction.max()
)


plt.figure(
    figsize=(7, 7)
)


plt.scatter(
    y_test,
    random_forest_prediction
)


plt.plot(
    [
        rf_minimum,
        rf_maximum
    ],
    [
        rf_minimum,
        rf_maximum
    ]
)


plt.title(
    "Actual vs Predicted - Random Forest"
)

plt.xlabel(
    "Actual Strength (MPa)"
)

plt.ylabel(
    "Predicted Strength (MPa)"
)

plt.tight_layout()

plt.savefig(
    output_dir
    / "06_actual_vs_predicted_random_forest.png",
    dpi=150
)

plt.close()


# =========================================================
# 34. Random Forest Feature Importance
# =========================================================

importance_plot = (
    importance_result.sort_values(
        "Importance",
        ascending=True
    )
)


plt.figure(
    figsize=(8, 6)
)


plt.barh(
    importance_plot["Variable"],
    importance_plot["Importance"]
)


plt.title(
    "Random Forest Feature Importance"
)


plt.xlabel(
    "Importance"
)


plt.ylabel(
    "Variable"
)


plt.tight_layout()

plt.savefig(
    output_dir
    / "07_random_forest_feature_importance.png",
    dpi=150
)

plt.close()


# =========================================================
# 35. 実行結果を表示
# =========================================================

print(
    "========================================"
)

print(
    "Concrete Strength Analysis"
)

print(
    "========================================"
)


print(
    f"\nデータ件数: {len(df)}"
)


print(
    f"欠損値合計: "
    f"{df.isnull().sum().sum()}"
)


print(
    "\n【重回帰分析】"
)

print(
    f"R²: "
    f"{ols_result.rsquared:.3f}"
)

print(
    f"Adjusted R²: "
    f"{ols_result.rsquared_adj:.3f}"
)


print(
    "\n【モデル比較】"
)

print(
    model_comparison.to_string(
        index=False
    )
)


print(
    "\n【Random Forest Feature Importance】"
)

print(
    importance_result.to_string(
        index=False
    )
)


print(
    "\n========================================"
)

print(
    "分析が完了しました。"
)

print(
    "========================================"
)


print(
    f"\n出力先:\n{output_dir}"
)