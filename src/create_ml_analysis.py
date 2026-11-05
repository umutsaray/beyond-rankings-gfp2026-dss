from __future__ import annotations

import argparse
import importlib.util
import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from extract_gfp_pdfs import DEFAULT_OUTPUT_DIR


TARGET = "pwrindx"
MISSINGNESS_THRESHOLD = 0.40
RANDOM_STATE = 42

EXCLUDE_COLUMNS = {
    "country",
    "year",
    "source_file",
    "source_url",
    "rank",
    "total_countries",
    "last_reviewed_date",
    "pwrindx",
    "extraction_engine",
    "text_line_count",
}

FEATURE_LABELS = {
    "defense_budget": "Savunma bütçesi",
    "purchasing_power": "Satın alma gücü",
    "foreign_reserves": "Döviz/altın rezervi",
    "external_debt": "Dış borç",
    "square_land_area": "Kara alanı",
    "coastline": "Kıyı şeridi",
    "shared_borders": "Kara sınırları",
    "waterways": "Su yolları",
    "total_population": "Toplam nüfus",
    "available_manpower": "Kullanılabilir insan gücü",
    "fit_for_service": "Hizmete uygun nüfus",
    "reaching_military_age_annually": "Askerlik yaşına gelen nüfus",
    "total_military_personnel": "Toplam askerî personel",
    "active_personnel": "Aktif personel",
    "reserve_personnel": "Yedek personel",
    "paramilitary_personnel": "Paramiliter personel",
    "air_force_personnel": "Hava kuvvetleri personeli",
    "army_personnel": "Kara kuvvetleri personeli",
    "navy_personnel": "Deniz kuvvetleri personeli",
    "yearly_mobilization_potential": "Yıllık mobilizasyon potansiyeli",
    "aircraft_total": "Toplam hava aracı",
    "aircraft_readiness": "Hava aracı hazırlık",
    "fighters": "Avcı uçakları",
    "fighters_readiness": "Avcı hazırlık",
    "attack_aircraft": "Taarruz uçakları",
    "attack_aircraft_readiness": "Taarruz uçağı hazırlık",
    "transports": "Nakliye uçakları",
    "transports_readiness": "Nakliye uçağı hazırlık",
    "trainers": "Eğitim uçakları",
    "trainers_readiness": "Eğitim uçağı hazırlık",
    "special_mission": "Özel görev uçakları",
    "special_mission_readiness": "Özel görev hazırlık",
    "tanker_fleet": "Tanker filosu",
    "tanker_fleet_readiness": "Tanker hazırlık",
    "helicopters": "Helikopterler",
    "helicopters_readiness": "Helikopter hazırlık",
    "attack_helicopters": "Taarruz helikopterleri",
    "attack_helicopters_readiness": "Taarruz helikopteri hazırlık",
    "tanks": "Tanklar",
    "tanks_readiness": "Tank hazırlık",
    "vehicles": "Zırhlı/askerî araçlar",
    "vehicles_readiness": "Araç hazırlık",
    "self_propelled_artillery": "Kundağı motorlu topçu",
    "self_propelled_artillery_readiness": "Kundağı motorlu topçu hazırlık",
    "towed_artillery": "Çekili topçu",
    "towed_artillery_readiness": "Çekili topçu hazırlık",
    "mlrs": "ÇNRA/MLRS",
    "mlrs_readiness": "ÇNRA/MLRS hazırlık",
    "naval_total_assets": "Deniz toplam unsur",
    "aircraft_carriers": "Uçak gemileri",
    "helicopter_carriers": "Helikopter gemileri",
    "destroyers": "Destroyerler",
    "frigates": "Fırkateynler",
    "corvettes": "Korvetler",
    "submarines": "Denizaltılar",
    "patrol_vessels": "Devriye gemileri",
    "mine_warfare": "Mayın harbi unsurları",
    "oil_production": "Petrol üretimi",
    "oil_consumption": "Petrol tüketimi",
    "oil_deficit": "Petrol açığı/fazlası",
    "oil_proven_reserves": "Petrol rezervi",
    "natural_gas_production": "Doğal gaz üretimi",
    "natural_gas_consumption": "Doğal gaz tüketimi",
    "natural_gas_deficit": "Doğal gaz açığı/fazlası",
    "natural_gas_proven_reserves": "Doğal gaz rezervi",
    "coal_production": "Kömür üretimi",
    "coal_consumption": "Kömür tüketimi",
    "coal_deficit": "Kömür açığı/fazlası",
    "coal_proven_reserves": "Kömür rezervi",
    "internet_coverage": "İnternet kapsama oranı",
    "labor_force": "İşgücü",
    "merchant_marine_fleet": "Ticaret filosu",
    "ports_trade_terminals": "Liman/ticaret terminali",
    "airports": "Havalimanları",
    "roadway_coverage": "Karayolu ağı",
    "railway_coverage": "Demiryolu ağı",
}


def has_package(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None


def label_for(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


def load_clean_dataset(output_dir: Path) -> pd.DataFrame:
    clean_path = output_dir / "tables" / "gfp_2026_clean.csv"
    if not clean_path.exists():
        raise SystemExit(f"Missing clean dataset: {clean_path}. Run clean_gfp_dataset.py first.")
    return pd.read_csv(clean_path)


def find_turkiye_index(df: pd.DataFrame) -> int:
    mask = df["country"].astype(str).str.casefold().isin({"turkiye", "turkey", "türkiye"})
    if not mask.any():
        mask = df["source_file"].astype(str).str.contains("Turkiye|Turkey|Türkiye", case=False, na=False)
    if not mask.any():
        raise SystemExit("Could not find Turkiye in the cleaned dataset.")
    return int(df.index[mask][0])


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    numeric_df = df.copy()
    for column in numeric_df.columns:
        if column not in EXCLUDE_COLUMNS:
            numeric_df[column] = pd.to_numeric(numeric_df[column], errors="coerce")

    candidate_columns = []
    feature_audit_rows = []
    for column in numeric_df.columns:
        if column in EXCLUDE_COLUMNS or column.endswith("_rank"):
            continue
        values = pd.to_numeric(numeric_df[column], errors="coerce")
        if values.notna().sum() == 0:
            continue
        missing_pct = values.isna().mean()
        excluded = missing_pct > MISSINGNESS_THRESHOLD
        feature_audit_rows.append(
            {
                "feature": column,
                "label": label_for(column),
                "missing_pct": round(float(missing_pct * 100), 2),
                "selected": not excluded,
            }
        )
        if not excluded:
            candidate_columns.append(column)

    y = pd.to_numeric(df[TARGET], errors="coerce")
    valid_rows = y.notna()
    X = numeric_df.loc[valid_rows, candidate_columns].apply(pd.to_numeric, errors="coerce")
    y = y.loc[valid_rows]
    audit_df = pd.DataFrame(feature_audit_rows).sort_values(["selected", "missing_pct", "feature"], ascending=[False, True, True])
    return X, y, audit_df


def get_model_specs(missing_packages: list[str]) -> dict[str, object]:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR

    models: dict[str, object] = {
        "Random Forest Regressor": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=500,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Support Vector Regressor": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", SVR(kernel="rbf", C=10.0, epsilon=0.03, gamma="scale")),
            ]
        ),
        "Ridge Regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
    }

    if has_package("xgboost"):
        from xgboost import XGBRegressor

        models["XGBoost Regressor"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=350,
                        max_depth=3,
                        learning_rate=0.04,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        objective="reg:squarederror",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    else:
        missing_packages.append("xgboost")

    if has_package("lightgbm"):
        from lightgbm import LGBMRegressor

        models["LightGBM Regressor"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMRegressor(
                        n_estimators=350,
                        learning_rate=0.04,
                        num_leaves=15,
                        random_state=RANDOM_STATE,
                        verbose=-1,
                    ),
                ),
            ]
        )
    else:
        missing_packages.append("lightgbm")

    if has_package("catboost"):
        from catboost import CatBoostRegressor

        models["CatBoost Regressor"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    CatBoostRegressor(
                        iterations=350,
                        depth=4,
                        learning_rate=0.04,
                        loss_function="RMSE",
                        random_seed=RANDOM_STATE,
                        verbose=False,
                    ),
                ),
            ]
        )
    else:
        missing_packages.append("catboost")

    if not has_package("shap"):
        missing_packages.append("shap")

    return models


def regression_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse,
        "R2": r2_score(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred) * 100,
    }


def evaluate_models(models: dict[str, object], X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    from sklearn.base import clone
    from sklearn.model_selection import KFold, cross_val_predict

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    predictions = {}
    for model_name, model in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred = cross_val_predict(clone(model), X, y, cv=cv, n_jobs=None)
        predictions[model_name] = y_pred
        metrics = regression_metrics(y, y_pred)
        rows.append(
            {
                "model": model_name,
                "validation": "5-fold CV",
                "n_countries": len(y),
                "n_features": X.shape[1],
                **metrics,
            }
        )
    performance = pd.DataFrame(rows).sort_values(["RMSE", "MAE"])
    return performance, predictions


def fit_model(model: object, X: pd.DataFrame, y: pd.Series) -> object:
    from sklearn.base import clone

    fitted = clone(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted.fit(X, y)
    return fitted


def transformed_feature_frame(fitted_pipeline: object, X: pd.DataFrame) -> pd.DataFrame:
    imputer = fitted_pipeline.named_steps.get("imputer")
    if imputer is None:
        return X.copy()
    transformed = imputer.transform(X)
    return pd.DataFrame(transformed, columns=X.columns, index=X.index)


def get_estimator(fitted_pipeline: object) -> object:
    if hasattr(fitted_pipeline, "named_steps") and "model" in fitted_pipeline.named_steps:
        return fitted_pipeline.named_steps["model"]
    return fitted_pipeline


def feature_importance_table(
    models: dict[str, object],
    performance: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, str, object]:
    preferred = ["Random Forest Regressor", "XGBoost Regressor", "LightGBM Regressor", "CatBoost Regressor", "Ridge Regression"]
    available_order = [name for name in preferred if name in models]
    if not available_order:
        available_order = performance["model"].tolist()

    for model_name in available_order:
        fitted = fit_model(models[model_name], X, y)
        estimator = get_estimator(fitted)
        if hasattr(estimator, "feature_importances_"):
            importances = np.asarray(estimator.feature_importances_, dtype=float)
            method = "model_feature_importance"
        elif hasattr(estimator, "coef_"):
            importances = np.abs(np.asarray(estimator.coef_, dtype=float)).ravel()
            method = "absolute_standardized_coefficient"
        else:
            continue

        if np.isclose(importances.sum(), 0):
            importances = np.repeat(1 / len(importances), len(importances))
        normalized = importances / importances.sum()
        table = pd.DataFrame(
            {
                "feature": X.columns,
                "label": [label_for(column) for column in X.columns],
                "importance": normalized,
                "raw_importance": importances,
                "model": model_name,
                "method": method,
            }
        ).sort_values("importance", ascending=False)
        return table.head(20), model_name, fitted

    raise RuntimeError("No available model exposes feature importances or coefficients.")


def turkiye_prediction_table(
    df: pd.DataFrame,
    y: pd.Series,
    predictions: dict[str, np.ndarray],
    performance: pd.DataFrame,
) -> pd.DataFrame:
    turkiye_idx = find_turkiye_index(df)
    y_index = list(y.index)
    turkiye_position = y_index.index(turkiye_idx)
    rows = []
    actual = float(y.loc[turkiye_idx])
    for model_name, y_pred in predictions.items():
        predicted = float(y_pred[turkiye_position])
        rows.append(
            {
                "country": df.loc[turkiye_idx, "country"],
                "model": model_name,
                "actual_pwrindx": actual,
                "predicted_pwrindx_5fold_oof": predicted,
                "prediction_error": predicted - actual,
                "absolute_error": abs(predicted - actual),
                "pct_error": abs(predicted - actual) / actual * 100,
                "interpretation": "Model Türkiye'yi olduğundan daha zayıf tahmin ediyor"
                if predicted > actual
                else "Model Türkiye'yi olduğundan daha güçlü tahmin ediyor",
            }
        )
    table = pd.DataFrame(rows)
    return table.merge(performance[["model", "RMSE", "R2"]], on="model", how="left").sort_values("absolute_error")


def save_actual_vs_predicted(
    best_model_name: str,
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    output_path: Path,
) -> None:
    from sklearn.base import clone
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)
    fitted = clone(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted.fit(X_train, y_train)
    y_pred = fitted.predict(X_test)
    metrics = regression_metrics(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter(y_test, y_pred, color="#2563eb", alpha=0.8, edgecolor="white", linewidth=0.5)
    low = min(float(np.min(y_test)), float(np.min(y_pred)))
    high = max(float(np.max(y_test)), float(np.max(y_pred)))
    ax.plot([low, high], [low, high], color="#991b1b", linestyle="--", linewidth=1.5)
    ax.set_title(f"Actual vs Predicted PwrIndx ({best_model_name})", weight="bold")
    ax.set_xlabel("Actual PwrIndx (lower is stronger)")
    ax.set_ylabel("Predicted PwrIndx")
    ax.text(
        0.04,
        0.96,
        f"Train/test split only for visualization\nRMSE: {metrics['RMSE']:.3f} | R2: {metrics['R2']:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cbd5e1"),
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_feature_importance_figure(table: pd.DataFrame, output_path: Path) -> None:
    plot_df = table.sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(plot_df["label"], plot_df["importance"], color="#0f766e")
    ax.set_title("Top 20 Feature Importance", weight="bold")
    ax.set_xlabel("Normalized importance")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def shap_or_proxy_values(
    fitted_model: object,
    X: pd.DataFrame,
    y: pd.Series,
    turkiye_idx: int,
    importance_table: pd.DataFrame,
    missing_packages: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if "shap" not in missing_packages:
        try:
            import shap

            estimator = get_estimator(fitted_model)
            X_transformed = transformed_feature_frame(fitted_model, X)
            explainer = shap.Explainer(estimator, X_transformed)
            shap_values = explainer(X_transformed)
            values = np.asarray(shap_values.values)
            if values.ndim == 3:
                values = values[:, :, 0]
            summary = pd.DataFrame(
                {
                    "feature": X.columns,
                    "label": [label_for(column) for column in X.columns],
                    "mean_abs_effect": np.abs(values).mean(axis=0),
                }
            ).sort_values("mean_abs_effect", ascending=False)
            turkiye_position = list(X.index).index(turkiye_idx)
            turkiye = pd.DataFrame(
                {
                    "feature": X.columns,
                    "label": [label_for(column) for column in X.columns],
                    "effect_on_predicted_pwrindx": values[turkiye_position],
                    "interpretation": np.where(
                        values[turkiye_position] < 0,
                        "PwrIndx'i düşürerek askerî gücü iyileştirir",
                        "PwrIndx'i yükselterek tahmini zayıflatır",
                    ),
                }
            ).assign(abs_effect=lambda frame: frame["effect_on_predicted_pwrindx"].abs())
            return summary, turkiye.sort_values("abs_effect", ascending=False), "SHAP"
        except Exception:
            # Fall through to proxy if SHAP cannot explain the selected estimator.
            pass

    imputed = X.fillna(X.median(numeric_only=True))
    standardized = (imputed - imputed.mean()) / imputed.std(ddof=0).replace(0, 1)
    corr = imputed.apply(lambda column: column.corr(y)).fillna(0)
    importance = importance_table.set_index("feature")["importance"].reindex(X.columns).fillna(0)
    proxy = standardized.mul(corr, axis=1).mul(importance, axis=1)
    summary = pd.DataFrame(
        {
            "feature": X.columns,
            "label": [label_for(column) for column in X.columns],
            "mean_abs_effect": proxy.abs().mean(axis=0),
        }
    ).sort_values("mean_abs_effect", ascending=False)
    turkiye = pd.DataFrame(
        {
            "feature": X.columns,
            "label": [label_for(column) for column in X.columns],
            "effect_on_predicted_pwrindx": proxy.loc[turkiye_idx],
        }
    )
    turkiye["interpretation"] = np.where(
        turkiye["effect_on_predicted_pwrindx"] < 0,
        "PwrIndx'i düşürerek askerî gücü iyileştirir",
        "PwrIndx'i yükselterek tahmini zayıflatır",
    )
    turkiye["abs_effect"] = turkiye["effect_on_predicted_pwrindx"].abs()
    return summary, turkiye.sort_values("abs_effect", ascending=False), "proxy_importance_correlation"


def save_shap_summary(summary: pd.DataFrame, method: str, output_path: Path) -> None:
    plot_df = summary.head(20).sort_values("mean_abs_effect")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(plot_df["label"], plot_df["mean_abs_effect"], color="#7c3aed")
    title = "SHAP Summary - Mean Absolute Effect" if method == "SHAP" else "SHAP unavailable - proxy effect summary"
    ax.set_title(title, weight="bold")
    ax.set_xlabel("Mean absolute effect on predicted PwrIndx")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_turkiye_explanation(turkiye_effects: pd.DataFrame, method: str, output_path: Path) -> None:
    plot_df = turkiye_effects.head(15).copy().sort_values("effect_on_predicted_pwrindx")
    colors = np.where(plot_df["effect_on_predicted_pwrindx"] < 0, "#0f766e", "#b91c1c")
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(plot_df["label"], plot_df["effect_on_predicted_pwrindx"], color=colors)
    ax.axvline(0, color="#334155", linewidth=1)
    title = "Türkiye SHAP Explanation" if method == "SHAP" else "Türkiye SHAP proxy explanation"
    ax.set_title(title, weight="bold")
    ax.set_xlabel("Effect on predicted PwrIndx (negative = stronger)")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def write_missing_packages(output_dir: Path, missing_packages: list[str]) -> None:
    path = output_dir / "article_report" / "ml_missing_packages.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if missing_packages:
        text = "\n".join(sorted(set(missing_packages))) + "\n"
    else:
        text = "All optional ML packages are installed.\n"
    path.write_text(text, encoding="utf-8")


def write_ml_report(
    output_dir: Path,
    performance: pd.DataFrame,
    importance: pd.DataFrame,
    turkiye_predictions: pd.DataFrame,
    turkiye_effects: pd.DataFrame,
    missing_packages: list[str],
    shap_method: str,
    n_features: int,
) -> None:
    report_path = output_dir / "article_report" / "ml_interpretation_turkiye.md"
    best = performance.iloc[0]
    best_turkiye = turkiye_predictions.iloc[0]
    negative_effects = turkiye_effects[turkiye_effects["effect_on_predicted_pwrindx"] < 0].head(6)
    positive_effects = turkiye_effects[turkiye_effects["effect_on_predicted_pwrindx"] > 0].head(6)

    lines = [
        "# Makine Öğrenmesi Analizi: Türkiye'nin PwrIndx Tahmini",
        "",
        "## Modelleme Tasarımı",
        (
            f"Hedef değişken `pwrindx` olarak tanımlanmıştır. Ülke kimliği, kaynak dosyası, genel sıra, yıl ve hedef değişken "
            f"model dışı bırakılmış; %40'tan fazla eksik değere sahip alanlar çıkarılmıştır. Kalan {n_features} sayısal gösterge "
            "basit medyan atama ile tamamlanmıştır. Random Forest, SVR ve Ridge modelleri çalıştırılmış; kurulu olan ek paketlere "
            "göre XGBoost, LightGBM ve CatBoost da otomatik olarak sürece dahil edilebilir."
        ),
        "",
        "Ana performans ölçümü 5-katlı çapraz doğrulama ile yapılmıştır. Eğitim/test ayrımı yalnızca gerçek-tahmin grafiği için kullanılmıştır.",
        "",
        "## 5-Katlı Çapraz Doğrulama Sonuçları",
        "",
        "| Model | MAE | RMSE | R2 | MAPE (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in performance.iterrows():
        lines.append(f"| {row['model']} | {row['MAE']:.4f} | {row['RMSE']:.4f} | {row['R2']:.4f} | {row['MAPE']:.2f} |")

    lines.extend(
        [
            "",
            (
                f"En düşük RMSE değerine göre öne çıkan model **{best['model']}** olmuştur "
                f"(RMSE={best['RMSE']:.4f}, R2={best['R2']:.4f}). PwrIndx değerinde daha düşük değer daha güçlü askerî konuma "
                "işaret ettiği için hataların yönü ayrıca yorumlanmalıdır."
            ),
            "",
            "## Türkiye İçin Tahmin - Gerçek Karşılaştırması",
            (
                f"Türkiye'nin gerçek PwrIndx değeri {best_turkiye['actual_pwrindx']:.4f}; en düşük hata veren modelin 5-katlı "
                f"dış-kat tahmini {best_turkiye['predicted_pwrindx_5fold_oof']:.4f} düzeyindedir. "
                f"Tahmin hatası {best_turkiye['prediction_error']:.4f} olarak hesaplanmıştır."
            ),
            "",
            "## Özellik Önemi ve Açıklanabilirlik",
            "En yüksek model önemine sahip göstergeler:",
        ]
    )
    for _, row in importance.head(10).iterrows():
        lines.append(f"- {row['label']}: {row['importance']:.4f}")

    lines.extend(
        [
            "",
            (
                "SHAP etkilerinde negatif değerler, tahmini PwrIndx'i düşürdüğü için Türkiye'nin askerî güç konumunu iyileştirici "
                "yönde yorumlanmalıdır. Pozitif değerler ise tahmini PwrIndx'i yükselterek daha zayıf bir görünüm üretir."
            ),
            "",
            f"Açıklanabilirlik yöntemi: `{shap_method}`.",
            "",
            "Türkiye için PwrIndx'i düşüren başlıca göstergeler:",
        ]
    )
    for _, row in negative_effects.iterrows():
        lines.append(f"- {row['label']}: {row['effect_on_predicted_pwrindx']:.4f}")

    lines.append("")
    lines.append("Türkiye için tahmini PwrIndx'i yükselten başlıca göstergeler:")
    for _, row in positive_effects.iterrows():
        lines.append(f"- {row['label']}: {row['effect_on_predicted_pwrindx']:.4f}")

    if missing_packages:
        lines.extend(
            [
                "",
                "## Eksik Paket Notu",
                (
                    "Aşağıdaki opsiyonel paketler bu ortamda kurulu olmadığı için ilgili modeller veya gerçek SHAP hesaplaması çalıştırılmamıştır: "
                    + ", ".join(sorted(set(missing_packages)))
                    + ". Bu bilgi `ml_missing_packages.txt` dosyasına da yazılmıştır."
                ),
            ]
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def create_ml_analysis(output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    from sklearn.model_selection import train_test_split  # noqa: F401 - validates sklearn availability early

    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    table_dir = output_dir / "article_tables"
    figure_dir = output_dir / "article_figures"
    report_dir = output_dir / "article_report"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    df = load_clean_dataset(output_dir)
    X, y, audit = prepare_features(df)
    if X.empty:
        raise SystemExit("No usable ML features after exclusions and missingness filtering.")

    missing_packages: list[str] = []
    models = get_model_specs(missing_packages)
    write_missing_packages(output_dir, missing_packages)

    performance, predictions = evaluate_models(models, X, y)
    performance.to_csv(table_dir / "table8_ml_model_performance.csv", index=False, encoding="utf-8-sig")

    importance, importance_model_name, fitted_importance_model = feature_importance_table(models, performance, X, y)
    importance.to_csv(table_dir / "table9_feature_importance_top20.csv", index=False, encoding="utf-8-sig")

    turkiye_predictions = turkiye_prediction_table(df, y, predictions, performance)
    turkiye_predictions.to_csv(table_dir / "table10_turkiye_predicted_vs_actual.csv", index=False, encoding="utf-8-sig")

    best_model_name = performance.iloc[0]["model"]
    save_actual_vs_predicted(best_model_name, models[best_model_name], X, y, figure_dir / "fig8_actual_vs_predicted_pwrindx.png")
    save_feature_importance_figure(importance, figure_dir / "fig9_feature_importance_top20.png")

    turkiye_idx = find_turkiye_index(df)
    shap_summary, turkiye_effects, shap_method = shap_or_proxy_values(
        fitted_importance_model,
        X,
        y,
        turkiye_idx,
        importance,
        missing_packages,
    )
    save_shap_summary(shap_summary, shap_method, figure_dir / "fig10_shap_summary.png")
    save_turkiye_explanation(turkiye_effects, shap_method, figure_dir / "fig11_turkiye_shap_explanation.png")

    audit.to_csv(report_dir / "ml_feature_audit.csv", index=False, encoding="utf-8-sig")
    write_ml_report(
        output_dir,
        performance,
        importance,
        turkiye_predictions,
        turkiye_effects,
        missing_packages,
        shap_method,
        X.shape[1],
    )

    print(f"Wrote ML tables to {table_dir}")
    print(f"Wrote ML figures to {figure_dir}")
    print(f"Wrote ML report to {report_dir / 'ml_interpretation_turkiye.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create ML analysis for Türkiye-focused GFP article.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Folder containing generated outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_ml_analysis(args.output_dir)


if __name__ == "__main__":
    main()
