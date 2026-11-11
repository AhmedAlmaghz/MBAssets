#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.9: مشروع تطبيقي متكامل: نظام تنبؤ بأسعار المنازل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-012-evaluate_model.py
"""

"""وحدة تدريب ومقارنة نماذج التنبؤ بأسعار المنازل."""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor

CANDIDATE_MODELS = {
    "الانحدار الخطي": LinearRegression(),
    "شجرة القرار": DecisionTreeRegressor(random_state=42),
    "الغابة العشوائية": RandomForestRegressor(n_estimators=100, random_state=42),
}

def evaluate_model(model, X_test, y_test) -> dict[str, float]:
    """تقييم نموذج مُدرَّب وإرجاع مقاييس الأداء."""
    predictions = model.predict(X_test)
    return {
        "MAE": mean_absolute_error(y_test, predictions),
        "R2": r2_score(y_test, predictions),
    }

def train_and_compare_models(X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """تدريب عدة نماذج ومقارنة أدائها.

    Returns:
        DataFrame يلخص أداء كل نموذج، مرتّب من الأفضل للأسوأ حسب R2.
    """
    results = []

    for name, model in CANDIDATE_MODELS.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results.append({"النموذج": name, **metrics})

    results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
    return results_df

def get_best_model(X_train, y_train):
    """تدريب أفضل نموذج (الغابة العشوائية) وإرجاعه جاهزًا للاستخدام."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, file_path: str) -> None:
    """حفظ النموذج المُدرَّب كملف."""
    joblib.dump(model, file_path)
