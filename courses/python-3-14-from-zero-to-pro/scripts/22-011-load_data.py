#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.9: مشروع تطبيقي متكامل: نظام تنبؤ بأسعار المنازل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-011-load_data.py
"""

"""وحدة تحضير بيانات المنازل للتدريب."""

import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = ["المساحة", "عدد_الغرف", "عمر_المبنى", "المدينة"]
TARGET_COLUMN = "السعر"

def load_data(file_path: str) -> pd.DataFrame:
    """تحميل بيانات المنازل من ملف CSV."""
    return pd.read_csv(file_path, encoding="utf-8-sig")

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """تحويل الأعمدة النصية إلى صيغة رقمية مناسبة للنموذج.

    نستخدم One-Hot Encoding للمدينة لأنها فئة غير مرتبة.
    """
    return pd.get_dummies(df[FEATURE_COLUMNS], columns=["المدينة"])

def split_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
):
    """تقسيم البيانات إلى مجموعتي تدريب واختبار.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = prepare_features(df)
    y = df[TARGET_COLUMN]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
