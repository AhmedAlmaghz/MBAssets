#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.13: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-017-model.py
"""

# ===== النمط الأساسي الموحّد لكل نموذج =====
from sklearn.SOME_MODULE import SomeAlgorithm
model = SomeAlgorithm()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# ===== تقسيم البيانات =====
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===== تحضير البيانات =====
import pandas as pd
pd.get_dummies(df, columns=["فئة_نصية"])        # One-Hot Encoding

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit_transform على التدريب فقط!
X_test_scaled = scaler.transform(X_test)         # transform فقط على الاختبار

# ===== Pipeline =====
from sklearn.pipeline import Pipeline
pipe = Pipeline([("scaler", StandardScaler()), ("model", SomeAlgorithm())])
pipe.fit(X_train, y_train)

# ===== التقييم =====
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
accuracy_score(y_test, predictions)        # للتصنيف
mean_absolute_error(y_test, predictions)   # للانحدار
r2_score(y_test, predictions)              # للانحدار

# ===== الحفظ والتحميل =====
import joblib
joblib.dump(model, "model.joblib")
loaded = joblib.load("model.joblib")
