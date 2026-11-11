#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.6: تحضير البيانات: الخطوة الأهم (غالبًا)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-004-ex22_6.py
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df = pd.DataFrame({
    "المدينة": ["الرياض", "جدة", "الرياض", "الدمام"],
    "السعر": [100, 150, 120, 90],
})

# الطريقة 1: Label Encoding -- يحوّل كل فئة لرقم (مناسب للفئات المرتبة فقط!)
encoder = LabelEncoder()
df["المدينة_مشفرة"] = encoder.fit_transform(df["المدينة"])
print(df)

# الطريقة 2: One-Hot Encoding -- عمود منفصل لكل فئة (الأنسب غالبًا للفئات غير المرتبة)
df_encoded = pd.get_dummies(df, columns=["المدينة"])
print(df_encoded)
