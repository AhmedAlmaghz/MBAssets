#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العشرون: مسار تحليل البيانات (pandas, numpy, matplotlib)
القسم 20.6: مشروع تطبيقي متكامل: تحليل مبيعات متجر إلكتروني

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 20-028-n_records.py
"""

"""سكربت مساعد لتوليد بيانات مبيعات تجريبية واقعية."""

import numpy as np
import pandas as pd

np.random.seed(42)  # لضمان نتائج قابلة لإعادة الإنتاج

n_records = 1000

products = ["لابتوب", "ماوس", "لوحة مفاتيح", "شاشة", "سماعات", "طابعة"]
product_prices = {
    "لابتوب": 3500, "ماوس": 50, "لوحة مفاتيح": 120,
    "شاشة": 900, "سماعات": 150, "طابعة": 600,
}
cities = ["الرياض", "جدة", "الدمام", "مكة", "المدينة"]

data = {
    "order_id": range(1, n_records + 1),
    "التاريخ": pd.date_range("2024-01-01", "2024-12-31", periods=n_records),
    "المنتج": np.random.choice(products, n_records),
    "المدينة": np.random.choice(cities, n_records, p=[0.35, 0.25, 0.15, 0.15, 0.10]),
    "الكمية": np.random.randint(1, 6, n_records),
}

df = pd.DataFrame(data)
df["سعر_الوحدة"] = df["المنتج"].map(product_prices)
df["الإجمالي"] = df["سعر_الوحدة"] * df["الكمية"]

# إدخال بعض القيم المفقودة عمدًا (كما يحدث في البيانات الحقيقية)
missing_indices = np.random.choice(df.index, size=20, replace=False)
df.loc[missing_indices, "المدينة"] = None

df.to_csv("data/sales_data.csv", index=False, encoding="utf-8-sig")
print(f"تم توليد {n_records} سجل مبيعات في data/sales_data.csv")
