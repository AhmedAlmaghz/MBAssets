#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.9: مشروع تطبيقي متكامل: نظام تنبؤ بأسعار المنازل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-010-ex22_9.py
"""

"""سكربت توليد بيانات منازل تجريبية واقعية."""

import numpy as np
import pandas as pd

np.random.seed(42)
n = 500

sizes = np.random.uniform(50, 400, n)               # المساحة بالمتر المربع
rooms = np.random.randint(1, 8, n)                   # عدد الغرف
age = np.random.uniform(0, 40, n)                    # عمر المبنى بالسنوات
city = np.random.choice(["الرياض", "جدة", "الدمام"], n, p=[0.5, 0.3, 0.2])

city_premium = {"الرياض": 1.3, "جدة": 1.15, "الدمام": 1.0}
base_price = (
    sizes * 8
    + rooms * 50
    - age * 10
    + np.random.normal(0, 100, n)  # ضوضاء عشوائية واقعية
)
price = base_price * [city_premium[c] for c in city]
price = np.clip(price, 100, None)  # لا أسعار سالبة

df = pd.DataFrame({
    "المساحة": sizes.round(1),
    "عدد_الغرف": rooms,
    "عمر_المبنى": age.round(1),
    "المدينة": city,
    "السعر": price.round(0),
})

df.to_csv("data/houses.csv", index=False, encoding="utf-8-sig")
print(f"تم توليد {n} عقار في data/houses.csv")
