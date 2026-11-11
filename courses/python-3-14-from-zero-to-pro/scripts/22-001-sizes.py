#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.3: التثبيت وأول نموذج

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-001-sizes.py
"""

import numpy as np
from sklearn.linear_model import LinearRegression

# بيانات تدريب: مساحة المنزل (م²) -> السعر (بالآلاف)
# Features يجب أن تكون دائمًا مصفوفة ثنائية الأبعاد: [[قيمة], [قيمة], ...]
sizes = np.array([[50], [70], [90], [120], [150], [200]])
prices = np.array([150, 200, 250, 320, 400, 500])  # Target أحادي البعد

# إنشاء النموذج وتدريبه
model = LinearRegression()
model.fit(sizes, prices)  # "fit" = التدريب

# التنبؤ بسعر منزل جديد مساحته 100 م²
new_size = np.array([[100]])
predicted_price = model.predict(new_size)

print(f"السعر المتوقع لمنزل 100م²: {predicted_price[0]:.1f} ألف")

# معاملات النموذج المُتعلَّمة (المعادلة: price = slope * size + intercept)
print(f"الميل (slope): {model.coef_[0]:.2f}")
print(f"التقاطع (intercept): {model.intercept_:.2f}")
