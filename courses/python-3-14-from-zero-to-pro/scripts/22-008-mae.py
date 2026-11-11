#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.7: مقاييس التقييم (Evaluation Metrics)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-008-mae.py
"""

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, predictions)   # متوسط الخطأ المطلق
mse = mean_squared_error(y_test, predictions)    # متوسط مربع الخطأ (يُعاقب الأخطاء الكبيرة أكثر)
r2 = r2_score(y_test, predictions)                # نسبة التباين المُفسَّر (0 إلى 1، كلما اقترب من 1 كان أفضل)

print(f"متوسط الخطأ المطلق: {mae:.2f}")
print(f"R²: {r2:.3f}")
