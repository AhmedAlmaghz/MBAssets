#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.6: تحضير البيانات: الخطوة الأهم (غالبًا)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-005-scaler.py
"""

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# fit_transform على بيانات التدريب فقط
X_train_scaled = scaler.fit_transform(X_train)

# transform فقط (وليس fit) على بيانات الاختبار!
X_test_scaled = scaler.transform(X_test)
