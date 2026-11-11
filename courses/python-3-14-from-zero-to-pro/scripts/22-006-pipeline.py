#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.6: تحضير البيانات: الخطوة الأهم (غالبًا)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-006-pipeline.py
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# الـ Pipeline يُطبّق كل الخطوات تلقائيًا وبالترتيب الصحيح
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", KNeighborsClassifier(n_neighbors=3)),
])

pipeline.fit(X_train, y_train)          # يُطبّق scaler ثم classifier تلقائيًا
predictions = pipeline.predict(X_test)  # ونفس الشيء هنا -- بدون تسرّب بيانات!
