#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.5: خوارزميات شائعة أخرى (نظرة سريعة)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-003-model.py
"""

from sklearn.ensemble import RandomForestClassifier

# نفس البيانات بالضبط، فقط غيّرنا الخوارزمية
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(f"دقة Random Forest: {accuracy_score(y_test, predictions):.2%}")
