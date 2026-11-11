#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.8: حفظ النموذج واستخدامه لاحقًا

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-009-loaded_model.py
"""

import joblib

# حفظ النموذج المُدرَّب
joblib.dump(pipeline, "model.joblib")

# تحميله لاحقًا (في سكربت آخر، أو حتى بعد أسابيع)
loaded_model = joblib.load("model.joblib")
new_prediction = loaded_model.predict(X_new)
