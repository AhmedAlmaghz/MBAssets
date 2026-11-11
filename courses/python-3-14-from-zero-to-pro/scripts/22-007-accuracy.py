#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني والعشرون: مسار الذكاء الاصطناعي — نافذة أولى (scikit-learn)
القسم 22.7: مقاييس التقييم (Evaluation Metrics)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 22-007-accuracy.py
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

accuracy = accuracy_score(y_test, predictions)        # نسبة التنبؤات الصحيحة
precision = precision_score(y_test, predictions, average="macro")  # دقة التنبؤات الإيجابية
recall = recall_score(y_test, predictions, average="macro")        # نسبة اكتشاف الحالات الإيجابية الفعلية
f1 = f1_score(y_test, predictions, average="macro")   # متوسط توافقي بين precision وrecall

print(classification_report(y_test, predictions))  # تقرير شامل بكل المقاييس دفعة واحدة
print(confusion_matrix(y_test, predictions))        # مصفوفة توضح أنواع الأخطاء بالتفصيل
