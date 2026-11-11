#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.2: أساسيات تعريف واستدعاء الدالة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-004-calculate_bmi.py
"""

def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """
    حساب مؤشر كتلة الجسم (BMI).

    المعاملات:
        weight_kg: الوزن بالكيلوغرام
        height_m: الطول بالمتر

    تُرجع:
        قيمة BMI كرقم عشري
    """
    return weight_kg / (height_m ** 2)

# يمكنك قراءة التوثيق في أي وقت باستخدام:
print(calculate_bmi.__doc__)
help(calculate_bmi)
