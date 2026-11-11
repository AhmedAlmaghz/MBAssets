#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس: الأرقام والعمليات الحسابية
القسم 5.5: فخ الدقة العشرية: المشكلة الأشهر في كل لغات البرمجة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 05-012-ex5_5.py
"""

price: float = 19.99
quantity: int = 3
total: float = round(price * quantity, 2)
print(total)  # 59.97
