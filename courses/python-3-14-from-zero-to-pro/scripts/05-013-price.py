#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس: الأرقام والعمليات الحسابية
القسم 5.5: فخ الدقة العشرية: المشكلة الأشهر في كل لغات البرمجة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 05-013-price.py
"""

from decimal import Decimal

price = Decimal("19.99")  # لاحظ: كنص! وليس float مباشرة
quantity = Decimal("3")
total = price * quantity

print(total)  # 59.97  ← دقيق تمامًا!
