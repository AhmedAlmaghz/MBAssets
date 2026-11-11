#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس: الأرقام والعمليات الحسابية
القسم 5.18: 📌 ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 05-037-population.py
"""

# العمليات الأساسية
a + b, a - b, a * b, a / b
a // b   # قسمة صحيحة
a % b    # باقي القسمة
a ** b   # أُس

# التقريب والدقة
round(3.14159, 2)     # 3.14
from decimal import Decimal
Decimal("19.99")      # دقة مالية تامة

# دوال مُدمجة
abs(-5), max(1,5,3), min(1,5,3)

# وحدة math
import math
math.sqrt(16), math.ceil(4.1), math.floor(4.9), math.pi

# عمليات مختصرة
x += 5   # x = x + 5
x *= 2   # x = x * 2

# فواصل الأرقام الكبيرة
population = 1_000_000

# مقارنة آمنة للأعداد العشرية
import math
math.isclose(0.1 + 0.2, 0.3)  # True
