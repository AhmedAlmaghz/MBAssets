#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس: الأرقام والعمليات الحسابية
القسم 5.12: الأعداد العشوائية: مقدمة سريعة لوحدة `random`

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 05-031-ex5_12.py
"""

import random

random_number: int = random.randint(1, 10)  # رقم عشوائي بين 1 و10 (شاملاً الطرفين)
print(random_number)

random_float: float = random.random()  # رقم عشوائي بين 0 و1
print(random_float)
