#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع: التكرار (Loops)
القسم 7.12: أداء الحلقات: نصائح احترافية مبكرة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 07-020-ex7_12.py
"""

import math

numbers: list[int] = [1, 2, 3, 4, 5]

# ❌ غير فعّال - يُعيد حساب len() في كل تكرار (رغم أنها سريعة، المبدأ عام)
for i in range(len(numbers)):
    print(numbers[i])

# ✅ أفضل - أسلوب Pythonic مباشر
for number in numbers:
    print(number)
