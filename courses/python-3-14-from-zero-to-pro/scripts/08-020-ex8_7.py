#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن: هياكل البيانات - القوائم (Lists)
القسم 8.7: ⚠️ الفخ الأشهر على الإطلاق: نسخ القوائم (Shallow Copy Trap)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 08-020-ex8_7.py
"""

import copy

original: list[list[int]] = [[1, 2], [3, 4]]
deep_copy: list[list[int]] = copy.deepcopy(original)

deep_copy[0].append(99)

print(original)   # [[1, 2], [3, 4]]  ← لم تتأثر الآن! ✅
print(deep_copy)  # [[1, 2, 99], [3, 4]]
