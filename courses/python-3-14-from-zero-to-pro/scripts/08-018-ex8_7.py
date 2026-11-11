#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن: هياكل البيانات - القوائم (Lists)
القسم 8.7: ⚠️ الفخ الأشهر على الإطلاق: نسخ القوائم (Shallow Copy Trap)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 08-018-ex8_7.py
"""

original: list[int] = [1, 2, 3]

# 3 طرق صحيحة لإنشاء نسخة فعلية مستقلة
copy1: list[int] = original.copy()
copy2: list[int] = original[:]  # تقطيع كامل
copy3: list[int] = list(original)

copy1.append(4)
print(original)  # [1, 2, 3]  ← لم تتأثر! ✅
print(copy1)     # [1, 2, 3, 4]
