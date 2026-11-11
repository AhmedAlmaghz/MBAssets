#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن: هياكل البيانات - القوائم (Lists)
القسم 8.13: أخطاء شائعة ونصائح احترافية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 08-042-ex8_13.py
"""

# ✅ الحل الصحيح: التكرار على نسخة، والتعديل على الأصل
numbers: list[int] = [1, 2, 3, 4, 5]

for number in numbers.copy():  # أو numbers[:]
    if number % 2 == 0:
        numbers.remove(number)

print(numbers)  # [1, 3, 5]  ← آمن وصحيح
