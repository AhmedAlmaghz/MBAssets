#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن: هياكل البيانات - القوائم (Lists)
القسم 8.5: أهم Methods القوائم

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 08-007-ex8_5.py
"""

numbers: list[int] = [1, 2, 3]

numbers.append([4, 5])
print(numbers)  # [1, 2, 3, [4, 5]]  ← قائمة متداخلة! ليس هذا المقصود غالبًا

numbers2: list[int] = [1, 2, 3]
numbers2.extend([4, 5])
print(numbers2)  # [1, 2, 3, 4, 5]  ← هذا هو المقصود عادةً
