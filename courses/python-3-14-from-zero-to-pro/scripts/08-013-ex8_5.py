#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن: هياكل البيانات - القوائم (Lists)
القسم 8.5: أهم Methods القوائم

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 08-013-ex8_5.py
"""

fruits: list[str] = ["تفاح", "موز", "تفاح", "عنب"]

print(fruits.count("تفاح"))   # 2  ← عدد مرات التكرار
print(fruits.index("موز"))    # 1  ← فهرس أول ظهور
print("عنب" in fruits)        # True
print("مانجو" in fruits)      # False
