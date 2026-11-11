#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع: التكرار (Loops)
القسم 7.6: `break`: الخروج الفوري من الحلقة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 07-010-found.py
"""

numbers: list[int] = [4, 7, 2, 9, 5, 1]  # سنتعلم القوائم بعمق قريبًا
target: int = 9
found: bool = False

for num in numbers:
    if num == target:
        found = True
        break  # لا داعي لمتابعة البحث بعد إيجاده!

if found:
    print("العنصر موجود!")
