#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع: التكرار (Loops)
القسم 7.8: `else` في الحلقات: ميزة نادرًا ما تُشرح بوضوح

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 07-014-ex7_8.py
"""

numbers: list[int] = [1, 3, 5, 7, 9]
target: int = 4

for num in numbers:
    if num == target:
        print("وُجد!")
        break
else:
    print("لم يُوجد العنصر في القائمة")  # ستُطبع هذه، لأن 4 غير موجود
