#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع: التكرار (Loops)
القسم 7.4: حلقة `for`: التكرار على عناصر معروفة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 07-006-ex7_4.py
"""

# range(end) - من 0 حتى end-1
for i in range(5):
    print(i)  # 0, 1, 2, 3, 4

# range(start, end) - من start حتى end-1
for i in range(2, 6):
    print(i)  # 2, 3, 4, 5

# range(start, end, step) - بقفزات محددة
for i in range(0, 10, 2):
    print(i)  # 0, 2, 4, 6, 8

# step سالب للعد التنازلي
for i in range(10, 0, -1):
    print(i)  # 10, 9, 8, 7, 6, 5, 4, 3, 2, 1
