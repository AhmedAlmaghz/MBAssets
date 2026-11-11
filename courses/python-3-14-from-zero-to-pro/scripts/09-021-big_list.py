#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.3: المجموعات (Sets)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-021-big_list.py
"""

import time

big_list = list(range(1_000_000))
big_set = set(range(1_000_000))

start = time.perf_counter()
999_999 in big_list
print(f"List: {time.perf_counter() - start:.6f} ثانية")

start = time.perf_counter()
999_999 in big_set
print(f"Set: {time.perf_counter() - start:.6f} ثانية")

# النتيجة التقريبية: List أبطأ بآلاف المرات!
