#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.5: المولّدات (Generators): الكفاءة في الذاكرة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-034-fib.py
"""

from itertools import islice, count, cycle

# islice: أخذ جزء محدد من generator لا نهائي
fib = infinite_fibonacci()
first_5 = list(islice(fib, 5))
print(first_5)  # [0, 1, 1, 2, 3]

# count: عداد لا نهائي يبدأ من رقم معين
for i in islice(count(10, 2), 5):  # يبدأ من 10، بخطوة 2
    print(i, end=" ")
print()  # 10 12 14 16 18

# cycle: تكرار عناصر قائمة بشكل لا نهائي
colors = cycle(["red", "green", "blue"])
for color in islice(colors, 7):
    print(color, end=" ")
print()  # red green blue red green blue red
