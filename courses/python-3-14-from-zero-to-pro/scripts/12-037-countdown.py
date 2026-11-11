#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.8: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-037-countdown.py
"""

# تمرين 1
fruits = ["banana", "kiwi", "apple", "fig"]
sorted_fruits = sorted(fruits, key=lambda word: len(word))
print(sorted_fruits)  # ['fig', 'kiwi', 'apple', 'banana']

# تمرين 2
def countdown(n: int):
    while n > 0:
        yield n
        n -= 1

for num in countdown(5):
    print(num, end=" ")  # 5 4 3 2 1
