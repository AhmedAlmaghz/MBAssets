#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.2: القواميس (Dictionaries)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-011-numbers.py
"""

# الصيغة: {key_expr: value_expr for item in iterable}

# مثال 1: تربيع الأرقام
numbers = [1, 2, 3, 4, 5]
squares = {n: n**2 for n in numbers}
print(squares)  # {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# مثال 2: مع شرط
even_squares = {n: n**2 for n in numbers if n % 2 == 0}
print(even_squares)  # {2: 4, 4: 16}

# مثال 3: عكس قاموس (قلب المفاتيح والقيم)
original = {"a": 1, "b": 2, "c": 3}
reversed_dict = {value: key for key, value in original.items()}
print(reversed_dict)  # {1: 'a', 2: 'b', 3: 'c'}

# مثال 4: تحويل قائمتين متوازيتين إلى قاموس باستخدام zip
names = ["Sara", "Omar", "Lina"]
scores = [90, 75, 88]
result = {name: score for name, score in zip(names, scores)}
print(result)  # {'Sara': 90, 'Omar': 75, 'Lina': 88}
