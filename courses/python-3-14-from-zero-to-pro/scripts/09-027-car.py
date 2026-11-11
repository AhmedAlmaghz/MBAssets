#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع: القواميس والمجموعات (Dict & Set)
القسم 9.6: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 09-027-car.py
"""

# تمرين 1
car = {"brand": "Toyota", "model": "Corolla", "year": 2023}
print(f"الماركة: {car['brand']}")
print(f"الموديل: {car['model']}")
print(f"السنة: {car['year']}")

# تمرين 2
visitors = ["Ali", "Sara", "Ali", "Omar", "Sara", "Ali"]
unique_visitors = set(visitors)
print(f"عدد الزوار الفريدين: {len(unique_visitors)}")  # 3

# تمرين 3
my_dict = {}
my_dict["a"] = 1
my_dict["b"] = 2
my_dict["c"] = 3
removed = my_dict.pop("b")
print(my_dict)  # {'a': 1, 'c': 3}
