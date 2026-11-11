#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن: هياكل البيانات - القوائم (Lists)
القسم 8.18: 📌 ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 08-045-copy.py
"""

# الإنشاء
my_list: list[int] = [1, 2, 3]
my_tuple: tuple[int, ...] = (1, 2, 3)

# الإضافة/الحذف
my_list.append(4)
my_list.insert(0, 99)
my_list.remove(2)       # بالقيمة
my_list.pop(0)           # بالفهرس، تُعيد القيمة

# الترتيب
my_list.sort()           # يُعدّل الأصل، يُعيد None
sorted(my_list)          # نسخة جديدة مُرتَّبة

# النسخ (مهم جدًا!)
copy = original          # ⚠️ مرجع فقط، ليس نسخة!
copy = original.copy()   # نسخة سطحية
import copy as cp
deep = cp.deepcopy(original)  # نسخة عميقة كاملة

# List Comprehension
[x**2 for x in range(5)]
[x for x in range(10) if x % 2 == 0]

# دوال مفيدة
len(my_list), sum(my_list), max(my_list), min(my_list)
zip(list1, list2)
enumerate(my_list, start=1)
