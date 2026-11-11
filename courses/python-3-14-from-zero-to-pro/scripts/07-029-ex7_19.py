#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع: التكرار (Loops)
القسم 7.19: 📌 ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 07-029-ex7_19.py
"""

# while - تكرار شرطي
while condition:
    # الكود
    update_condition()  # لا تنسَ هذا!

# for - تكرار على عناصر معروفة
for item in collection:
    print(item)

for i in range(5):        # 0 إلى 4
for i in range(2, 8):     # 2 إلى 7
for i in range(0, 10, 2): # بقفزة 2

# التحكم بالحلقة
break      # خروج فوري كامل
continue   # تخطّي التكرار الحالي فقط

# else مع الحلقة (نادر لكن مفيد)
for item in items:
    if condition:
        break
else:
    print("لم يحدث break")

# enumerate - فهرس + قيمة
for index, value in enumerate(items):
    print(index, value)
