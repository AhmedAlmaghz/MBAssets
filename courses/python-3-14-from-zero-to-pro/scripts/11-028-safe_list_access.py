#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.11: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-028-safe_list_access.py
"""

# تمرين 1
try:
    num = int(input("أدخل رقماً: "))
    print("رقم صحيح")
except ValueError:
    print("خطأ: ليس رقماً")

# تمرين 2
def safe_list_access(lst: list, index: int):
    try:
        return lst[index]
    except IndexError:
        return None

print(safe_list_access([1, 2, 3], 1))   # 2
print(safe_list_access([1, 2, 3], 10))  # None
