#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس: اتخاذ القرار (Control Flow)
القسم 6.17: 📌 ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 06-035-result.py
"""

# الشرط الأساسي
if condition:
    pass
elif other_condition:
    pass
else:
    pass

# العوامل المنطقية
x and y   # كلاهما True
x or y    # أحدهما True
not x     # عكس x

# Ternary Expression
result = "A" if condition else "B"

# match-case (Python 3.10+)
match value:
    case 1:
        pass
    case 2 | 3:  # أو
        pass
    case _:  # الحالة الافتراضية
        pass

# Walrus Operator
if (n := get_value()) > 5:
    print(n)

# Truthy / Falsy
if my_string:  # True إن لم يكن فارغًا
    pass
