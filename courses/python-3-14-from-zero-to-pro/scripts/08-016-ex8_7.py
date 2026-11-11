#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن: هياكل البيانات - القوائم (Lists)
القسم 8.7: ⚠️ الفخ الأشهر على الإطلاق: نسخ القوائم (Shallow Copy Trap)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 08-016-ex8_7.py
"""

original: list[int] = [1, 2, 3]
copy: list[int] = original  # ⚠️ هذا لا ينسخ القائمة!

copy.append(4)

print(original)  # [1, 2, 3, 4]  ← تغيّرت أيضًا!! رغم أننا عدّلنا copy فقط!
print(copy)      # [1, 2, 3, 4]
