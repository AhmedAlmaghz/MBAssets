#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن: هياكل البيانات - القوائم (Lists)
القسم 8.12: Tuples: الأخ «غير القابل للتعديل» للقائمة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 08-038-ex8_12.py
"""

point: tuple[int, int] = (10, 20)

print(point[0])     # 10  ← الفهرسة تعمل
print(len(point))   # 2   ← len تعمل

x, y = point  # Unpacking - تفكيك القيم مباشرة لمتغيرات!
print(x, y)   # 10 20
