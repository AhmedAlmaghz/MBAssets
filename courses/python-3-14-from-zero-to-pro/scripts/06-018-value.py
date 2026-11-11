#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس: اتخاذ القرار (Control Flow)
القسم 6.7: القيم «الصادقة» و«الكاذبة» ضمنيًا (Truthy & Falsy)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 06-018-value.py
"""

value = 5
print(value == True)  # False!  لأن 5 لا يساوي True حرفيًا رغم أنه Truthy
print(bool(value))     # True   لأن 5 يُقيَّم كـ Truthy
