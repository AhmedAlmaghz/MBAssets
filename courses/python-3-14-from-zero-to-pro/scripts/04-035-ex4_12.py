#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع: التعامل مع النصوص (Strings)
القسم 4.12: أخطاء شائعة ونصائح احترافية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 04-035-ex4_12.py
"""

age: int = 25
message: str = "عمري: " + age  # ❌ TypeError
message: str = "عمري: " + str(age)  # ✅ صحيح
message: str = f"عمري: {age}"       # ✅ الأفضل والأبسط
