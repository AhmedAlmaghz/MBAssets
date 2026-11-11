#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع: التعامل مع النصوص (Strings)
القسم 4.8: أهم Methods النصوص (الأكثر استخدامًا عمليًا)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 04-025-ex4_8.py
"""

text: str = "البرمجة بلغة Python ممتعة"

print("Python" in text)        # True
print("Java" in text)          # False
print(text.find("Python"))     # 11 (الفهرس حيث بدأت الكلمة)
print(text.find("Java"))       # -1 (غير موجودة)
print(text.count("ة"))         # عدد مرات تكرار الحرف
