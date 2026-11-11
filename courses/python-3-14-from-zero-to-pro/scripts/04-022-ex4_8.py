#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع: التعامل مع النصوص (Strings)
القسم 4.8: أهم Methods النصوص (الأكثر استخدامًا عمليًا)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 04-022-ex4_8.py
"""

text: str = "Hello World"

print(text.upper())       # HELLO WORLD
print(text.lower())       # hello world
print(text.title())       # Hello World (كل كلمة تبدأ بحرف كبير)
print(text.capitalize())  # Hello world (أول حرف فقط كبير)
