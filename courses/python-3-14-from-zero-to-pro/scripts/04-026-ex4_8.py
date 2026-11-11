#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع: التعامل مع النصوص (Strings)
القسم 4.8: أهم Methods النصوص (الأكثر استخدامًا عمليًا)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 04-026-ex4_8.py
"""

text: str = "أنا أحب Java"
new_text: str = text.replace("Java", "Python")
print(new_text)  # أنا أحب Python
