#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع: التعامل مع النصوص (Strings)
القسم 4.16: 📌 ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 04-038-ex4_16.py
"""

# الإنشاء
text: str = "مرحبا"
multiline: str = """نص متعدد الأسطر"""
raw: str = r"C:\path"

# f-strings
f"{variable}"
f"{value:.2f}"      # خانتان عشريتان
f"{value:,}"        # فاصلة آلاف
f"{variable=}"      # تشخيص سريع

# الفهرسة والتقطيع
text[0]          # الحرف الأول
text[-1]         # الحرف الأخير
text[2:5]        # من الفهرس 2 حتى 4
text[::-1]       # عكس النص

# أهم Methods
text.upper() / text.lower()
text.strip()
text.split(",") / "-".join(list)
text.replace(old, new)
text.find(sub)
"sub" in text
len(text)
