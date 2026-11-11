#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع: التعامل مع النصوص (Strings)
القسم 4.8: أهم Methods النصوص (الأكثر استخدامًا عمليًا)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 04-028-ex4_8.py
"""

print("12345".isdigit())      # True - كلها أرقام؟
print("Hello".isalpha())      # True - كلها حروف؟
print("Hello123".isalnum())   # True - حروف أو أرقام فقط؟
print("   ".isspace())        # True - كلها مسافات؟
print("Hello".startswith("He"))  # True
print("Hello".endswith("lo"))    # True
