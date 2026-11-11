#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.3: الصيغة الأساسية: try / except

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-009-value.py
"""

# ترتيب except مهم جداً! الأكثر تحديداً أولاً
try:
    value = my_list[10]
except IndexError:
    print("فهرس خارج النطاق")
except LookupError:  # تأتي بعد IndexError لأنها أعم منها
    print("خطأ بحث عام")
