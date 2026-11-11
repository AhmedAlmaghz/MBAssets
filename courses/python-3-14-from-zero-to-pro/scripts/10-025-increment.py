#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.5: النطاق (Scope): أين «تعيش» المتغيرات؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-025-increment.py
"""

counter = 0

def increment():
    global counter  # أخبر Python: استخدم المتغير العام، لا تنشئ واحداً محلياً
    counter += 1

increment()
increment()
print(counter)  # 2
