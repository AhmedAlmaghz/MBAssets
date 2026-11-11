#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.5: النطاق (Scope): أين «تعيش» المتغيرات؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-023-my_function.py
"""

x = 10  # متغير عام (Global)

def my_function():
    y = 20  # متغير محلي (Local) - يعيش فقط داخل الدالة
    print(x)  # يمكن قراءة المتغير العام من داخل الدالة
    print(y)

my_function()
print(x)  # 10 - متاح
# print(y)  
# 💥 NameError: name 'y' is not defined -- y لا توجد خارج الدالة!
