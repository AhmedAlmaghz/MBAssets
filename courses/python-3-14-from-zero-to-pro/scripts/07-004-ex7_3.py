#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع: التكرار (Loops)
القسم 7.3: ⚠️ الفخ الأخطر: الحلقات اللا نهائية (Infinite Loops)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 07-004-ex7_3.py
"""

while True:  # لا نهائية عمدًا
    user_input: str = input("اكتب 'خروج' لإنهاء البرنامج: ")
    if user_input == "خروج":
        break  # نخرج من الحلقة هنا
    print(f"كتبت: {user_input}")
