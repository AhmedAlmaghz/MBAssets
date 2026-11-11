#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.1: لماذا يحتاج كل برنامج احترافي لمعالجة الأخطاء؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-001-balance.py
"""

balance = 1000
withdrawal = input("أدخل مبلغ السحب: ")  # المستخدم يكتب "مئة"
balance -= int(withdrawal)  # 💥 ValueError: invalid literal for int()
# البرنامج ينهار، والمستخدم يرى رسالة تقنية غير مفهومة
