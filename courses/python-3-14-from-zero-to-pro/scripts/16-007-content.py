#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.2: الملفات النصية: الأساسيات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-007-content.py
"""

# ❌ خطير: الاعتماد على الترميز الافتراضي لنظام التشغيل
with open("notes.txt", "r") as file:
    content = file.read()

# ✅ آمن: تحديد الترميز صراحة
with open("notes.txt", "r", encoding="utf-8") as file:
    content = file.read()
