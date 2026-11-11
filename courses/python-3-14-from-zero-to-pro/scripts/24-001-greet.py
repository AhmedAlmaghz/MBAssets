#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.2: PEP 8: دليل الأسلوب الرسمي لبايثون

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-001-greet.py
"""

# ✅ صحيح: مسافة واحدة حول عوامل التشغيل
total = price + tax

# ❌ خطأ: بدون مسافات، أو مسافات زائدة
total=price+tax
total  =  price  +  tax

# ✅ صحيح: بدون مسافة قبل الفاصلة، مسافة بعدها
def greet(name, age):
    pass

# ✅ صحيح: بدون مسافات حول = في المعاملات الافتراضية (دون type hint)
def greet(name="أحمد"):
    pass

# ✅ لكن مع type hint، نضيف مسافات حول =
def greet(name: str = "أحمد"):
    pass
