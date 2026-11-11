#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.3: الإغلاقات (Closures): الدوال التي تتذكر

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-010-ex12_3.py
"""

# تأكيد عملي أن كل closure يحتفظ بنسخته الخاصة
print(double(10))  # 20 - لا تزال تتذكر factor=2
print(triple(10))  # 30 - لا تزال تتذكر factor=3
