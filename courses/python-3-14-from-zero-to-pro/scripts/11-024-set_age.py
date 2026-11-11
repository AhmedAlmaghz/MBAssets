#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.8: دالة assert: التحقق أثناء التطوير

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-024-set_age.py
"""

# ❌ خطأ: استخدام assert للتحقق من مدخلات مستخدم
def set_age(age: int) -> None:
    assert age >= 0, "العمر يجب أن يكون موجباً"  # قد يُحذف بالكامل!

# ✅ صحيح: استخدام raise للتحقق من مدخلات مستخدم
def set_age(age: int) -> None:
    if age < 0:
        raise ValueError("العمر يجب أن يكون موجباً")
