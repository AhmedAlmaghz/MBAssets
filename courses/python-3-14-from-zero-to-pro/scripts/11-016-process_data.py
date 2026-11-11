#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.5: إطلاق الأخطاء بنفسك: raise

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-016-process_data.py
"""

def process_data(data: list) -> float:
    try:
        return sum(data) / len(data)
    except ZeroDivisionError:
        print("⚠️ تحذير: القائمة فارغة، سيتم إعادة إطلاق الخطأ.")
        raise  # raise بدون معامل = إعادة إطلاق نفس الخطأ الملتقط

try:
    process_data([])
except ZeroDivisionError:
    print("❌ المستوى الأعلى: تم التعامل مع القائمة الفارغة.")
