#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.6: تمرير المعاملات: القيمة مقابل المرجع

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-031-process_safely.py
"""

def process_safely(lst: list) -> list:
    lst_copy = lst.copy()  # اعمل على نسخة
    lst_copy.append(999)
    return lst_copy
