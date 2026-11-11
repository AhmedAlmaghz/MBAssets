#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.3: المعاملات (Parameters) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-008-greet.py
"""

def greet(name: str) -> str:  # name = parameter
    return f"Hello, {name}"

greet("Ahmed")  # "Ahmed" = argument
