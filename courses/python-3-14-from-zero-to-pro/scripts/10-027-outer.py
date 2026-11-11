#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.5: النطاق (Scope): أين «تعيش» المتغيرات؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-027-outer.py
"""

x = "global x"

def outer():
    x = "enclosing x"

    def inner():
        x = "local x"
        print(x)  # local x - توجد في المستوى المحلي، تتوقف هنا

    inner()
    print(x)  # enclosing x

outer()
print(x)  # global x
