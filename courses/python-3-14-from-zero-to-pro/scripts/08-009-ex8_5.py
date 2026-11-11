#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن: هياكل البيانات - القوائم (Lists)
القسم 8.5: أهم Methods القوائم

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 08-009-ex8_5.py
"""

fruits: list[str] = ["تفاح", "موز"]
fruits.remove("مانجو")  # ❌ ValueError: list.remove(x): x not in list
