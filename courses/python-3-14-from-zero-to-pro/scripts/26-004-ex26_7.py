#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس والعشرون: من الكود إلى المنتج (Packaging, Deployment، وخارطة الطريق)
القسم 26.7: ملفات لا غنى عنها في أي حزمة احترافية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 26-004-ex26_7.py
"""

from textkit import slugify, truncate

slugify("Hello World!")       # "hello-world"
truncate("نص طويل جداً", 5)   # "نص طو..."
