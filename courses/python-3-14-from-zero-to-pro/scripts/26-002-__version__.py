#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس والعشرون: من الكود إلى المنتج (Packaging, Deployment، وخارطة الطريق)
القسم 26.4: كتابة الكود القابل للنشر

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 26-002-__version__.py
"""

"""textkit: مجموعة أدوات بسيطة لمعالجة النصوص."""

from textkit.formatters import is_palindrome, slugify, truncate, word_count

__version__ = "0.1.0"

__all__ = [
    "slugify",
    "truncate",
    "word_count",
    "is_palindrome",
]
