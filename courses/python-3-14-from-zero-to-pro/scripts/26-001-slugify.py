#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس والعشرون: من الكود إلى المنتج (Packaging, Deployment، وخارطة الطريق)
القسم 26.4: كتابة الكود القابل للنشر

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 26-001-slugify.py
"""

"""أدوات معالجة وتنسيق النصوص."""

import re

def slugify(text: str) -> str:
    """
    يحوّل نصًا إلى صيغة "slug" مناسبة للروابط.

    >>> slugify("Hello World!")
    'hello-world'
    >>> slugify("  Python 3.14  Guide  ")
    'python-3-14-guide'
    """
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text.strip("-")

def truncate(text: str, max_length: int, suffix: str = "...") -> str:
    """
    يختصر نصًا طويلاً ويضيف لاحقة إن تجاوز الطول المحدد.

    >>> truncate("Python is amazing", 10)
    'Python is...'
    """
    if len(text) <= max_length:
        return text
    return text[:max_length].rstrip() + suffix

def word_count(text: str) -> int:
    """يُحصي عدد الكلمات في نص."""
    return len(text.split())

def is_palindrome(text: str) -> bool:
    """
    يتحقق إن كان النص متناظرًا (يُقرأ نفسه عكسيًا).

    >>> is_palindrome("Was it a car or a cat I saw")
    True
    """
    cleaned = re.sub(r"[^\w]", "", text.lower())
    return cleaned == cleaned[::-1]
