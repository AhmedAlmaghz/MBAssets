#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن عشر: إدارة المشاريع كمحترف
القسم 18.6: مشروع تطبيقي: تجهيز مشروع كامل من الصفر

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 18-004-__version__.py
"""

# src/currency_converter/__init__.py
from currency_converter.converter import convert

__version__ = "0.1.0"
__all__ = ["convert"]
