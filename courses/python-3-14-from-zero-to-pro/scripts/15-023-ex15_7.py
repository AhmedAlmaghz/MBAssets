#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.7: الدوال كقيم: Callable

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-023-ex15_7.py
"""

# دالة بدون معاملات تُرجع نصًا
greeting_provider: Callable[[], str]

# دالة تستقبل int و str وتُرجع bool
validator: Callable[[int, str], bool]

# دالة بأي عدد ونوع من المعاملات (تجنّب هذا إن أمكن، لكنه موجود)
from typing import Callable
any_callable: Callable[..., int]  # ... تعني "أي معاملات"
