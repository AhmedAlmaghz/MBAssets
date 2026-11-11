#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.12: أداة mypy: التحقق الساكن من الأنواع

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-040-result.py
"""

import some_untyped_library  # type: ignore[import-untyped]

result = some_untyped_library.do_something()
