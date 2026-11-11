#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.12: أداة mypy: التحقق الساكن من الأنواع

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-036-user.py
"""

user = find_user(5, {1: "أحمد"})
if user is not None:
    print(user.upper())
else:
    print("المستخدم غير موجود")
