#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.12: أداة mypy: التحقق الساكن من الأنواع

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-035-find_user.py
"""

def find_user(user_id: int, users: dict[int, str]) -> str | None:
    return users.get(user_id)

user = find_user(5, {1: "أحمد"})
print(user.upper())  # ❌ خطأ محتمل!
