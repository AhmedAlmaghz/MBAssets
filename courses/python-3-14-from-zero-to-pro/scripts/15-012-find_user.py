#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.4: القيم الاختيارية: Optional و None

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-012-find_user.py
"""

def find_user(user_id: int, users: dict[int, str]) -> str:
    return users.get(user_id)  # قد تُرجع None إن لم يوجد المستخدم!
