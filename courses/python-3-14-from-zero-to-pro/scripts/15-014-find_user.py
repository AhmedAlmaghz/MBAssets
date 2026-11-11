#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.4: القيم الاختيارية: Optional و None

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-014-find_user.py
"""

def find_user(user_id: int, users: dict[int, str]) -> str | None:
    return users.get(user_id)

result: str | None = find_user(5, {1: "أحمد", 2: "سارة"})

if result is not None:
    print(result.upper())  # آمن تمامًا، لأننا تحققنا أولًا
else:
    print("المستخدم غير موجود")
