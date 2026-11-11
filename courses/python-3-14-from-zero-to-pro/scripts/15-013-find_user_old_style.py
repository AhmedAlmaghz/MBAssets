#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.4: القيم الاختيارية: Optional و None

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-013-find_user_old_style.py
"""

from typing import Optional

def find_user_old_style(user_id: int, users: dict[int, str]) -> Optional[str]:
    return users.get(user_id)
