#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.8: القيم الثابتة المحدودة: Literal

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-025-set_status.py
"""

from typing import Literal

def set_status(status: Literal["active", "inactive", "pending"]) -> None:
    print(f"الحالة الجديدة: {status}")

set_status("active")    # صحيح
set_status("pending")   # صحيح
set_status("deleted")   # ❌ mypy يرفض هذا! "deleted" ليست من القيم المسموحة
