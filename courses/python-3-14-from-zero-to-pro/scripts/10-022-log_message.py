#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل العاشر: الدوال - تنظيم الكود
القسم 10.4: القيمة المُرجعة (Return Values) بعمق

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 10-022-log_message.py
"""

def log_message(message: str) -> None:
    """تسجل رسالة. لا تُرجع شيئاً (None صراحة)."""
    print(f"[LOG]: {message}")
    return None  # اختياري لكنه أوضح من حيث القصد

def validate_age(age: int) -> bool | None:
    """تتحقق من صحة العمر؛ تُرجع None إذا كانت القيمة غير منطقية."""
    if age < 0 or age > 150:
        return None
    return age >= 18
