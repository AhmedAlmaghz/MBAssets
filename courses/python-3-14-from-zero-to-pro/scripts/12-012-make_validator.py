#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.3: الإغلاقات (Closures): الدوال التي تتذكر

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-012-make_validator.py
"""

def make_validator(min_length: int):
    """تُنشئ دالة تحقق مخصصة بحد أدنى معين."""
    def validate(text: str) -> bool:
        return len(text) >= min_length
    return validate

validate_username = make_validator(5)
validate_password = make_validator(8)

print(validate_username("ali"))      # False (3 أحرف فقط)
print(validate_username("ahmed99"))  # True
print(validate_password("short"))    # False
print(validate_password("secure123"))  # True
