#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.11: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-029-calculate_square_root.py
"""

import math

# تمرين 3
class NegativeValueError(Exception):
    pass

def calculate_square_root(n: float) -> float:
    if n < 0:
        raise NegativeValueError(f"لا يمكن حساب جذر رقم سالب: {n}")
    return math.sqrt(n)

try:
    print(calculate_square_root(-9))
except NegativeValueError as e:
    print(f"❌ {e}")

# تمرين 4
class MissingConfigError(Exception):
    def __init__(self, key: str):
        self.key = key
        super().__init__(f"الإعداد المطلوب '{key}' غير موجود.")

def parse_config(data: dict) -> tuple[str, int]:
    try:
        host = data["host"]
    except KeyError:
        raise MissingConfigError("host")

    try:
        port = data["port"]
    except KeyError:
        raise MissingConfigError("port")

    return host, port

try:
    parse_config({"host": "localhost"})
except MissingConfigError as e:
    print(f"❌ {e}")  # ❌ الإعداد المطلوب 'port' غير موجود.
