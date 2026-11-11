#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.6: إنشاء استثناءات مخصصة (Custom Exceptions)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-020-__init__.py
"""

class ValidationError(Exception):
    """خطأ تحقق من صحة البيانات، مع تفاصيل الحقل المُخالف."""

    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"خطأ في حقل '{field}': {message}")

def validate_age(age: int) -> None:
    if age < 0 or age > 150:
        raise ValidationError(field="age", message="يجب أن يكون بين 0 و150")

try:
    validate_age(-5)
except ValidationError as e:
    print(f"الحقل المُخالف: {e.field}")
    print(f"الرسالة: {e.message}")
