#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.5: Methods بأنواعها الثلاثة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-017-is_prime.py
"""

class MathHelper:
    @staticmethod
    def is_prime(n: int) -> bool:
        """دالة مساعدة لا تحتاج الوصول لأي بيانات خاصة بالفئة أو الكائن."""
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True

# تُستدعى عبر الفئة مباشرة، بدون إنشاء كائن!
print(MathHelper.is_prime(17))  # True
print(MathHelper.is_prime(18))  # False
