#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس: اتخاذ القرار (Control Flow)
القسم 6.15: مشروع تطبيقي: «نظام تقييم كلمة المرور الذكي»

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 06-034-strength_level.py
"""

# password_checker.py
"""نظام تقييم قوة كلمة المرور."""

SPECIAL_CHARS: str = "!@#$%^&*()_+-="

password: str = input("أدخل كلمة مرور لتقييمها: ")

# فحص كل شرط على حدة
has_min_length: bool = len(password) >= 8
has_digit: bool = any(char.isdigit() for char in password)
has_upper: bool = any(char.isupper() for char in password)
has_special: bool = any(char in SPECIAL_CHARS for char in password)

# حساب عدد الشروط المُستوفاة
strength_score: int = sum([has_min_length, has_digit, has_upper, has_special])

# تحديد مستوى القوة
if strength_score <= 1:
    strength_level: str = "ضعيفة ❌"
elif strength_score <= 3:
    strength_level = "متوسطة ⚠️"
else:
    strength_level = "قوية ✅"

# طباعة التقرير
print("\n" + "=" * 40)
print("🔐  تقرير تقييم كلمة المرور")
print("=" * 40)
print(f"الطول (8+ أحرف): {'✅' if has_min_length else '❌'}")
print(f"يحتوي رقمًا: {'✅' if has_digit else '❌'}")
print(f"يحتوي حرفًا كبيرًا: {'✅' if has_upper else '❌'}")
print(f"يحتوي رمزًا خاصًا: {'✅' if has_special else '❌'}")
print("-" * 40)
print(f"النتيجة: {strength_score}/4")
print(f"مستوى القوة: {strength_level}")
print("=" * 40)
