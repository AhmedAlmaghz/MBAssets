#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.10: مشروع تطبيقي: نظام تسجيل دخول آمن

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-027-reset_password.py
"""

def reset_password(email: str, new_password: str) -> None:
    """إعادة تعيين كلمة مرور، مع إلغاء القفل إن وُجد."""
    try:
        user = users_db[email]
    except KeyError:
        raise UserNotFoundError(f"لا يوجد مستخدم بالبريد: {email}")

    if len(new_password) < 8 or not any(c.isdigit() for c in new_password):
        raise WeakPasswordError("كلمة المرور الجديدة ضعيفة جداً.")

    user["password"] = new_password
    user["failed_attempts"] = 0
    user["locked"] = False
    print(f"🔓 تم إعادة تعيين كلمة المرور وفك قفل الحساب لـ {email}")

try:
    reset_password("ahmed@test.com", "NewPass123")
except AuthError as e:
    print(f"❌ {e}")
