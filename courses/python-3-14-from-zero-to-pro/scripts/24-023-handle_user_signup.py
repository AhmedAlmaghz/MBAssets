#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.9: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-023-handle_user_signup.py
"""

def handle_user_signup(email, password):
    if "@" not in email:
        raise ValueError("بريد غير صالح")
    if len(password) < 8:
        raise ValueError("كلمة مرور قصيرة")
    hashed = password[::-1]  # تشفير وهمي للتبسيط
    print(f"حفظ المستخدم: {email}")
    print(f"إرسال بريد ترحيبي إلى: {email}")
    return {"email": email, "password": hashed}
