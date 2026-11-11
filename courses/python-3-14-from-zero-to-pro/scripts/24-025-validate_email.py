#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.9: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-025-validate_email.py
"""

# 4
def validate_email(email: str) -> None:
    if "@" not in email:
        raise ValueError("بريد غير صالح")

def validate_password(password: str) -> None:
    if len(password) < 8:
        raise ValueError("كلمة مرور قصيرة")

def hash_password(password: str) -> str:
    return password[::-1]  # تشفير وهمي للتبسيط

def save_user(email: str, hashed_password: str) -> None:
    print(f"حفظ المستخدم: {email}")

def send_welcome_email(email: str) -> None:
    print(f"إرسال بريد ترحيبي إلى: {email}")

def handle_user_signup(email: str, password: str) -> dict:
    validate_email(email)
    validate_password(password)
    hashed = hash_password(password)
    save_user(email, hashed)
    send_welcome_email(email)
    return {"email": email, "password": hashed}

# 5
def get_shipping_cost(order) -> float:
    if order is None:
        return 0
    if order.weight <= 0:
        return 0
    if order.destination == "local":
        return order.weight * 2
    return order.weight * 5
