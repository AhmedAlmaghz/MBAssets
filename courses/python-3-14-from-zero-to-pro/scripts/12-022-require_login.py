#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثاني عشر: الدوال المتقدمة (Lambda, Decorators, Generators, Closures)
القسم 12.4: المزخرفات (Decorators): تعديل سلوك الدوال

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 12-022-require_login.py
"""

from functools import wraps

def require_login(func):
    @wraps(func)
    def wrapper(user: dict, *args, **kwargs):
        if not user.get("is_logged_in"):
            print("❌ يجب تسجيل الدخول أولاً.")
            return None
        return func(user, *args, **kwargs)
    return wrapper

@require_login
def view_profile(user: dict) -> None:
    print(f"👤 ملف {user['name']} الشخصي")

guest = {"name": "زائر", "is_logged_in": False}
logged_in_user = {"name": "أحمد", "is_logged_in": True}

view_profile(guest)          # ❌ يجب تسجيل الدخول أولاً.
view_profile(logged_in_user)  # 👤 ملف أحمد الشخصي
