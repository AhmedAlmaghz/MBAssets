#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.4: مكتبة `requests`: طلب الصفحات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-007-login_response.py
"""

import requests

# Session يحافظ تلقائيًا على الكوكيز بين الطلبات (مفيد بعد تسجيل الدخول مثلاً)
with requests.Session() as session:
    session.headers.update({"User-Agent": "Educational Bot"})

    login_response = session.post(
        "https://example.com/login",
        data={"user": "ahmad", "pass": "1234"},
        timeout=10,
    )

    # هذا الطلب سيستخدم نفس الكوكيز تلقائيًا
    profile_response = session.get("https://example.com/profile", timeout=10)
