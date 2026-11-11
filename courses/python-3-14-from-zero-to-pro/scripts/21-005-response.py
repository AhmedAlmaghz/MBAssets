#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.4: مكتبة `requests`: طلب الصفحات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-005-response.py
"""

import requests

# GET مع query parameters (مثل الفصل 19)
response = requests.get(
    "https://api.example.com/search",
    params={"q": "python", "page": 1},
    timeout=10,
)
# الرابط الفعلي سيكون: https://api.example.com/search?q=python&page=1

# POST لإرسال بيانات (مثلاً تسجيل الدخول أو إرسال نموذج)
response = requests.post(
    "https://api.example.com/login",
    json={"username": "ahmad", "password": "secret"},
    timeout=10,
)
