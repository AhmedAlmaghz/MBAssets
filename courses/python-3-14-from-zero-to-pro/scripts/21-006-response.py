#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.4: مكتبة `requests`: طلب الصفحات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-006-response.py
"""

import requests

response = requests.get("https://api.github.com/users/python", timeout=10)
data = response.json()  # تحويل تلقائي من JSON إلى dict/list بايثون

print(data["name"])
print(data["public_repos"])
