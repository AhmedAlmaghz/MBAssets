#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.4: مكتبة `requests`: طلب الصفحات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-003-headers.py
"""

import requests

headers = {
    "User-Agent": "Mozilla/5.0 (Educational Scraper Bot - contact@example.com)"
}

response = requests.get("https://books.toscrape.com/", headers=headers)
print(response.status_code)
