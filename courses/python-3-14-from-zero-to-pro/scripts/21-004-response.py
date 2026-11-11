#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.4: مكتبة `requests`: طلب الصفحات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-004-response.py
"""

import requests

try:
    response = requests.get(
        "https://books.toscrape.com/",
        timeout=10,  # ثوانٍ -- إذا لم يرد الخادم خلالها، يُرفع استثناء
    )
except requests.Timeout:
    print("⏱️ انتهت المهلة الزمنية -- الخادم بطيء جدًا أو لا يستجيب")
except requests.ConnectionError:
    print("🔌 تعذّر الاتصال -- تحقق من الإنترنت أو الرابط")
