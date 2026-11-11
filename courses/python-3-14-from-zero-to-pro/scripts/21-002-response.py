#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.4: مكتبة `requests`: طلب الصفحات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-002-response.py
"""

import requests

response = requests.get("https://books.toscrape.com/")

if response.status_code == 200:
    print("✅ نجح الطلب!")
elif response.status_code == 404:
    print("❌ الصفحة غير موجودة")
else:
    print(f"⚠️ حدث خطأ: {response.status_code}")

# طريقة أنظف وأكثر احترافية
try:
    response.raise_for_status()  # يرفع استثناء تلقائيًا إذا كان status_code خطأ
    print("✅ نجح الطلب!")
except requests.HTTPError as e:
    print(f"❌ خطأ HTTP: {e}")
