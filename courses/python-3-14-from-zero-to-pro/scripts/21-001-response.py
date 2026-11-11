#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.4: مكتبة `requests`: طلب الصفحات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-001-response.py
"""

import requests

response = requests.get("https://books.toscrape.com/")

print(response.status_code)   # 200 يعني نجاح (كما تعلمنا في الفصل 19)
print(response.text[:500])    # أول 500 حرف من محتوى HTML الخام
