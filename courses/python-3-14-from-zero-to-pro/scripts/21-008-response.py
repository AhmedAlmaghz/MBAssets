#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.5: مكتبة BeautifulSoup: تحليل واستخراج البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-008-response.py
"""

import requests
from bs4 import BeautifulSoup

response = requests.get("https://books.toscrape.com/", timeout=10)
soup = BeautifulSoup(response.text, "lxml")  # "lxml" هو المحلل (parser) المستخدم

print(soup.title)        # <title>All products | Books to Scrape...</title>
print(soup.title.text)   # All products | Books to Scrape - Sandbox
