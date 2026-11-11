#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.12: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-022-response.py
"""

import requests
from bs4 import BeautifulSoup

response = requests.get("https://books.toscrape.com/", timeout=10)
print(response.status_code)
print(len(response.text))

soup = BeautifulSoup(response.text, "lxml")
print(soup.title.get_text(strip=True))

links = soup.find_all("a", href=True)
for link in links[:5]:
    print(link["href"])
