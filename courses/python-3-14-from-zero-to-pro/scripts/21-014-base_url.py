#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.7: التعامل مع التصفّح متعدد الصفحات (Pagination)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-014-base_url.py
"""

import time

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/catalogue/page-{}.html"
headers = {"User-Agent": "Educational Scraper"}

all_titles = []
page = 1

while True:
    url = BASE_URL.format(page)
    response = requests.get(url, headers=headers, timeout=10)

    if response.status_code == 404:
        print(f"انتهت الصفحات عند الصفحة {page - 1}")
        break

    soup = BeautifulSoup(response.text, "lxml")
    books = soup.select("article.product_pod")

    for book in books:
        title = book.select_one("h3 a").get("title")
        all_titles.append(title)

    print(f"✅ تمت معالجة الصفحة {page} ({len(books)} كتاب)")
    page += 1
    time.sleep(1)  # ⚠️ احترام الخادم -- تأخير بين الطلبات إلزامي

print(f"\nإجمالي الكتب المُستخرجة: {len(all_titles)}")
