#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.6: نمط العمل الكامل: من الطلب إلى البيانات المنظّمة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-013-url.py
"""

import requests
from bs4 import BeautifulSoup

url = "https://books.toscrape.com/"
headers = {"User-Agent": "Educational Scraper (learning purposes)"}

response = requests.get(url, headers=headers, timeout=10)
response.raise_for_status()

soup = BeautifulSoup(response.text, "lxml")

books = soup.select("article.product_pod")
print(f"تم العثور على {len(books)} كتاب في الصفحة")

for book in books[:3]:  # أول 3 كتب فقط للعرض
    title = book.select_one("h3 a").get("title")
    price = book.select_one("p.price_color").get_text(strip=True)
    availability = book.select_one("p.instock.availability").get_text(strip=True)
    rating_classes = book.select_one("p.star-rating")["class"]
    rating = [c for c in rating_classes if c != "star-rating"][0]  # مثل "Three"

    print(f"📖 {title}")
    print(f"   السعر: {price}")
    print(f"   التوفر: {availability}")
    print(f"   التقييم: {rating}")
    print("-" * 40)
