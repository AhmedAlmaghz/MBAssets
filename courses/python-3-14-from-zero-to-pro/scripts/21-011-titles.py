#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.5: مكتبة BeautifulSoup: تحليل واستخراج البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-011-titles.py
"""

# محدد CSS: وسم.class
titles = soup.select("h3 a")               # كل وسم <a> داخل <h3>
prices = soup.select("p.price_color")       # كل <p> بخاصية class="price_color"
first_price = soup.select_one("p.price_color")  # أول عنصر مطابق فقط

# محددات أكثر تعقيدًا
all_product_links = soup.select("article.product_pod h3 a")

for link in all_product_links:
    print(link.get("title"), "->", link.get("href"))
