#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.5: مكتبة BeautifulSoup: تحليل واستخراج البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-009-first_product.py
"""

# العثور على أول عنصر مطابق
first_product = soup.find("article", class_="product_pod")

# العثور على كل العناصر المطابقة -> تُرجع قائمة
all_products = soup.find_all("article", class_="product_pod")
print(f"عدد المنتجات: {len(all_products)}")

# البحث بالوسم فقط
all_headings = soup.find_all("h3")

# البحث بخاصية id
header = soup.find(id="messages")

# البحث بعدة شروط
links = soup.find_all("a", href=True)  # كل روابط <a> التي تملك خاصية href
