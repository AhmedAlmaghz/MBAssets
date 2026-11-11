#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.5: مكتبة BeautifulSoup: تحليل واستخراج البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-012-product.py
"""

product = soup.find("article", class_="product_pod")

print(product.parent)         # العنصر الأب
print(product.find_next_sibling())  # العنصر الشقيق التالي
print(product.contents)       # كل العناصر الفرعية المباشرة كقائمة
