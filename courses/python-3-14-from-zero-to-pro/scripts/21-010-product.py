#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.5: مكتبة BeautifulSoup: تحليل واستخراج البيانات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-010-product.py
"""

product = soup.find("article", class_="product_pod")

title_tag = product.find("h3").find("a")
title = title_tag["title"]           # الوصول لخاصية HTML (title attribute)
# أو
title = title_tag.get("title")       # أكثر أمانًا -- يُرجع None إن لم توجد بدلاً من خطأ

price_tag = product.find("p", class_="price_color")
price_text = price_tag.text          # النص الظاهر للمستخدم: "£51.77"
price_text_stripped = price_tag.get_text(strip=True)  # يزيل المسافات الزائدة تلقائيًا
