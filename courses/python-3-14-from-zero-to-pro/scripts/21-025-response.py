#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.14: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-025-response.py
"""

# ===== requests: الطلبات الأساسية =====
import requests
response = requests.get(url, headers={"User-Agent": "..."}, timeout=10)
response.status_code         # 200, 404, 500...
response.text                 # محتوى HTML/نص خام
response.json()               # تحويل تلقائي من JSON
response.raise_for_status()   # يرفع استثناء عند 4xx/5xx

try:
    response = requests.get(url, timeout=10)
except requests.Timeout:
    ...
except requests.ConnectionError:
    ...

# ===== BeautifulSoup: التحليل =====
from bs4 import BeautifulSoup
soup = BeautifulSoup(response.text, "lxml")

soup.find("div", class_="price")       # أول عنصر مطابق
soup.find_all("article")               # كل العناصر المطابقة -> قائمة
soup.select("article.product_pod h3 a")  # محدد CSS -> قائمة
soup.select_one("title")               # محدد CSS -> أول عنصر

tag.text / tag.get_text(strip=True)    # النص الظاهر
tag.get("href")                        # خاصية HTML بأمان (None إن لم توجد)
tag["href"]                            # خاصية HTML (يرفع خطأ إن لم توجد)

# ===== الأتمتة الآمنة =====
import time
time.sleep(1)                          # تأخير بين الطلبات (إلزامي أخلاقيًا)
