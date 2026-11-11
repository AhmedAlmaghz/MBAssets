#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.9: معالجة الأخطاء بصلابة (Robust Scraping)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-017-fetch_page.py
"""

import time

import requests
from bs4 import BeautifulSoup

def fetch_page(url: str, retries: int = 3) -> requests.Response | None:
    """جلب صفحة مع إعادة محاولة عند الفشل.

    Args:
        url: الرابط المطلوب جلبه.
        retries: عدد محاولات إعادة الطلب عند الفشل.

    Returns:
        كائن Response عند النجاح، أو None عند فشل كل المحاولات.
    """
    headers = {"User-Agent": "Educational Scraper"}

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response
        except requests.Timeout:
            print(f"⏱️ محاولة {attempt}: انتهت المهلة الزمنية لـ {url}")
        except requests.HTTPError as e:
            print(f"❌ محاولة {attempt}: خطأ HTTP {e}")
        except requests.ConnectionError:
            print(f"🔌 محاولة {attempt}: تعذّر الاتصال بـ {url}")

        if attempt < retries:
            time.sleep(2 * attempt)  # تأخير متزايد (Exponential backoff مبسّط)

    print(f"🚫 فشلت كل المحاولات لـ {url}")
    return None

def extract_book_data(book_element) -> dict[str, str] | None:
    """استخراج بيانات كتاب واحد بأمان، مع التعامل مع عناصر مفقودة.

    Args:
        book_element: عنصر BeautifulSoup يمثّل بطاقة كتاب واحدة.

    Returns:
        قاموس ببيانات الكتاب، أو None إذا كانت البيانات الأساسية مفقودة.
    """
    title_tag = book_element.select_one("h3 a")
    price_tag = book_element.select_one("p.price_color")

    if title_tag is None or price_tag is None:
        return None  # تخطَّ هذا العنصر بدلاً من رفع خطأ يوقف البرنامج كله

    return {
        "title": title_tag.get("title", "غير معروف"),
        "price": price_tag.get_text(strip=True),
    }
