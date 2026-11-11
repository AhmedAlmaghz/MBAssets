#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.11: مشروع تطبيقي متكامل: مستخرج بيانات كتب شامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-018-fetch_page.py
"""

"""وحدة جلب صفحات الويب مع معالجة الأخطاء وإعادة المحاولة."""

import time

import requests

USER_AGENT = "Educational Book Scraper (learning purposes, github.com/example)"
REQUEST_TIMEOUT = 10
DELAY_BETWEEN_REQUESTS = 1.0  # ثانية -- احترام الخادم

def fetch_page(url: str, retries: int = 3) -> str | None:
    """جلب محتوى HTML لصفحة، مع إعادة محاولة عند الفشل.

    Args:
        url: رابط الصفحة المطلوبة.
        retries: عدد محاولات إعادة الطلب عند حدوث خطأ مؤقت.

    Returns:
        محتوى HTML كنص، أو None إذا فشلت كل المحاولات أو الصفحة غير موجودة.
    """
    headers = {"User-Agent": USER_AGENT}

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)

            if response.status_code == 404:
                return None  # نهاية الصفحات -- ليس خطأ، بل إشارة توقف طبيعية

            response.raise_for_status()
            return response.text

        except requests.Timeout:
            print(f"⏱️ [محاولة {attempt}/{retries}] انتهت المهلة الزمنية: {url}")
        except requests.ConnectionError:
            print(f"🔌 [محاولة {attempt}/{retries}] تعذّر الاتصال: {url}")
        except requests.HTTPError as e:
            print(f"❌ [محاولة {attempt}/{retries}] خطأ HTTP: {e}")

        if attempt < retries:
            time.sleep(2 * attempt)

    print(f"🚫 فشلت كل المحاولات لجلب: {url}")
    return None

def polite_delay() -> None:
    """تأخير مهذّب بين الطلبات المتتالية لاحترام خادم الموقع."""
    time.sleep(DELAY_BETWEEN_REQUESTS)
