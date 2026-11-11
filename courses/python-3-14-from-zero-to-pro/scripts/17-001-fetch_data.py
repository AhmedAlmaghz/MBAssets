#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.2: فهم المشكلة أولًا: لماذا الكود العادي «بطيء» أحيانًا؟

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-001-fetch_data.py
"""

import time

def fetch_data(source: str, delay: float) -> str:
    """يُحاكي طلب بيانات بطيء من مصدر خارجي (مثل API أو قاعدة بيانات)."""
    print(f"🔄 بدء تحميل البيانات من {source}...")
    time.sleep(delay)  # محاكاة وقت الانتظار (شبكة، قرص صلب، إلخ)
    print(f"✅ انتهى تحميل البيانات من {source}")
    return f"بيانات من {source}"

def main() -> None:
    start = time.perf_counter()

    result1 = fetch_data("الخادم أ", 2)
    result2 = fetch_data("الخادم ب", 2)
    result3 = fetch_data("الخادم ج", 2)

    elapsed = time.perf_counter() - start
    print(f"\nالوقت الإجمالي: {elapsed:.2f} ثانية")

main()
