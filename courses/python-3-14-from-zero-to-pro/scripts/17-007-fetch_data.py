#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.4: تحقيق التزامن الحقيقي: `asyncio.gather()`

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-007-fetch_data.py
"""

import asyncio
import time

async def fetch_data(source: str, delay: float) -> str:
    print(f"🔄 بدء تحميل البيانات من {source}...")
    await asyncio.sleep(delay)
    print(f"✅ انتهى تحميل البيانات من {source}")
    return f"بيانات من {source}"

async def main() -> None:
    start = time.perf_counter()

    results = await asyncio.gather(
        fetch_data("الخادم أ", 2),
        fetch_data("الخادم ب", 2),
        fetch_data("الخادم ج", 2),
    )

    elapsed = time.perf_counter() - start
    print(f"\nالنتائج: {results}")
    print(f"الوقت الإجمالي: {elapsed:.2f} ثانية")

asyncio.run(main())
