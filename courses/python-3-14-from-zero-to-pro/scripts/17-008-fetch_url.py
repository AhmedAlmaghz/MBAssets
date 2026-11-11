#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.5: مثال عملي حقيقي: تحميل بيانات من الإنترنت بـ `aiohttp`

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-008-fetch_url.py
"""

import asyncio
import time
import aiohttp

async def fetch_url(session: aiohttp.ClientSession, url: str) -> int:
    """يُحمّل صفحة ويب ويُرجع رمز الحالة (Status Code) فقط، للتبسيط."""
    async with session.get(url) as response:
        return response.status

async def main() -> None:
    urls = [
        "https://www.python.org",
        "https://www.github.com",
        "https://www.wikipedia.org",
    ]

    start = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - start

    for url, status in zip(urls, results):
        print(f"{url} → {status}")

    print(f"\nالوقت الإجمالي: {elapsed:.2f} ثانية")

asyncio.run(main())
