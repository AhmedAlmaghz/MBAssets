#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.12: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-024-fetch_page_async.py
"""

import asyncio

import aiohttp

SEMAPHORE_LIMIT = 3  # عدد الطلبات المتزامنة القصوى -- احترامًا للخادم

async def fetch_page_async(session: aiohttp.ClientSession, url: str, sem: asyncio.Semaphore) -> str | None:
    """جلب صفحة بشكل غير متزامن مع تحديد التزامن عبر Semaphore."""
    async with sem:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 404:
                    return None
                response.raise_for_status()
                await asyncio.sleep(1)  # تأخير مهذّب حتى في الوضع المتزامن
                return await response.text()
        except aiohttp.ClientError as e:
            print(f"خطأ: {e}")
            return None

async def scrape_pages_async(urls: list[str]) -> list[str | None]:
    """جلب عدة صفحات بالتوازي مع تحديد عدد الطلبات المتزامنة."""
    sem = asyncio.Semaphore(SEMAPHORE_LIMIT)
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_page_async(session, url, sem) for url in urls]
        return await asyncio.gather(*tasks)

# urls = [BASE_URL.format(i) for i in range(1, 6)]
# results = asyncio.run(scrape_pages_async(urls))
