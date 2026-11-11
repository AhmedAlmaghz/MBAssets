#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.10: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-015-download_image.py
"""

# تمرين 4
import asyncio
import time

async def download_image() -> str:
    await asyncio.sleep(3)
    return "تم تحميل الصورة"

async def download_video() -> str:
    await asyncio.sleep(5)
    return "تم تحميل الفيديو"

async def download_document() -> str:
    await asyncio.sleep(1)
    return "تم تحميل المستند"

async def main() -> None:
    start = time.perf_counter()
    results = await asyncio.gather(
        download_image(), download_video(), download_document()
    )
    elapsed = time.perf_counter() - start
    print(results)
    print(f"الوقت: {elapsed:.2f} ثانية")  # قريب من 5 ثوانٍ

asyncio.run(main())

# تمرين 5
import asyncio

async def download_image() -> str:
    await asyncio.sleep(3)
    return "تم تحميل الصورة"

async def download_video() -> str:
    await asyncio.sleep(2)
    raise ConnectionError("فشل الاتصال بخادم الفيديو")

async def download_document() -> str:
    await asyncio.sleep(1)
    return "تم تحميل المستند"

async def main() -> None:
    results = await asyncio.gather(
        download_image(), download_video(), download_document(),
        return_exceptions=True,
    )
    for result in results:
        if isinstance(result, Exception):
            print(f"❌ فشل: {result}")
        else:
            print(f"✅ نجح: {result}")

asyncio.run(main())

# تمرين 6
import asyncio

async def fetch_with_timeout(delay: float, timeout: float) -> str:
    try:
        async def task() -> str:
            await asyncio.sleep(delay)
            return "تم التحميل بنجاح"
        return await asyncio.wait_for(task(), timeout=timeout)
    except asyncio.TimeoutError:
        return "انتهت المهلة"

async def main() -> None:
    print(await fetch_with_timeout(1, 3))   # "تم التحميل بنجاح"
    print(await fetch_with_timeout(5, 2))   # "انتهت المهلة"

asyncio.run(main())
