#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.7: معالجة الأخطاء في الكود غير المتزامن

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-011-task_that_fails.py
"""

import asyncio

async def task_that_fails() -> str:
    await asyncio.sleep(1)
    raise ValueError("فشلت هذه المهمة!")

async def task_that_succeeds() -> str:
    await asyncio.sleep(1)
    return "نجحت هذه المهمة"

async def main() -> None:
    results = await asyncio.gather(
        task_that_succeeds(),
        task_that_fails(),
        return_exceptions=True,  # ← مهم جدًا!
    )

    for result in results:
        if isinstance(result, Exception):
            print(f"خطأ: {result}")
        else:
            print(f"نجاح: {result}")

asyncio.run(main())
