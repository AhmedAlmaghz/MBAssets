#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.7: معالجة الأخطاء في الكود غير المتزامن

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-010-risky_operation.py
"""

import asyncio

async def risky_operation(should_fail: bool) -> str:
    await asyncio.sleep(1)
    if should_fail:
        raise ValueError("حدث خطأ أثناء العملية!")
    return "نجحت العملية"

async def main() -> None:
    try:
        result = await risky_operation(should_fail=True)
        print(result)
    except ValueError as e:
        print(f"تم اصطياد الخطأ: {e}")

asyncio.run(main())
