#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.10: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-014-countdown.py
"""

# تمرين 1
import asyncio

async def countdown(n: int) -> None:
    for i in range(n, 0, -1):
        print(i)
        await asyncio.sleep(1)
    print("انتهى!")

asyncio.run(countdown(5))

# تمرين 2
# await func(): ينفّذ الـ Coroutine وينتظر نتيجتها فورًا قبل المتابعة.
# asyncio.create_task(func()): يبدأ تنفيذ الـ Coroutine "في الخلفية" فورًا،
# ويسمح لك بمتابعة كود آخر، على أن تنتظر نتيجتها لاحقًا بـ await عند الحاجة.

# تمرين 3 - الكود المصحح
import asyncio

async def get_data() -> str:
    return "بيانات"

async def main() -> None:
    result = await get_data()  # يجب await لتنفيذ الـ Coroutine فعليًا
    print(result)

asyncio.run(main())  # يجب استدعاء main عبر asyncio.run، وليس مباشرة
