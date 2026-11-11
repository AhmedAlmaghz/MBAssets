#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.6: `asyncio.create_task()`: بدء المهام دون انتظار فوري

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-009-background_job.py
"""

import asyncio

async def background_job(name: str, delay: float) -> str:
    print(f"🚀 بدأت المهمة: {name}")
    await asyncio.sleep(delay)
    print(f"🏁 انتهت المهمة: {name}")
    return f"نتيجة {name}"

async def main() -> None:
    # إنشاء المهمة تبدأ تنفيذها فورًا "في الخلفية"
    task1 = asyncio.create_task(background_job("أ", 3))
    task2 = asyncio.create_task(background_job("ب", 1))

    print("البرنامج يستمر بعمل أشياء أخرى بينما المهام تعمل في الخلفية...")
    await asyncio.sleep(0.5)
    print("لا يزال البرنامج يعمل...")

    # الآن ننتظر نتائج المهام عند الحاجة الفعلية إليها
    result1 = await task1
    result2 = await task2

    print(result1, result2)

asyncio.run(main())
