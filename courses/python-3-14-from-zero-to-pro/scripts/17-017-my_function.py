#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.12: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-017-my_function.py
"""

import asyncio

# ===== تعريف وتشغيل Coroutine =====
async def my_function() -> str:
    await asyncio.sleep(1)   # انتظار غير متزامن (وليس time.sleep!)
    return "النتيجة"

asyncio.run(my_function())   # نقطة الدخول الرئيسية - تُستدعى مرة واحدة فقط

# ===== انتظار متتالٍ (عند الاعتماد الحقيقي بين العمليات) =====
async def main() -> None:
    result1 = await my_function()
    result2 = await my_function()  # يبدأ فقط بعد انتهاء الأولى

# ===== تزامن حقيقي (عمليات مستقلة) =====
async def main() -> None:
    results = await asyncio.gather(
        my_function(),
        my_function(),
        return_exceptions=True,  # لمنع توقف الكل عند فشل واحدة
    )

# ===== بدء مهمة في الخلفية =====
async def main() -> None:
    task = asyncio.create_task(my_function())
    # ... كود آخر يعمل الآن ...
    result = await task  # ننتظر النتيجة عند الحاجة فقط

# ===== فرض مهلة قصوى =====
try:
    result = await asyncio.wait_for(my_function(), timeout=2.0)
except asyncio.TimeoutError:
    print("انتهت المهلة")

# ===== معالجة الأخطاء =====
try:
    result = await risky_coroutine()
except SomeError as e:
    print(f"خطأ: {e}")
