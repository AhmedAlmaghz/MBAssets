#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.3: المفاهيم الأساسية: `async def`, `await`, Coroutines

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-004-say_hello.py
"""

import asyncio

async def say_hello() -> None:
    print("مرحبًا!")

asyncio.run(say_hello())  # الآن تُطبع "مرحبًا!" فعليًا
