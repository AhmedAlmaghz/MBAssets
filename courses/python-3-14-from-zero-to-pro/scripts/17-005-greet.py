#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.3: المفاهيم الأساسية: `async def`, `await`, Coroutines

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-005-greet.py
"""

import asyncio

async def greet(name: str) -> None:
    print(f"مرحبًا، {name}!")

async def main() -> None:
    await greet("أحمد")
    await greet("سارة")

asyncio.run(main())
