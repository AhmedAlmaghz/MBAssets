#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.3: المفاهيم الأساسية: `async def`, `await`, Coroutines

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-003-result.py
"""

result = say_hello()
print(result)  # <coroutine object say_hello at 0x...>
# لاحظ: "مرحبًا!" لم تُطبع إطلاقًا!
