#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.8: دعم async بشكل كامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-009-slow_operation.py
"""

import asyncio

@app.get("/slow-operation")
async def slow_operation() -> dict[str, str]:
    await asyncio.sleep(2)  # محاكاة عملية بطيئة (استدعاء API خارجي، قاعدة بيانات، إلخ)
    return {"message": "اكتملت العملية البطيئة"}
