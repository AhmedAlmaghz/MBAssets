#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.9: اختبار الكود غير المتزامن (async)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-014-fetch_data.py
"""

# src/async_utils.py
import asyncio

async def fetch_data() -> str:
    await asyncio.sleep(0.1)
    return "بيانات"
