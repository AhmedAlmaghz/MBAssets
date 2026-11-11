#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس والعشرون: الاختبار (Testing) باستخدام pytest
القسم 25.9: اختبار الكود غير المتزامن (async)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 25-015-test_fetch_data.py
"""

# tests/test_async_utils.py
import pytest
from async_utils import fetch_data

@pytest.mark.asyncio
async def test_fetch_data():
    result = await fetch_data()
    assert result == "بيانات"
