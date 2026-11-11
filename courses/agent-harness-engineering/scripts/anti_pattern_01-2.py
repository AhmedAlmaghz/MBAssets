# -*- coding: utf-8 -*-
"""
❌ ANTI-PATTERN: خداع النفس بـ async def مع كود متزامن يجمد النظام بالكامل!

مقتطف من الفصل 2 — «تسخير قوة Python 3.14+ لهندسة الوكلاء».
النوع: anti_pattern. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch02_python314_environment.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

# ❌ ANTI-PATTERN: خداع النفس بـ async def مع كود متزامن يجمد النظام بالكامل!
import requests


async def bad_web_search_tool(query: str) -> str:
    # كارثة: مكتبة requests متزامنة! هذا السطر سيجمد كل الوكلاء والمستخدمين على الخادم
    # حتى يرد الموقع الخارجي بعد 5 ثوانٍ!
    response = requests.get(f"https://api.example.com/search?q={query}", timeout=5)
    return response.text
