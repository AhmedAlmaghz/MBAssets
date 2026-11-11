# -*- coding: utf-8 -*-
"""
✅ BEST PRACTICE: استخدام مكتبات غير متزامنة أصيلة (مثل httpx) للـ I/O، و to_thread للـ CPU

مقتطف من الفصل 2 — «تسخير قوة Python 3.14+ لهندسة الوكلاء».
النوع: example. قد يكون مقتطفاً جزئياً يعتمد على تعريفات الملف الرئيسي
(ch02_python314_environment.py); راجع سياقه في الكتاب قبل التشغيل المستقل.
"""

# ✅ BEST PRACTICE: استخدام مكتبات غير متزامنة أصيلة (مثل httpx) للـ I/O، و to_thread للـ CPU
import httpx


async def good_web_search_tool(query: str, client: httpx.AsyncClient) -> str:
    # ممتاز: httpx غير متزامن ويحرر حلقة الأحداث لخدمة مهام أخرى أثناء انتظار الشبكة
    response = await client.get(f"https://api.example.com/search?q={query}", timeout=5.0)
    return response.text
