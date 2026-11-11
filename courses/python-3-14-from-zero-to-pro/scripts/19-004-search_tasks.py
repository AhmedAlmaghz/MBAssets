#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.5: المعاملات: Path Parameters و Query Parameters

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-004-search_tasks.py
"""

@app.get("/search")
def search_tasks(keyword: str | None = None) -> dict[str, str | None]:
    if keyword is None:
        return {"message": "لم يُحدد كلمة بحث"}
    return {"message": f"البحث عن: {keyword}"}
