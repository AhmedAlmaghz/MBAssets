#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.5: المعاملات: Path Parameters و Query Parameters

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-003-list_tasks.py
"""

@app.get("/tasks")
def list_tasks(skip: int = 0, limit: int = 10) -> dict[str, int]:
    return {"skip": skip, "limit": limit}
