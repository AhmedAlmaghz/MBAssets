#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.5: المعاملات: Path Parameters و Query Parameters

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-002-get_task.py
"""

from fastapi import FastAPI

app = FastAPI()

@app.get("/tasks/{task_id}")
def get_task(task_id: int) -> dict[str, int]:
    return {"task_id": task_id}
