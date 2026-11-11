#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.7: معالجة الأخطاء: `HTTPException`

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-008-get_task.py
"""

from fastapi import FastAPI, HTTPException

app = FastAPI()

tasks_db: dict[int, str] = {1: "تعلم Python", 2: "تعلم FastAPI"}

@app.get("/tasks/{task_id}")
def get_task(task_id: int) -> dict[str, str]:
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="المهمة غير موجودة")
    return {"title": tasks_db[task_id]}
