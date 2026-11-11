#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.10: مشروع تطبيقي متكامل: واجهة API لإدارة المهام

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-015-read_root.py
"""

"""نقطة الدخول الرئيسية لتطبيق FastAPI."""
from fastapi import FastAPI

from task_api.routers import tasks

app = FastAPI(
    title="واجهة API لإدارة المهام",
    description="مشروع تطبيقي من كتاب تعلّم Python",
    version="1.0.0",
)

app.include_router(tasks.router)

@app.get("/", tags=["Root"])
def read_root() -> dict[str, str]:
    return {"message": "مرحبًا بك في واجهة API لإدارة المهام! زُر /docs للتوثيق التفاعلي"}
