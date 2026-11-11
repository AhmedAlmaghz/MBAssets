#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.9: تنظيم مشروع أكبر: `APIRouter`

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-010-list_tasks.py
"""

# src/task_api/routers/tasks.py
from fastapi import APIRouter

router = APIRouter(prefix="/tasks", tags=["Tasks"])

@router.get("/")
def list_tasks() -> list[str]:
    return ["مهمة 1", "مهمة 2"]
