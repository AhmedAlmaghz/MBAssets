#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.9: تنظيم مشروع أكبر: `APIRouter`

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-011-app.py
"""

# src/task_api/main.py
from fastapi import FastAPI
from task_api.routers import tasks

app = FastAPI()
app.include_router(tasks.router)
