#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.6: Pydantic: نمذجة البيانات الواردة (Request Body)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-006-task_create.py
"""

from pydantic import BaseModel, Field

class TaskCreate(BaseModel):
    title: str = Field(min_length=1, max_length=100)
    priority: str = Field(default="متوسطة", pattern="^(منخفضة|متوسطة|عالية)$")
    estimated_hours: float = Field(default=1.0, gt=0, le=100)
