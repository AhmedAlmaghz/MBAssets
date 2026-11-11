#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.10: مشروع تطبيقي متكامل: واجهة API لإدارة المهام

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-012-task_create.py
"""

"""نماذج البيانات (Pydantic Models)."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field

type Priority = Literal["منخفضة", "متوسطة", "عالية"]

class TaskCreate(BaseModel):
    """البيانات المطلوبة لإنشاء مهمة جديدة."""
    title: str = Field(min_length=1, max_length=200)
    priority: Priority = "متوسطة"

class TaskUpdate(BaseModel):
    """البيانات القابلة للتحديث في مهمة موجودة (كلها اختيارية)."""
    title: str | None = Field(default=None, min_length=1, max_length=200)
    priority: Priority | None = None
    done: bool | None = None

class Task(BaseModel):
    """نموذج المهمة الكامل كما يُخزَّن ويُرجَع للعميل."""
    id: int
    title: str
    priority: Priority
    done: bool = False
