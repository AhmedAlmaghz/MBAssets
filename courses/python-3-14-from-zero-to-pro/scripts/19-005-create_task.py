#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.6: Pydantic: نمذجة البيانات الواردة (Request Body)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-005-create_task.py
"""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class TaskCreate(BaseModel):
    title: str
    priority: str = "متوسطة"
    done: bool = False

@app.post("/tasks")
def create_task(task: TaskCreate) -> TaskCreate:
    return task
