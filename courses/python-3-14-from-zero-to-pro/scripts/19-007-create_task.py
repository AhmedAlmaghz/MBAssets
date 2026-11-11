#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.6: Pydantic: نمذجة البيانات الواردة (Request Body)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-007-create_task.py
"""

class TaskOut(BaseModel):
    id: int
    title: str
    priority: str
    done: bool

@app.post("/tasks", response_model=TaskOut)
def create_task(task: TaskCreate) -> TaskOut:
    new_id = 1  # لاحقًا سيأتي من قاعدة بيانات حقيقية
    return TaskOut(id=new_id, title=task.title, priority=task.priority, done=task.done)
