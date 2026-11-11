#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.10: مشروع تطبيقي متكامل: واجهة API لإدارة المهام

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-014-list_tasks.py
"""

"""نقاط النهاية الخاصة بإدارة المهام."""
from fastapi import APIRouter, HTTPException, status

from task_api.models import Task, TaskCreate, TaskUpdate
from task_api.storage import storage

router = APIRouter(prefix="/tasks", tags=["Tasks"])

@router.get("/", response_model=list[Task])
def list_tasks(done: bool | None = None) -> list[Task]:
    """يُرجع كل المهام، مع إمكانية التصفية حسب حالة الإنجاز."""
    all_tasks = storage.get_all()
    if done is None:
        return all_tasks
    return [task for task in all_tasks if task.done == done]

@router.get("/{task_id}", response_model=Task)
def get_task(task_id: int) -> Task:
    """يُرجع مهمة واحدة بمعرّفها."""
    task = storage.get_by_id(task_id)
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"لا توجد مهمة بالمعرّف {task_id}",
        )
    return task

@router.post("/", response_model=Task, status_code=status.HTTP_201_CREATED)
def create_task(task_data: TaskCreate) -> Task:
    """ينشئ مهمة جديدة."""
    return storage.create(title=task_data.title, priority=task_data.priority)

@router.patch("/{task_id}", response_model=Task)
def update_task(task_id: int, task_data: TaskUpdate) -> Task:
    """يُحدّث حقولًا معيّنة في مهمة موجودة."""
    updated = storage.update(
        task_id,
        title=task_data.title,
        priority=task_data.priority,
        done=task_data.done,
    )
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"لا توجد مهمة بالمعرّف {task_id}",
        )
    return updated

@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_task(task_id: int) -> None:
    """يحذف مهمة موجودة."""
    deleted = storage.delete(task_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"لا توجد مهمة بالمعرّف {task_id}",
        )
