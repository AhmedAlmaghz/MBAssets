#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.10: مشروع تطبيقي متكامل: واجهة API لإدارة المهام

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-013-__init__.py
"""

"""طبقة تخزين بسيطة تعتمد على ملف JSON (كما تعلمنا في الفصل 16)."""
from __future__ import annotations

import json
from pathlib import Path

from task_api.models import Task

class TaskStorage:
    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self._tasks: dict[int, Task] = {}
        self._next_id: int = 1
        self._load()

    def _load(self) -> None:
        if not self.file_path.exists():
            return
        try:
            raw_data = json.loads(self.file_path.read_text(encoding="utf-8"))
            self._tasks = {item["id"]: Task(**item) for item in raw_data}
            if self._tasks:
                self._next_id = max(self._tasks.keys()) + 1
        except (json.JSONDecodeError, KeyError):
            self._tasks = {}

    def _save(self) -> None:
        data = [task.model_dump() for task in self._tasks.values()]
        self.file_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_all(self) -> list[Task]:
        return list(self._tasks.values())

    def get_by_id(self, task_id: int) -> Task | None:
        return self._tasks.get(task_id)

    def create(self, title: str, priority: str) -> Task:
        task = Task(id=self._next_id, title=title, priority=priority, done=False)  # type: ignore[arg-type]
        self._tasks[task.id] = task
        self._next_id += 1
        self._save()
        return task

    def update(self, task_id: int, **changes: object) -> Task | None:
        task = self._tasks.get(task_id)
        if task is None:
            return None
        updated_task = task.model_copy(update={k: v for k, v in changes.items() if v is not None})
        self._tasks[task_id] = updated_task
        self._save()
        return updated_task

    def delete(self, task_id: int) -> bool:
        if task_id not in self._tasks:
            return False
        del self._tasks[task_id]
        self._save()
        return True

# نسخة واحدة مشتركة للتطبيق بالكامل (Singleton بسيط)
storage = TaskStorage(Path("tasks.json"))
