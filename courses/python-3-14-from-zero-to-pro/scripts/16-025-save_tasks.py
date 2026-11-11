#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.4: JSON: لغة تبادل البيانات العالمية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-025-save_tasks.py
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import json

@dataclass
class Task:
    title: str
    done: bool = False

def save_tasks(tasks: list[Task], file_path: Path) -> None:
    data = [asdict(task) for task in tasks]
    file_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def load_tasks(file_path: Path) -> list[Task]:
    if not file_path.exists():
        return []
    raw_data = json.loads(file_path.read_text(encoding="utf-8"))
    return [Task(**item) for item in raw_data]

tasks_file = Path("tasks.json")

tasks = [
    Task("تعلم JSON"),
    Task("تعلم CSV", done=True),
]
save_tasks(tasks, tasks_file)

loaded_tasks = load_tasks(tasks_file)
for task in loaded_tasks:
    status = "✅" if task.done else "⬜"
    print(f"{status} {task.title}")
