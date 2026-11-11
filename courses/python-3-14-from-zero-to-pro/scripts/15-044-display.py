#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.14: مشروع تطبيقي: نظام إدارة مهام مُنمَّط بالكامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-044-display.py
"""

"""
نظام إدارة المهام - مثال تطبيقي شامل على Type Hints
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Protocol, Callable

# ---------------------------------------------------------
# 1. تعريف الأنواع الأساسية باستخدام Literal و Type Alias
# ---------------------------------------------------------
type Priority = Literal["منخفضة", "متوسطة", "عالية"]
type Status = Literal["معلّقة", "قيد التنفيذ", "منجزة"]

# ---------------------------------------------------------
# 2. Protocol لأي شيء "قابل للعرض"
# ---------------------------------------------------------
class Displayable(Protocol):
    def display(self) -> str:
        ...

# ---------------------------------------------------------
# 3. فئة المهمة (Task) باستخدام dataclass
# ---------------------------------------------------------
@dataclass
class Task:
    title: str
    priority: Priority = "متوسطة"
    status: Status = "معلّقة"
    due_date: date | None = None
    tags: list[str] = field(default_factory=list)

    def display(self) -> str:
        due_info = f" (موعد التسليم: {self.due_date})" if self.due_date else ""
        tags_info = f" [{', '.join(self.tags)}]" if self.tags else ""
        return f"📋 {self.title} - {self.priority} - {self.status}{due_info}{tags_info}"

    def mark_as(self, new_status: Status) -> None:
        self.status = new_status

# ---------------------------------------------------------
# 4. مستودع عام (Generic Repository) لإدارة أي نوع من العناصر
# ---------------------------------------------------------
class Repository[T]:
    """مستودع عام يدير مجموعة من العناصر من أي نوع."""

    def __init__(self) -> None:
        self._items: list[T] = []

    def add(self, item: T) -> None:
        self._items.append(item)

    def remove(self, item: T) -> None:
        self._items.remove(item)

    def all(self) -> list[T]:
        return self._items.copy()

    def filter(self, predicate: Callable[[T], bool]) -> list[T]:
        return [item for item in self._items if predicate(item)]

    def find_first(self, predicate: Callable[[T], bool]) -> T | None:
        for item in self._items:
            if predicate(item):
                return item
        return None

    def count(self) -> int:
        return len(self._items)

# ---------------------------------------------------------
# 5. مدير المهام (TaskManager) - يستخدم Repository
# ---------------------------------------------------------
class TaskManager:
    def __init__(self) -> None:
        self._repo: Repository[Task] = Repository()

    def add_task(
        self,
        title: str,
        priority: Priority = "متوسطة",
        due_date: date | None = None,
        tags: list[str] | None = None,
    ) -> Task:
        task = Task(
            title=title,
            priority=priority,
            due_date=due_date,
            tags=tags if tags is not None else [],
        )
        self._repo.add(task)
        return task

    def get_by_status(self, status: Status) -> list[Task]:
        return self._repo.filter(lambda t: t.status == status)

    def get_by_priority(self, priority: Priority) -> list[Task]:
        return self._repo.filter(lambda t: t.priority == priority)

    def find_task_by_title(self, title: str) -> Task | None:
        return self._repo.find_first(lambda t: t.title == title)

    def complete_task(self, title: str) -> bool:
        task = self.find_task_by_title(title)
        if task is None:
            return False
        task.mark_as("منجزة")
        return True

    def summary(self) -> dict[Status, int]:
        statuses: list[Status] = ["معلّقة", "قيد التنفيذ", "منجزة"]
        return {
            status: len(self.get_by_status(status))
            for status in statuses
        }

    def display_all(self) -> None:
        if self._repo.count() == 0:
            print("لا توجد مهام بعد.")
            return
        for task in self._repo.all():
            print(task.display())

# ---------------------------------------------------------
# 6. دالة عامة تعمل مع أي كائن Displayable (Protocol)
# ---------------------------------------------------------
def print_report(items: list[Displayable], title: str = "تقرير") -> None:
    print(f"=== {title} ===")
    for item in items:
        print(item.display())
    print(f"عدد العناصر: {len(items)}")

# ---------------------------------------------------------
# 7. برنامج تجريبي
# ---------------------------------------------------------
def main() -> None:
    manager = TaskManager()

    manager.add_task(
        "كتابة تقرير شهري",
        priority="عالية",
        due_date=date(2025, 6, 30),
        tags=["عمل", "عاجل"],
    )
    manager.add_task("قراءة كتاب Python", priority="منخفضة")
    manager.add_task(
        "مراجعة الكود مع الفريق",
        priority="متوسطة",
        tags=["فريق"],
    )

    print("كل المهام:")
    manager.display_all()
    print()

    manager.complete_task("قراءة كتاب Python")

    print("المهام العالية الأولوية:")
    for task in manager.get_by_priority("عالية"):
        print(task.display())
    print()

    print("ملخص الحالات:")
    for status, count in manager.summary().items():
        print(f"  {status}: {count}")
    print()

    high_priority_tasks: list[Displayable] = list(manager.get_by_priority("عالية"))
    print_report(high_priority_tasks, title="تقرير المهام العاجلة")

if __name__ == "__main__":
    main()
