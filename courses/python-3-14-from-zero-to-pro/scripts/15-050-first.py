#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.17: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-050-first.py
"""

# ===== الأنواع الأساسية =====
age: int = 25
price: float = 19.99
name: str = "سارة"
active: bool = True

# ===== الهياكل المركّبة =====
numbers: list[int] = [1, 2, 3]
point: tuple[float, float] = (1.0, 2.0)
scores: tuple[int, ...] = (1, 2, 3, 4)
ages: dict[str, int] = {"أحمد": 30}
tags: set[str] = {"a", "b"}

# ===== القيم الاختيارية والمركّبة =====
result: str | None = None          # اختياري (الصيغة الحديثة)
value: int | str = 5               # إمّا هذا أو ذاك

# ===== Any (استخدمها بحذر!) =====
from typing import Any
data: Any = {"whatever": "shape"}

# ===== Callable (الدوال كقيم) =====
from typing import Callable
operation: Callable[[int, int], int]

# ===== Literal (قيم محددة فقط) =====
from typing import Literal
Status = Literal["pending", "done"]

# ===== TypedDict (قاموس مُهيكَل) =====
from typing import TypedDict
class User(TypedDict):
    name: str
    age: int

# ===== Generics (البرمجة العامة) =====
def first[T](items: list[T]) -> T:
    return items[0]

class Box[T]:
    def __init__(self, content: T) -> None:
        self.content = content

# ===== Protocol (duck typing مُهيكَل) =====
from typing import Protocol
class Drawable(Protocol):
    def draw(self) -> str: ...

# ===== mypy =====
# التثبيت:
#   uv add --dev mypy
# التشغيل:
#   uv run mypy file.py
#   uv run mypy . --strict
# الإعداد في pyproject.toml:
#   [tool.mypy]
#   strict = true
