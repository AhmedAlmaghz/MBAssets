#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.8: القيم الثابتة المحدودة: Literal

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-026-move.py
"""

Direction = Literal["up", "down", "left", "right"]

def move(direction: Direction, steps: int = 1) -> tuple[int, int]:
    deltas: dict[Direction, tuple[int, int]] = {
        "up": (0, 1),
        "down": (0, -1),
        "left": (-1, 0),
        "right": (1, 0),
    }
    dx, dy = deltas[direction]
    return (dx * steps, dy * steps)

print(move("up", 3))  # (0, 3)
