#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.11: البروتوكولات: Protocol (Duck Typing المُهيكَل)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-033-draw.py
"""

from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> str:
        ...

class Circle:
    def draw(self) -> str:
        return "رسم دائرة ⭕"

class Square:
    def draw(self) -> str:
        return "رسم مربع ⬜"

def render(shape: Drawable) -> None:
    print(shape.draw())

render(Circle())  # صحيح! Circle تملك draw()
render(Square())  # صحيح! Square تملك draw() أيضًا
