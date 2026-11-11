#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.10: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-027-__init__.py
"""

# ============ الوراثة ============
class Parent:
    def __init__(self, x):
        self.x = x
    def greet(self):
        return f"Parent: {self.x}"

class Child(Parent):
    def __init__(self, x, y):
        super().__init__(x)    # استدعاء __init__ الأساسي
        self.y = y
    def greet(self):
        return f"{super().greet()}, Child: {self.y}"  # تمديد السلوك

# ============ تعدد الأشكال ============
for obj in [ClassA(), ClassB(), ClassC()]:
    obj.common_method()  # كل فئة تُنفّذ بطريقتها

# ============ الفئات المجردة ============
from abc import ABC, abstractmethod

class Base(ABC):
    @abstractmethod
    def required_method(self): ...

# Base()  # 💥 TypeError - لا يمكن إنشاء كائن مباشرة

# ============ Dataclasses ============
from dataclasses import dataclass, field

@dataclass
class Point:
    x: float
    y: float
    tags: list = field(default_factory=list)  # ⚠️ ليس []

@dataclass(frozen=True)   # immutable
@dataclass(order=True)    # يدعم < <= > >=

# ============ Magic Methods إضافية ============
def __call__(self, *args): ...      # obj(args)
def __getitem__(self, i): ...       # obj[i]
def __enter__(self): ...            # with obj as x:
def __exit__(self, *args): ...

# ============ isinstance ============
isinstance(obj, SomeClass)          # ✅ يأخذ الوراثة بعين الاعتبار
type(obj) == SomeClass              # ❌ تطابق دقيق فقط
