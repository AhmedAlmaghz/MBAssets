#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.5: Dataclasses: اختصار كتابة الفئات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-015-student.py
"""

from dataclasses import dataclass, field

@dataclass
class Student:
    name: str
    age: int = 18  # قيمة افتراضية بسيطة
    grades: list[float] = field(default_factory=list)  # للقيم القابلة للتعديل!

s1 = Student("أحمد")
s1.grades.append(90)
print(s1)  # Student(name='أحمد', age=18, grades=[90])

s2 = Student("سارة")
print(s2)  # Student(name='سارة', age=18, grades=[])  -- مستقلة تماماً!
