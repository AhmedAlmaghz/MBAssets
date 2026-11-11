#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.3: تعدد الأشكال (Polymorphism)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-011-make_sound.py
"""

class Duck:
    def make_sound(self) -> str:
        return "Quack!"

class Robot:
    def make_sound(self) -> str:
        return "Beep Boop!"

def animate(entity) -> None:
    """لا تهتم بنوع entity، فقط تحتاج أن تملك make_sound()"""
    print(entity.make_sound())

animate(Duck())    # Quack!
animate(Robot())   # Beep Boop! -- Robot ليست حيواناً إطلاقاً لكنها تعمل!
