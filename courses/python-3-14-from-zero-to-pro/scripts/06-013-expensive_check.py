#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس: اتخاذ القرار (Control Flow)
القسم 6.6: العوامل المنطقية: `and`, `or`, `not`

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 06-013-expensive_check.py
"""

def expensive_check() -> bool:
    print("تم استدعاء الفحص المكلف!")
    return True

# مع and: إذا كان الشرط الأول False، لن يُستدعى الثاني إطلاقًا
if False and expensive_check():
    pass
# لن تُطبع "تم استدعاء الفحص المكلف!" أبدًا!

# مع or: إذا كان الشرط الأول True، لن يُستدعى الثاني إطلاقًا
if True or expensive_check():
    pass
# لن تُطبع "تم استدعاء الفحص المكلف!" أبدًا!
