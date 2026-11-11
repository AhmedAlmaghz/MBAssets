#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس: اتخاذ القرار (Control Flow)
القسم 6.12: أخطاء شائعة ونصائح احترافية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 06-030-ex6_12.py
"""

is_active: bool = True

# ❌ غير احترافي (زائد عن الحاجة)
if is_active == True:
    print("نشط")

# ✅ احترافي (Pythonic)
if is_active:
    print("نشط")
