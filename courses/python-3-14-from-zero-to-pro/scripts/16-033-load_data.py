#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.6: أفضل الممارسات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-033-load_data.py
"""

# ✅ بنية جيدة: دوال واضحة ومسؤولة عن مهمة واحدة
def load_data(path: Path) -> list[dict[str, object]]:
    ...

def save_data(data: list[dict[str, object]], path: Path) -> None:
    ...

def process_data(data: list[dict[str, object]]) -> list[dict[str, object]]:
    ...  # المنطق الفعلي منفصل تمامًا عن I/O
