#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث والعشرون: مسار بناء أدوات سطر الأوامر (CLI)
القسم 23.7: مشروع تطبيقي متكامل: أداة تنظيم الملفات (File Organizer)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 23-009-get_category_for_file.py
"""

"""قواعد تصنيف أنواع الملفات إلى فئات."""

FILE_CATEGORIES: dict[str, list[str]] = {
    "الصور": [".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".bmp"],
    "المستندات": [".pdf", ".docx", ".doc", ".txt", ".md", ".xlsx", ".pptx"],
    "الفيديوهات": [".mp4", ".mkv", ".avi", ".mov", ".webm"],
    "الصوتيات": [".mp3", ".wav", ".flac", ".aac"],
    "الأرشيف": [".zip", ".rar", ".tar", ".gz", ".7z"],
    "الأكواد": [".py", ".js", ".html", ".css", ".json", ".java", ".cpp"],
}

DEFAULT_CATEGORY = "أخرى"

def get_category_for_file(extension: str) -> str:
    """تحديد الفئة المناسبة لامتداد ملف معيّن.

    Args:
        extension: امتداد الملف (مثل '.jpg'), بأي حالة أحرف.

    Returns:
        اسم الفئة المناسبة، أو DEFAULT_CATEGORY إن لم يُعرف الامتداد.
    """
    extension = extension.lower()
    for category, extensions in FILE_CATEGORIES.items():
        if extension in extensions:
            return category
    return DEFAULT_CATEGORY
