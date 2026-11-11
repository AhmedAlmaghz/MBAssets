#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.11: مشروع تطبيقي متكامل: مستخرج بيانات كتب شامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-020-save_books_to_csv.py
"""

"""وحدة حفظ بيانات الكتب المُستخرجة."""

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from book_scraper.parser import Book

def save_books_to_csv(books: list[Book], file_path: str | Path) -> None:
    """حفظ قائمة الكتب في ملف CSV باستخدام pandas.

    Args:
        books: قائمة كائنات Book المُستخرجة.
        file_path: مسار ملف الحفظ.
    """
    records = [asdict(book) for book in books]
    df = pd.DataFrame(records)
    df.to_csv(file_path, index=False, encoding="utf-8-sig")
