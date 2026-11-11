#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.8: مشروع تطبيقي: نظام إدارة مكتبة شخصية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-030-sort_by_pages.py
"""

def sort_by_pages(self, reverse: bool = False) -> list[Book]:
    """ترتيب كتب المكتبة حسب عدد الصفحات."""
    return sorted(self.books, key=lambda book: book.pages, reverse=reverse)

# إضافة هذا كـ method داخل Library، ثم:
sorted_books = my_library.sort_by_pages()
for book in sorted_books:
    print(f"{book.title}: {book.pages} صفحة")
