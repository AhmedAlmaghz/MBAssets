#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.12: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-023-filter_books_by_rating.py
"""

# تمرين 5 -- سهل التطبيق مباشرة على كائنات Book
def filter_books_by_rating(books: list[Book], min_rating: int) -> list[Book]:
    """إرجاع الكتب التي تقييمها أكبر من أو يساوي الحد الأدنى المطلوب."""
    return [book for book in books if book.rating >= min_rating]

# مثال استخدام
high_rated = filter_books_by_rating(books, min_rating=4)
print(f"عدد الكتب ذات التقييم 4+: {len(high_rated)}")
