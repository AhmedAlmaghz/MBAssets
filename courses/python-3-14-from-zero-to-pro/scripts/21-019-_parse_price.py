#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.11: مشروع تطبيقي متكامل: مستخرج بيانات كتب شامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-019-_parse_price.py
"""

"""وحدة تحليل صفحات الكتب واستخراج البيانات منها."""

from dataclasses import dataclass

from bs4 import BeautifulSoup

RATING_WORDS_TO_NUMBERS = {
    "One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5,
}

@dataclass
class Book:
    """يمثّل بيانات كتاب واحد مُستخرجة من الموقع."""

    title: str
    price: float
    rating: int
    availability: str

def _parse_price(price_text: str) -> float:
    """تحويل نص السعر مثل '£51.77' إلى رقم عشري."""
    cleaned = price_text.replace("£", "").strip()
    return float(cleaned)

def _parse_rating(star_rating_tag) -> int:
    """استخراج رقم التقييم من اسم الـ class (مثل 'Three' -> 3)."""
    classes = star_rating_tag.get("class", [])
    for cls in classes:
        if cls in RATING_WORDS_TO_NUMBERS:
            return RATING_WORDS_TO_NUMBERS[cls]
    return 0

def extract_books_from_page(html: str) -> list[Book]:
    """استخراج كل الكتب من صفحة HTML واحدة.

    Args:
        html: محتوى HTML الخام للصفحة.

    Returns:
        قائمة من كائنات Book، مع تخطي أي عنصر بيانات ناقصة.
    """
    soup = BeautifulSoup(html, "lxml")
    book_elements = soup.select("article.product_pod")

    books: list[Book] = []

    for element in book_elements:
        title_tag = element.select_one("h3 a")
        price_tag = element.select_one("p.price_color")
        rating_tag = element.select_one("p.star-rating")
        availability_tag = element.select_one("p.instock.availability")

        if not all([title_tag, price_tag, rating_tag, availability_tag]):
            continue  # تخطَّ أي كتاب ببيانات ناقصة بدلاً من إيقاف البرنامج

        book = Book(
            title=title_tag.get("title", "غير معروف"),
            price=_parse_price(price_tag.get_text(strip=True)),
            rating=_parse_rating(rating_tag),
            availability=availability_tag.get_text(strip=True),
        )
        books.append(book)

    return books

def has_next_page(html: str) -> bool:
    """التحقق من وجود صفحة تالية في نظام التصفّح."""
    soup = BeautifulSoup(html, "lxml")
    return soup.select_one("li.next") is not None
