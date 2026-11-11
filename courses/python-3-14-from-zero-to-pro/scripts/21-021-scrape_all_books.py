#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي والعشرون: مسار الأتمتة — استخراج البيانات من الويب (Web Scraping)
القسم 21.11: مشروع تطبيقي متكامل: مستخرج بيانات كتب شامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 21-021-scrape_all_books.py
"""

"""نقطة الدخول الرئيسية لأداة استخراج بيانات الكتب."""

from pathlib import Path

import pandas as pd

from book_scraper.fetcher import fetch_page, polite_delay
from book_scraper.parser import Book, extract_books_from_page, has_next_page
from book_scraper.storage import save_books_to_csv

BASE_URL = "https://books.toscrape.com/catalogue/page-{}.html"
OUTPUT_FILE = Path("output/books.csv")
MAX_PAGES = 5  # نحدد سقفًا احترازيًا لتجنب حلقة لا نهائية في حال خطأ غير متوقع

def scrape_all_books() -> list[Book]:
    """استخراج كل الكتب من كل الصفحات المتاحة.

    Returns:
        قائمة شاملة بكل الكتب المُستخرجة من كل الصفحات.
    """
    all_books: list[Book] = []
    page_number = 1

    while page_number <= MAX_PAGES:
        url = BASE_URL.format(page_number)
        print(f"📄 جلب الصفحة {page_number}...")

        html = fetch_page(url)
        if html is None:
            print("🏁 لا مزيد من الصفحات أو حدث خطأ غير قابل للتعافي. توقف.")
            break

        books_in_page = extract_books_from_page(html)
        all_books.extend(books_in_page)
        print(f"   ✅ تم استخراج {len(books_in_page)} كتاب")

        if not has_next_page(html):
            print("🏁 تم الوصول لآخر صفحة.")
            break

        page_number += 1
        polite_delay()

    return all_books

def print_analysis(books: list[Book]) -> None:
    """طباعة تحليل سريع لبيانات الكتب المُستخرجة."""
    if not books:
        print("لا توجد بيانات للتحليل.")
        return

    df = pd.DataFrame([{"title": b.title, "price": b.price, "rating": b.rating} for b in books])

    print("\n" + "=" * 50)
    print(" " * 12 + "تحليل بيانات الكتب المُستخرجة")
    print("=" * 50)
    print(f"إجمالي الكتب: {len(df)}")
    print(f"متوسط السعر: £{df['price'].mean():.2f}")
    print(f"أغلى كتاب: £{df['price'].max():.2f}")
    print(f"أرخص كتاب: £{df['price'].min():.2f}")
    print(f"متوسط التقييم: {df['rating'].mean():.1f} / 5")
    print("=" * 50)

def main() -> None:
    """تشغيل عملية الاستخراج الكاملة."""
    print("🚀 بدء استخراج بيانات الكتب من books.toscrape.com\n")

    books = scrape_all_books()

    print(f"\n💾 حفظ {len(books)} كتاب في {OUTPUT_FILE}...")
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    save_books_to_csv(books, OUTPUT_FILE)

    print_analysis(books)
    print(f"\n✅ اكتمل! راجع الملف: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
