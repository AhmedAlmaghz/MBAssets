#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث عشر: البرمجة الكائنية - الأساس (Class, Object, __init__, self)
القسم 13.8: مشروع تطبيقي: نظام إدارة مكتبة شخصية

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 13-029-__init__.py
"""

class Book:
    """يمثل كتاباً واحداً في المكتبة."""

    total_books = 0  # Class Attribute - عداد لكل الكتب المُنشأة

    def __init__(self, title: str, author: str, pages: int, is_read: bool = False):
        self.title = title
        self.author = author
        self.pages = pages
        self._is_read = is_read
        Book.total_books += 1

    @property
    def is_read(self) -> bool:
        return self._is_read

    def mark_as_read(self) -> None:
        self._is_read = True
        print(f"✅ '{self.title}' تم وضع علامة 'مقروء' عليه.")

    def __str__(self) -> str:
        status = "📖 مقروء" if self._is_read else "📕 غير مقروء"
        return f"{self.title} - {self.author} ({self.pages} صفحة) [{status}]"

    def __repr__(self) -> str:
        return f"Book(title='{self.title}', author='{self.author}')"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Book):
            return False
        return self.title == other.title and self.author == other.author

    @classmethod
    def from_string(cls, data: str) -> "Book":
        """ينشئ كتاباً من نص بصيغة: 'العنوان|المؤلف|الصفحات'"""
        title, author, pages = data.split("|")
        return cls(title.strip(), author.strip(), int(pages.strip()))

    @staticmethod
    def estimate_reading_hours(pages: int, pages_per_hour: int = 30) -> float:
        """تقدير عدد الساعات اللازمة لقراءة كتاب بعدد صفحات معين."""
        return round(pages / pages_per_hour, 1)

class Library:
    """تمثل مكتبة تحتوي على مجموعة من الكتب."""

    def __init__(self, name: str):
        self.name = name
        self.books: list[Book] = []

    def add_book(self, book: Book) -> None:
        if book in self.books:
            print(f"⚠️ '{book.title}' موجود مسبقاً في المكتبة.")
            return
        self.books.append(book)
        print(f"✅ تمت إضافة '{book.title}' إلى مكتبة '{self.name}'.")

    def remove_book(self, title: str) -> None:
        for book in self.books:
            if book.title == title:
                self.books.remove(book)
                print(f"🗑️ تم حذف '{title}'.")
                return
        print(f"❌ لم يُعثر على '{title}'.")

    def get_unread_books(self) -> list[Book]:
        return [book for book in self.books if not book.is_read]

    def total_pages(self) -> int:
        return sum(book.pages for book in self.books)

    def find_by_author(self, author: str) -> list[Book]:
        return [book for book in self.books if book.author == author]

    def __len__(self) -> int:
        return len(self.books)

    def __str__(self) -> str:
        return f"مكتبة '{self.name}' - {len(self.books)} كتاب"

    def print_all(self) -> None:
        print(f"\n📚 {self}")
        print("-" * 50)
        for i, book in enumerate(self.books, start=1):
            print(f"{i}. {book}")
        print("-" * 50)

# --- التشغيل التجريبي ---
if __name__ == "__main__":
    my_library = Library("مكتبة أحمد")

    book1 = Book("نادي السيارات", "خالد عبدالله", 250)
    book2 = Book.from_string("الخيميائي | باولو كويلو | 180")
    book3 = Book("Clean Code", "Robert Martin", 400, is_read=True)

    my_library.add_book(book1)
    my_library.add_book(book2)
    my_library.add_book(book3)
    my_library.add_book(book1)  # محاولة إضافة مكررة

    my_library.print_all()

    book1.mark_as_read()

    print(f"\n📖 عدد الكتب غير المقروءة: {len(my_library.get_unread_books())}")
    print(f"📄 إجمالي الصفحات: {my_library.total_pages()}")
    print(f"📊 إجمالي الكتب في كل المكتبات: {Book.total_books}")

    hours = Book.estimate_reading_hours(book2.pages)
    print(f"⏱️ وقت قراءة متوقع لـ '{book2.title}': {hours} ساعة")

    print(f"\n{my_library}")  # يستخدم __str__
    print(f"عدد الكتب (len): {len(my_library)}")  # يستخدم __len__
