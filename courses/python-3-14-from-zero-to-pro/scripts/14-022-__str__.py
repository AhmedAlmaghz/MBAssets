#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.7: مشروع تطبيقي متكامل: نظام حجز مكتبة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-022-__str__.py
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto

# ============ الطبقة 1: نماذج البيانات (Dataclasses) ============

class ItemStatus(Enum):
    """حالة العنصر في المكتبة."""
    AVAILABLE = auto()
    BORROWED = auto()
    RESERVED = auto()

@dataclass
class Member:
    """يمثل عضواً في المكتبة."""
    member_id: int
    name: str
    borrowed_items: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"عضو #{self.member_id}: {self.name}"

# ============ الطبقة 2: الفئة المجردة للعناصر القابلة للاستعارة ============

class LibraryItem(ABC):
    """فئة مجردة أساسية لأي عنصر في المكتبة."""

    def __init__(self, item_id: str, title: str, loan_days: int):
        self.item_id = item_id
        self.title = title
        self.loan_days = loan_days  # عدد الأيام المسموحة للاستعارة
        self.status = ItemStatus.AVAILABLE
        self.borrowed_by: Member | None = None
        self.due_date: datetime | None = None

    @abstractmethod
    def get_category(self) -> str:
        """كل فئة فرعية يجب أن تحدد فئتها."""
        pass

    @abstractmethod
    def display_details(self) -> str:
        """كل فئة فرعية تعرض تفاصيلها الخاصة."""
        pass

    def borrow(self, member: Member) -> bool:
        """استعارة العنصر - سلوك مشترك لكل الأنواع."""
        if self.status != ItemStatus.AVAILABLE:
            print(f"❌ '{self.title}' غير متاح حالياً (الحالة: {self.status.name}).")
            return False

        self.status = ItemStatus.BORROWED
        self.borrowed_by = member
        self.due_date = datetime.now() + timedelta(days=self.loan_days)
        member.borrowed_items.append(self.item_id)

        print(f"✅ استعار {member.name} '{self.title}'. "
              f"الإرجاع المتوقع: {self.due_date.strftime('%Y-%m-%d')}")
        return True

    def return_item(self) -> None:
        """إرجاع العنصر."""
        if self.status != ItemStatus.BORROWED:
            print(f"⚠️ '{self.title}' ليس مُستعاراً أصلاً.")
            return

        if self.borrowed_by:
            self.borrowed_by.borrowed_items.remove(self.item_id)
            print(f"✅ تم إرجاع '{self.title}' من {self.borrowed_by.name}.")

        self.status = ItemStatus.AVAILABLE
        self.borrowed_by = None
        self.due_date = None

    def is_overdue(self) -> bool:
        """التحقق من تأخر الإرجاع - سلوك مشترك."""
        if self.status != ItemStatus.BORROWED or self.due_date is None:
            return False
        return datetime.now() > self.due_date

    def __str__(self) -> str:
        status_display = self.status.name
        return f"[{self.get_category()}] {self.title} ({self.item_id}) - {status_display}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, LibraryItem):
            return False
        return self.item_id == other.item_id

    def __hash__(self) -> int:
        return hash(self.item_id)

# ============ الطبقة 3: الفئات الفرعية (تطبيق تعدد الأشكال) ============

class Book(LibraryItem):
    def __init__(self, item_id: str, title: str, author: str, pages: int):
        super().__init__(item_id, title, loan_days=14)  # الكتب تُستعار 14 يوماً
        self.author = author
        self.pages = pages

    def get_category(self) -> str:
        return "كتاب"

    def display_details(self) -> str:
        return f"📖 {self.title} - تأليف: {self.author} ({self.pages} صفحة)"

class DVD(LibraryItem):
    def __init__(self, item_id: str, title: str, duration_minutes: int):
        super().__init__(item_id, title, loan_days=5)  # الأفلام تُستعار 5 أيام فقط
        self.duration_minutes = duration_minutes

    def get_category(self) -> str:
        return "دي في دي"

    def display_details(self) -> str:
        return f"🎬 {self.title} - المدة: {self.duration_minutes} دقيقة"

class Magazine(LibraryItem):
    def __init__(self, item_id: str, title: str, issue_number: int):
        super().__init__(item_id, title, loan_days=3)  # المجلات 3 أيام فقط
        self.issue_number = issue_number

    def get_category(self) -> str:
        return "مجلة"

    def display_details(self) -> str:
        return f"📰 {self.title} - العدد: {self.issue_number}"

# ============ الطبقة 4: استثناءات مخصصة ============

class LibraryError(Exception):
    """الفئة الأساسية لكل أخطاء نظام المكتبة."""
    pass

class ItemNotFoundError(LibraryError):
    def __init__(self, item_id: str):
        super().__init__(f"العنصر '{item_id}' غير موجود في المكتبة.")

class MemberNotFoundError(LibraryError):
    def __init__(self, member_id: int):
        super().__init__(f"العضو رقم {member_id} غير موجود.")

# ============ الطبقة 5: النظام الرئيسي ============

class Library:
    """يُدير كل عمليات المكتبة: الأعضاء، العناصر، الاستعارة، الإرجاع."""

    def __init__(self, name: str):
        self.name = name
        self.items: dict[str, LibraryItem] = {}
        self.members: dict[int, Member] = {}

    # --- إدارة العناصر ---
    def add_item(self, item: LibraryItem) -> None:
        self.items[item.item_id] = item
        print(f"📚 أُضيف '{item.title}' إلى مكتبة '{self.name}'.")

    def find_item(self, item_id: str) -> LibraryItem:
        try:
            return self.items[item_id]
        except KeyError:
            raise ItemNotFoundError(item_id)

    # --- إدارة الأعضاء ---
    def register_member(self, member: Member) -> None:
        self.members[member.member_id] = member
        print(f"👤 تم تسجيل {member.name} كعضو جديد.")

    def find_member(self, member_id: int) -> Member:
        try:
            return self.members[member_id]
        except KeyError:
            raise MemberNotFoundError(member_id)

    # --- العمليات الأساسية ---
    def borrow_item(self, item_id: str, member_id: int) -> None:
        item = self.find_item(item_id)
        member = self.find_member(member_id)
        item.borrow(member)

    def return_item(self, item_id: str) -> None:
        item = self.find_item(item_id)
        item.return_item()

    # --- تقارير (تطبيق تعدد الأشكال بوضوح) ---
    def list_all_items(self) -> None:
        print(f"\n📋 كل عناصر مكتبة '{self.name}':")
        print("-" * 60)
        for item in self.items.values():
            print(item.display_details())  # كل نوع يعرض تفاصيله الخاصة!
        print("-" * 60)

    def list_available_items(self) -> list[LibraryItem]:
        return [item for item in self.items.values()
                if item.status == ItemStatus.AVAILABLE]

    def list_overdue_items(self) -> list[LibraryItem]:
        return [item for item in self.items.values() if item.is_overdue()]

    def items_by_category(self) -> dict[str, int]:
        """عدد العناصر حسب الفئة - باستخدام polymorphism (get_category)."""
        counts: dict[str, int] = {}
        for item in self.items.values():
            category = item.get_category()
            counts[category] = counts.get(category, 0) + 1
        return counts

# ============ التشغيل التجريبي ============

if __name__ == "__main__":
    library = Library("مكتبة الأمل العامة")

    # إضافة عناصر متنوعة (تعدد الأشكال في العمل)
    library.add_item(Book("B001", "بايثون للمحترفين", "محمد أحمد", 350))
    library.add_item(Book("B002", "Clean Code", "Robert Martin", 400))
    library.add_item(DVD("D001", "رحلة إلى الفضاء", 120))
    library.add_item(Magazine("M001", "التقنية اليوم", 45))

    # تسجيل أعضاء
    library.register_member(Member(1, "أحمد"))
    library.register_member(Member(2, "سارة"))

    print()
    library.list_all_items()

    print("\n--- عمليات الاستعارة ---")
    library.borrow_item("B001", 1)
    library.borrow_item("D001", 2)

    try:
        library.borrow_item("B001", 2)  # محاولة استعارة كتاب مُستعار بالفعل
    except LibraryError as e:
        print(f"❌ {e}")

    try:
        library.borrow_item("X999", 1)  # عنصر غير موجود
    except ItemNotFoundError as e:
        print(f"❌ {e}")

    print("\n--- حالة المكتبة ---")
    library.list_all_items()

    available = library.list_available_items()
    print(f"\n✅ العناصر المتاحة حالياً: {len(available)}")
    for item in available:
        print(f"  - {item}")

    print(f"\n📊 توزيع العناصر حسب الفئة: {library.items_by_category()}")

    print("\n--- إرجاع عنصر ---")
    library.return_item("B001")

    library.list_all_items()
