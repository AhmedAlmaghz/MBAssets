#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس عشر: الملفات وتنسيقات البيانات
القسم 16.7: مشروع تطبيقي: مدير جهات اتصال متكامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 16-034-display.py
"""

"""
مدير جهات الاتصال - مشروع تطبيقي شامل
يغطي: الملفات النصية، pathlib، JSON، CSV، ومعالجة الأخطاء
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Literal
import json
import csv

type ContactGroup = Literal["عائلة", "عمل", "أصدقاء", "أخرى"]

# ---------------------------------------------------------
# نموذج البيانات
# ---------------------------------------------------------
@dataclass
class Contact:
    name: str
    phone: str
    email: str = ""
    group: ContactGroup = "أخرى"

    def display(self) -> str:
        return f"📇 {self.name} | {self.phone} | {self.email} | {self.group}"

# ---------------------------------------------------------
# استثناءات مخصصة
# ---------------------------------------------------------
class ContactNotFoundError(Exception):
    """يُرفع عند عدم العثور على جهة اتصال بالاسم المطلوب."""

# ---------------------------------------------------------
# مدير جهات الاتصال
# ---------------------------------------------------------
class ContactManager:
    def __init__(self, storage_path: Path) -> None:
        self.storage_path = storage_path
        self._contacts: list[Contact] = []
        self._load()

    # ----- التحميل والحفظ -----
    def _load(self) -> None:
        """يحمّل جهات الاتصال من ملف JSON إن وُجد، وإلا يبدأ بقائمة فارغة."""
        if not self.storage_path.exists():
            self._contacts = []
            return

        try:
            raw_text = self.storage_path.read_text(encoding="utf-8")
            raw_data = json.loads(raw_text)
            self._contacts = [Contact(**item) for item in raw_data]
        except json.JSONDecodeError:
            print("⚠️ تحذير: ملف البيانات تالف. سيتم البدء بقائمة فارغة.")
            self._contacts = []
        except TypeError:
            print("⚠️ تحذير: بنية البيانات غير متوافقة. سيتم البدء بقائمة فارغة.")
            self._contacts = []

    def save(self) -> None:
        """يحفظ كل جهات الاتصال الحالية إلى ملف JSON."""
        data = [asdict(contact) for contact in self._contacts]
        self.storage_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ----- العمليات الأساسية -----
    def add_contact(
        self,
        name: str,
        phone: str,
        email: str = "",
        group: ContactGroup = "أخرى",
    ) -> Contact:
        contact = Contact(name=name, phone=phone, email=email, group=group)
        self._contacts.append(contact)
        self.save()
        return contact

    def find_by_name(self, name: str) -> Contact:
        for contact in self._contacts:
            if contact.name == name:
                return contact
        raise ContactNotFoundError(f"لا توجد جهة اتصال باسم '{name}'")

    def delete_contact(self, name: str) -> None:
        contact = self.find_by_name(name)  # يرفع ContactNotFoundError تلقائيًا إن لم يوجد
        self._contacts.remove(contact)
        self.save()

    def search(self, keyword: str) -> list[Contact]:
        keyword_lower = keyword.lower()
        return [
            c for c in self._contacts
            if keyword_lower in c.name.lower() or keyword_lower in c.phone
        ]

    def get_by_group(self, group: ContactGroup) -> list[Contact]:
        return [c for c in self._contacts if c.group == group]

    def all_contacts(self) -> list[Contact]:
        return self._contacts.copy()

    def count(self) -> int:
        return len(self._contacts)

    # ----- التصدير -----
    def export_to_csv(self, csv_path: Path) -> None:
        if not self._contacts:
            print("لا توجد جهات اتصال للتصدير.")
            return

        fieldnames = [f.name for f in fields(Contact)]
        with open(csv_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for contact in self._contacts:
                writer.writerow(asdict(contact))
        print(f"✅ تم تصدير {len(self._contacts)} جهة اتصال إلى {csv_path}")

    def import_from_csv(self, csv_path: Path) -> int:
        if not csv_path.exists():
            raise FileNotFoundError(f"ملف الاستيراد غير موجود: {csv_path}")

        imported_count = 0
        with open(csv_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                self._contacts.append(Contact(**row))  # type: ignore[arg-type]
                imported_count += 1
        self.save()
        return imported_count

# ---------------------------------------------------------
# برنامج تجريبي
# ---------------------------------------------------------
def main() -> None:
    manager = ContactManager(Path("contacts.json"))

    manager.add_contact("أحمد علي", "0501234567", "ahmed@example.com", "عمل")
    manager.add_contact("سارة محمد", "0559876543", group="عائلة")
    manager.add_contact("خالد يوسف", "0507654321", "khaled@example.com", "أصدقاء")

    print(f"إجمالي جهات الاتصال: {manager.count()}")
    print()

    print("كل جهات الاتصال:")
    for contact in manager.all_contacts():
        print(contact.display())
    print()

    print("البحث عن 'سارة':")
    for contact in manager.search("سارة"):
        print(contact.display())
    print()

    try:
        manager.find_by_name("غير موجود")
    except ContactNotFoundError as e:
        print(f"خطأ متوقع: {e}")
    print()

    manager.export_to_csv(Path("contacts_export.csv"))

if __name__ == "__main__":
    main()
