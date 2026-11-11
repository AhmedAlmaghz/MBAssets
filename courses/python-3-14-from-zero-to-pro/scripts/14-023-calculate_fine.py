#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.7: مشروع تطبيقي متكامل: نظام حجز مكتبة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-023-calculate_fine.py
"""

class LibraryItem(ABC):
    # ... (الكود السابق كما هو) ...
    FINE_PER_DAY = 2.0  # ريال لكل يوم تأخير

    def calculate_fine(self) -> float:
        """حساب الغرامة إذا كان العنصر متأخراً."""
        if not self.is_overdue() or self.due_date is None:
            return 0.0
        days_late = (datetime.now() - self.due_date).days
        return days_late * self.FINE_PER_DAY

# استخدام:
item = library.find_item("B001")
fine = item.calculate_fine()
if fine > 0:
    print(f"⚠️ غرامة تأخير: {fine} ر.س")
