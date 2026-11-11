#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع عشر: البرمجة الكائنية
القسم 14.6: مزيد من Magic Methods: `__call__`, `__getitem__`, Context Managers

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 14-021-__enter__.py
"""

class Timer:
    """context manager يقيس زمن تنفيذ كتلة كود."""

    def __enter__(self):
        import time
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        import time
        self.end = time.perf_counter()
        print(f"⏱️ استغرق التنفيذ: {self.end - self.start:.4f} ثانية")

with Timer():
    total = sum(i ** 2 for i in range(1_000_000))
    print(f"النتيجة: {total}")
