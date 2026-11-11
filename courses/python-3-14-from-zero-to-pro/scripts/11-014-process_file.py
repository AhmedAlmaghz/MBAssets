#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.4: else و finally: إكمال الصورة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-014-process_file.py
"""

def process_file(filename: str) -> None:
    file = None
    try:
        file = open(filename, "r", encoding="utf-8")
        content = file.read()
        print(f"تم قراءة {len(content)} حرف.")
    except FileNotFoundError:
        print(f"❌ الملف '{filename}' غير موجود.")
    finally:
        if file is not None:
            file.close()
            print("✅ تم إغلاق الملف.")  # يحدث دائماً!

process_file("not_exist.txt")
