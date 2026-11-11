#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الحادي عشر: معالجة الأخطاء (Error Handling)
القسم 11.4: else و finally: إكمال الصورة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 11-012-read_and_process_number.py
"""

def read_and_process_number(text: str) -> None:
    try:
        number = int(text)
    except ValueError:
        print(f"❌ '{text}' ليس رقماً صحيحاً.")
    else:
        # يُنفَّذ فقط إذا نجح التحويل بدون خطأ
        print(f"✅ تم تحويل الرقم بنجاح: {number}")
        print(f"   ضعفه: {number * 2}")
    finally:
        # يُنفَّذ دائماً - مثالي للتنظيف
        print("--- انتهت المحاولة ---")

read_and_process_number("42")
print()
read_and_process_number("abc")
