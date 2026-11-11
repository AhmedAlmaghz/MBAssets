#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث والعشرون: مسار بناء أدوات سطر الأوامر (CLI)
القسم 23.2: البداية البدائية: وحدة `argparse` المدمجة

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 23-001-parser.py
"""

# greet.py
import argparse

parser = argparse.ArgumentParser(description="أداة ترحيب بسيطة")
parser.add_argument("name", help="اسم الشخص المراد الترحيب به")
parser.add_argument("--greeting", default="مرحبًا", help="كلمة الترحيب")
parser.add_argument("--loud", action="store_true", help="اطبع بأحرف كبيرة")

args = parser.parse_args()

message = f"{args.greeting}، {args.name}!"
if args.loud:
    message = message.upper()

print(message)
