#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع: التعامل مع النصوص (Strings)
القسم 4.14: مشروع تطبيقي: «محلل النصوص الذكي»

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 04-037-ex4_14.py
"""

# text_analyzer.py
"""برنامج تحليل النصوص - يعرض إحصائيات شاملة عن أي نص مُدخَل."""

# استقبال النص من المستخدم
text: str = input("أدخل نصًا لتحليله: ").strip()

# تقسيم النص لكلمات لحساب الإحصائيات
words: list[str] = text.split()

# الحسابات الأساسية
total_chars: int = len(text.replace(" ", ""))
word_count: int = len(words)
space_count: int = text.count(" ")

# إيجاد أطول كلمة
longest_word: str = max(words, key=len) if words else ""

# تحويلات النص
title_case: str = text.title()
reversed_text: str = text[::-1]

# طباعة التقرير
print("\n" + "=" * 40)
print("📊  تقرير تحليل النص")
print("=" * 40)
print(f"📝 النص الأصلي: {text}")
print(f"🔤 عدد الحروف (بدون مسافات): {total_chars}")
print(f"📦 عدد الكلمات: {word_count}")
print(f"␣  عدد المسافات: {space_count}")
print(f"🏆 أطول كلمة: {longest_word}")
print(f"🔠 بصيغة العنوان: {title_case}")
print(f"🔄 النص معكوسًا: {reversed_text}")
print("=" * 40)
