#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثالث: المتغيرات - أول لبنة
القسم 3.15: مشروع تطبيقي: «بطاقة التعريف الذكية - النسخة المُطوَّرة»

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 03-036-ex3_15.py
"""

# advanced_id_card.py

# استقبال البيانات
full_name: str = input("ما اسمك الكامل؟ ")
birth_year_text: str = input("ما سنة ميلادك؟ ")
height_text: str = input("ما طولك بالمتر؟ (مثال: 1.75) ")
is_employed_text: str = input("هل أنت موظف؟ (نعم/لا) ")

# تحويل الأنواع
birth_year: int = int(birth_year_text)
height: float = float(height_text)
is_employed: bool = is_employed_text.strip() == "نعم"

# الحسابات
CURRENT_YEAR: int = 2025
age: int = CURRENT_YEAR - birth_year

# تحويل bool لنص عربي مفهوم
employment_status: str = "نعم ✅" if is_employed else "لا ❌"

# طباعة البطاقة
print("=" * 35)
print("🪪  بطاقة التعريف الشخصية")
print("=" * 35)
print(f"👤 الاسم: {full_name}")
print(f"🎂 العمر: {age} سنة")
print(f"📏 الطول: {height} متر")
print(f"💼 موظف: {employment_status}")
print("=" * 35)
