#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع: التكرار (Loops)
القسم 7.16: مشروع تطبيقي 1: «لعبة تخمين الأرقام»

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 07-027-ex7_16.py
"""

# number_guessing_game.py
"""لعبة تخمين الأرقام - احزر الرقم السري!"""

import random

MAX_ATTEMPTS: int = 7
secret_number: int = random.randint(1, 100)
attempts: int = 0

print("=" * 40)
print("🎯  لعبة تخمين الأرقام")
print("=" * 40)
print(f"خمّن رقمًا بين 1 و100! لديك {MAX_ATTEMPTS} محاولات")

while attempts < MAX_ATTEMPTS:
    guess_text: str = input("\nخمّن الرقم: ").strip()

    if not guess_text.isdigit():
        print("⚠️ من فضلك أدخل رقمًا صحيحًا فقط!")
        continue

    guess: int = int(guess_text)
    attempts += 1

    if guess == secret_number:
        print(f"\n🎉 مبروك! خمّنت الرقم الصحيح في {attempts} محاولة!")
        break
    elif guess < secret_number:
        print("📈 الرقم السري أكبر من تخمينك")
    else:
        print("📉 الرقم السري أصغر من تخمينك")

    remaining: int = MAX_ATTEMPTS - attempts
    if remaining > 0:
        print(f"محاولاتك المتبقية: {remaining}")
else:
    print(f"\n😢 انتهت محاولاتك! الرقم السري كان: {secret_number}")

print("=" * 40)
