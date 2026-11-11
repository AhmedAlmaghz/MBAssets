#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الثامن: هياكل البيانات - القوائم (Lists)
القسم 8.16: مشروع تطبيقي: «نظام إدارة قائمة مهام (To-Do List)»

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 08-044-display_menu.py
"""

# todo_list.py
"""نظام إدارة قائمة مهام تفاعلي بسيط."""

tasks: list[str] = []

def display_menu() -> None:
    print("\n" + "=" * 35)
    print("📋  قائمة المهام")
    print("=" * 35)
    print("1. إضافة مهمة جديدة")
    print("2. عرض كل المهام")
    print("3. حذف مهمة")
    print("4. إنهاء البرنامج")
    print("=" * 35)

while True:
    display_menu()
    choice: str = input("اختر رقمًا: ").strip()

    if choice == "1":
        new_task: str = input("اكتب المهمة الجديدة: ").strip()
        if new_task:
            tasks.append(new_task)
            print(f"✅ تمت إضافة: {new_task}")
        else:
            print("⚠️ لا يمكن إضافة مهمة فارغة!")

    elif choice == "2":
        if not tasks:
            print("📭 لا توجد مهام حاليًا!")
        else:
            print("\n📌 مهامك الحالية:")
            for index, task in enumerate(tasks, start=1):
                print(f"  {index}. {task}")

    elif choice == "3":
        if not tasks:
            print("📭 لا توجد مهام لحذفها!")
        else:
            for index, task in enumerate(tasks, start=1):
                print(f"  {index}. {task}")

            task_number: str = input("أدخل رقم المهمة المراد حذفها: ").strip()

            if task_number.isdigit() and 1 <= int(task_number) <= len(tasks):
                removed: str = tasks.pop(int(task_number) - 1)
                print(f"🗑️ تم حذف: {removed}")
            else:
                print("⚠️ رقم غير صالح!")

    elif choice == "4":
        print("👋 إلى اللقاء!")
        break

    else:
        print("⚠️ اختيار غير صحيح، حاول مرة أخرى")
