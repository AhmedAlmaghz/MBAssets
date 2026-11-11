#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السابع عشر: البرمجة غير المتزامنة - مقدمة عملية (async/await)
القسم 17.10: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 17-013-get_data.py
"""

async def get_data() -> str:
    return "بيانات"

def main():
    result = get_data()
    print(result)

main()
