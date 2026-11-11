#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس: اتخاذ القرار (Control Flow)
القسم 6.9: match-case: الأسلوب الحديث الثوري (Python 3.10+)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 06-022-ex6_9.py
"""

# سيناريو مثالي لـ match-case: فحص قيمة واحدة ضد عدة احتمالات محددة
http_status: int = 404

match http_status:
    case 200:
        print("نجاح")
    case 404:
        print("غير موجود")
    case 500:
        print("خطأ في الخادم")
    case _:
        print("حالة غير معروفة")
