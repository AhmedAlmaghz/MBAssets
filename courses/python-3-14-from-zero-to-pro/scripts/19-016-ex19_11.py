#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.11: أفضل الممارسات

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-016-ex19_11.py
"""

status.HTTP_200_OK          # نجاح عام
status.HTTP_201_CREATED     # بعد إنشاء مورد جديد (POST)
status.HTTP_204_NO_CONTENT  # نجاح بدون محتوى مُرجَع (DELETE)
status.HTTP_404_NOT_FOUND   # المورد غير موجود
status.HTTP_422_UNPROCESSABLE_ENTITY  # بيانات غير صالحة (يتم تلقائيًا من Pydantic)
