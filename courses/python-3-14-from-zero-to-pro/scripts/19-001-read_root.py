#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.4: أول واجهة API: «مرحبًا أيها العالم»

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-001-read_root.py
"""

# main.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "مرحبًا بك في واجهة API!"}
