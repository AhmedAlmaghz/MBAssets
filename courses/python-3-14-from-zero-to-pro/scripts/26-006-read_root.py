#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس والعشرون: من الكود إلى المنتج (Packaging, Deployment، وخارطة الطريق)
القسم 26.14: مشروع تطبيقي متكامل: من مكتبة محلية إلى منتج منشور بالكامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 26-006-read_root.py
"""

"""واجهة API بسيطة تستخدم مكتبة textkit."""

from fastapi import FastAPI
from pydantic import BaseModel

from textkit import slugify, truncate

app = FastAPI(
    title="Textkit API",
    description="واجهة ويب لأدوات معالجة النصوص",
    version="0.1.0",
)

class TextRequest(BaseModel):
    text: str

class TruncateRequest(BaseModel):
    text: str
    max_length: int = 50

@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "مرحبًا بك في Textkit API"}

@app.post("/slugify")
def slugify_endpoint(request: TextRequest) -> dict[str, str]:
    return {"result": slugify(request.text)}

@app.post("/truncate")
def truncate_endpoint(request: TruncateRequest) -> dict[str, str]:
    return {"result": truncate(request.text, request.max_length)}
