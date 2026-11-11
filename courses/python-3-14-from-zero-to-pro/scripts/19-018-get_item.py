#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل التاسع عشر: مسار تطوير الويب - بناء واجهة API لإدارة المهام (FastAPI)
القسم 19.14: ملخص الفصل (Cheat Sheet)

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 19-018-get_item.py
"""

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

app = FastAPI()

class ItemCreate(BaseModel):
    name: str = Field(min_length=1)
    price: float = Field(gt=0)

@app.get("/items/{item_id}")
def get_item(item_id: int, q: str | None = None) -> dict:
    ...

@app.post("/items", status_code=status.HTTP_201_CREATED)
def create_item(item: ItemCreate) -> ItemCreate:
    ...

@app.patch("/items/{item_id}")
def update_item(item_id: int, item: ItemCreate) -> ItemCreate:
    ...

@app.delete("/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_item(item_id: int) -> None:
    ...

# تشغيل:
# uv run uvicorn main:app --reload
# التوثيق: http://127.0.0.1:8000/docs
