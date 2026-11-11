#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل السادس والعشرون: من الكود إلى المنتج (Packaging, Deployment، وخارطة الطريق)
القسم 26.14: مشروع تطبيقي متكامل: من مكتبة محلية إلى منتج منشور بالكامل

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 26-007-test_read_root.py
"""

from fastapi.testclient import TestClient

from textkit_api.main import app

client = TestClient(app)

def test_read_root() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_slugify_endpoint() -> None:
    response = client.post("/slugify", json={"text": "Hello World!"})
    assert response.status_code == 200
    assert response.json() == {"result": "hello-world"}

def test_truncate_endpoint() -> None:
    response = client.post(
        "/truncate", json={"text": "Python is amazing", "max_length": 10}
    )
    assert response.status_code == 200
    assert response.json() == {"result": "Python is..."}
