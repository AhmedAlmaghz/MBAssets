#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.6: مقدمة مبسّطة لمبادئ SOLID

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-015-__init__.py
"""

# ❌ ينتهك DIP: الصنف مرتبط مباشرة بتفصيل محدد (MySQL)
class UserRepository:
    def __init__(self):
        self.db = MySQLDatabase()  # اعتماد مباشر وصلب على تفصيل محدد

    def get_user(self, user_id: int):
        return self.db.query(f"SELECT * FROM users WHERE id={user_id}")

# ✅ يتبع DIP: الاعتماد على تجريد (Protocol كما تعلمنا في الفصل 15)
from typing import Protocol

class Database(Protocol):
    def query(self, sql: str) -> list: ...

class UserRepository:
    def __init__(self, db: Database):  # نعتمد على أي شيء يطابق الواجهة
        self.db = db

    def get_user(self, user_id: int):
        return self.db.query(f"SELECT * FROM users WHERE id={user_id}")

# الآن يمكن تمرير MySQL أو PostgreSQL أو حتى قاعدة بيانات وهمية للاختبار
repo = UserRepository(db=MySQLDatabase())
repo_test = UserRepository(db=FakeDatabaseForTesting())
