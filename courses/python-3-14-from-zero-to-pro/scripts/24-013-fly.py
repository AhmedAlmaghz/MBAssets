#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.6: مقدمة مبسّطة لمبادئ SOLID

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-013-fly.py
"""

# ❌ ينتهك LSP: الصنف الفرعي يُغيّر السلوك المتوقع بشكل مفاجئ
class Bird:
    def fly(self) -> str:
        return "أطير في السماء"

class Penguin(Bird):
    def fly(self) -> str:
        raise NotImplementedError("البطريق لا يطير!")  # مفاجأة سيئة لأي كود يتوقع fly()

# ✅ يتبع LSP: إعادة تصميم الهرمية لتعكس الواقع بدقة
class Bird(ABC):
    @abstractmethod
    def move(self) -> str:
        ...

class FlyingBird(Bird):
    def move(self) -> str:
        return "أطير في السماء"

class Penguin(Bird):
    def move(self) -> str:
        return "أسبح في الماء"  # سلوك صحيح ومتوقع، بدون كسر توقعات المستخدم
