#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الرابع والعشرون: كتابة كود نظيف (Clean Code)
القسم 24.6: مقدمة مبسّطة لمبادئ SOLID

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 24-014-work.py
"""

# ❌ ينتهك ISP: واجهة ضخمة تُجبر كل شيء على تطبيق كل شيء
class Worker(ABC):
    @abstractmethod
    def work(self) -> str: ...
    @abstractmethod
    def eat(self) -> str: ...

class RobotWorker(Worker):
    def work(self) -> str:
        return "أعمل 24 ساعة"

    def eat(self) -> str:
        raise NotImplementedError("الروبوت لا يأكل!")  # إجبار غير منطقي

# ✅ يتبع ISP: واجهات صغيرة ومتخصصة، يطبّق كل صنف ما يحتاجه فقط
class Workable(ABC):
    @abstractmethod
    def work(self) -> str: ...

class Eatable(ABC):
    @abstractmethod
    def eat(self) -> str: ...

class HumanWorker(Workable, Eatable):
    def work(self) -> str:
        return "أعمل 8 ساعات"

    def eat(self) -> str:
        return "أتناول الغداء"

class RobotWorker(Workable):  # لا حاجة لتطبيق eat() أبدًا
    def work(self) -> str:
        return "أعمل 24 ساعة"
