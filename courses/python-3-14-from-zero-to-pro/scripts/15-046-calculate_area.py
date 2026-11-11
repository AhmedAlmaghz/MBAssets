#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
الفصل الخامس عشر: Type Hints — نحو برمجة أكثر وضوحًا ودقة (باستخدام mypy)
القسم 15.15: تمارين

المصدر: كتاب «بايثون 3.14 — من الصفر إلى الاحتراف»
الملف: 15-046-calculate_area.py
"""

# تمرين 1
def calculate_area(width: float, height: float) -> float:
    return width * height

# تمرين 2
from typing import Literal

Grade = Literal["ممتاز", "جيد", "راسب"]

def get_grade(score: int) -> Grade:
    if score >= 90:
        return "ممتاز"
    elif score >= 60:
        return "جيد"
    else:
        return "راسب"

# تمرين 3
# لا فرق في المعنى: كلاهما يعني "قد يكون int أو None".
# Optional[int] هي الصيغة القديمة (تحتاج: from typing import Optional).
# int | None هي الصيغة الحديثة الموصى بها في Python 3.10+ (لا تحتاج استيراد).
# في كتابنا نستخدم دائمًا: int | None
